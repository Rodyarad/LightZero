"""
This script is an extension of the original transformer.py from karpathy/nanoGPT.
It incorporates LoRA (Low-Rank Adaptation) for fine-tuning and introduces a
Curriculum Learning mechanism that activates different LoRA adapters sequentially.

Key features:
- Adds `CurriculumLoRALinear`, a custom linear layer with multiple LoRA adapters.
- Controls which modules to apply LoRA to via configuration (e.g., attention and feed-forward layers).
- Maintains the extensibility and readability of the original nanoGPT codebase.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from ding.torch_utils.network import GRUGatingUnit
from einops import rearrange
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache, SlotKVCache
from lzero.model.common import SimNorm
from .action import ContinuousActionAdapter, DiscreteActionAdapter


class LearnableScale(nn.Module):
    """
    A learnable scalar parameter constrained within a specific range.

    The formula `s = offset + scale * tanh(ŝ)` maps an unbounded logit `ŝ`
    to the range (offset - scale, offset + scale). Using tanh can sometimes
    provide more stable gradients than sigmoid.

    For example, to achieve a range of (0.8, 1.2), one would use
    `init=1.0` and `s_range=0.2`.
    """

    def __init__(self, init: float = 1.0, s_range: float = 0.2) -> None:
        """
        Overview:
            Initializes the LearnableScale module.
        Arguments:
            - init (:obj:`float`): The initial value of the scalar, which also serves as the center of the range.
            - s_range (:obj:`float`): The scale factor that determines the range (init - s_range, init + s_range).
        """
        super().__init__()
        assert s_range > 0, "The scaling range must be positive."
        self.offset = init
        self.scale = s_range

        # Initialize the logit to 0, so the initial output is exactly `init`.
        self.logit = nn.Parameter(torch.tensor(0.0))
        # TODO: Initially frozen, activated by a CurriculumController.
        self.logit.requires_grad = False

    def forward(self) -> torch.Tensor:
        """
        Overview:
            Computes the scaled value.
        Returns:
            - torch.Tensor: The learnable scalar, constrained to the specified range.
        """
        return self.offset + self.scale * torch.tanh(self.logit)

##############################################
# Optimized CurriculumLoRALinear Implementation (Recommended Version)
##############################################

class CurriculumLoRALinear(nn.Module):
    """
    Optimized CurriculumLoRALinear.
    
    Effective weight at stage s:
    W_eff = α₀*W₀ + Σ_{j=1 to s} αⱼ*Δθⱼ

    Optimization logic at stage s (s >= 1):
    - Train: Δθₛ, α₀, and {αⱼ | 1 <= j < s}
    - Freeze: W₀, {Δθⱼ | 1 <= j < s}, and αₛ
    
    This avoids the redundancy of training αₛ alongside Δθₛ.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
                 curriculum_stage_num: int = 1, lora_scale_init: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.curriculum_stage_num = curriculum_stage_num
        self.curriculum_stage = 0

        # Base weights (W₀ and bias)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Learnable scale for the base weight (α₀)
        self.base_weight_scale = LearnableScale(init=1.0, s_range=0.2)
        
        # A scale for each adapter (α₁, α₂, ...)
        self.adapters = nn.ModuleList()
        self.adapter_scales = nn.ModuleList()

        if r > 0 and (curriculum_stage_num - 1) > 0:
            for _ in range(curriculum_stage_num - 1):
                adapter = nn.ParameterDict({
                    'lora_A': nn.Parameter(torch.randn(r, in_features) * 0.01),
                    'lora_B': nn.Parameter(torch.zeros(out_features, r))
                })
                self.adapters.append(adapter)
                self.adapter_scales.append(LearnableScale(lora_scale_init, s_range=0.2))
        else:
            self.adapters = None

        self.set_curriculum_stage(0)

    def set_curriculum_stage(self, stage: int) -> None:
        assert 0 <= stage < self.curriculum_stage_num, f"Stage must be within [0, {self.curriculum_stage_num-1}]"
        self.curriculum_stage = stage
        module_id = f"({self.in_features}x{self.out_features})"
        
        # --- Stage 0: Base Training ---
        if stage == 0:
            self.weight.requires_grad = True
            if self.bias is not None: self.bias.requires_grad = True
            
            # Freeze everything else
            self.base_weight_scale.logit.requires_grad = False
            if self.adapters:
                for adapter in self.adapters:
                    adapter['lora_A'].requires_grad = False
                    adapter['lora_B'].requires_grad = False
                for scale in self.adapter_scales:
                    scale.logit.requires_grad = False

            # Log only from rank 0 to avoid excessive output
            from ding.utils import get_rank
            if get_rank() == 0:
                logging.info(f"[CurriculumLoRALinear {module_id}] Stage 0: Base layer trainable.")

        # --- Stage >= 1: Adaptation ---
        else:
            # Freeze base model
            self.weight.requires_grad = False
            if self.bias is not None: self.bias.requires_grad = False
            
            # α₀ is trainable from stage 1 onwards
            self.base_weight_scale.logit.requires_grad = True
            
            if self.adapters:
                # Set trainability for LoRA adapters
                for idx, adapter in enumerate(self.adapters):
                    is_current_adapter = (idx == stage - 1)
                    adapter['lora_A'].requires_grad = is_current_adapter
                    adapter['lora_B'].requires_grad = is_current_adapter
                
                # --- OPTIMIZED LOGIC FOR SCALES ---
                # Set trainability for adapter scales {α_j}
                for idx, scale in enumerate(self.adapter_scales):
                    # A scale α_j is trainable if it belongs to a *previous* stage (j < s).
                    # The current stage's scale α_s (idx = stage - 1) is NOT trained.
                    is_previous_scale = (idx < stage - 1)
                    scale.logit.requires_grad = is_previous_scale

            # Log only from rank 0 to avoid excessive output
            from ding.utils import get_rank
            if get_rank() == 0:
                logging.info(f"[CurriculumLoRALinear {module_id}] Stage {stage}: Activating adapter {stage - 1} and scales for stages < {stage - 1}.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply scaling to base weight if in an adaptation stage
        if self.curriculum_stage > 0:
            alpha_0 = self.base_weight_scale()
            scaled_weight = self.weight * alpha_0
            baseline_out = F.linear(x, scaled_weight, self.bias)
        else:
            baseline_out = F.linear(x, self.weight, self.bias)

        if self.curriculum_stage == 0 or self.adapters is None:
            return baseline_out

        adapter_out = 0
        # Iterate through all adapters up to the current stage
        for idx in range(self.curriculum_stage):
            if idx >= len(self.adapters):
                break
            
            adapter = self.adapters[idx]
            scale = self.adapter_scales[idx]()
            
            lora_x = self.lora_dropout(x)
            out = F.linear(lora_x, adapter['lora_A'])
            out = F.linear(out, adapter['lora_B'])
            
            # The forward pass is a simple sum. The magic happens in `set_curriculum_stage`
            # which controls `requires_grad`. No need for `.detach()` here.
            # Gradients will naturally flow only to parameters with `requires_grad=True`.
            adapter_out = adapter_out + self.scaling * out * scale

        return baseline_out + adapter_out
    

# ##############################################
# # CurriculumLoRALinear Implementation
# ##############################################

# class CurriculumLoRALinear(nn.Module):
#     """
#     CurriculumLoRALinear extends a standard linear layer with curriculum-based LoRA adapters.

#     This module internally stores a base weight and bias. It also initializes multiple
#     LoRA adapters (number = curriculum_stage_num - 1), which are activated sequentially.

#     Forward pass logic:
#     - If `curriculum_stage == 0`:
#         Output = F.linear(x, W, bias)
#     - If `curriculum_stage >= 1`:
#         Output = base_output + sum_{i=0}^{curriculum_stage-1} scaling * adapter_i(x)
#       where only the adapter for the current stage (index == curriculum_stage - 1) is trainable.
#       Previous adapters contribute to the forward pass but their gradients are detached.

#     Note:
#     - The `set_curriculum_stage(stage)` method must be called externally to switch between stages.
#     - Logging messages indicate the module's dimensions and the freeze/unfreeze status of its parameters.
#     """

#     def __init__(self, in_features: int, out_features: int, bias: bool = True,
#                  r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
#                  curriculum_stage_num: int = 1, lora_scale_init: float = 1.0) -> None:
#         """
#         Overview:
#             Initializes the CurriculumLoRALinear layer. If `curriculum_stage_num > 1`,
#             it creates `curriculum_stage_num - 1` LoRA adapters.
#         Arguments:
#             - in_features (:obj:`int`): Size of each input sample.
#             - out_features (:obj:`int`): Size of each output sample.
#             - bias (:obj:`bool`): If True, adds a learnable bias to the output.
#             - r (:obj:`int`): The rank of the LoRA decomposition. If 0, LoRA is disabled.
#             - lora_alpha (:obj:`int`): The alpha parameter for LoRA scaling.
#             - lora_dropout (:obj:`float`): The dropout probability for LoRA layers.
#             - curriculum_stage_num (:obj:`int`): The total number of curriculum stages.
#             - lora_scale_init (:obj:`float`): The initial value for the learnable scale of each adapter.
#         """
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.scaling = lora_alpha / r if r > 0 else 1.0
#         self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
#         self.curriculum_stage_num = curriculum_stage_num
#         self.curriculum_stage = 0  # Initial stage is 0

#         # Initialize base weights (part of the base transformer), trainable by default
#         self.weight = nn.Parameter(torch.empty(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_features))
#         else:
#             self.register_parameter('bias', None)
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#         # Initialize LoRA adapters, which exist only if r > 0 and curriculum_stage_num > 1
#         self.adapters = nn.ModuleList()
#         self.adapter_scales = nn.ModuleList()

#         if r > 0 and (curriculum_stage_num - 1) > 0:
#             for _ in range(curriculum_stage_num - 1):
#                 adapter = nn.ParameterDict({
#                     'lora_A': nn.Parameter(torch.randn(r, in_features) * 0.01),
#                     'lora_B': nn.Parameter(torch.zeros(out_features, r))
#                 })
#                 self.adapters.append(adapter)
#                 self.adapter_scales.append(LearnableScale(lora_scale_init, s_range=0.2))

#         else:
#             self.adapters = None

#         # Initially (stage 0), the base layer is trainable, and all adapters are frozen
#         self.weight.requires_grad = True
#         if self.bias is not None:
#             self.bias.requires_grad = True
#         if self.adapters is not None:
#             for adapter in self.adapters:
#                 adapter['lora_A'].requires_grad = False
#                 adapter['lora_B'].requires_grad = False

#     def set_curriculum_stage(self, stage: int) -> None:
#         """
#         Overview:
#             Sets the current curriculum stage and updates the `requires_grad` status of parameters accordingly.
#             - Stage 0: The base layer is trainable; all adapters are frozen.
#             - Stage >= 1: The base layer is frozen. Only the current adapter (index = stage - 1) is trainable.
#                           Previous adapters contribute to the forward pass but do not propagate gradients.
#         Arguments:
#             - stage (:obj:`int`): The curriculum stage to set, in the range [0, curriculum_stage_num - 1].
#         """
#         assert 0 <= stage < self.curriculum_stage_num, f"Stage must be within [0, {self.curriculum_stage_num-1}]"
#         self.curriculum_stage = stage

#         module_id = f"({self.in_features}x{self.out_features})"
#         if stage == 0:
#             self.weight.requires_grad = True
#             if self.bias is not None:
#                 self.bias.requires_grad = True
#             if self.adapters is not None:
#                 for adapter in self.adapters:
#                     adapter['lora_A'].requires_grad = False
#                     adapter['lora_B'].requires_grad = False
#             logging.info(f"[CurriculumLoRALinear {module_id}] Stage 0: Base layer is trainable, all adapters are frozen.")
#         else:
#             # For stages > 0, freeze the base layer
#             self.weight.requires_grad = False
#             if self.bias is not None:
#                 self.bias.requires_grad = False
            
#             if self.adapters is not None:
#                 for idx, adapter in enumerate(self.adapters):
#                     is_current_adapter = (idx == stage - 1)
#                     adapter['lora_A'].requires_grad = is_current_adapter
#                     adapter['lora_B'].requires_grad = is_current_adapter
#                     status = "activated (trainable)" if is_current_adapter else "frozen (forward-only)"
#                     logging.info(f"[CurriculumLoRALinear {module_id}] Stage {stage}: Adapter {idx} is {status}.")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Overview:
#             Performs the forward pass of the CurriculumLoRALinear layer.
#         Arguments:
#             - x (:obj:`torch.Tensor`): The input tensor.
#         Returns:
#             - torch.Tensor: The output tensor.
#         """
#         baseline_out = F.linear(x, self.weight, self.bias)
#         if self.curriculum_stage == 0 or self.adapters is None:
#             return baseline_out

#         adapter_out = 0
#         # For the first `curriculum_stage` adapters, only the last one backpropagates.
#         # Others are detached to contribute only to the forward pass.
#         for idx in range(self.curriculum_stage):
#             if idx >= len(self.adapters):
#                 break
#             adapter = self.adapters[idx]
#             lora_x = self.lora_dropout(x)
#             out = F.linear(lora_x, adapter['lora_A'])
#             out = F.linear(out, adapter['lora_B'])
            
#             scale = self.adapter_scales[idx]()

#             # NOTE: All adapter scales are currently trainable.
#             if idx == self.curriculum_stage - 1:
#                 # Only the current adapter's output contributes to the gradient computation.
#                 adapter_out = adapter_out + self.scaling * out * scale
#             else:
#                 # Outputs from previous adapters are detached.
#                 adapter_out = adapter_out + self.scaling * out.detach() * scale

#         return baseline_out + adapter_out


##############################################
# Helper function to wrap linear layers
##############################################

def _maybe_wrap_linear(linear: nn.Linear, config, module_label: str) -> nn.Module:
    """
    Overview:
        A helper function that wraps an `nn.Linear` layer with `CurriculumLoRALinear`
        if LoRA and curriculum learning are enabled for the specified module.
    Arguments:
        - linear (:obj:`nn.Linear`): The original linear layer to be potentially wrapped.
        - config: The model configuration object.
        - module_label (:obj:`str`): A label identifying the module type (e.g., "attn", "feed_forward").
    Returns:
        - nn.Module: The wrapped `CurriculumLoRALinear` layer or the original `nn.Linear` layer.
    """
    use_curriculum_lora = (
        config.lora_r > 0 and
        module_label in config.lora_target_modules and
        getattr(config, "curriculum_stage_num", 1) > 1
    )
    if use_curriculum_lora:
        new_linear = CurriculumLoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            curriculum_stage_num=config.curriculum_stage_num,
            lora_scale_init=config.lora_scale_init
        )
        new_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.data.copy_(linear.bias.data)
        return new_linear
    else:
        return linear


##############################################
# Helper function to set curriculum stage
##############################################

def set_curriculum_stage(model: nn.Module, stage: int) -> None:
    """
    Overview:
        Recursively traverses all submodules of a given model, finds all instances
        of `CurriculumLoRALinear`, and calls their `set_curriculum_stage` method.
        This function is generic and can be applied to any model structure.
    Arguments:
        - model (:obj:`nn.Module`): The model to update (e.g., a Transformer or Vision Transformer).
        - stage (:obj:`int`): The curriculum stage to set.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, CurriculumLoRALinear):
            module.set_curriculum_stage(stage)
            count += 1

    # Log only from rank 0 to avoid excessive output
    from ding.utils import get_rank
    if count > 0 and get_rank() == 0:
        logging.info(f"[Curriculum] Updated {count} CurriculumLoRALinear modules in {type(model).__name__} to stage {stage}.")

# Alias for backward compatibility
set_curriculum_stage_for_transformer = set_curriculum_stage


##############################################
# Transformer Configuration
##############################################
@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    # LoRA parameters
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_target_modules: list = None

    # Curriculum Learning parameters
    # `curriculum_stage_num` is the total number of stages (e.g., 3 means stages 0, 1, 2)
    curriculum_stage_num: int = 1  # 1 (base) + number of available LoRA adapters
    min_stage0_iters: int = 10_000     # Minimum iterations for stage 0
    max_stage_iters: int = 20_000     # Maximum iterations per stage
    lora_scale_init: float = 1.0      # Initial value for learnable adapter scales

    # Other configurations
    task_embed_option: str = "none"
    register_token_num: int = 4
    register_token_shared: bool = True

    gru_gating: bool = False
    moe_in_transformer: bool = False
    multiplication_moe_in_transformer: bool = False
    num_experts_of_moe_in_transformer: int = 1

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the model can handle."""
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    """
    A Transformer model implementation.
    """

    def __init__(self, config: TransformerConfig, task_embed: Optional[nn.Module] = None) -> None:
        """
        Overview:
            Initializes the Transformer model.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration object for the model.
            - task_embed (:obj:`Optional[nn.Module]`): An optional module for generating task embeddings.
        """
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.task_embed = task_embed
        self.task_embed_option = self.config.task_embed_option
        self.use_register_token = (self.task_embed_option == "register_task_embed")

        if self.use_register_token:
            self.register_token_num = getattr(config, "register_token_num", 4)
            self.register_token_shared = getattr(config, "register_token_shared", True)
            
            if self.register_token_shared:
                # Shared mode: all tasks use the same register_tokens parameter.
                self.register_tokens = nn.Parameter(torch.empty(self.register_token_num, config.embed_dim))
                nn.init.xavier_uniform_(self.register_tokens)
            else:
                # Non-shared mode: relies on the external `task_embed` module to generate
                # task-specific embeddings, which are then normalized and expanded.
                self.task_embed = task_embed
                self.sim_norm = SimNorm(simnorm_dim=config.embed_dim)

    def add_register_tokens(self, sequences: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Overview:
            Prepends or appends register tokens to the input sequences.
        Arguments:
            - sequences (:obj:`torch.Tensor`): The input sequences, with shape (B, T, C).
            - task_id (:obj:`int`): The ID of the current task.
        Returns:
            - torch.Tensor: The sequences with register tokens concatenated, shape (B, T + register_token_num, C).
        """
        B = sequences.size(0)
        device = sequences.device

        if self.register_token_shared:
            # Shared mode: use the same set of register tokens for all batches.
            register_tokens = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            # Non-shared mode: dynamically generate task embedding and expand it.
            task_embedding = self.task_embed(torch.tensor([task_id], device=device))
            task_embedding = self.sim_norm(task_embedding.view(1, -1)).view(-1)
            register_tokens = task_embedding.unsqueeze(0).expand(self.register_token_num, -1)
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)

        # Concatenate register tokens at the end of the sequence.
        new_sequences = torch.cat([sequences, register_tokens], dim=1)
        return new_sequences

    def remove_register_tokens_from_kv(self, past_keys_values: Optional[KeysValues]) -> None:
        """
        Overview:
            Removes the register tokens from the key-value cache of all layers.
            This is called at the end of the forward pass during inference.
        Arguments:
            - past_keys_values (:obj:`Optional[KeysValues]`): The key-value cache.
        """
        if past_keys_values is not None:
            past_keys_values.remove_register_tokens(self.register_token_num)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        Overview:
            Generates a placeholder for the key-value cache.
        Arguments:
            - n (:obj:`int`): The batch size.
            - max_tokens (:obj:`int`): The maximum number of tokens in the sequence.
        Returns:
            - KeysValues: An object containing empty tensors for keys and values.
        """
        device = self.ln_f.weight.device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(
        self,
        sequences: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        task_id: int = 0,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass of the Transformer model.
        Arguments:
            - sequences (:obj:`torch.Tensor`): The input tensor of shape (B, T, C).
            - past_keys_values (:obj:`Optional[KeysValues]`): An optional cache for keys and values to speed up inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Tensor indicating the valid length of the context for each sample.
            - task_id (:obj:`int`): The ID of the current task.
            - start_pos (:obj:`int`): The starting position for the current sequence (used with kv-caching).
        Returns:
            - torch.Tensor: The output tensor of shape (B, T, C).
        """
        if self.use_register_token:
            sequences = self.add_register_tokens(sequences, task_id)

        x = self.drop(sequences)

        for i, block in enumerate(self.blocks):
            kv_cache_layer = None if past_keys_values is None else past_keys_values[i]
            x = block(x, kv_cache_layer, valid_context_lengths)

        x = self.ln_f(x)

        if self.use_register_token:
            # During inference, remove register tokens from the KV cache to maintain consistency
            # for external logic that does not expect them.
            if past_keys_values is not None:
                self.remove_register_tokens_from_kv(past_keys_values)
            
            # TODO: Remove register tokens from the final output to match the input sequence length.
            x = x[:, :-self.register_token_num, :]

        return x


class Block(nn.Module):
    """
    A single Transformer block, consisting of self-attention and a feed-forward network.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Overview:
            Initializes a Transformer block.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration object for the block.
        """
        super().__init__()
        self.gru_gating = config.gru_gating
        if self.gru_gating:
            # As in GTrXL, for stabilizing training with recurrence
            self.gate1 = GRUGatingUnit(config.embed_dim, bias_init=2.0)
            self.gate2 = GRUGatingUnit(config.embed_dim, bias_init=2.0)

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)

        if config.moe_in_transformer:
            from .moe import MoELayer
            # Create multiple independent MLP instances as experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.embed_dim, 4 * config.embed_dim),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(4 * config.embed_dim, config.embed_dim),
                    nn.Dropout(config.resid_pdrop),
                ) for _ in range(config.num_experts_of_moe_in_transformer)
            ])
            self.feed_forward = MoELayer(
                config,
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=config.num_experts_per_tok,
            )
            # Log only from rank 0 to avoid excessive output
            from ding.utils import get_rank
            if get_rank() == 0:
                logging.info(f"Using MoE in transformer feed-forward with {config.num_experts_of_moe_in_transformer} experts.")
        elif config.multiplication_moe_in_transformer:
            from .moe import MoELayer, MultiplicationFeedForward
            # Create multiple FeedForward instances for multiplication-based MoE
            self.experts = nn.ModuleList([
                MultiplicationFeedForward(config) for _ in range(config.num_experts_of_moe_in_transformer)
            ])
            self.feed_forward = MoELayer(
                config,
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=config.num_experts_per_tok,
            )
            # Log only from rank 0 to avoid excessive output
            from ding.utils import get_rank
            if get_rank() == 0:
                logging.info(f"Using Multiplication MoE in transformer feed-forward with {config.num_experts_of_moe_in_transformer} experts.")
        else:
            # Standard MLP, with linear layers potentially wrapped for LoRA.
            self.feed_forward = nn.Sequential(
                _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim), config, "feed_forward"),
                nn.GELU(approximate='tanh'),
                _maybe_wrap_linear(nn.Linear(4 * config.embed_dim, config.embed_dim), config, "feed_forward"),
                nn.Dropout(config.resid_pdrop),
            )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass of the Transformer block.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking.
        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        attn_output = self.attn(self.ln1(x), past_keys_values, valid_context_lengths)
        if self.gru_gating:
            x = self.gate1(x, attn_output)
            ff_output = self.feed_forward(self.ln2(x))
            x = self.gate2(x, ff_output)
        else:
            x = x + attn_output
            x = x + self.feed_forward(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism for a Transformer.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Overview:
            Initializes the SelfAttention module.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration object for the attention module.
        """
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        
        self.task_embed_option = self.config.task_embed_option
        self.use_register_token = (self.task_embed_option == "register_task_embed")
        if self.use_register_token:
            self.register_token_num = getattr(config, "register_token_num", 4)

        # Wrap linear layers if LoRA is enabled for the attention module
        self.key = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.query = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.value = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        self.proj = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # TODO: The mask size is conservatively large to accommodate register tokens.
        # This could be made more dynamic.
        mask_size = config.max_tokens
        if self.use_register_token:
            mask_size += self.register_token_num * 5
        causal_mask = torch.tril(torch.ones(mask_size, mask_size))
        self.register_buffer('mask', causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass for the self-attention mechanism.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (B, T, C).
            - kv_cache (:obj:`Optional[KeysValues]`): Optional key-value cache for faster inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Optional tensor containing valid context lengths.
        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        head_size = C // self.num_heads
        
        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.shape[2]

        q = self.query(x).view(B, T, self.num_heads, head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, head_size).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        current_len = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Construct the attention mask
        mask = self.mask[past_len:past_len + T, :current_len]

        if valid_context_lengths is not None:
            # This logic is for a specific use case and may need adjustment.
            # It creates a custom mask for each item in the batch.
            batch_mask = torch.zeros(B, T, current_len, device=att.device)
            for i in range(B):
                batch_mask[i] = mask.clone()
                # Zero out attention to invalid past context
                batch_mask[i, :, :(past_len - valid_context_lengths[i])] = 0
            mask = batch_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Adjust mask for register tokens if they are in use
        if self.use_register_token and self.register_token_num > 0:
            # Allow all positions to attend to register tokens and vice-versa
            register_mask = mask.clone()
            # Register tokens are at the end of the sequence
            register_indices_start = current_len - self.register_token_num
            register_mask[..., register_indices_start:] = 1  # All can see registers
            # This part is more complex if T is not the full sequence length
            if T > self.register_token_num:
                 # Only the actual register tokens in the current input `x` can see everything
                 register_mask[..., -self.register_token_num:, :] = 1
            mask = register_mask
            
            if kv_cache is not None:
                # Ensure mask dimensions match the potentially smaller KV cache length
                new_L = kv_cache.shape[2]
                mask = mask[..., :new_L]

        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))

        return y

    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Computes the attention map for visualization, without computing the final output.
        Arguments:
            - x (:obj:`torch.Tensor`): Input sequence with shape (B, T, C).
            - kv_cache (:obj:`Optional[KeysValues]`): Cached keys and values for long sequence inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for variable-length inputs.
        Returns:
            - torch.Tensor: Attention map of shape (B, num_heads, T, L + T).
        """
        B, T, C = x.size()
        head_size = C // self.num_heads

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.shape[2]

        q = self.query(x).view(B, T, self.num_heads, head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, head_size).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        current_len = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        mask = self.mask[past_len:past_len + T, :current_len]
        if valid_context_lengths is not None:
            batch_mask = torch.zeros(B, T, current_len, device=att.device)
            for i in range(B):
                batch_mask[i] = mask.clone()
                batch_mask[i, :, :(past_len - valid_context_lengths[i])] = 0
            mask = batch_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att


@dataclass
class SlotTransformerConfig:
    """
    Configuration for the SlotTransformer and ObjectWorldModel.
    
    This config consolidates all parameters needed for:
    - SlotTransformer architecture
    - ObjectWorldModel training and inference
    - Auxiliary transformers (dynamics, prediction)
    """
    num_slots: int = 7
    slots_dim: int = 64
    tokens_dim: int = 256
    max_timestep: int = 100
    
    action_space_size: int = 6
    action_type: str = 'discrete'
    continuous_action_space: bool = False
    
    num_heads: int = 4
    num_layers: int = 3
    hidden_mult: int = 4
    embed_dim: int = 256

    use_time_encoding: bool = True
    use_actions: bool = True
    use_slot_temporal_block: bool = True
    use_slot_interaction_block: bool = True
    
    dynamics_transformer_blocks: int = 2
    dynamics_transformer_heads: int = 4
    dynamics_transformer_dropout: float = 0.0
    
    prediction_transformer_blocks: int = 2
    prediction_transformer_heads: int = 4
    prediction_transformer_dropout: float = 0.0
    
    env_num: int = 8
    device: str = 'cuda'
    gamma: float = 0.997
    context_length: int = 100
    max_cache_size: int = 5000
    
    policy_entropy_weight: float = 5e-3
    predict_latent_loss_type: str = 'group_kl'
    latent_recon_loss_weight: float = 0.0
    perceptual_loss_weight: float = 0.0
    
    support_size: int = 101

    group_size: int = 8
    norm_type: str = 'BN'
    
    game_segment_length: int = 20
    num_simulations: int = 50
    num_unroll_steps: int = 5
    
    use_policy_logits_clip: bool = False
    policy_logits_clip_min: float = -10.0
    policy_logits_clip_max: float = 10.0
    use_policy_loss_temperature: bool = False
    policy_loss_temperature: float = 1.0
    use_target_policy_resmooth: bool = False
    target_policy_resmooth_eps: float = 0.05
    
    use_priority: bool = False
    
    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens."""
        return self.tokens_per_block * self.max_blocks


def get_sin_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]

class CrossAttention(nn.Module):
    """
    Cross-attention module for SlotTransformer action attention with KV caching support.
    Based on the existing SelfAttention implementation but adapted for cross-attention.
    
    Query comes from slots, Key/Value come from actions.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, 
                 max_tokens: int = 100) -> None:
        """
        Overview:
            Initializes the SlotCrossAttention module.
        Arguments:
            - embed_dim (:obj:`int`): The dimension of embeddings.
            - num_heads (:obj:`int`): The number of attention heads.
            - dropout (:obj:`float`): Dropout probability.
            - max_tokens (:obj:`int`): Maximum number of tokens for the causal mask.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection layers for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
    
    def forward(
        self, 
        query_input: torch.Tensor,  # Slots: (B, T_q, C)
        key_value_input: torch.Tensor,  # Actions: (B, T_kv, C)
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        valid_context_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Overview:
            Performs cross-attention with optional KV caching.
        Arguments:
            - query_input (:obj:`torch.Tensor`): Query tensor from slots, shape (B, T_q, C).
            - key_value_input (:obj:`torch.Tensor`): Key/Value tensor from actions, shape (B, T_kv, C).
            - kv_cache (:obj:`Optional[KVCache]`): Optional KV cache for caching action keys/values.
            - attn_mask (:obj:`Optional[torch.Tensor]`): Optional attention mask.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for each batch element.
        Returns:
            - torch.Tensor: Output tensor of shape (B, T_q, C).
        """
        B, T_q, C = query_input.size()
        T_kv = key_value_input.size(1)
        
        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.shape[2]
        
        # Compute Q from slots
        q = self.query(query_input).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute K, V from actions
        k = self.key(key_value_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(key_value_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use KV cache if available
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply mask
        att = att + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply valid_context_lengths mask if provided
        if valid_context_lengths is not None:
            # Create batch-specific mask: block attention to invalid past context
            for i in range(B):
                invalid_len = past_len - valid_context_lengths[i]
                if invalid_len > 0:
                    att[i, :, :, :invalid_len] = float('-inf')
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))
        
        return y


class CompasParLayer(nn.TransformerEncoderLayer):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    d_model: int
        Dimensionality of the input tokens
    nhead: int
        Number of heads in multi-head attention
    dim_feedforward: int
        Hidden dimension in the MLP
    dropout: float
        Amount of dropout to apply. Default is 0.1
    activation: int
        Nonlinear activation in the MLP. Default is ReLU
    layer_norm_eps: int
        Epsilon value in the layer normalization components
    batch_first: int
        If True, shape is (B, num_tokens, token_dim); otherwise, it is (num_tokens, B, token_dim)
    norm_first: int
        If True, transformer is in mode pre-norm: otherwise, it is post-norm
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=True, device=None, dtype=None,
                 use_actions: bool = True,
                 use_slot_temporal_block: bool = True,
                 use_slot_interaction_block: bool = True,
                 max_timestep: int = 100):
        """
        Module initializer
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )

        self.use_actions = use_actions
        self.use_slot_temporal_block = use_slot_temporal_block
        self.use_slot_interaction_block = use_slot_interaction_block

        self.self_attn_obj = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        
        # Create TransformerConfig for SelfAttention from layer parameters
        temporal_attn_config = TransformerConfig(
            tokens_per_block=max_timestep,
            max_blocks=1,
            attention='causal',
            num_layers=1,
            num_heads=nhead,
            embed_dim=d_model,
            embed_pdrop=0.0,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.self_attn_time = SelfAttention(temporal_attn_config)
        
        # Action attention - use SlotCrossAttention with KV-caching support
        self.act_attn = CrossAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            max_tokens=max_timestep
        )

        self.act_norm = nn.LayerNorm(
            d_model,
            eps=layer_norm_eps,
        )

    def forward(self, src, action, time_mask=None, kv_cache=None, valid_context_lengths=None):
        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        action: torch Tensor
            Action embeddings. Shape is (B, N_imgs, Dim)
        time_mask: torch Tensor, optional
            Temporal mask for causal attention
        kv_cache: tuple, optional
            Tuple of (temporal_cache, action_cache) for KV caching
        valid_context_lengths: torch Tensor, optional
            Valid context lengths for each batch element
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), self.act_norm(action), time_mask, kv_cache, valid_context_lengths)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, action, time_mask, kv_cache, valid_context_lengths))
            x = self.norm2(x + self._ff_block(x))

        return x

    def gen_act_causal_mask(self, slots: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        time_steps = slots.size(1)
        num_slots = slots.size(2)
        total_len = past_len + time_steps
        
        row_indices = torch.arange(time_steps, device=slots.device).unsqueeze(1) + past_len
        col_indices = torch.arange(total_len, device=slots.device).unsqueeze(0)
        causal_mask = col_indices <= row_indices

        attn_mask = torch.zeros(time_steps, total_len, device=slots.device, dtype=slots.dtype)
        attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))
        
        attn_mask = attn_mask.repeat_interleave(num_slots, 0)
        
        return attn_mask

    def _sa_block(self, x, action, time_mask, kv_cache=None, valid_context_lengths=None):
        """
        Forward pass through the parallel attention branches with KV-caching support.
        
        Arguments:
            - x: Slot tokens, shape (B, num_imgs, num_slots, dim)
            - action: Action embeddings, shape (B, num_imgs, dim)
            - time_mask: Optional temporal mask
            - kv_cache: Optional tuple of (temporal_cache, action_cache)
            - valid_context_lengths: Optional tensor of valid context lengths for each batch element
        """
        B, num_imgs, num_slots, dim = x.shape
        temporal_cache, action_cache = kv_cache if kv_cache else (None, None)

        # object-attention
        if self.use_slot_interaction_block:
            x_aux = x.clone().view(B * num_imgs, num_slots, dim)
            x_obj = self.self_attn_obj(
                query=x_aux,
                key=x_aux,
                value=x_aux,
                need_weights=False
            )[0]
            x_obj = x_obj.view(B, num_imgs, num_slots, dim)
        else:
            x_obj = 0

        if self.use_actions:
            # action-attention
            past_len = action_cache.shape[2] if action_cache is not None else 0
            attn_mask = self.gen_act_causal_mask(x, past_len=past_len)
            x_aux = x.clone().view(B, num_imgs * num_slots, dim)
            x_act = self.act_attn(
                query_input=x_aux,
                key_value_input=action,
                kv_cache=action_cache,
                attn_mask=attn_mask,
                valid_context_lengths=valid_context_lengths
            )
            x_act = x_act.view(B, num_imgs, num_slots, dim)
        else:
            x_act = 0

        if self.use_slot_temporal_block:
            # temporal-attention
            # Reshape: process each slot independently over time
            # From (B, num_imgs, num_slots, dim) to (B * num_slots, num_imgs, dim)
            x_reshaped = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
            
            # Expand valid_context_lengths for each slot if provided
            temporal_valid_lengths = None
            if valid_context_lengths is not None:
                # Repeat for each slot: (B,) -> (B * num_slots,)
                temporal_valid_lengths = valid_context_lengths.repeat_interleave(num_slots)
            
            # Use SelfAttention with KV cache
            x_time = self.self_attn_time(
                x=x_reshaped,
                kv_cache=temporal_cache,
                valid_context_lengths=temporal_valid_lengths
            )
            
            # Reshape back: (B * num_slots, num_imgs, dim) -> (B, num_imgs, num_slots, dim)
            x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)
        else:
            x_time = 0

        y = self.dropout1(x_obj + x_time + x_act)
        return y

class SlotTransformer(nn.Module):
    """
    SlotTransformer: Object-centric transformer for slot-based world models.
    
    This transformer processes slot representations with temporal, object-interaction,
    and action attention mechanisms.
    """
    
    def __init__(self, config: SlotTransformerConfig):
        """
        Initialize SlotTransformer from configuration.
        
        Arguments:
            - config (:obj:`SlotTransformerConfig`): Configuration object containing all parameters.
        """
        super().__init__()
        self.config = config
        
        self.num_slots = config.num_slots
        self.slots_dim = config.slots_dim
        self.tokens_dim = config.tokens_dim
        self.max_timestep = config.max_timestep
        self.action_dim = config.action_space_size
        self.action_type = config.action_type
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.hidden_mult = config.hidden_mult
        self.use_time_encoding = config.use_time_encoding
        self.use_actions = config.use_actions
        self.use_slot_temporal_block = config.use_slot_temporal_block
        self.use_slot_interaction_block = config.use_slot_interaction_block
        self.in_proj = nn.Linear(self.slots_dim, self.tokens_dim, bias=False)
        
        # Action adapter
        if self.action_type == "discrete":
            self.act_adapter = DiscreteActionAdapter(num_actions=self.action_dim, actions_dim=self.tokens_dim)
        else:
            self.act_adapter = ContinuousActionAdapter(
                actions_dim=self.action_dim, projected_actions_dim=self.tokens_dim, norm_actions=True
            )

        self.blocks = nn.ModuleList([
            CompasParLayer(
                d_model=self.tokens_dim,
                nhead=self.num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=self.tokens_dim * self.hidden_mult,
                use_actions=self.use_actions,
                use_slot_temporal_block=self.use_slot_temporal_block,
                use_slot_interaction_block=self.use_slot_interaction_block,
                max_timestep=self.max_timestep,
            ) for _ in range(self.num_layers)
        ])

        self.time_pos_encoding = nn.Parameter(
            get_sin_pos_enc(self.max_timestep, self.tokens_dim), requires_grad=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(self.tokens_dim, self.slots_dim),
        )

    # def load_state_from_method(self, state_dict: dict):
    #     state_dict = {k.replace('transition_model.', ''): v for k, v in state_dict.items() if
    #                   k.startswith('transition_model.')}
    #     self.load_state_dict(state_dict)
    
    def generate_empty_kv_cache(self, batch_size: int) -> SlotKVCache:
        """
        Overview:
            Creates an empty KV cache for inference.
        Arguments:
            - batch_size (:obj:`int`): The batch size.
        Returns:
            - SlotKVCache: An empty KV cache object.
        """
        device = next(self.parameters()).device
        
        return SlotKVCache(
            batch_size=batch_size,
            num_slots=self.num_slots,
            num_heads=self.num_heads,
            max_timesteps=self.max_timestep,
            embed_dim=self.tokens_dim,
            num_layers=self.num_layers,
            device=device
        )

    # def auto_predict_trajectory(self, slots: torch.Tensor, actions: torch.Tensor):
    #     pred_len = slots.size(1) - self.max_timestep
    #     return self.predict_trajectory(slots, actions, length=pred_len), self.max_timestep

    # def predict_trajectory(self, slots: torch.Tensor, actions: torch.Tensor, length: int = 2):
    #     next_slots = []
    #     max_t = min(self.max_timestep, slots.size(1))
    #     slots = slots[:, :max_t]
    #     offset = 0

    #     # print(actions.shape)
    #     for i in range(length):
    #         act_step = i - offset
    #         act = actions[:, act_step:max_t + act_step]
    #         next_slot = self.forward(slots, act)
    #         if max_t == self.max_timestep:
    #             slots = slots[:, 1:]
    #         else:
    #             max_t += 1
    #             offset += 1
    #         slots = torch.cat((slots, next_slot.unsqueeze(1)), dim=1)
    #         next_slots.append(next_slot)

    #     return torch.stack(next_slots, dim=1)

    # def _par_forward_blocks(self, slots: torch.Tensor, actions: torch.Tensor):
    #     blocks = zip(self.slots_action_blocks, self.slots_pred_blocks)
    #     for act_block, pred_block in blocks:
    #         act_slots = act_block(slots, actions)
    #         pred_slots = pred_block(slots)
    #         slots = act_slots + pred_slots
    #     return slots

    # def _seq_forward_blocks(self, slots: torch.Tensor, actions: torch.Tensor):
    #     blocks = zip(self.slots_action_blocks, self.slots_pred_blocks)
    #     for act_block, pred_block in blocks:
    #         slots = act_block(slots, actions)
    #         slots = pred_block(slots)
    #     return slots

    def forward(self, slots: torch.Tensor, actions: torch.Tensor, 
                kv_cache: Optional[SlotKVCache] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Forward pass through the SlotTransformer with optional KV-caching.
        Arguments:
            - slots (:obj:`torch.Tensor`): Slot representations, shape (B, T, N, slots_dim).
            - actions (:obj:`torch.Tensor`): Action tokens, shape (B, T, action_dim).
            - kv_cache (:obj:`Optional[SlotKVCache]`): Optional KV cache for inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for each batch element.
        Returns:
            - torch.Tensor: Predicted next slots, shape (B, N, slots_dim).
        """
        B, T, N, _ = slots.shape
        in_slots = slots
        slots = self.in_proj(slots)

        time_enc = self.time_pos_encoding[:, -T:]
        actions = self.act_adapter(actions)

        slots = slots + time_enc.unsqueeze(2).expand(B, -1, self.num_slots, -1)
        actions = actions + time_enc.expand(B, -1, -1)

        # Pass through transformer blocks with KV cache
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_cache[layer_idx] if kv_cache else None
            slots = block(slots, actions, kv_cache=layer_cache, valid_context_lengths=valid_context_lengths)

        slots = in_slots + self.mlp_out(slots)
        return slots[:, -1]