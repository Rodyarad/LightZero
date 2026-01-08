import logging
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical, Independent, Normal, TransformedDistribution, TanhTransform

from lzero.model.utils import calculate_dormant_ratio, compute_average_weight_magnitude, compute_effective_rank
from .kv_caching import KeysValues, SlotKVCache
from .tokenizer import Tokenizer
from .transformer import SlotTransformer, SlotTransformerConfig
from .aux_transformer import TransformerEncoder
from .utils import LossWithIntermediateLosses, init_weights, WorldModelOutput, hash_state
logging.getLogger().setLevel(logging.DEBUG)

import torch
import torch.nn as nn

logging.getLogger().setLevel(logging.DEBUG)


class ObjectWorldModel(nn.Module):

    def __init__(self, config: SlotTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = SlotTransformer(self.config)

        self.env_num = self.config.env_num
        if self.config.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move all modules to the specified device
        logging.info(f"self.device: {self.device}")
        self.to(self.device)

        # Initialize configuration parameters
        self._initialize_config_parameters()

        #TODO check and fix params
        # Dynamics transformer: predicts next step slots from current slots
        # Used for dynamics/observations prediction
        self.dynamics_transformer = TransformerEncoder(
            dim=self.config.slots_dim,
            n_blocks=self.config.dynamics_transformer_blocks,
            n_heads=self.config.dynamics_transformer_heads,
            qkv_dim=None,
            memory_dim=None,
            qkv_bias=True,
            dropout=self.config.dynamics_transformer_dropout,
            hidden_dim=None,
            initial_residual_scale=None,
            frozen=False
        )
        
        # Prediction transformer: processes slots for reward/value/policy prediction
        # Output is aggregated and fed to MLP heads
        self.prediction_transformer = TransformerEncoder(
            dim=self.config.slots_dim,
            n_blocks=self.config.prediction_transformer_blocks,
            n_heads=self.config.prediction_transformer_heads,
            qkv_dim=None,
            memory_dim=None,
            qkv_bias=True,
            dropout=self.config.prediction_transformer_dropout,
            activation='gelu',
            hidden_dim=None,
            initial_residual_scale=None,
            frozen=False
        )

        self.hidden_size = config.embed_dim // config.num_heads

        self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', 'LayerNorm')

        # Head modules
        self.head_rewards = self._create_head( self.support_size)
        self.head_policy = self._create_head(self.action_space_size)
        self.head_value = self._create_head(self.support_size)

        self.head_dict = {}
        for name, module in self.named_children():
            if name.startswith("head_"):
                self.head_dict[name] = module
        if self.head_dict:
            self.head_dict = nn.ModuleDict(self.head_dict)

        # Apply weight initialization, the order is important
        self.apply(lambda module: init_weights(module, norm_type=self.config.norm_type))

        self._initialize_last_layer()

        # Initialize keys and values for transformer
        self._initialize_transformer_keys_values()

        self.shared_pool_size_init = int(self.config.game_segment_length)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?

        # for self.kv_cache_recurrent_infer
        # If needed, recurrent_infer should store the results of the one MCTS search.
        self.num_simulations = getattr(self.config, 'num_simulations', 50)


        self.shared_pool_size_recur = int(self.num_simulations*self.env_num)
        self.shared_pool_recur_infer = [None] * self.shared_pool_size_recur
        self.shared_pool_index = 0

        # Cache structures
        self._initialize_cache_structures()
        
        # for self.kv_cache_init_infer
        # In contrast, init_infer only needs to retain the results of the most recent step.
        self.shared_pool_init_infer = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        self.shared_pool_index_init_envs = [0 for _ in range(self.env_num)]

        # for self.kv_cache_wm
        self.shared_pool_size_wm = int(self.env_num)
        self.shared_pool_wm = [None] * self.shared_pool_size_wm
        self.shared_pool_index_wm = 0

        self.reanalyze_phase = False

    def _initialize_cache_structures(self) -> None:
        """Initialize cache structures for past keys and values."""

        # Use old cache system (original implementation)
        self.past_kv_cache_recurrent_infer = {}
        self.pool_idx_to_key_map_recur_infer = [None] * self.shared_pool_size_recur
        self.past_kv_cache_init_infer_envs = [{} for _ in range(self.env_num)]
        self.pool_idx_to_key_map_init_envs = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        logging.info("Using OLD cache system (original implementation)")

    def _inspect_and_log_head_params(self, head_name: str, head_module: nn.Module, status: str):
        """
        检查并记录指定Head模块的参数统计信息。
        
        Args:
            head_name (str): 要检查的Head的名称 (例如, "Value Head")。
            head_module (nn.Module): Head的实际nn.Sequential模块。
            status (str): 描述当前状态的字符串 (例如, "Before Re-init")。
        """
        logging.info(f"--- 检查 {head_name} 参数 ({status}) ---")
        with torch.no_grad():
            for param_name, param in head_module.named_parameters():
                if param.numel() > 0:
                    stats = {
                        "mean": param.mean().item(),
                        "std": param.std().item(),
                        "abs_mean": param.abs().mean().item(),
                        "max": param.max().item(),
                        "min": param.min().item(),
                    }
                    logging.info(
                        f"  -> {param_name:<20} | "
                        f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, "
                        f"AbsMean: {stats['abs_mean']:.4f}, "
                        f"Max: {stats['max']:.4f}, Min: {stats['min']:.4f}"
                    )
        logging.info("-" * (23 + len(head_name) + len(status)))


    def custom_copy_kv_cache_to_shared_init_envs(self, src_kv: SlotKVCache, env_id: int) -> int:
        """
        Overview:
            Efficiently copies the contents of a SlotKVCache object to the shared pool 
            for a specific environment in the init_infer stage.
            
            SlotKVCache contains two types of caches:
            - temporal_keys_values: for self-attention over time (shape: B * num_slots)
            - action_keys_values: for cross-attention with actions (shape: B)
            
        Arguments:
            - src_kv (:obj:`SlotKVCache`): The source SlotKVCache object from which data is copied.
            - env_id (:obj:`int`): The identifier of the environment for which the cache is being copied.
        Returns:
            - index (:obj:`int`): The index in the shared pool where the SlotKVCache object is stored.
        """
        pool_index = self.shared_pool_index_init_envs[env_id]
        
        if self.shared_pool_init_infer[env_id][pool_index] is None:
            self.shared_pool_init_infer[env_id][pool_index] = SlotKVCache(
                batch_size=src_kv.batch_size,
                num_slots=src_kv.num_slots,
                num_heads=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads,
                max_timesteps=src_kv._temporal_keys_values._keys_values[0]._k_cache._max_tokens,
                embed_dim=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads * 
                          src_kv._temporal_keys_values._keys_values[0]._k_cache._head_dim,
                num_layers=src_kv.num_layers,
                device=src_kv._temporal_keys_values._keys_values[0]._k_cache._device
            )
        
        dst_kv = self.shared_pool_init_infer[env_id][pool_index]
        
        # Copy temporal caches (for self-attention over time)
        for src_layer, dst_layer in zip(src_kv._temporal_keys_values._keys_values, 
                                        dst_kv._temporal_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        # Copy action caches (for cross-attention with actions)
        for src_layer, dst_layer in zip(src_kv._action_keys_values._keys_values, 
                                        dst_kv._action_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = pool_index
        self.shared_pool_index_init_envs[env_id] = (pool_index + 1) % self.shared_pool_size_init
        
        return index

    def custom_copy_kv_cache_to_shared_wm(self, src_kv: SlotKVCache) -> SlotKVCache:
        """
        Overview:
            Efficiently copies the contents of a SlotKVCache object to the shared pool for world model usage.
            This is used for batch processing multiple environments simultaneously.
            
        Arguments:
            - src_kv (:obj:`SlotKVCache`): The source SlotKVCache object from which data is copied.
        Returns:
            - dst_kv (:obj:`SlotKVCache`): The destination SlotKVCache object in the shared pool.
        """
        pool_index = self.shared_pool_index_wm
        
        if self.shared_pool_wm[pool_index] is None:
            self.shared_pool_wm[pool_index] = SlotKVCache(
                batch_size=src_kv.batch_size,
                num_slots=src_kv.num_slots,
                num_heads=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads,
                max_timesteps=src_kv._temporal_keys_values._keys_values[0]._k_cache._max_tokens,
                embed_dim=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads * 
                          src_kv._temporal_keys_values._keys_values[0]._k_cache._head_dim,
                num_layers=src_kv.num_layers,
                device=src_kv._temporal_keys_values._keys_values[0]._k_cache._device
            )
        
        dst_kv = self.shared_pool_wm[pool_index]
        
        # Copy temporal caches
        for src_layer, dst_layer in zip(src_kv._temporal_keys_values._keys_values, 
                                        dst_kv._temporal_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        # Copy action caches
        for src_layer, dst_layer in zip(src_kv._action_keys_values._keys_values, 
                                        dst_kv._action_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        self.shared_pool_index_wm = (pool_index + 1) % self.shared_pool_size_wm
        
        return dst_kv

    def custom_copy_kv_cache_to_shared_recur(self, src_kv: SlotKVCache) -> int:
        """
        Overview:
            Efficiently copies the contents of a SlotKVCache object to the shared pool for recurrent inference.
            This is used during MCTS simulations to cache intermediate states.
            
        Arguments:
            - src_kv (:obj:`SlotKVCache`): The source SlotKVCache object from which data is copied.
        Returns:
            - index (:obj:`int`): The index in the shared pool where the SlotKVCache object is stored.
        """
        pool_index = self.shared_pool_index
        
        if self.shared_pool_recur_infer[pool_index] is None:
            self.shared_pool_recur_infer[pool_index] = SlotKVCache(
                batch_size=src_kv.batch_size,
                num_slots=src_kv.num_slots,
                num_heads=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads,
                max_timesteps=src_kv._temporal_keys_values._keys_values[0]._k_cache._max_tokens,
                embed_dim=src_kv._temporal_keys_values._keys_values[0]._k_cache._num_heads * 
                          src_kv._temporal_keys_values._keys_values[0]._k_cache._head_dim,
                num_layers=src_kv.num_layers,
                device=src_kv._temporal_keys_values._keys_values[0]._k_cache._device
            )
        
        dst_kv = self.shared_pool_recur_infer[pool_index]
        
        # Copy temporal caches
        for src_layer, dst_layer in zip(src_kv._temporal_keys_values._keys_values, 
                                        dst_kv._temporal_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        # Copy action caches
        for src_layer, dst_layer in zip(src_kv._action_keys_values._keys_values, 
                                        dst_kv._action_keys_values._keys_values):
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = pool_index
        self.shared_pool_index = (pool_index + 1) % self.shared_pool_size_recur
        
        return index

    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
        self.policy_entropy_weight = self.config.policy_entropy_weight
        self.predict_latent_loss_type = self.config.predict_latent_loss_type
        self.group_size = self.config.group_size
        self.num_groups = self.config.embed_dim // self.group_size
        self.obs_type = self.config.obs_type
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.gamma = self.config.gamma
        self.context_length = self.config.context_length
        self.dormant_threshold = self.config.dormant_threshold
        self.analysis_dormant_ratio_weight_rank = self.config.analysis_dormant_ratio_weight_rank
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.slot_dim = self.config.slot_dim
        self.num_slots = self.config.num_slots

        # ==================== [NEW] Policy Stability Fix Options ====================
        # Load fix options from config (with defaults for backward compatibility)
        self.use_policy_logits_clip = getattr(self.config, 'use_policy_logits_clip', False)
        self.policy_logits_clip_min = getattr(self.config, 'policy_logits_clip_min', -10.0)
        self.policy_logits_clip_max = getattr(self.config, 'policy_logits_clip_max', 10.0)

        # [NEW] Fix5: Temperature scaling for policy loss
        self.use_policy_loss_temperature = getattr(self.config, 'use_policy_loss_temperature', False)
        self.policy_loss_temperature = getattr(self.config, 'policy_loss_temperature', 1.0)

        # [NEW] Fix3: Check if target policy re-smooth is enabled (now deprecated in favor of Fix2)
        use_target_policy_resmooth = getattr(self.config, 'use_target_policy_resmooth', False)
        if use_target_policy_resmooth:
            logging.warning(
                "[DEPRECATED] use_target_policy_resmooth=True is deprecated! "
                "Policy label smoothing should now be controlled by 'continuous_ls_eps' in policy config. "
                "Fix3 (use_target_policy_resmooth) creates redundant smoothing with Fix2. "
                "Please set use_target_policy_resmooth=False and use continuous_ls_eps instead."
            )

        # [NEW] Debug: Print configuration on initialization
        if self.use_policy_logits_clip:
            logging.info(f"[Policy Logits Clip] ENABLED: range=[{self.policy_logits_clip_min}, {self.policy_logits_clip_max}]")
        else:
            logging.warning(f"[Policy Logits Clip] DISABLED! Using default values.")

        if self.use_policy_loss_temperature and self.policy_loss_temperature != 1.0:
            logging.info(f"[Policy Loss Temperature] ENABLED: temperature={self.policy_loss_temperature}")
        # =============================================================================

    def _create_head(self, output_dim: int, norm_layer=None) -> nn.Module:
        """Create head modules for the transformer."""
        modules = [
            nn.LayerNorm(self.config.slot_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim*4),
            nn.LayerNorm(self.config.embed_dim*4),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim*4, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return nn.Sequential(*modules)
        

    def _create_head_cont(self, output_dim: int) -> nn.Module:
        """Create head modules for the transformer."""
        from ding.model.common import ReparameterizationHead
        self.fc_policy_head = ReparameterizationHead(
            input_size=self.config.embed_dim,
            output_size=output_dim,
            layer_num=2,
            sigma_type=self.sigma_type,
            activation=nn.GELU(approximate='tanh'),
            fixed_sigma_value=self.config.fixed_sigma_value if self.sigma_type == 'fixed' else 0.5,
            norm_type=None,
            bound_type=self.bound_type
        )
        return self.fc_policy_head

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True
        if last_linear_layer_init_zero:
            module_to_initialize = [self.head_policy, self.head_value, self.head_rewards]
            for head in module_to_initialize:
                for layer in reversed(head):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_transformer_keys_values(self) -> None:
        """Initialize keys and values for the transformer."""
        self.keys_values_wm_single_env = self.transformer.generate_empty_kv_cache(batch_size=1)
        self.keys_values_wm_single_env_tmp = self.transformer.generate_empty_kv_cache(batch_size=1)
        self.keys_values_wm = self.transformer.generate_empty_kv_cache(batch_size=self.env_num)

    def forward(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        past_keys_values: Optional[Union[SlotKVCache, List[SlotKVCache]]] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        # kvcache_independent: bool = False,
        # is_init_infer: bool = True,
        # search_depth: Optional[List[int]] = None
    ) -> "WorldModelOutput":
        """
        Overview:
            Forward pass for the object world model using SlotTransformer.
        Arguments:
            - slots (:obj:`torch.Tensor`): Slot representations, shape (B, T, N, slots_dim).
            - actions (:obj:`torch.Tensor`): Action tokens, shape (B, T, action_dim).
            - past_keys_values (:obj:`Optional[Union[SlotKVCache, List[SlotKVCache]]]`): 
                KV cache(s) for inference. Can be a single cache or a list for independent caching.
            - kvcache_independent (:obj:`bool`): Whether each batch element has its own KV cache. Defaults to False.
            - is_init_infer (:obj:`bool`): Whether this is initial inference (root of MCTS). Defaults to True.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for each batch element.
            - search_depth (:obj:`Optional[List[int]]`): Search depth for each batch element in MCTS.
        Returns:
            - WorldModelOutput: Output containing slots, logits for observations, rewards, policy, and value.
        """
        B, T, N, _ = slots.shape
        

        x = self.transformer(slots, actions, past_keys_values, valid_context_lengths)  # (B, N, slots_dim)
        
        # Dynamics transformer: predict next step slots (for dynamics/observations)
        # slots_out: (B, N, slots_dim) -> dynamics_transformer -> predicted_slots: (B, N, slots_dim)
        predicted_slots = self.dynamics_transformer(x)  # (B, N, slots_dim)
        
        # Prediction transformer: process slots for reward/value/policy prediction
        # slots_out: (B, N, slots_dim) -> prediction_transformer -> slots_for_prediction: (B, N, slots_dim)
        slots_for_prediction = self.prediction_transformer(x)  # (B, N, slots_dim)
        
        # Aggregate slots (sum over slots dimension) for prediction heads
        slots_agg = slots_for_prediction.sum(dim=1)  # (B, slots_dim)
        
        # Apply prediction heads
        logits_value = self.head_value(slots_agg)  # (B, support_size)
        logits_rewards = self.head_rewards(slots_agg)  # (B, support_size)
        logits_policy = self.head_policy(slots_agg)  # (B, action_space_size) or (B, 2*action_space_size) for continuous
        
        # Clip policy logits if enabled
        if self.use_policy_logits_clip:
            logits_policy = torch.clamp(
                logits_policy,
                min=self.policy_logits_clip_min,
                max=self.policy_logits_clip_max
            )
        
        # Return WorldModelOutput
        return WorldModelOutput(
            x, 
            logits_observations=predicted_slots, 
            logits_rewards=logits_rewards, 
            logits_ends=None, 
            logits_policy=logits_policy, 
            logits_value=logits_value
        )

    #@profile
    @torch.no_grad()
    def reset_for_initial_inference(self, slot_act_dict: torch.FloatTensor, start_pos: int = 0) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Arguments:
            - obs_act_dict (:obj:`torch.FloatTensor`): A dictionary containing 'obs', 'action', and 'current_obs'.
        Returns:
            - torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(slot_act_dict, dict):
            batch_slots = slot_act_dict['obs']  # obs_act_dict['obs'] is at timestep t
            batch_action = slot_act_dict['action'] # obs_act_dict['action'] is at timestep t
            batch_current_slots = slot_act_dict['current_obs'] # obs_act_dict['current_obs'] is at timestep t+1

        if batch_current_slots is not None:
            # ================ Collect and Evaluation Phase ================
            self.latent_state = batch_current_slots
            outputs_wm = self.wm_forward_for_initial_infererence(batch_slots, batch_action,
                                                                                   batch_current_slots, start_pos)
        else:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            self.latent_state = batch_slots
            outputs_wm = self.wm_forward_for_initial_infererence(batch_slots, batch_action, None, start_pos)

        return outputs_wm, self.latent_state

    #@profile
    @torch.no_grad()
    def wm_forward_for_initial_infererence(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None, start_pos: int = 0) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - last_obs_embeddings (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = last_obs_embeddings.shape
        if n <= self.env_num and current_obs_embeddings is not None:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                 # Determine whether it is the first step in an episode.
                first_step_flag = max(batch_action) == -1
                if first_step_flag:
                    # ------------------------- First Step of an Episode -------------------------
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=current_obs_embeddings.shape[0],
                                                                                      max_tokens=self.context_length)
                    # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, start_pos=start_pos)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)
                else:
                    # --------------------- Continuing an Episode (Multi-environment) ---------------------
                    # current_obs_embeddings is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []

                    for i in range(ready_env_num):
                        # Retrieve latent state for a single environment

                        state_single_env = last_obs_embeddings[i]
                        # Compute hash value using latent state for a single environment
                        cache_key = hash_state(state_single_env.view(-1).cpu().numpy())  # last_obs_embeddings[i] is torch.Tensor

                        # ==================== Phase 1.6: Storage Layer Integration ====================
                        # Retrieve cached value
                        # OLD SYSTEM: Use legacy cache dictionaries
                        cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                        if cache_index is not None:
                            matched_value = self.shared_pool_init_infer[i][cache_index]
                        else:
                            matched_value = None
                        # =============================================================================

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # If a matching value is found, add it to the list
                            self.root_hit_cnt += 1
                            # ==================== BUG FIX: Cache Corruption Prevention ====================
                            # Perform a deep copy because the transformer's forward pass modifies matched_value in-place.
                            # OLD SYSTEM: Use custom_copy_kv_cache_to_shared_wm
                            self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                            # =============================================================================
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # Reset using zero values
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                            # If using RoPE positional encoding, then at reset, the pos_embed should use the absolute position start_pos[i].
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)},
                                                      past_keys_values=self.keys_values_wm_single_env,
                                                      is_init_infer=True, start_pos=start_pos[i].item())
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # Input self.keys_values_wm_list, output self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    start_pos = start_pos[:ready_env_num]

                    batch_action = batch_action[:ready_env_num]
                
                    act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)
                    
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm,
                                              is_init_infer=True, start_pos=start_pos)
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, start_pos=start_pos)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, num_observations_tokens,
                                                          self.config.embed_dim)  # (BL, K) for unroll_step=1

            last_obs_embeddings = last_obs_embeddings[:, :-1, :]
            batch_action = torch.from_numpy(batch_action).to(last_obs_embeddings.device)
            act_tokens = rearrange(batch_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            # Each sample in the batch (last_obs_embeddings, act_tokens) corresponds to the same time step, and start_pos also corresponds to each sample's respective t.
            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)}, start_pos=start_pos)

            # select the last timestep for each sample
            last_steps_value = outputs_wm.logits_value[:, -1:, :]
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            # outputs_wm.logits_value.shape (B, H, 101) = (B*H, 101)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')

        return outputs_wm

    #@profile
    @torch.no_grad()
    def forward_initial_inference(self, obs_act_dict, start_pos: int = 0):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict, start_pos)

        # ==================== BUG FIX: Clear Cache Using Correct API ====================
        # OLD SYSTEM: Clear using legacy attribute
        self.past_kv_cache_recurrent_infer.clear()
        # =============================================================================

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    #@profile
    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    search_depth=[], start_pos: int = 0):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of state and action history.
            - simulation_index (:obj:`int`, optional): Index of the current simulation. Defaults to 0.
            - search_depth (:obj:`list`, optional): List containing depth of latent states in the search tree. 
        Returns:
            - tuple: A tuple containing output sequence, updated latent state, reward, logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index, start_pos)

        latent_state_list = []
        token = action.reshape(-1, self.action_space_size)

        # Trim and pad kv_cache: modify self.keys_values_wm in-place
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False)
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list

        for k in range(2):
            # action_token obs_token
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            # Perform forward pass
            outputs_wm = self.forward(
                obs_embeddings_or_act_tokens,
                past_keys_values=self.keys_values_wm,
                kvcache_independent=False,
                is_init_infer=False,
                start_pos=start_pos,
                search_depth=search_depth # List containing depth of latent states in the search tree. 
            )

            self.keys_values_wm_size_list_current = [i + 1 for i in self.keys_values_wm_size_list_current]

            if k == 0:
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        del self.latent_state  # Very important to minimize cuda memory usage
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(
            self.latent_state,
            is_init_infer=False,
            simulation_index=simulation_index,
        )

        return (outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value)


    #@profile
    def trim_and_pad_kv_cache(self, is_init_infer=True) -> list:
        """
        Adjusts the key-value cache for each environment to ensure they all have the same size.

        In a multi-environment setting, the key-value cache (kv_cache) for each environment is stored separately.
        During recurrent inference, the kv_cache sizes may vary across environments. This method pads each kv_cache
        to match the largest size found among them, facilitating batch processing in the transformer forward pass.

        Arguments:
            - is_init_infer (:obj:`bool`): Indicates if this is an initial inference. Default is True.
        Returns:
            - list: Updated sizes of the key-value caches.
        """
        # Find the maximum size among all key-value caches
        max_size = max(self.keys_values_wm_size_list)

        # Iterate over each layer of the transformer
        for layer in range(self.num_layers):
            kv_cache_k_list = []
            kv_cache_v_list = []

            # Enumerate through each environment's key-value pairs
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                effective_size = self.keys_values_wm_size_list[idx]
                pad_size = max_size - effective_size

                # If padding is required, trim the end and pad the beginning of the cache
                if pad_size > 0:
                    k_cache_trimmed = k_cache[:, :, :-pad_size, :]
                    v_cache_trimmed = v_cache[:, :, :-pad_size, :]
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # Stack the caches along a new dimension and remove any extra dimensions
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # Update the cache size to the maximum size
            self.keys_values_wm._keys_values[layer]._k_cache._size = max_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = max_size

        return self.keys_values_wm_size_list

    #@profile
    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             search_depth=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state tensor.
            - is_init_infer (:obj:`bool`): Flag to indicate if this is the initial inference.
            - simulation_index (:obj:`int`): Index of the simulation.
            - search_depth (:obj:`list`): List of depth indices in the search tree.
            - valid_context_lengths (:obj:`list`): List of valid context lengths.
        """
        if self.context_length <= 2:
            # No context to update if the context length is less than or equal to 2.
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            cache_key = hash_state(latent_state[i].view(-1).cpu().numpy())  # latent_state[i] is torch.Tensor
            context_length = self.context_length

            if not is_init_infer:
                # ============ Internal Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding
                current_max_context_length = max(self.keys_values_wm_size_list_current)
                trim_size = current_max_context_length - self.keys_values_wm_size_list_current[i]
                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============
                    # cache shape [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                    if trim_size > 0:
                        # Trim invalid leading zeros as per effective length
                        # Remove the first trim_size zero kv items
                        k_cache_trimmed = k_cache_current[:, trim_size:, :]
                        v_cache_trimmed = v_cache_current[:, trim_size:, :]
                        # If effective length < current_max_context_length, pad the end of cache with 'trim_size' zeros
                        k_cache_padded = F.pad(k_cache_trimmed, (0, 0, 0, trim_size), "constant",
                                               0)  # Pad with 'trim_size' zeros at end of cache
                        v_cache_padded = F.pad(v_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)
                    else:
                        k_cache_padded = k_cache_current
                        v_cache_padded = v_cache_current

                    # Update cache of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    # Update size of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm_size_list_current[i]
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm_size_list_current[i]

                    # ============ NOTE: Very Important ============
                    if self.keys_values_wm_single_env._keys_values[layer]._k_cache._size >= context_length - 1:
                        # Keep only the last self.context_length-3 timesteps of context
                        # For memory environments, training is for H steps, recurrent_inference might exceed H steps
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length - 1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length - 1, :].squeeze(0)

                        if not self.config.rotary_emb:
                            # Index pre-computed positional encoding differences
                            pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                            pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                            # ============ NOTE: Very Important ============
                            # Apply positional encoding correction to k and v
                            k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                            v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update single environment cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)

                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            else:
                # ============ Root Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding

                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============

                    if self.keys_values_wm._keys_values[layer]._k_cache._size < context_length - 1:  # Keep only the last self.context_length-1 timesteps of context
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(
                            0)  # Shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm._keys_values[layer]._k_cache._size
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm._keys_values[layer]._v_cache._size
                    else:
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                        v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, 2:context_length - 1, :]
                        v_cache_trimmed = v_cache_current[:, 2:context_length - 1, :]

                        if not self.config.rotary_emb:
                            # Index pre-computed positional encoding differences
                            pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                            pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                            # ============ NOTE: Very Important ============
                            # Apply positional encoding correction to k and v
                            k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                            v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update cache of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        # Update size of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3


            # ==================== Phase 1.5: Storage Layer Integration ====================
            # OLD SYSTEM: Use legacy cache with manual eviction
            if is_init_infer:
                # ==================== 主动淘汰修复逻辑 ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index_init_envs[i]
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_init_envs[i][index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    # 确保要删除的键确实存在，避免意外错误
                    if old_key_to_evict in self.past_kv_cache_init_infer_envs[i]:
                        del self.past_kv_cache_init_infer_envs[i][old_key_to_evict]

                # 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)

                # 4. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
                self.pool_idx_to_key_map_init_envs[i][index_to_write] = cache_key
            else:
                # ==================== RECURRENT INFER FIX ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_recur_infer[index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    if old_key_to_evict in self.past_kv_cache_recurrent_infer:
                        del self.past_kv_cache_recurrent_infer[old_key_to_evict]

                # 4. 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)

                # 5. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_recurrent_infer[cache_key] = cache_index
                self.pool_idx_to_key_map_recur_infer[index_to_write] = cache_key
            # =============================================================================



    #@profile
    def retrieve_or_generate_kvcache(self, latent_state: list, ready_env_num: int,
                                     simulation_index: int = 0, start_pos: int = 0) -> list:
        """
        Retrieves or generates key-value caches for each environment based on the latent state.

        For each environment, this method either retrieves a matching cache from the predefined
        caches if available, or generates a new cache if no match is found. The method updates
        the internal lists with these caches and their sizes.

        Arguments:
            - latent_state (:obj:`list`): List of latent states for each environment.
            - ready_env_num (:obj:`int`): Number of environments ready for processing.
            - simulation_index (:obj:`int`, optional): Index for simulation tracking. Default is 0.
        Returns:
            - list: Sizes of the key-value caches for each environment.
        """
        for index in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[index]  # latent_state[i] is np.array
            cache_key = hash_state(state_single_env)

            if self.reanalyze_phase:
                matched_value = None
            else:
                # ==================== Phase 1.6: Storage Layer Integration (Refactored) ====================
                # OLD SYSTEM: Use legacy cache dictionaries and pools
                # Try to retrieve the cached value from past_kv_cache_init_infer_envs
                cache_index = self.past_kv_cache_init_infer_envs[index].get(cache_key)
                if cache_index is not None:
                    matched_value = self.shared_pool_init_infer[index][cache_index]
                else:
                    matched_value = None

                # 仅当在 init_infer 中未找到时，才尝试从 recurrent_infer 缓存中查找
                if matched_value is None:
                    # 安全地从字典中获取索引，它可能返回 None
                    recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
                    # 只有在索引有效（不是 None）的情况下，才使用它来从物理池中检索值
                    if recur_cache_index is not None:
                        matched_value = self.shared_pool_recur_infer[recur_cache_index]

                    if recur_cache_index is None:
                        logging.debug(f"[OLD CACHE MISS] Not found for key={cache_key} in recurrent infer. Generating new cache.")
                # =============================================================================

            if matched_value is not None:
                # If a matching cache is found, add it to the lists
                self.hit_count += 1
                # ==================== BUG FIX: Cache Corruption Prevention ====================
                # Perform a deep copy because the transformer's forward pass modifies matched_value in-place.
                # Without cloning, the original cache in init_pool or recur_pool would be polluted,
                # causing incorrect predictions in subsequent queries.
                # OLD SYSTEM: Use custom_copy_kv_cache_to_shared_wm
                self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                # =============================================================================
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # If no matching cache is found, generate a new one using zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(
                    n=1, max_tokens=self.context_length
                )
                
                # Determine the absolute start position based on the reanalyze phase flag.
                if self.reanalyze_phase:
                    num_rows, num_cols = start_pos.shape  # Original start_pos shape is (batch, num_columns)
                    total_cols = num_cols + 1             # Each logical row is extended by one column.
                    row_idx = index // total_cols
                    col_idx = index % total_cols
                    # If the column index equals the original number of columns, this indicates the added column; set to 0.
                    start_pos_adjusted: int = 0 if col_idx == num_cols else int(start_pos[row_idx, col_idx])
                else:
                    start_pos_adjusted = int(start_pos[index].item())

                self.forward(
                    {'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)},
                    past_keys_values=self.keys_values_wm_single_env, is_init_infer=True, start_pos=start_pos_adjusted
                )
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

        return self.keys_values_wm_size_list


    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        start_pos = batch['timestep']
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # ======================== Logging for Analysis ========================
        # This block calculates various metrics for model analysis if the corresponding config flag is enabled.
        # These metrics help in debugging and understanding model behavior during training.
        if self.analysis_dormant_ratio_weight_rank:
            # --- Dormant Ratio Calculation ---
            # Calculate the dormant ratio of the encoder to monitor neuron activity.
            shape = batch['observations'].shape  # Original shape, e.g., (B, T, C, H, W)
            # Reshape observations to create a single large batch for the encoder.
            # E.g., (32, 5, 3, 64, 64) -> (160, 3, 64, 64)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])
            
            dormant_ratio_encoder_dict = calculate_dormant_ratio(
                self.tokenizer.encoder, inputs.detach(), dormant_threshold=self.dormant_threshold
            )
            dormant_ratio_encoder = dormant_ratio_encoder_dict['global']

            # --- Average Weight Magnitude Calculation ---
            # Calculate the global average absolute weight magnitude for different model components.
            # This is a useful metric for monitoring training stability.
            avg_weight_mag_encoder = compute_average_weight_magnitude(self.tokenizer.encoder)
            avg_weight_mag_transformer = compute_average_weight_magnitude(self.transformer)
            avg_weight_mag_head = compute_average_weight_magnitude(self.head_dict)

            # --- Effective Rank Calculation ---
            # Calculate the effective rank of representations from specific layers in the encoder.
            # This metric helps analyze the dimensionality and information content of the learned features.
            # The 'representation_layer_name' argument specifies the target layer within the model's named modules.
            
            # Effective rank for the final linear layer of the encoder.
            e_rank_last_linear = compute_effective_rank(
                self.tokenizer.encoder, inputs, representation_layer_name="last_linear"
            )
            # Effective rank for the SimNorm layer of the encoder.
            e_rank_sim_norm = compute_effective_rank(
                self.tokenizer.encoder, inputs, representation_layer_name="sim_norm"
            )

            # ==================== BUG FIX: Clear Cache Using Correct API ====================
            self.past_kv_cache_recurrent_infer.clear()
            # =============================================================================
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)
            avg_weight_mag_encoder = torch.tensor(0.)
            avg_weight_mag_transformer = torch.tensor(0.)
            avg_weight_mag_head = torch.tensor(0.)
            e_rank_last_linear = torch.tensor(0.)
            e_rank_sim_norm = torch.tensor(0.)

        # Calculate the L2 norm of the latent state roots
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()

        # Action tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, start_pos=start_pos)

        intermediate_tensor_x = outputs.output_sequence.detach()

        global_step = kwargs.get('global_step', 0)

        if global_step > 0 and global_step % 100000000000 == 0:

            with torch.no_grad():
                batch_size, seq_len = batch['actions'].shape[0], batch['actions'].shape[1]

                pred_val_logits = outputs.logits_value.view(batch_size * seq_len, -1)
                pred_rew_logits = outputs.logits_rewards.view(batch_size * seq_len, -1)

                scalar_values = inverse_scalar_transform_handle(pred_val_logits).squeeze(-1)
                scalar_rewards = inverse_scalar_transform_handle(pred_rew_logits).squeeze(-1)

                self._analyze_latent_representation(
                    latent_states=obs_embeddings,
                    timesteps=batch['timestep'],
                    game_states=batch['observations'],
                    predicted_values=scalar_values, # 传入预测的Value
                    predicted_rewards=scalar_rewards, # 传入预测的Reward
                    step_counter=global_step
                )

        if self.config.use_priority:
            # ==================== START MODIFICATION 5 ====================
            # Calculate value_priority, similar to MuZero.
            with torch.no_grad():
                # 1. Get the predicted value logits for the first step of the sequence (t=0).
                # The shape is (B, support_size).
                predicted_value_logits_step0 = outputs.logits_value[:, 0, :]

                # 2. Convert the categorical prediction to a scalar value.
                # The shape becomes (B, 1).
                predicted_scalar_value_step0 = inverse_scalar_transform_handle(predicted_value_logits_step0)

                # 3. Get the target scalar value for the first step from the batch.
                # The shape is (B, num_unroll_steps), so we take the first column.
                target_scalar_value_step0 = batch['scalar_target_value'][:, 0]

                # 4. Calculate the L1 loss (absolute difference) between prediction and target.
                # This is the priority. We use reduction='none' to get per-sample priorities.
                value_priority = F.l1_loss(predicted_scalar_value_step0.squeeze(-1), target_scalar_value_step0, reduction='none')
            # ===================== END MODIFICATION 5 =====================
        else:
            value_priority = torch.tensor(0.)

        if self.obs_type == 'image':
            if self.config.latent_recon_loss_weight > 0:

                # Reconstruct observations from latent state representations
                reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
                latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
                perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            else:
                latent_recon_loss = self.latent_recon_loss
                perceptual_loss = self.perceptual_loss

        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))

            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = self.latent_recon_loss

        elif self.obs_type == 'text':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=torch.float32)
            decode_loss_mode = self.config.decode_loss_mode 

            # Reconstruction loss for predicting the next latent (via backbone)
            # input -> encoder -> backbone(unizero) -> decoder -> latent_recon_loss
            if decode_loss_mode == "after_backbone":
                next_latent_state = outputs.logits_observations[:, :-1, :]
                next_target_ids = batch['observations'][:, 1:, :] 
                
                latent_recon_loss = self.tokenizer.decode_to_reconstruction_outputs(
                    embeddings=next_latent_state,
                    target_ids=next_target_ids,
                ).loss

            #Reconstruction loss for predicting the current latent (without using the backbone)
            # input -> encoder -> decoder -> latent_recon_loss
            elif decode_loss_mode == "before_backbone":
                latent_recon_loss = self.tokenizer.decode_to_reconstruction_outputs(
                    embeddings=obs_embeddings,
                    target_ids=batch['observations'],
                ).loss

            else:
                latent_recon_loss = self.latent_recon_loss

        elif self.obs_type == 'image_memory':
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio_weight_rank:
            # Calculate dormant ratio of the world model
            dormant_ratio_world_model = calculate_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          dormant_threshold=self.dormant_threshold)
            dormant_ratio_transformer = dormant_ratio_world_model['transformer']
            dormant_ratio_head = dormant_ratio_world_model['head']

            # ==================== BUG FIX: Clear Cache Using Correct API ====================
            self.past_kv_cache_recurrent_infer.clear()
            # =============================================================================
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_transformer = torch.tensor(0.)
            dormant_ratio_head = torch.tensor(0.)

        # For training stability, use target_tokenizer to compute the true next latent state representations
        with torch.no_grad():
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, _ = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        # Compute prediction loss for observations. Options: MSE and Group KL
        if self.predict_latent_loss_type == 'mse':
            # MSE loss, directly compare logits and labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(
                -1)
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL loss, group features and calculate KL divergence within each group
            batch_size, num_features = logits_observations.shape
            epsilon = 1e-6
            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)
        elif self.predict_latent_loss_type == 'cos_sim':
            cosine_sim_loss = 1 - F.cosine_similarity(logits_observations, labels_observations, dim=-1)
            loss_obs = cosine_sim_loss

        # Apply mask to loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # ==================== [NEW] Fix3: Load re-smooth options from config ====================
        use_target_policy_resmooth = getattr(self.config, 'use_target_policy_resmooth', False)
        target_policy_resmooth_eps = getattr(self.config, 'target_policy_resmooth_eps', 0.05)
        # ======================================================================================

        # Compute labels for policy and value (with optional re-smoothing)
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(
            batch['target_value'],
            batch['target_policy'],
            batch['mask_padding'],
            use_target_policy_resmooth=use_target_policy_resmooth,
            target_policy_resmooth_eps=target_policy_resmooth_eps
        )

        # Compute losses for rewards, policy, and value
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')

        loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy,
                                                                                            batch,
                                                                                            element='policy')

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # Compute timesteps
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # Compute discount coefficients for each timestep
        discounts = self.gamma ** timesteps

        # Group losses into first step, middle step, and last step
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        # batch['mask_padding'] indicates mask status for future H steps, exclude masked losses to maintain accurate mean statistics
        # Group losses for each loss item
        for loss_name, loss_tmp in zip(
                ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
                [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]

            # Adjust loss shape to (batch_size, seq_len)
            loss_tmp = loss_tmp.view(-1, seq_len)

            # First step loss
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()

            # Middle step loss
            middle_timestep = seq_len // 2
            middle_step_mask = mask_padding[:, middle_timestep]
            middle_step_losses[loss_name] = loss_tmp[:, middle_timestep][middle_step_mask].mean()

            # Last step loss
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # Discount reconstruction loss and perceptual loss
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss
        # Calculate overall discounted loss
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).sum()/ batch['mask_padding'][:,1:].sum()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        detached_obs_embeddings = obs_embeddings.detach()


        return LossWithIntermediateLosses(
            latent_recon_loss_weight=self.latent_recon_loss_weight,
            perceptual_loss_weight=self.perceptual_loss_weight,
            continuous_action_space=False,
            loss_obs=discounted_loss_obs,
            loss_rewards=discounted_loss_rewards,
            loss_value=discounted_loss_value,
            loss_policy=discounted_loss_policy,
            latent_recon_loss=discounted_latent_recon_loss,
            perceptual_loss=discounted_perceptual_loss,
            orig_policy_loss=discounted_orig_policy_loss,
            policy_entropy=discounted_policy_entropy,
            first_step_losses=first_step_losses,
            middle_step_losses=middle_step_losses,
            last_step_losses=last_step_losses,
            dormant_ratio_encoder=dormant_ratio_encoder,
            dormant_ratio_transformer=dormant_ratio_transformer,
            dormant_ratio_head=dormant_ratio_head,
            avg_weight_mag_encoder = avg_weight_mag_encoder,
            avg_weight_mag_transformer = avg_weight_mag_transformer,
            avg_weight_mag_head = avg_weight_mag_head,
            e_rank_last_linear = e_rank_last_linear,
            e_rank_sim_norm = e_rank_sim_norm,
            latent_state_l2_norms=latent_state_l2_norms,

            value_priority=value_priority,
            intermediate_tensor_x=intermediate_tensor_x,
            obs_embeddings=detached_obs_embeddings, # <-- 新增
            logits_value=outputs.logits_value.detach(),  # 使用detach()，因为它仅用于分析和裁剪，不参与梯度计算
            logits_reward=outputs.logits_rewards.detach(),
            logits_policy=outputs.logits_policy.detach(),
            )

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs is an object with logits attributes like 'rewards', 'policy', and 'value'.
        # labels is a target tensor for comparison. batch is a dictionary with a mask indicating valid timesteps.

        logits = getattr(outputs, f'logits_{element}')

        # ==================== [NEW] Fix5: Temperature Scaling for Policy ====================
        if element == 'policy' and self.use_policy_loss_temperature and self.policy_loss_temperature != 1.0:
            # Apply temperature scaling to soften the distribution
            logits = logits / self.policy_loss_temperature
        # ===================================================================================

        if torch.isnan(logits).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if torch.isnan(labels).any():
            raise ValueError(f"NaN detected in labels_value for batch {batch} and element '{element}'")

        # Reshape your tensors
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assume labels initially have shape [batch, time, dim]

        # Reshape your mask. True indicates valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # Compute cross-entropy loss
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if torch.isnan(loss).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if element == 'policy':
            # Compute policy entropy loss
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # Combine losses with specified weight
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    #@profile
    def compute_policy_entropy_loss(self, logits, mask):
        # Compute entropy of the policy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # Apply mask and return average entropy loss
        entropy_loss = (entropy * mask)
        return entropy_loss

    #@profile
    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # Each sequence sample should have at most one 'done' flag
        mask_fill = torch.logical_not(mask_padding)

        # Prepare observation labels
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]

        # Fill the masked areas of rewards
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        # Fill the masked areas of ends
        # labels_endgs = ends.masked_fill(mask_fill, -100)

        # return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)
        return labels_observations, labels_rewards.view(-1, self.support_size), None


    #@profile
    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor,
                                                use_target_policy_resmooth: bool = False,
                                                target_policy_resmooth_eps: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute labels for value and policy predictions. """
        mask_fill = torch.logical_not(mask_padding)

        # ==================== [NEW] Fix3: Re-smooth Target Policy ====================
        # Re-smooth target_policy to prevent extreme distributions in buffer
        if use_target_policy_resmooth and target_policy_resmooth_eps > 0:
            num_actions = target_policy.shape[-1]
            uniform_dist = torch.ones_like(target_policy) / num_actions
            target_policy = (1 - target_policy_resmooth_eps) * target_policy + \
                           target_policy_resmooth_eps * uniform_dist
        # =============================================================================

        # Fill the masked areas of policy
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        # Fill the masked areas of value
        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)

        return labels_policy.reshape(-1, self.action_space_size), labels_value.reshape(-1, self.support_size)

    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        # Use old cache clearing logic
        for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        self.past_kv_cache_recurrent_infer.clear()
        self.keys_values_wm_list.clear()
        print(f'Cleared {self.__class__.__name__} past_kv_cache (OLD system).')

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"
