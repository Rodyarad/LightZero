import logging
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical, Independent, Normal, TransformedDistribution, TanhTransform

from lzero.model.common import SimNorm
from lzero.model.utils import calculate_dormant_ratio, compute_average_weight_magnitude, compute_effective_rank
from .kv_caching import KeysValues
from .slicer import Head, PolicyHeadCont, SlotHead
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights, WorldModelOutput, hash_state
from collections import OrderedDict 
logging.getLogger().setLevel(logging.DEBUG)

from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import datetime
import torch
import torch.nn as nn

logging.getLogger().setLevel(logging.DEBUG)


class WorldModel(nn.Module):
    """
    Overview:
        The WorldModel class is responsible for the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667),
        which is used to predict the next latent state, rewards, policy, and value based on the current latent state and action.
        The world model consists of three main components:
            - a tokenizer, which encodes observations into embeddings,
            - a transformer, which processes the input sequences,
            - and heads, which generate the logits for observations, rewards, policy, and value.
    """

    def __init__(self, config: TransformerConfig, tokenizer) -> None:
        """
        Overview:
            Initialize the WorldModel class.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration for the transformer.
            - tokenizer (:obj:`Tokenizer`): The tokenizer.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.task_embed_option = self.config.task_embed_option  # Strategy for task embeddings

        self.transformer = Transformer(self.config)
        self.task_num = 1
        self.env_num = self.config.env_num
        if self.config.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move all modules to the specified device
        logging.info(f"self.device: {self.device}")
        self.to(self.device)

        self.task_embed_dim = config.task_embed_dim if hasattr(config, "task_embed_dim") else 96

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        self.hidden_size = config.embed_dim // config.num_heads

        # Position embedding
        if not self.config.rotary_emb:
            if self.model_type == 'slot':
                self.pos_emb = nn.Embedding(config.max_blocks, config.embed_dim, device=self.device)
            else:
                self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim, device=self.device)
            print(f"self.pos_emb.weight.device: {self.pos_emb.weight.device}")

        self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4
        if self.task_embed_option == "concat_task_embed":
            self.obs_per_embdding_dim = self.config.embed_dim - self.task_embed_dim
        else:
            self.obs_per_embdding_dim = self.config.embed_dim
        self.continuous_action_space = self.config.continuous_action_space

        # Initialize action embedding table
        if self.continuous_action_space:
            # TODO: check the effect of SimNorm
            self.act_embedding_table = nn.Sequential(
                nn.Linear(config.action_space_size, config.embed_dim, device=self.device, bias=False),
                SimNorm(simnorm_dim=self.group_size))
        else:
            # for discrete action space
            self.act_embedding_table = nn.Embedding(config.action_space_size, config.embed_dim, device=self.device)
            logging.info(f"self.act_embedding_table.weight.device: {self.act_embedding_table.weight.device}")

        self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', None)

        # Head modules
        self.head_rewards = self._create_head(self.act_tokens_pattern, self.support_size)
        # self.head_observations = self._create_head(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim, \
        #                                             self._get_final_norm(self.final_norm_option_in_obs_head)  # NOTE: using the specified normalization method for observations head
        #                                           )
        # if self.model_type == 'slot':
        #     self.head_observations = self._create_head_for_slots(self.act_tokens_pattern, self.obs_per_embdding_dim, \
        #                                             self._get_final_norm(self.final_norm_option_in_obs_head)  # NOTE: using the specified normalization method for observations head
        #                                            )
        # else:
        self.head_observations = self._create_head_for_latent(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim, \
                                                self._get_final_norm(self.final_norm_option_in_obs_head)  # NOTE: using the specified normalization method for observations head
                                               )
        if self.continuous_action_space:
            self.sigma_type = self.config.sigma_type
            self.bound_type = self.config.bound_type
            self.head_policy = self._create_head_cont(self.value_policy_tokens_pattern, self.action_space_size)
        else:
            if self.model_type == 'slot':
                self.head_policy = self._create_slot_head(self.value_policy_tokens_pattern, self.action_space_size)
            else:
                self.head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_space_size)
        
        if self.model_type == 'slot':
            self.head_value = self._create_slot_head(self.value_policy_tokens_pattern, self.support_size)
        else:
            self.head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size)

        self.head_dict = {}
        for name, module in self.named_children():
            if name.startswith("head_"):
                self.head_dict[name] = module
        if self.head_dict:
            self.head_dict = nn.ModuleDict(self.head_dict)

        # Apply weight initialization, the order is important
        # self.apply(lambda module: init_weights(module, norm_type=self.config.norm_type))

        # Build the set of modules to skip during re-initialization.
        # This is compatible with cases where self.tokenizer.encoder does not have 'pretrained_model',
        # or self.tokenizer does not have 'decoder_network'.
        # NOTE: This step is crucial — without skipping, pretrained modules (e.g., encoder/decoder) would be unintentionally re-initialized
        skip_modules = set()
        if hasattr(self.tokenizer.encoder, 'pretrained_model'):
            skip_modules.update(self.tokenizer.encoder.pretrained_model.modules())
        if hasattr(self.tokenizer, 'decoder_network') and self.tokenizer.decoder_network is not None:
            skip_modules.update(self.tokenizer.decoder_network.modules())

        def custom_init(module):
            # If the current module is part of the skip list, return without reinitializing
            if module in skip_modules:
                return
            # Otherwise, apply the specified initialization method
            init_weights(module, norm_type=self.config.norm_type)

        # Recursively apply `custom_init` to all submodules of the model
        self.apply(custom_init)

        self._initialize_last_layer()

        # # Cache structures
        # self._initialize_cache_structures()

        # Projection input dimension
        self._initialize_projection_input_dim()

        # Hit count and query count statistics
        self._initialize_statistics()

        self.obs_history: List = []
        self.act_history: List = []
        self.context_length_in_blocks = self.context_length // self.tokens_per_block

        self.latent_recon_loss = torch.tensor(0., device=self.device)
        self.perceptual_loss = torch.tensor(0., device=self.device)

        self.num_simulations = getattr(self.config, 'num_simulations', 50)
        self.reanalyze_phase = False

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

    def _analyze_latent_representation(
            self, 
            latent_states: torch.Tensor, 
            timesteps: torch.Tensor, 
            game_states: torch.Tensor, 
            predicted_values: torch.Tensor,
            predicted_rewards: torch.Tensor,
            step_counter: int
        ):
            """
            分析并记录 latent states 的统计信息和t-SNE可视化。
            【新功能】：在t-SNE图上显示对应的游戏图像，并标注预测的Value和Reward。
            【已修改】：如果保存路径已存在同名文件，则在文件名后附加时间戳。
            
            Args:
                latent_states (torch.Tensor): Encoder的输出, shape (B*L, 1, E)
                timesteps (torch.Tensor): 对应的时间步, shape (B, L)
                game_states (torch.Tensor): 原始的游戏观测, shape (B, L, C, H, W)
                predicted_values (torch.Tensor): 预测的标量Value, shape (B*L,)
                predicted_rewards (torch.Tensor): 预测的标量Reward, shape (B*L,)
                step_counter (int): 全局训练步数
            """
            # ... (统计分析部分保持不变) ...
            # (确保 latent_states 和 game_states 的形状为 (N, ...))
            if latent_states.dim() > 2:
                latent_states = latent_states.reshape(-1, latent_states.shape[-1])
            num_c, num_h, num_w = game_states.shape[-3:]
            game_states = game_states.reshape(-1, num_c, num_h, num_w)

            with torch.no_grad():
                l2_norm = torch.norm(latent_states, p=2, dim=1).mean()
                mean = latent_states.mean()
                std = latent_states.std()
                print(f"[Step {step_counter}] Latent Stats | L2 Norm: {l2_norm:.4f}, Mean: {mean:.4f}, Std: {std:.4f}")

            # 带图像和V/R值的 t-SNE 可视化
            if step_counter >= 0:
            # if step_counter > 0 and step_counter % 200 == 0:

                print(f"[Step {step_counter}] Performing t-SNE analysis with images, values, and rewards...")

                # 将数据转换到CPU
                latents_np = latent_states.detach().cpu().numpy()
                images_np = game_states.detach().cpu().numpy()
                values_np = predicted_values.detach().cpu().numpy()
                rewards_np = predicted_rewards.detach().cpu().numpy()

                tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
                tsne_results = tsne.fit_transform(latents_np)

                # --- 绘制带图像和标注的散点图 ---

                # 减少图像数量以保持清晰
                num_points_to_plot = min(len(latents_np), 70) # 减少到70个点
                indices = np.random.choice(len(latents_np), num_points_to_plot, replace=False)

                fig, ax = plt.subplots(figsize=(20, 18)) # 增大画布尺寸

                # 先画出所有点的散点图作为背景
                ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=values_np, cmap='viridis', alpha=0.3, s=10)

                for i in indices:
                    x, y = tsne_results[i]
                    img = images_np[i].transpose(1, 2, 0)
                    img = np.clip(img, 0, 1)

                    # 放置图像
                    im = OffsetImage(img, zoom=0.7) # 稍微放大图像
                    ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.0, bboxprops=dict(edgecolor='none'))
                    ax.add_artist(ab)

                    # 在图像下方添加文字标注
                    text_label = f"V:{values_np[i]:.1f} R:{rewards_np[i]:.1f}"
                    ax.text(x, y - 1.0, text_label, ha='center', va='top', fontsize=8, color='red',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

                ax.update_datalim(tsne_results)
                ax.autoscale()

                ax.set_title(f't-SNE of Latent States (Value as Color) at Step {step_counter}', fontsize=16)
                ax.set_xlabel('t-SNE dimension 1', fontsize=12)
                ax.set_ylabel('t-SNE dimension 2', fontsize=12)

                # 添加colorbar来解释背景点的颜色
                norm = plt.Normalize(values_np.min(), values_np.max())
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label='Predicted Value')

                # --- 修改部分：检查文件是否存在，如果存在则添加时间戳 ---
                # 1. 构建基础路径
                # base_save_path = (
                #     f'/mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/unizero_mspacman_analyze/'
                #     f'tsne_with_vr_{self.config.optim_type}_lr{self.config.learning_rate}_step_{step_counter}.png'
                # )
                base_save_path = (
                    f'/mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/unizero_mspacman_analyze/'
                    f'tsne_with_vr_{self.config.optim_type}_step_{step_counter}.png'
                )

                # 2. 检查文件是否存在，并确定最终保存路径
                if os.path.exists(base_save_path):
                    # 如果文件已存在，则生成时间戳并附加到文件名
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path_root, path_ext = os.path.splitext(base_save_path)
                    save_path = f"{path_root}_{timestamp}{path_ext}"
                    print(f"File '{base_save_path}' already exists. Saving to new path with timestamp.")
                else:
                    # 如果文件不存在，则使用原始路径
                    save_path = base_save_path

                # 3. 保存图像
                plt.savefig(save_path)
                plt.close(fig) # 明确关闭图形对象
                print(f"t-SNE plot with V/R annotations saved to {save_path}")

    def _get_final_norm(self, norm_option: Optional[str]) -> Optional[nn.Module]:
        """
        Return the corresponding normalization module based on the specified normalization option.
        """
        if norm_option == 'LayerNorm':
            return nn.LayerNorm(self.config.embed_dim, eps=1e-5)
        elif norm_option == 'SimNorm':
            return SimNorm(simnorm_dim=self.config.group_size)
        elif norm_option is None:
            return None
        else:
            raise ValueError(f"Unsupported final_norm_option_in_obs_head: {norm_option}")

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
        self.tokens_per_block = self.config.tokens_per_block
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)
        self.model_type = self.config.model_type
        self.max_blocks = self.config.max_blocks
        self.max_tokens = self.config.max_tokens

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

    def _initialize_patterns(self) -> None:
        """Initialize patterns for block masks."""
        if self.model_type == 'slot':
            self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
            self.all_but_last_latent_state_pattern[-1] = 0
            self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
            self.act_tokens_pattern[-1] = 1
            self.value_policy_tokens_pattern = 1 - self.act_tokens_pattern
        else:
            self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
            self.all_but_last_latent_state_pattern[-2] = 0
            self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
            self.act_tokens_pattern[-1] = 1
            self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
            self.value_policy_tokens_pattern[-2] = 1 

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            # nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim*4),
            nn.LayerNorm(self.config.embed_dim*4),      # 2. <-- 新增！稳定内部激活
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim*4, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_head_for_slots(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            # nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim*4),
            nn.LayerNorm(self.config.embed_dim*4),      # 2. <-- 新增！稳定内部激活
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim*4, self.num_observations_tokens * output_dim),
            nn.Unflatten(-1, (self.num_observations_tokens, output_dim))
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )
    
    def _create_head_for_latent(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            nn.Linear(self.config.embed_dim, self.config.embed_dim*4),
            nn.LayerNorm(self.config.embed_dim*4),      # 2. <-- 新增！稳定内部激活
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim*4, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_slot_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> SlotHead:
        """
        Create head module for slot-based models (policy/value).
        Aggregates K slots per block using mean pooling, then passes to MLP.
        """
        modules = [
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim*4),
            nn.LayerNorm(self.config.embed_dim*4),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim*4, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return SlotHead(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_head_cont(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        from ding.model.common import ReparameterizationHead
        self.fc_policy_head = ReparameterizationHead(
            input_size=self.config.embed_dim,
            output_size=output_dim,
            layer_num=2,  # TODO: check the effect of layer_num
            sigma_type=self.sigma_type,
            activation=nn.GELU(approximate='tanh'),
            fixed_sigma_value=self.config.fixed_sigma_value if self.sigma_type == 'fixed' else 0.5,
            norm_type=None,
            bound_type=self.bound_type
        )
        return PolicyHeadCont(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=self.fc_policy_head
        )

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            if self.continuous_action_space:
                module_to_initialize = [self.head_value, self.head_rewards, self.head_observations]
            else:
                module_to_initialize = [self.head_policy, self.head_value, self.head_rewards, self.head_observations]
            for head in module_to_initialize:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break



    def _initialize_projection_input_dim(self) -> None:
        """Initialize the projection input dimension based on the number of observation tokens."""
        if self.num_observations_tokens == 16:
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:
            # self.projection_input_dim = self.config.embed_dim
            if self.task_embed_option == "concat_task_embed":
                self.projection_input_dim = self.config.embed_dim - self.task_embed_dim
            elif self.task_embed_option == "register_task_embed":
                self.projection_input_dim = self.config.embed_dim
            elif self.task_embed_option == "add_task_embed":
                self.projection_input_dim = self.config.embed_dim
            else:
                self.projection_input_dim = self.config.embed_dim
        else:
            self.projection_input_dim = self.config.embed_dim


    def _initialize_statistics(self) -> None:
        """Initialize counters for hit count and query count statistics."""
        self.hit_count = 0
        self.total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

    #@profile
    def _get_positional_embedding(self, layer, attn_type) -> torch.Tensor:
        """
         Helper function to get positional embedding for a given layer and attention type.

         Arguments:
         - layer (:obj:`int`): Layer index.
         - attn_type (:obj:`str`): Attention type, either 'key' or 'value'.

         Returns:
         - torch.Tensor: The positional embedding tensor.
         """
        attn_func = getattr(self.transformer.blocks[layer].attn, attn_type)
        if self.model_type == 'slot':
            token_positions = torch.arange(self.config.max_tokens, device=self.pos_emb.weight.device)
            block_positions = torch.div(token_positions, self.tokens_per_block, rounding_mode='floor')
            pos_matrix = self.pos_emb(block_positions)
        else:
            pos_matrix = self.pos_emb.weight

        out = attn_func(pos_matrix).view(
            1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
        ).transpose(1, 2)
        if torch.cuda.is_available():
            out = out.to(self.device)
        return out.detach()

    def forward(
        self,
        obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, Tuple]],
        is_init_infer: bool = True,
    ) -> "WorldModelOutput":
        """
        Overview:
            Forward pass for the world model. This method processes observation embeddings and/or action tokens,
            optionally adds position encodings (with or without rotary position embeddings), passes the resulting
            sequences through the transformer, and finally generates logits for observations, rewards, policy, and value.
        
        Arguments:
            - obs_embeddings_or_act_tokens (dict): Dictionary containing one or more of the following keys:
                - 'obs_embeddings': torch.Tensor representing observation embeddings.
                - 'act_tokens': torch.Tensor representing action tokens.
                - 'obs_embeddings_and_act_tokens': Combined data for both observations and actions.
            - past_keys_values (Optional[torch.Tensor]): Cached key-value pairs for the transformer. Defaults to None.
            - kvcache_independent (bool): Flag to indicate whether key-value caching is independent. Defaults to False.
            - is_init_infer (bool): Flag to indicate if this is the initial inference step. Defaults to True.
            - valid_context_lengths (Optional[torch.Tensor]): Valid lengths for the context. Defaults to None.
            - search_depth (Optional[List[int]]): List representing the search depth for each batch element, used for
                position encoding adjustment. Defaults to None.
        
        Returns:
            WorldModelOutput: An output instance containing:
                - x: Output features from the transformer.
                - logits for observations.
                - logits for rewards.
                - logits_ends (None).
                - logits for policy.
                - logits for value.
        """

        # Reset valid context lengths during initial inference phase.
        if is_init_infer:
            valid_context_lengths = None

        # sequences: torch.Tensor  # Output sequence to feed into transformer
        # num_steps: int           # Number of timesteps in the sequence

        # Process observation embeddings if available.
        if "obs_embeddings" in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens["obs_embeddings"]
            # If the observation embeddings have 2 dimensions, expand them to include a time dimension.
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)
            
            if not self.config.rotary_emb:
                # Add traditional position embeddings if not using rotary embeddings.
                sequences = self._add_position_embeddings(
                    obs_embeddings, num_steps,
                )

        # Process action tokens if available.
        elif "act_tokens" in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens["act_tokens"]
            if self.continuous_action_space:
                num_steps = 1
                act_tokens = act_tokens.float()
                if len(act_tokens.shape) == 2:
                    act_tokens = act_tokens.unsqueeze(1)
            else:
                if len(act_tokens.shape) == 3:
                    act_tokens = act_tokens.squeeze(1)
                num_steps = act_tokens.size(1)
            # Convert action tokens to embeddings using the action embedding table.
            act_embeddings = self.act_embedding_table(act_tokens)
            if not self.config.rotary_emb:
                sequences = self._add_position_embeddings(
                    act_embeddings, num_steps,
                )

        # Process combined observation embeddings and action tokens.
        elif "obs_embeddings_and_act_tokens" in obs_embeddings_or_act_tokens:
            # Process combined inputs to calculate either the target value (for training)
            # or target policy (for reanalyze phase).
            if self.continuous_action_space:
                sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens)
            else:
                sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens)

        elif "last_obs_embeddings_act_tokens_and_current_obs" in obs_embeddings_or_act_tokens:
            # Process combined inputs for continue epsiodes for root in mcts
            # if self.continuous_action_space:
            #     sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens)
            # else:
            sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens, True)

        else:
            raise ValueError("Input dictionary must contain one of 'obs_embeddings', 'act_tokens', or 'obs_embeddings_and_act_tokens'.")

        # Pass the sequence through the transformer.
        x = self._transformer_pass(sequences)
        
        # Generate logits for various components.
        logits_observations = self.head_observations(x, num_steps, 0)
        logits_rewards = self.head_rewards(x, num_steps, 0)
        logits_policy = self.head_policy(x, num_steps, 0)

        # ==================== [NEW] Fix1: Clip Policy Logits ====================
        # Prevent policy logits from exploding, which can cause gradient issues
        if self.use_policy_logits_clip:
            logits_policy = torch.clamp(
                logits_policy,
                min=self.policy_logits_clip_min,
                max=self.policy_logits_clip_max
            )
        # ========================================================================

        logits_value = self.head_value(x, num_steps, 0)

        if "last_obs_embeddings_act_tokens_and_current_obs" in obs_embeddings_or_act_tokens:
            logits_observations = logits_observations[:,-self.num_observations_tokens:,:]
            logits_rewards = logits_rewards[:,-1:,:]
            logits_policy = logits_policy[:,-1:,:]
            logits_value = logits_value[:,-1:,:]

        # The 'logits_ends' is intentionally set to None.
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    #@profile
    def _add_position_embeddings(self, embeddings, num_steps):
        """
        Add position embeddings to the input embeddings.

        Arguments:
            - embeddings (:obj:`torch.Tensor`): Input embeddings.
            - num_steps (:obj:`int`): Number of steps.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - is_init_infer (:obj:`bool`): Initialize inference.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Embeddings with position information added.
        """
        def _token_to_pos_index(token_indices: torch.Tensor) -> torch.Tensor:
            if self.model_type == 'slot':
                return torch.div(token_indices, self.tokens_per_block, rounding_mode='floor')
            else:
                return token_indices

        token_indices = torch.arange(num_steps, device=self.device).unsqueeze(0)
        pos_indices = _token_to_pos_index(token_indices)
        position_embeddings = self.pos_emb(pos_indices)
        return embeddings + position_embeddings

    #@profile
    def _process_obs_act_combined_cont(self, obs_embeddings_or_act_tokens):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        if self.continuous_action_space:
            act_tokens = act_tokens.float()
            if len(act_tokens.shape) == 2:  # TODO
                act_tokens = act_tokens.unsqueeze(-1)

        # B, L, E
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        # B, L*2, E
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return_result = obs_act_embeddings
        if not self.config.rotary_emb:
            token_indices = torch.arange(num_steps, device=self.device)
            if self.model_type == 'slot':
                pos_indices = torch.div(token_indices, self.tokens_per_block, rounding_mode='floor')
            else:
                pos_indices = token_indices
            return_result += self.pos_emb(pos_indices)
        return return_result, num_steps

    #@profile
    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens, eval_init_inference = False):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        if eval_init_inference:
            obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['last_obs_embeddings_act_tokens_and_current_obs']
            current_obs_embeddings = obs_embeddings[:,-1,:,:]
            obs_embeddings = obs_embeddings[:,:-1,:,:]
        else:
            obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
            if len(obs_embeddings.shape) == 3:
                obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                     -1)

        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        if eval_init_inference:
            return_result = torch.cat((obs_act_embeddings, current_obs_embeddings), dim=1)
            num_steps = return_result.size(1)
        else:
            return_result = obs_act_embeddings
            num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))

        if not self.config.rotary_emb:
            token_indices = torch.arange(num_steps, device=self.device)
            if self.model_type == 'slot':
                pos_indices = torch.div(token_indices, self.tokens_per_block, rounding_mode='floor')
            else:
                pos_indices = token_indices
            return_result += self.pos_emb(pos_indices)
        return return_result, num_steps

    def _transformer_pass(self, sequences):
        """
        Pass sequences through the transformer.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input sequences.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): Previous keys and values for transformer.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Transformer output.
        """
        return self.transformer(sequences)

    #@profile
    @torch.no_grad()
    def reset_for_initial_inference(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Arguments:
            - obs_act_dict (:obj:`torch.FloatTensor`): A dictionary containing 'obs', 'action', and 'current_obs'.
        Returns:
            - torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(obs_act_dict, dict):
            batch_obs = obs_act_dict['obs']  # obs_act_dict['obs'] is at timestep t
            batch_action = obs_act_dict['action'] # obs_act_dict['action'] is at timestep t
            batch_current_obs = obs_act_dict['current_obs'] # obs_act_dict['current_obs'] is at timestep t+1

        # Encode observations to latent embeddings.
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_obs)

        if batch_current_obs is not None:
            # ================ Collect and Evaluation Phase ================
            # Encode current observations to latent embeddings
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_current_obs)
            # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
            self.latent_state = current_obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action,
                                                                                   current_obs_embeddings)
        else:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            self.latent_state = obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action, None)

        return outputs_wm, self.latent_state

    #@profile
    @torch.no_grad()
    def wm_forward_for_initial_infererence(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - last_obs_embeddings (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n = last_obs_embeddings.shape[0]
        if n <= self.env_num and current_obs_embeddings is not None:
            if self.continuous_action_space:
                first_step_flag = not isinstance(batch_action[0], np.ndarray)
            else:
                first_step_flag = max(batch_action) == -1
            if first_step_flag:
                self._reset_env_history()
                outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings}, is_init_infer=True)
                self._append_obs_to_history(current_obs_embeddings)
            else:
                ready_env_num = current_obs_embeddings.shape[0]
                batch_action = batch_action[:ready_env_num]

                self._append_obs_to_history(current_obs_embeddings)

                if self.continuous_action_space:
                    act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(1)
                else:
                    act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)

                self._append_act_to_history(act_tokens)

                obs_seq = torch.stack(self.obs_history, dim=1).to(self.device)
                act_seq = torch.stack(self.act_history, dim=1).to(self.device)

                outputs_wm = self.forward({'last_obs_embeddings_act_tokens_and_current_obs': (obs_seq, act_seq)}, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, self.num_observations_tokens,
                                                          self.config.embed_dim)  # (BL, K) for unroll_step=1

            last_obs_embeddings = last_obs_embeddings[:, :-1, :]
            batch_action = torch.from_numpy(batch_action).to(last_obs_embeddings.device)
            if self.continuous_action_space:
                act_tokens = batch_action
            else:
                act_tokens = rearrange(batch_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            # Each sample in the batch (last_obs_embeddings, act_tokens) corresponds to the same time step
            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)})

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
    def forward_initial_inference(self, obs_act_dict):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict)
        # =============================================================================

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    def _reset_env_history(self) -> None:
            self.obs_history = []
            self.act_history = []

    def _append_obs_to_history(self, next_obs_embedding: torch.Tensor) -> None:

        self.obs_history.append(next_obs_embedding.detach())

        if len(self.obs_history) > self.context_length_in_blocks:
            excess = len(self.obs_history) - self.context_length_in_blocks
            self.obs_history = self.obs_history[excess:]

    def _append_act_to_history(self, action: Any) -> None:
        self.act_history.append(action.detach())

        if len(self.act_history) == self.context_length_in_blocks:
            self.act_history = self.act_history[1:]

    #@profile
    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    search_depth=[]):
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

        max_history_len = self.max_blocks
        if len(state_action_history) > max_history_len:
            state_action_history = state_action_history[-max_history_len:]

        history_states = []
        history_actions = []

        for state, act in state_action_history:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(self.device)
            if isinstance(act, np.ndarray):
                act = torch.from_numpy(act).to(self.device)

            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device)
            if not isinstance(act, torch.Tensor):
                act = torch.tensor(act, device=self.device)
            history_states.append(state)
            history_actions.append(act)

        obs_embeddings = torch.stack(history_states, dim=1)  # (B, L, K, E)

        if self.continuous_action_space:
            act_tokens = torch.stack(history_actions, dim=1)  # (B, L, action_space_size)
        else:
            act_tokens = torch.stack(history_actions, dim=1)  # (B, L)
            act_tokens = act_tokens.unsqueeze(-1)  # (B, L, 1)

        outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})

        reward = outputs_wm.logits_rewards[:, -1, :]  # (B, support_size)
        policy = outputs_wm.logits_policy[:, -1, :]  # (B, action_space_size)
        value = outputs_wm.logits_value[:, -1, :]  # (B, support_size)
        next_latent = outputs_wm.logits_observations[:, -self.num_observations_tokens:, :]

        self.latent_state = next_latent

        return (None, self.latent_state, reward, policy, value)


    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # ========= for visual analysis =========
        # Uncomment the lines below for visual analysis in Pong
        # self.plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong_H10_H4_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='pong_H10_H4_tsne')
        # Uncomment the lines below for visual analysis in visual match
        # self.plot_latent_tsne_each_and_all(obs_embeddings, suffix='visual_match_memlen1-60-15_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='visual_match_memlen1-60-15_tsne')

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
        if self.continuous_action_space:
            act_tokens = batch['actions']
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})
        
        # [新增] 从模型输出中获取中间张量 x，并分离计算图
        intermediate_tensor_x = outputs.output_sequence.detach()

        global_step = kwargs.get('global_step', 0)
        # if global_step >= 0 and global_step % 10000 == 0: # 20k
        if global_step > 0 and global_step % 100000000000 == 0: # 20k # TODO

            with torch.no_grad():
                # 将logits转换为标量值
                # 注意：outputs的形状是(B, L, E)，我们需要reshape
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

                #  ========== for visualization ==========
                # Uncomment the lines below for visual analysis
                # original_images, reconstructed_images = batch['observations'], reconstructed_images
                # target_policy = batch['target_policy']
                # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
                #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
                # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
                #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
                #  ========== for visualization ==========
                # ========== Calculate reconstruction loss and perceptual loss ============
                latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
                perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            else:
                # TODO:
                latent_recon_loss = self.latent_recon_loss
                perceptual_loss = self.perceptual_loss

        elif self.obs_type == 'slot':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            latent_recon_loss = self.latent_recon_loss


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
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            # original_images, reconstructed_images = batch['observations'], reconstructed_images

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
            #                                                        reconstructed_images)
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

            torch.cuda.empty_cache()
        else:
            dormant_ratio_transformer = torch.tensor(0.)
            dormant_ratio_head = torch.tensor(0.)

        #  ========== for visualization ==========
        # Uncomment the lines below for visualization
        # predict_policy = outputs.logits_policy
        # predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        # predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # import pdb; pdb.set_trace()
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=[], suffix='pong_H10_H4_0613')

        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_success_episode')
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_fail_episode')
        #  ========== for visualization ==========

        # For training stability, use target_tokenizer to compute the true next latent state representations
        with torch.no_grad():
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, _ = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-self.num_observations_tokens], 'b t o -> (b t) o')
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

            #  ========== for debugging ==========
            # print('loss_obs:', loss_obs.mean())
            # assert not torch.isnan(loss_obs).any(), "loss_obs contains NaN values"
            # assert not torch.isinf(loss_obs).any(), "loss_obs contains Inf values"
            # for name, param in self.tokenizer.encoder.named_parameters():
            #     print('name, param.mean(), param.std():', name, param.mean(), param.std())
        elif self.predict_latent_loss_type == 'cos_sim':
            # --- 修复后的代码 (推荐方案) ---
            # 使用余弦相似度损失 (Cosine Similarity Loss)
            # F.cosine_similarity 计算的是相似度，范围是 [-1, 1]。我们希望最大化它，
            # 所以最小化 1 - similarity。
            # reduction='none' 使得我们可以像原来一样处理mask
            # print("predict_latent_loss_type == 'cos_sim'")
            cosine_sim_loss = 1 - F.cosine_similarity(logits_observations, labels_observations, dim=-1)
            loss_obs = cosine_sim_loss

        # Apply mask to loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].unsqueeze(-1).repeat(1, 1, self.num_observations_tokens).contiguous().view(-1)
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

        if not self.continuous_action_space:
            loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy,
                                                                                            batch,
                                                                                            element='policy')
        else:
            # NOTE: for continuous action space
            if self.config.policy_loss_type == 'simple':
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont_simple(outputs, batch)
            else:
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(outputs, batch)
            
            loss_policy = orig_policy_loss + self.policy_entropy_weight * policy_entropy_loss
            policy_entropy = - policy_entropy_loss

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # ==== TODO: calculate the new priorities for each transition. ====
        # value_priority = L1Loss(reduction='none')(labels_value.squeeze(-1), outputs['logits_value'][:, 0])
        # value_priority = value_priority.data.cpu().numpy() + 1e-6

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
                mask_padding = batch['mask_padding'][:, 1:seq_len].unsqueeze(-1).repeat(1, 1, self.num_observations_tokens)
                # Adjust loss shape to (batch_size, seq_len)
                loss_tmp = loss_tmp.view(batch['actions'].shape[0], seq_len, -1)
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

        # 为了让外部的训练循环能够获取encoder的输出，我们将其加入返回字典
        # 使用 .detach() 是因为这个张量仅用于后续的clip操作，不应影响梯度计算
        detached_obs_embeddings = obs_embeddings.detach()

        if self.continuous_action_space:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                continuous_action_space=True,
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
                policy_mu=mu,
                policy_sigma=sigma,
                target_sampled_actions=target_sampled_actions,
                
                value_priority=value_priority,
                intermediate_tensor_x=intermediate_tensor_x,
                obs_embeddings=detached_obs_embeddings, # <-- 新增

                # logits_value_mean=outputs.logits_value.mean(),
                # logits_value_max=outputs.logits_value.max(),
                # logits_value_min=outputs.logits_value.min(),
                # logits_policy_mean=outputs.logits_policy.mean(),
                # logits_policy_max=outputs.logits_policy.max(),
                # logits_policy_min=outputs.logits_policy.min(),
                logits_value=outputs.logits_value.detach(),  # 使用detach()，因为它仅用于分析和裁剪，不参与梯度计算
                logits_reward=outputs.logits_rewards.detach(),
                logits_policy=outputs.logits_policy.detach(),
            )
        else:
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

                # logits_value_mean=outputs.logits_value.mean(),
                # logits_value_max=outputs.logits_value.max(),
                # logits_value_min=outputs.logits_value.min(),
                # logits_policy_mean=outputs.logits_policy.mean(),
                # logits_policy_max=outputs.logits_policy.max(),
                # logits_policy_min=outputs.logits_policy.min(),
                logits_value=outputs.logits_value.detach(),  # 使用detach()，因为它仅用于分析和裁剪，不参与梯度计算
                logits_reward=outputs.logits_rewards.detach(),
                logits_policy=outputs.logits_policy.detach(),
            )

    
    # TODO: test correctness
    def _calculate_policy_loss_cont_simple(self, outputs, batch: dict):
        """
        Simplified policy loss calculation for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.

        Returns:
            - policy_loss (:obj:`torch.Tensor`): The simplified policy loss.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        # Get the policy logits and batch data
        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding'].contiguous().view(-1)
        target_policy = batch['target_policy'].contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = batch['child_sampled_actions'].contiguous().view(batch_size * num_unroll_steps, -1, action_space_size)

        # Flatten for vectorized computation
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        
        # Extract mean and standard deviation from logits
        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        dist = Independent(Normal(mu, sigma), 1)  # Create the normal distribution

        # Find the indices of the maximum values in the target policy
        target_best_action_idx = torch.argmax(target_policy, dim=1)

        # Select the best actions based on the indices
        target_best_action = target_sampled_actions[torch.arange(target_best_action_idx.size(0)), target_best_action_idx]

        # Clip the target actions to prevent numerical issues during arctanh
        # target_best_action_clamped = torch.clamp(target_best_action, -1 + 1e-6, 1 - 1e-6)
        target_best_action_clamped = torch.clamp(target_best_action, -0.999, 0.999)
        target_best_action_before_tanh = torch.arctanh(target_best_action_clamped)

        # Calculate the log probability of the best action
        log_prob_best_action = dist.log_prob(target_best_action_before_tanh)

        # Mask the log probability with the padding mask
        log_prob_best_action = log_prob_best_action * mask_batch

        # Return the negative log probability as the policy loss (we want to maximize log_prob)
        # policy_loss = -log_prob_best_action.mean()
        policy_loss = -log_prob_best_action

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -policy_entropy * mask_batch
        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def _calculate_policy_loss_cont(self, outputs, batch: dict, task_id=None) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the policy loss for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.
        Returns:
            - policy_loss (:obj:`torch.Tensor`): The calculated policy loss.
            - policy_entropy_loss (:obj:`torch.Tensor`): The entropy loss of the policy.
            - target_policy_entropy (:obj:`float`): The entropy of the target policy distribution.
            - target_sampled_actions (:obj:`torch.Tensor`): The actions sampled from the target policy.
            - mu (:obj:`torch.Tensor`): The mean of the normal distribution.
            - sigma (:obj:`torch.Tensor`): The standard deviation of the normal distribution.
        """
        if  task_id is None:
            batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size
        else:
            batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
                0], self.config.num_unroll_steps, self.config.action_space_size_list[task_id]
        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding']
        child_sampled_actions_batch = batch['child_sampled_actions']
        target_policy = batch['target_policy']

        # Flatten the unroll step dimension for easier vectorized operations
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        mask_batch = mask_batch.contiguous().view(-1)
        child_sampled_actions_batch = child_sampled_actions_batch.contiguous().view(batch_size * num_unroll_steps, -1,
                                                                                    action_space_size)

        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        mu = mu.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        dist = Independent(Normal(mu, sigma), 1)

        target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = child_sampled_actions_batch

        policy_entropy = dist.entropy().mean(dim=1)
        policy_entropy_loss = -policy_entropy * mask_batch

        # NOTE： Alternative way to calculate the log probability of the target actions
        # y = 1 - target_sampled_actions.pow(2)
        # target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -1 + 1e-6, 1 - 1e-6)
        # target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)
        # log_prob = dist.log_prob(target_sampled_actions_before_tanh)
        # log_prob = log_prob - torch.log(y + 1e-6).sum(-1)
        # log_prob_sampled_actions = log_prob

        base_dist = Normal(mu, sigma)
        tanh_transform = TanhTransform()
        dist = TransformedDistribution(base_dist, [tanh_transform])
        dist = Independent(dist, 1)
        target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -0.999, 0.999)
        # assert torch.all(target_sampled_actions_clamped < 1) and torch.all(target_sampled_actions_clamped > -1), "Actions are not properly clamped."
        log_prob = dist.log_prob(target_sampled_actions_clamped)
        log_prob_sampled_actions = log_prob

        # KL as projector
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)

        # KL as projector
        policy_loss = -torch.sum(
            torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
        ) * mask_batch

        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

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
        obs_embeddings = obs_embeddings[:, 1:]
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)

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

        if self.continuous_action_space:
            return None, labels_value.reshape(-1, self.support_size)
        else:
            return labels_policy.reshape(-1, self.action_space_size), labels_value.reshape(-1, self.support_size)

    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"
