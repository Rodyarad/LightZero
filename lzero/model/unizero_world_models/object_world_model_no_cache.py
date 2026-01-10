import logging
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lzero.model.utils import calculate_dormant_ratio, compute_average_weight_magnitude, compute_effective_rank
from .tokenizer import Tokenizer
from .transformer import SlotTransformer, SlotTransformerConfig
from .aux_transformer import TransformerEncoder
from .utils import LossWithIntermediateLosses, init_weights, WorldModelOutput

logging.getLogger().setLevel(logging.DEBUG)


class ObjectWorldModelNoCache(nn.Module):

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

        # Latent state placeholder for inference
        self.latent_state = None
        
        logging.info("ObjectWorldModelNoCache initialized (NO KV caching)")

    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
        self.policy_entropy_weight = self.config.policy_entropy_weight
        self.predict_latent_loss_type = self.config.predict_latent_loss_type
        self.group_size = self.config.group_size
        self.num_groups = self.config.embed_dim // self.group_size
        self.obs_type = getattr(self.config, 'obs_type', 'slots')
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
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.slots_dim = self.config.slots_dim
        self.num_slots = self.config.num_slots

        # Policy stability options
        self.use_policy_logits_clip = getattr(self.config, 'use_policy_logits_clip', False)
        self.policy_logits_clip_min = getattr(self.config, 'policy_logits_clip_min', -10.0)
        self.policy_logits_clip_max = getattr(self.config, 'policy_logits_clip_max', 10.0)

        self.use_policy_loss_temperature = getattr(self.config, 'use_policy_loss_temperature', False)
        self.policy_loss_temperature = getattr(self.config, 'policy_loss_temperature', 1.0)


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
            nn.LayerNorm(self.config.slots_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim * 4),
            nn.LayerNorm(self.config.embed_dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim * 4, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return nn.Sequential(*modules)

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

    def forward(
        self,
        slots: torch.Tensor,
        actions: torch.Tensor,
        valid_context_lengths: Optional[torch.Tensor] = None,
    ) -> "WorldModelOutput":
        """
        Overview:
            Forward pass for the object world model using SlotTransformer.
        Arguments:
            - slots: Slot representations, shape (B, T, N, slots_dim).
            - actions: Action tokens, shape (B, T) or (B, T, action_dim).
            - valid_context_lengths: Valid context lengths for each batch element.
        Returns:
            - WorldModelOutput: Output containing slots, logits for observations, rewards, policy, and value.
        """
        B, T, N, _ = slots.shape
        
        # Process through SlotTransformer (no KV cache)
        x = self.transformer(slots, actions, kv_cache=None, valid_context_lengths=valid_context_lengths)
        
        # Dynamics transformer: predict next step slots
        predicted_slots = self.dynamics_transformer(x)  # (B, N, slots_dim)
        
        # Prediction transformer: process slots for reward/value/policy
        slots_for_prediction = self.prediction_transformer(x)  # (B, N, slots_dim)
        
        # Aggregate slots (sum over slots dimension) for prediction heads
        slots_agg = slots_for_prediction.sum(dim=1)  # (B, slots_dim)
        
        # Apply prediction heads
        logits_value = self.head_value(slots_agg)
        logits_rewards = self.head_rewards(slots_agg)
        logits_policy = self.head_policy(slots_agg)
        
        # Clip policy logits if enabled
        if self.use_policy_logits_clip:
            logits_policy = torch.clamp(
                logits_policy,
                min=self.policy_logits_clip_min,
                max=self.policy_logits_clip_max
            )
        
        return WorldModelOutput(
            x, 
            logits_observations=predicted_slots, 
            logits_rewards=logits_rewards, 
            logits_ends=None, 
            logits_policy=logits_policy, 
            logits_value=logits_value
        )

    @torch.no_grad()
    def forward_initial_inference(self, slot_act_dict):
        """
        Perform initial inference based on the given slot-action dictionary.

        Arguments:
            - slot_act_dict (:obj:`dict`): Dictionary containing 'obs' (slots), 'action', and 'current_obs'.
        Returns:
            - tuple: (output_sequence, latent_state, logits_rewards, logits_policy, logits_value)
        """


        return (
            outputs_wm.output_sequence,
            self.latent_state,
            outputs_wm.logits_rewards,
            outputs_wm.logits_policy,
            outputs_wm.logits_value
        )

    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of (state, action) history.
        Returns:
            - tuple: (output_sequence, latent_state, reward, logits_policy, logits_value)
        """

        
        return (
            outputs_wm.output_sequence,
            self.latent_state,
            outputs_wm.logits_rewards,
            outputs_wm.logits_policy,
            outputs_wm.logits_value
        )


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
