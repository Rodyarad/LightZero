import pytest
import torch
from easydict import EasyDict

from lzero.model.unizero_model import UniZeroModel
from lzero.policy.scaling_transform import DiscreteSupport, InverseScalarTransform

NUM_SLOTS = 2
SLOT_DIM = 16
ACTION_SPACE = 4
UNROLL = 4
SUPPORT_SIZE = 21
BATCH = 2


def make_slot_world_model_cfg():
    return EasyDict(
        encoder_type='resnet',
        continuous_action_space=False,
        model_type='slot',
        tokens_per_block=NUM_SLOTS * 2,
        max_blocks=UNROLL,
        max_tokens=NUM_SLOTS * 2 * UNROLL,
        context_length=NUM_SLOTS * 2 * 2,
        gru_gating=False,
        device='cpu',
        analysis_sim_norm=False,
        analysis_dormant_ratio_weight_rank=False,
        action_space_size=ACTION_SPACE,
        group_size=8,
        attention='causal',
        num_layers=1,
        num_heads=2,
        embed_dim=SLOT_DIM,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        support_size=SUPPORT_SIZE,
        max_cache_size=50,
        env_num=2,
        latent_recon_loss_weight=0.,
        perceptual_loss_weight=0.,
        policy_entropy_weight=1e-4,
        final_norm_option_in_head='LayerNorm',
        final_norm_option_in_encoder='LayerNorm',
        predict_latent_loss_type='mse',
        obs_type='slot',
        gamma=1,
        dormant_threshold=0.025,
        rotary_emb=False,
        rope_theta=10000,
        max_seq_len=8192,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        decode_loss_mode=None,
        task_embed_option=None,
        use_task_embed=False,
        use_normal_head=True,
        use_softmoe_head=False,
        use_moe_head=False,
        num_experts_in_moe_head=4,
        moe_in_transformer=False,
        multiplication_moe_in_transformer=False,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_experts_of_moe_in_transformer=8,
        use_priority=True,
        num_slots=NUM_SLOTS,
        norm_type='LN',
        num_simulations=2,
        game_segment_length=5,
    )


def make_batch(mask_padding: torch.Tensor):
    T = mask_padding.shape[1]
    rewards = torch.zeros(BATCH, T, SUPPORT_SIZE)
    rewards[:, :, SUPPORT_SIZE // 2] = 1.0
    target_policy = torch.full((BATCH, T, ACTION_SPACE), 1.0 / ACTION_SPACE)
    return {
        'observations': torch.randn(BATCH, T, NUM_SLOTS, SLOT_DIM),
        'actions': torch.randint(0, ACTION_SPACE, (BATCH, T)),
        'timestep': torch.arange(T).unsqueeze(0).repeat(BATCH, 1),
        'rewards': rewards,
        'mask_padding': mask_padding,
        'ends': torch.zeros(BATCH, T, dtype=torch.long),
        'target_value': rewards.clone(),
        'target_policy': target_policy,
        'scalar_target_value': torch.zeros(BATCH, T),
    }


@pytest.mark.unittest
class TestSlotWorldModelLoss:

    def setup_method(self):
        torch.manual_seed(0)
        self.model = UniZeroModel(
            observation_shape=(NUM_SLOTS, SLOT_DIM),
            action_space_size=ACTION_SPACE,
            model_type='slot',
            world_model_cfg=make_slot_world_model_cfg(),
            norm_type='LN',
        )
        self.model.eval()
        value_support = DiscreteSupport(-10., 11., 1., 'cpu')
        self.inverse_handle = InverseScalarTransform(value_support, True)

    def compute_loss(self, mask_padding):
        batch = make_batch(mask_padding)
        return self.model.world_model.compute_loss(batch, self.model.world_model.tokenizer, self.inverse_handle)

    def test_loss_finite_with_normal_mask(self):
        mask_padding = torch.ones(BATCH, UNROLL, dtype=torch.bool)
        mask_padding[:, -1] = False
        losses = self.compute_loss(mask_padding)
        assert torch.isfinite(losses.loss_total).all()

    def test_loss_finite_when_only_first_step_valid(self):
        """Regression test: a batch where every sample sits at an episode's last valid
        step has mask_padding[:, 1:] all False, so the obs-loss normalizer
        mask_padding[:, 1:].sum() is zero and used to yield loss_total = 0/0 = nan."""
        mask_padding = torch.zeros(BATCH, UNROLL, dtype=torch.bool)
        mask_padding[:, 0] = True
        losses = self.compute_loss(mask_padding)
        assert torch.isfinite(losses.loss_total).all()
        assert losses.intermediate_losses['loss_obs'] == 0.0
