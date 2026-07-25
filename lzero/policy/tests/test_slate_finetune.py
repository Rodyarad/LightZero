"""
Unit tests for online SLATE fine-tuning during the UniZero train phase
(see ``lzero/policy/slate_finetune.py``).

Run with:
    pytest lzero/policy/tests/test_slate_finetune.py -sv -m unittest
"""
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lzero.policy.slate_finetune import (CollectorSlotEncoder, SlateEnvConfig,
                                         build_finetune_slate,
                                         encode_flat_windows_to_slots,
                                         encode_image_sequences_to_slots,
                                         encode_images_to_slots,
                                         prepare_slate_obs_batch,
                                         slate_finetune_step)
from zoo.ocr.slate.slate import SLATE

OBS_SIZE = 16
OBS_CHANNELS = 3

TINY_OCR_CONFIG = dict(
    name='SLATE',
    tau_start=1.0,
    tau_final=0.1,
    tau_steps=10,
    hard=False,
    use_cnn_feat=False,
    use_bcdec=False,
    dvae=dict(vocab_size=3, d_model=8),
    cnn=dict(hidden_size=6),
    slotattr=dict(
        num_iterations=2,
        num_slots=2,
        num_slot_heads=1,
        slot_size=4,
        mlp_hidden_size=5,
        pos_channels=4,
    ),
    tfdec=dict(num_dec_blocks=1, num_dec_heads=1),
    learning=dict(
        lr_half_life=100,
        lr_dvae=3e-4,
        lr_enc=1e-4,
        lr_dec=3e-4,
        lr_warmup_steps=4,
        dropout=0.0,
        clip=0.05,
    ),
)


def make_tiny_slate(clip: float = None) -> SLATE:
    config = copy.deepcopy(TINY_OCR_CONFIG)
    if clip is not None:
        config['learning']['clip'] = clip
    config_ocr = OmegaConf.create(config)
    config_env = SlateEnvConfig(OBS_SIZE, OBS_CHANNELS)
    return SLATE(config_ocr, config_env, observation_space=None)


def make_images(n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand((n, OBS_CHANNELS, OBS_SIZE, OBS_SIZE), generator=generator)


def make_raw_obs_batch(batch_size: int, num_steps: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(
        0, 256, size=(batch_size, num_steps, OBS_SIZE, OBS_SIZE, OBS_CHANNELS), dtype=np.uint8
    )


@pytest.mark.unittest
class TestPrepareSlateObsBatch:

    def test_shape_and_range(self):
        raw = make_raw_obs_batch(2, 3)
        mask = np.ones((2, 3))
        images = prepare_slate_obs_batch(raw, mask, device='cpu')
        assert images.shape == (6, OBS_CHANNELS, OBS_SIZE, OBS_SIZE)
        assert images.dtype == torch.float32
        assert images.min() >= 0.0 and images.max() <= 1.0
        assert torch.allclose(images[0, :, 0, 0], torch.from_numpy(raw[0, 0, 0, 0]).float() / 255.0)

    def test_mask_filters_padded_frames(self):
        raw = make_raw_obs_batch(2, 4)
        mask = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        images = prepare_slate_obs_batch(raw, mask, device='cpu')
        assert images.shape[0] == 3
        expected = np.stack([raw[0, 0], raw[0, 1], raw[1, 0]])
        expected = torch.from_numpy(expected).permute(0, 3, 1, 2).float() / 255.0
        assert torch.allclose(images, expected)

    def test_max_images_subsampling(self):
        raw = make_raw_obs_batch(3, 4)
        mask = np.ones((3, 4))
        images = prepare_slate_obs_batch(raw, mask, device='cpu', max_images=5)
        assert images.shape == (5, OBS_CHANNELS, OBS_SIZE, OBS_SIZE)
        images = prepare_slate_obs_batch(raw, mask, device='cpu', max_images=100)
        assert images.shape[0] == 12


@pytest.mark.unittest
class TestSlateFinetuneStep:

    def test_update_changes_slate_params_and_returns_metrics(self):
        slate = make_tiny_slate()
        params_before = copy.deepcopy(slate._module.state_dict())
        images = make_images(4)

        log = slate_finetune_step(slate, images, step=0)

        for key in ['slate/loss', 'slate/dvae_mse', 'slate/cross_entropy', 'slate/tau',
                    'slate/lr_dvae', 'slate/lr_enc', 'slate/lr_dec', 'slate/norm',
                    'slate/num_images', 'slate/step']:
            assert key in log, f"missing metric {key}"
            assert np.isfinite(log[key]), f"metric {key} is not finite"
        assert log['slate/num_images'] == 4
        assert log['slate/step'] == 0

        params_after = slate._module.state_dict()
        changed = [
            name for name, before in params_before.items()
            if not torch.equal(before, params_after[name])
        ]
        assert len(changed) > 0, "SLATE update did not change any parameter"
        assert any(name.startswith('_dvae') for name in changed)
        assert any(name.startswith(('_enc', '_slotattn', '_slotproj')) for name in changed)
        assert any(name.startswith(('_tfdec', '_dict', '_bos_token', '_z_pos', '_out')) for name in changed)

    def test_learning_clip_is_used(self):
        clip_value = 1e-3
        slate = make_tiny_slate(clip=clip_value)
        images = make_images(4)
        log = slate_finetune_step(slate, images, step=1)

        assert log['slate/norm'] > clip_value
        max_abs_grad = max(
            param.grad.detach().abs().max().item()
            for param in slate._module.parameters() if param.grad is not None
        )
        assert max_abs_grad <= clip_value * (1 + 1e-4)

    def test_isolation_from_other_modules(self):
        torch.manual_seed(0)
        other_model = torch.nn.Linear(8, 8)
        other_params_before = copy.deepcopy(other_model.state_dict())

        slate = make_tiny_slate()
        slate_finetune_step(slate, make_images(4), step=1)

        for name, param in other_model.named_parameters():
            assert param.grad is None
            assert torch.equal(param.data, other_params_before[name])

    def test_slate_params_untouched_by_external_update(self):
        slate = make_tiny_slate()
        slate_params_before = copy.deepcopy(slate._module.state_dict())

        torch.manual_seed(0)
        other_model = torch.nn.Linear(8, 8)
        opt = torch.optim.SGD(other_model.parameters(), lr=0.1)
        loss = other_model(torch.randn(4, 8)).pow(2).mean()
        loss.backward()
        opt.step()

        for name, param in slate._module.state_dict().items():
            assert torch.equal(param, slate_params_before[name])

    def test_tau_and_lr_schedules_progress(self):
        slate = make_tiny_slate()
        images = make_images(2)
        log_start = slate_finetune_step(slate, images, step=0)
        log_end = slate_finetune_step(slate, images, step=TINY_OCR_CONFIG['tau_steps'])

        assert log_start['slate/tau'] == pytest.approx(TINY_OCR_CONFIG['tau_start'], abs=0.2)
        assert log_end['slate/tau'] == pytest.approx(TINY_OCR_CONFIG['tau_final'])
        learning = TINY_OCR_CONFIG['learning']
        assert log_start['slate/lr_dvae'] == pytest.approx(learning['lr_dvae'])
        assert log_end['slate/lr_dvae'] == pytest.approx(learning['lr_dvae'])
        assert log_start['slate/lr_enc'] < log_end['slate/lr_enc']
        assert log_end['slate/lr_enc'] <= learning['lr_enc']


@pytest.mark.unittest
class TestEncodeImagesToSlots:
    """
    With finetune_slate=True the replay buffer stores only raw images, and these encoders
    produce the slots wherever they are needed, using the current SLATE weights.
    """

    NUM_SLOTS = TINY_OCR_CONFIG['slotattr']['num_slots']
    SLOT_DIM = TINY_OCR_CONFIG['slotattr']['slot_size']

    def test_single_frames_shape_and_detached(self):
        slate = make_tiny_slate()
        raw = make_raw_obs_batch(1, 5)[0]
        slots = encode_images_to_slots(slate, raw, device='cpu')
        assert slots.shape == (5, self.NUM_SLOTS, self.SLOT_DIM)
        assert slots.dtype == torch.float32
        assert not slots.requires_grad
        with pytest.raises(RuntimeError):
            slots.sum().backward()
        assert slate._module.training

    def test_single_frames_chunked(self):
        slate = make_tiny_slate()
        raw = make_raw_obs_batch(1, 5)[0]
        slots = encode_images_to_slots(slate, raw, device='cpu', chunk_size=2)
        assert slots.shape == (5, self.NUM_SLOTS, self.SLOT_DIM)

    def test_sequences_shape_and_detached(self):
        slate = make_tiny_slate()
        raw = make_raw_obs_batch(2, 4)
        slots = encode_image_sequences_to_slots(slate, raw, device='cpu')
        assert slots.shape == (2, 4, self.NUM_SLOTS, self.SLOT_DIM)
        assert not slots.requires_grad
        assert slate._module.training

    def test_flat_windows_shape_and_divisibility(self):
        slate = make_tiny_slate()
        window_len = 3
        raw = make_raw_obs_batch(1, 2 * window_len)[0]
        slots = encode_flat_windows_to_slots(slate, raw, window_len, device='cpu')
        assert slots.shape == (6, self.NUM_SLOTS, self.SLOT_DIM)
        assert not slots.requires_grad
        with pytest.raises(AssertionError):
            encode_flat_windows_to_slots(slate, raw[:5], window_len, device='cpu')

    def test_encoding_does_not_change_slate_params(self):
        slate = make_tiny_slate()
        params_before = copy.deepcopy(slate._module.state_dict())
        encode_image_sequences_to_slots(slate, make_raw_obs_batch(2, 3), device='cpu')
        encode_images_to_slots(slate, make_raw_obs_batch(1, 3)[0], device='cpu')
        for name, param in slate._module.state_dict().items():
            assert torch.equal(param, params_before[name])


def make_frame(i: int) -> np.ndarray:
    rng = np.random.RandomState(i)
    return rng.randint(0, 256, size=(OBS_SIZE, OBS_SIZE, OBS_CHANNELS), dtype=np.uint8)


def spy_on_get_slots(slate):
    """Wrap slate._module._get_slots to record the prev_slots argument of each call."""
    calls = []
    original = slate._module._get_slots

    def spy(obs, prev_slots=None, **kwargs):
        calls.append(None if prev_slots is None else prev_slots.detach().clone())
        return original(obs, prev_slots=prev_slots, **kwargs)

    slate._module._get_slots = spy
    return calls


@pytest.mark.unittest
class TestCollectorSlotEncoder:
    """
    The collector/evaluator-side replacement for the env-side SlotExtractorWrapper:
    encodes raw frames with the policy's current SLATE weights, maintaining the
    per-env prev_slots recurrence.
    """

    NUM_SLOTS = TINY_OCR_CONFIG['slotattr']['num_slots']
    SLOT_DIM = TINY_OCR_CONFIG['slotattr']['slot_size']

    def test_encode_shapes_and_state(self):
        encoder = CollectorSlotEncoder(make_tiny_slate(), device='cpu')
        slots = encoder.encode({0: make_frame(0), 3: make_frame(3)})
        assert set(slots.keys()) == {0, 3}
        for value in slots.values():
            assert isinstance(value, np.ndarray)
            assert value.shape == (self.NUM_SLOTS, self.SLOT_DIM)
            assert value.dtype == np.float32
        assert encoder.has_state(0) and encoder.has_state(3)
        assert not encoder.has_state(1)
        assert encoder.encode({}) == {}

    def test_prev_slots_recurrence(self):
        slate = make_tiny_slate()
        calls = spy_on_get_slots(slate)
        encoder = CollectorSlotEncoder(slate, device='cpu')

        out_step0 = encoder.encode({0: make_frame(0)})
        assert len(calls) == 2
        assert calls[0] is None
        assert calls[1] is not None
        calls.clear()

        out_step1 = encoder.encode({0: make_frame(1)})
        assert len(calls) == 1
        assert np.allclose(calls[0][0].numpy(), out_step0[0])
        calls.clear()

        encoder.reset(0)
        encoder.encode({0: make_frame(2)})
        assert len(calls) == 2 and calls[0] is None
        assert out_step1[0].shape == (self.NUM_SLOTS, self.SLOT_DIM)

    def test_mixed_fresh_and_continuing_envs(self):
        slate = make_tiny_slate()
        encoder = CollectorSlotEncoder(slate, device='cpu')
        encoder.encode({0: make_frame(0)})
        calls = spy_on_get_slots(slate)

        slots = encoder.encode({0: make_frame(1), 1: make_frame(2)})
        assert set(slots.keys()) == {0, 1}
        assert len(calls) == 2
        assert calls[0] is None and calls[1] is not None

    def test_encode_init_reuses_cached_slots(self):
        slate = make_tiny_slate()
        encoder = CollectorSlotEncoder(slate, device='cpu')
        out = encoder.encode({0: make_frame(0)})

        calls = spy_on_get_slots(slate)
        init_slots = encoder.encode_init({0: make_frame(0), 1: make_frame(1)})
        assert np.array_equal(init_slots[0], out[0])
        assert len(calls) == 2
        assert encoder.has_state(1)

    def test_reset(self):
        encoder = CollectorSlotEncoder(make_tiny_slate(), device='cpu')
        encoder.encode({0: make_frame(0), 1: make_frame(1)})
        encoder.reset(0)
        assert not encoder.has_state(0) and encoder.has_state(1)
        encoder.reset()
        assert not encoder.has_state(1)

    def test_encoding_is_isolated_from_training_state(self):
        slate = make_tiny_slate()
        params_before = copy.deepcopy(slate._module.state_dict())
        encoder = CollectorSlotEncoder(slate, device='cpu')
        encoder.encode({0: make_frame(0)})
        assert slate._module.training
        for name, param in slate._module.state_dict().items():
            assert torch.equal(param, params_before[name])

    def test_requires_slate(self):
        with pytest.raises(AssertionError):
            CollectorSlotEncoder(None, device='cpu')


@pytest.mark.unittest
class TestBuildFinetuneSlate:

    def test_build_loads_pretrained_checkpoint(self, tmp_path):
        config_path = tmp_path / 'tiny_slate.yaml'
        OmegaConf.save(OmegaConf.create(TINY_OCR_CONFIG), config_path)

        source = make_tiny_slate()
        checkpoint_path = tmp_path / 'tiny_slate.pth'
        torch.save(source.save(), checkpoint_path)

        slate = build_finetune_slate(
            ocr_config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            obs_size=OBS_SIZE,
            obs_channels=OBS_CHANNELS,
            device='cpu',
        )

        source_state = source._module.state_dict()
        for name, param in slate._module.state_dict().items():
            assert torch.equal(param, source_state[name]), f"parameter {name} was not loaded"
        assert all(
            param.requires_grad for param in slate._module.parameters()
            if param.dtype.is_floating_point
        )
        assert slate._module.training

    def test_build_without_checkpoint(self, tmp_path):
        config_path = tmp_path / 'tiny_slate.yaml'
        OmegaConf.save(OmegaConf.create(TINY_OCR_CONFIG), config_path)
        slate = build_finetune_slate(
            ocr_config_path=str(config_path),
            checkpoint_path=None,
            obs_size=OBS_SIZE,
            obs_channels=OBS_CHANNELS,
            device='cpu',
        )
        log = slate_finetune_step(slate, make_images(2), step=0)
        assert np.isfinite(log['slate/loss'])
