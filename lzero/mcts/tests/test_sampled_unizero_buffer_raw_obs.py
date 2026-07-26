"""
Unit tests for the raw (image) observation stream of ``SampledUniZeroGameBuffer``
(``store_raw_obs=True`` / ``finetune_slate=True``), used for online SLATE
fine-tuning with the sampled (continuous-action) UniZero policy.

Run with:
    pytest lzero/mcts/tests/test_sampled_unizero_buffer_raw_obs.py -sv -m unittest
"""
import gym
import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_sampled_unizero import SampledUniZeroGameBuffer
from lzero.mcts.buffer.game_segment import GameSegment

NUM_SLOTS, SLOT_DIM = 2, 4
IMG_SHAPE = (8, 8, 3)
ACTION_DIM = 2
K = 3  # num_of_sampled_actions
GAME_SEGMENT_LENGTH = 6
NUM_UNROLL_STEPS = 3
TD_STEPS = 2
BATCH_SIZE = 2


def make_config(store_raw_obs: bool, finetune_slate: bool) -> EasyDict:
    return EasyDict(dict(
        env_type='not_board_games',
        action_type='fixed_action_space',
        replay_buffer_size=1000,
        batch_size=BATCH_SIZE,
        priority_prob_alpha=1.0,
        priority_prob_beta=1.0,
        sample_type='transition',
        device='cpu',
        use_priority=False,
        reanalyze_outdated=False,
        use_root_value=False,
        mcts_ctree=True,
        num_unroll_steps=NUM_UNROLL_STEPS,
        td_steps=TD_STEPS,
        discount_factor=0.997,
        game_segment_length=GAME_SEGMENT_LENGTH,
        store_raw_obs=store_raw_obs,
        finetune_slate=finetune_slate,
        sampled_algo=True,
        gumbel_algo=False,
        use_ture_chance_label_in_chance_encoder=False,
        gray_scale=False,
        transform2string=False,
        model=dict(
            model_type='slot',
            continuous_action_space=True,
            action_space_size=ACTION_DIM,
            num_of_sampled_actions=K,
            observation_shape=(NUM_SLOTS, SLOT_DIM),
            frame_stack_num=1,
            image_channel=3,
            value_support_range=(-10., 11., 1.),
            reward_support_range=(-10., 11., 1.),
        ),
    ))


def slot_obs(i: int) -> np.ndarray:
    return np.full((NUM_SLOTS, SLOT_DIM), i, dtype=np.float32)


def raw_obs(i: int) -> np.ndarray:
    return np.full(IMG_SHAPE, i % 255, dtype=np.uint8)


def make_segment(cfg: EasyDict, start: int = 0) -> GameSegment:
    segment = GameSegment(
        gym.spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32),
        game_segment_length=GAME_SEGMENT_LENGTH,
        config=cfg,
    )
    segment.reset([slot_obs(start)], init_raw_observations=[raw_obs(start)])
    for i in range(start, start + GAME_SEGMENT_LENGTH):
        segment.append(
            action=np.random.randn(ACTION_DIM).astype(np.float32),
            obs=slot_obs(i + 1),
            reward=1.0,
            action_mask=None,
            to_play=-1,
            timestep=i - start,
            raw_obs=raw_obs(i + 1),
        )
        segment.store_search_stats(
            visit_counts=[1] * K,
            root_value=0.5,
            root_sampled_actions=np.random.rand(K, ACTION_DIM),
        )
    segment.game_segment_to_array()
    return segment


def make_buffer(store_raw_obs: bool, finetune_slate: bool) -> SampledUniZeroGameBuffer:
    cfg = make_config(store_raw_obs, finetune_slate)
    buffer = SampledUniZeroGameBuffer(cfg)
    for start in (0, 100):
        buffer._push_game_segment(make_segment(cfg, start=start), meta=dict(done=True, priorities=None))
    return buffer


@pytest.mark.unittest
class TestSampledUniZeroBufferRawObs:

    def setup_method(self):
        np.random.seed(0)

    def test_make_batch_finetune_slate_returns_raw_obs(self):
        buffer = make_buffer(store_raw_obs=True, finetune_slate=True)
        reward_value_context, _, policy_non_re_context, current_batch = buffer._make_batch(BATCH_SIZE, 0)

        obs_batch = current_batch[0]
        assert obs_batch.shape == (BATCH_SIZE, NUM_UNROLL_STEPS + 1, *IMG_SHAPE)
        assert obs_batch.dtype == np.uint8

        # the value context must carry raw frames as well (bootstrap obs are
        # re-encoded to slots with the current SLATE weights at target time)
        value_obs_list = reward_value_context[0]
        assert np.asarray(value_obs_list[0]).shape == (1, *IMG_SHAPE)

    def test_make_batch_store_raw_obs_keeps_slots_and_stashes_raw(self):
        buffer = make_buffer(store_raw_obs=True, finetune_slate=False)
        _, _, _, current_batch = buffer._make_batch(BATCH_SIZE, 0)

        # network inputs stay slot observations
        obs_batch = current_batch[0]
        assert obs_batch.shape[0] == BATCH_SIZE
        assert obs_batch.shape[-1] == SLOT_DIM

        # the raw stream is stashed for sample() to append to train_data
        assert buffer._raw_obs_batch_tmp.shape == (BATCH_SIZE, NUM_UNROLL_STEPS + 1, *IMG_SHAPE)

    def test_make_batch_default_has_no_raw_stream(self):
        buffer = make_buffer(store_raw_obs=False, finetune_slate=False)
        _, _, _, current_batch = buffer._make_batch(BATCH_SIZE, 0)

        assert current_batch[0].shape[-1] == SLOT_DIM
        assert not hasattr(buffer, '_raw_obs_batch_tmp')
