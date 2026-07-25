"""
Unit tests for the raw (image) observation stream of ``GameSegment``
(``store_raw_obs=True``), used for online SLATE fine-tuning.

Run with:
    pytest lzero/mcts/tests/test_game_segment_raw_obs.py -sv -m unittest
"""
import gym
import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_segment import GameSegment

NUM_SLOTS, SLOT_DIM = 2, 4
IMG_SHAPE = (8, 8, 3)
GAME_SEGMENT_LENGTH = 4
NUM_UNROLL_STEPS = 3
TD_STEPS = 2


def make_config(store_raw_obs: bool, finetune_slate: bool = False) -> EasyDict:
    return EasyDict(dict(
        num_unroll_steps=NUM_UNROLL_STEPS,
        td_steps=TD_STEPS,
        discount_factor=0.997,
        gray_scale=False,
        transform2string=False,
        sampled_algo=False,
        gumbel_algo=False,
        use_ture_chance_label_in_chance_encoder=False,
        store_raw_obs=store_raw_obs,
        finetune_slate=finetune_slate,
        model=dict(
            frame_stack_num=1,
            action_space_size=4,
            observation_shape=(NUM_SLOTS, SLOT_DIM),
            image_channel=3,
        ),
    ))


def make_segment(store_raw_obs: bool = True, finetune_slate: bool = False) -> GameSegment:
    return GameSegment(
        gym.spaces.Discrete(4),
        game_segment_length=GAME_SEGMENT_LENGTH,
        config=make_config(store_raw_obs, finetune_slate),
    )


def slot_obs(i: int) -> np.ndarray:
    return np.full((NUM_SLOTS, SLOT_DIM), i, dtype=np.float32)


def raw_obs(i: int) -> np.ndarray:
    return np.full(IMG_SHAPE, i, dtype=np.uint8)


def fill_segment(segment: GameSegment, num_steps: int, start: int = 0) -> None:
    for i in range(start, start + num_steps):
        segment.append(
            action=i % 4,
            obs=slot_obs(i + 1),
            reward=1.0,
            action_mask=np.ones(4, dtype=np.int8),
            to_play=-1,
            timestep=i,
            raw_obs=raw_obs(i + 1),
        )


@pytest.mark.unittest
class TestGameSegmentRawObs:

    def test_reset_and_append_keep_streams_aligned(self):
        segment = make_segment()
        segment.reset([slot_obs(0)], init_raw_observations=[raw_obs(0)])
        fill_segment(segment, GAME_SEGMENT_LENGTH)

        assert len(segment.raw_obs_segment) == len(segment.obs_segment)
        for slots, image in zip(segment.obs_segment, segment.raw_obs_segment):
            assert slots[0, 0] == image[0, 0, 0]

    def test_get_unroll_raw_obs_aligns_with_get_unroll_obs(self):
        segment = make_segment()
        segment.reset([slot_obs(0)], init_raw_observations=[raw_obs(0)])
        fill_segment(segment, GAME_SEGMENT_LENGTH)

        for pos in range(GAME_SEGMENT_LENGTH):
            unroll_obs = segment.get_unroll_obs(pos, num_unroll_steps=NUM_UNROLL_STEPS, padding=True)
            unroll_raw = segment.get_unroll_raw_obs(pos, num_unroll_steps=NUM_UNROLL_STEPS, padding=True)
            assert len(unroll_raw) == len(unroll_obs) == 1 + NUM_UNROLL_STEPS
            for slots, image in zip(unroll_obs, unroll_raw):
                assert slots[0, 0] == image[0, 0, 0]

    def test_pad_over_extends_raw_stream(self):
        segment = make_segment()
        segment.reset([slot_obs(0)], init_raw_observations=[raw_obs(0)])
        fill_segment(segment, GAME_SEGMENT_LENGTH)

        next_segment = make_segment()
        next_segment.reset([slot_obs(100)], init_raw_observations=[raw_obs(100)])
        fill_segment(next_segment, GAME_SEGMENT_LENGTH, start=100)

        beg, end = 1, 1 + NUM_UNROLL_STEPS + TD_STEPS
        segment.pad_over(
            next_segment.obs_segment[beg:end],
            next_segment.reward_segment[0:NUM_UNROLL_STEPS + TD_STEPS - 1],
            next_segment.action_segment[0:NUM_UNROLL_STEPS + TD_STEPS],
            [0.0] * (NUM_UNROLL_STEPS + TD_STEPS),
            [[0.25] * 4] * (NUM_UNROLL_STEPS + TD_STEPS),
            next_segment_raw_observations=next_segment.raw_obs_segment[beg:end],
        )
        segment.game_segment_to_array()

        assert isinstance(segment.raw_obs_segment, np.ndarray)
        assert segment.raw_obs_segment.shape[0] == segment.obs_segment.shape[0]
        for slots, image in zip(segment.obs_segment, segment.raw_obs_segment):
            assert slots[0, 0] == image[0, 0, 0]

    def test_finetune_slate_drops_slot_obs_on_save(self):
        segment = make_segment(finetune_slate=True)
        segment.reset([slot_obs(0)], init_raw_observations=[raw_obs(0)])
        fill_segment(segment, GAME_SEGMENT_LENGTH)

        assert len(segment.obs_segment) == GAME_SEGMENT_LENGTH + 1
        assert len(segment.get_obs()) == 1

        segment.game_segment_to_array()
        assert segment.obs_segment.size == 0
        assert isinstance(segment.raw_obs_segment, np.ndarray)
        assert segment.raw_obs_segment.shape == (GAME_SEGMENT_LENGTH + 1, *IMG_SHAPE)
        unroll_raw = segment.get_unroll_raw_obs(0, num_unroll_steps=NUM_UNROLL_STEPS, padding=True)
        assert len(unroll_raw) == 1 + NUM_UNROLL_STEPS

    def test_finetune_slate_requires_store_raw_obs(self):
        with pytest.raises(AssertionError):
            make_segment(store_raw_obs=False, finetune_slate=True)

    def test_zero_raw_obs(self):
        segment = make_segment(finetune_slate=True)
        segment.reset([slot_obs(1)], init_raw_observations=[raw_obs(1)])
        zeros = segment.zero_raw_obs()
        assert len(zeros) == 1
        assert zeros[0].shape == IMG_SHAPE
        assert zeros[0].dtype == np.uint8
        assert not zeros[0].any()

    def test_store_raw_obs_disabled(self):
        segment = make_segment(store_raw_obs=False)
        segment.reset([slot_obs(0)], init_raw_observations=[raw_obs(0)])
        fill_segment(segment, GAME_SEGMENT_LENGTH)

        assert len(segment.raw_obs_segment) == 0
        with pytest.raises(AssertionError):
            segment.get_unroll_raw_obs(0, num_unroll_steps=NUM_UNROLL_STEPS, padding=True)
