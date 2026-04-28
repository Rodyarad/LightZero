import argparse
import copy
import json
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from ding.config import compile_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed

from lzero.mcts.tree_search import mcts_ctree as mcts_ctree_module
from lzero.mcts.utils import prepare_observation
from zoo.ocrl.entry.ocrl_eval import build_config


class SearchDepthTracker:
    """Collect per-root max search depth across simulations in one MCTS search."""

    def __init__(self) -> None:
        self._active = False
        self._max_depths: Optional[np.ndarray] = None

    def start(self) -> None:
        self._active = True
        self._max_depths = None

    def record(self, search_lens: List[int]) -> None:
        if not self._active:
            return
        depths = np.asarray(search_lens, dtype=np.int32)
        if self._max_depths is None:
            self._max_depths = depths.copy()
        else:
            self._max_depths = np.maximum(self._max_depths, depths)

    def finish(self) -> np.ndarray:
        self._active = False
        if self._max_depths is None:
            return np.asarray([], dtype=np.int32)
        return self._max_depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate average MCTS max-depth on sampled OCRL states."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default='/home/rodya-rad/Desktop/work/LightZero/oc_agents_weights/oz_stica_goal_seed=0.pth.tar',
        help="Path to trained UniZero checkpoint (*.pth.tar).",
    )
    parser.add_argument("--env-id", type=str, default="TargetEnv-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-states",
        type=int,
        default=30,
        help="How many states to sample and evaluate.",
    )
    parser.add_argument(
        "--sampling-policy",
        type=str,
        default="policy",
        choices=["policy", "random"],
        help="Action source for stepping between sampled states.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Preferred device: cuda or cpu.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output path for per-state depths and summary.",
    )
    return parser.parse_args()


def _build_env_and_policy(args: argparse.Namespace):
    main_config, create_config = build_config(env_id=args.env_id)
    main_config = copy.deepcopy(main_config)
    create_config = copy.deepcopy(create_config)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    main_config.policy.cuda = use_cuda
    main_config.policy.device = "cuda" if use_cuda else "cpu"
    main_config.policy.eval_num_simulations = main_config.policy.num_simulations

    # We only need one synchronous evaluator env for state sampling.
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    create_config.env_manager.type = "base"

    cfg = compile_config(
        main_config, seed=args.seed, env=None, auto=True, create_cfg=create_config, save_cfg=False
    )
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env = env_fn(cfg=evaluator_env_cfg[0])
    env.seed(args.seed)
    set_pkg_seed(args.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval"])
    policy.learn_mode.load_state_dict(torch.load(args.model_path, map_location=cfg.policy.device))
    return cfg, env, policy


def _stack_obs(obs_buffer: Deque[np.ndarray], model_type: str) -> torch.Tensor:
    stack_np = np.stack(list(obs_buffer), axis=0)
    batch_np = np.asarray([stack_np])
    batch_np = prepare_observation(batch_np, model_type=model_type)
    return torch.from_numpy(batch_np).float()


def main() -> None:
    args = parse_args()
    cfg, env, policy = _build_env_and_policy(args)
    device = cfg.policy.device

    tracker = SearchDepthTracker()
    original_batch_traverse = mcts_ctree_module.tree_muzero.batch_traverse

    def wrapped_batch_traverse(*bt_args, **bt_kwargs):
        output = original_batch_traverse(*bt_args, **bt_kwargs)
        # Signature: (roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, to_play_batch)
        results = bt_args[5]
        tracker.record(results.get_search_len())
        return output

    mcts_ctree_module.tree_muzero.batch_traverse = wrapped_batch_traverse

    sampled_depths: List[int] = []
    sampled_timesteps: List[int] = []
    sampled_actions: List[int] = []

    try:
        obs_dict = env.reset()
        frame_stack_num = cfg.policy.model.frame_stack_num
        obs_buffer: Deque[np.ndarray] = deque(
            [np.asarray(obs_dict["observation"]).copy() for _ in range(frame_stack_num)],
            maxlen=frame_stack_num,
        )
        policy.eval_mode.reset([0])

        while len(sampled_depths) < args.num_states:
            action_mask = [np.asarray(obs_dict["action_mask"])]
            to_play = [np.asarray(obs_dict["to_play"])]
            timestep = [np.asarray(obs_dict.get("timestep", -1))]
            timestep_value = int(np.asarray(timestep[0]).item())

            obs_tensor = _stack_obs(obs_buffer, cfg.policy.model.model_type).to(device)

            tracker.start()
            policy_output: Dict[int, Dict] = policy.eval_mode.forward(
                obs_tensor,
                action_mask,
                to_play,
                ready_env_id=np.array([0]),
                timestep=timestep,
            )
            max_depth = tracker.finish()
            if max_depth.size == 0:
                raise RuntimeError("Depth tracker received no search_lens; check MCTS path.")

            # Report depth including the newly expanded child.
            depth_value = int(max_depth[0]) + 1
            sampled_depths.append(depth_value)
            sampled_timesteps.append(timestep_value)

            out0 = policy_output[0]
            policy_action = int(out0["action"])
            action = policy_action if args.sampling_policy == "policy" else int(env.action_space.sample())
            sampled_actions.append(action)

            timestep_obj = env.step(action)
            obs_dict = timestep_obj.obs
            obs_buffer.append(np.asarray(obs_dict["observation"]).copy())

            if timestep_obj.done:
                policy.eval_mode.reset([0])
                obs_dict = env.reset()
                obs_buffer = deque(
                    [np.asarray(obs_dict["observation"]).copy() for _ in range(frame_stack_num)],
                    maxlen=frame_stack_num,
                )

    finally:
        mcts_ctree_module.tree_muzero.batch_traverse = original_batch_traverse
        env.close()

    depths_np = np.asarray(sampled_depths, dtype=np.float32)
    summary = {
        "num_states": int(depths_np.shape[0]),
        "avg_max_depth": float(depths_np.mean()) if depths_np.size else 0.0,
        "sampling_policy": args.sampling_policy,
        "model_path": args.model_path,
        "env_id": args.env_id,
        "seed": int(args.seed),
    }

    print("MCTS max-depth summary:")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        payload = {
            "summary": summary,
            "depths": sampled_depths,
            "timesteps": sampled_timesteps,
            "actions": sampled_actions,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved details to {args.output_json}")


if __name__ == "__main__":
    main()
