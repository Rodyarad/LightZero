from typing import Any, Dict

import gym
import numpy as np

import gymnasium
import mani_skill.envs
import torch
from mani_skill import PushCubeEnv, register_env
from mani_skill.utils.structs import Array, Pose


@register_env(uid='PushCubeCustom-v1', max_episode_steps=50)
class PushCubeCustomEnv(PushCubeEnv):
    def __init__(self, *args, pose_reward_coef=1., place_reward_coef=1., **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_reward_coef = pose_reward_coef
        self._place_reward_coef = place_reward_coef

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = self._pose_reward_coef * reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += self._place_reward_coef * place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward


class ManiSkill(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, reward_mode, image_size, pose_reward_coef=1., place_reward_coef=1.):
        self.env = gymnasium.make(
            'PushCubeCustom-v1',
            pose_reward_coef=pose_reward_coef,
            place_reward_coef=place_reward_coef,
            obs_mode='rgbd',
            control_mode='pd_joint_delta_pos',
            render_mode='rgb_array',
            reward_mode=reward_mode,
            sensor_configs=dict(width=image_size, height=image_size),
        ).env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )
        self._last_observation = None

    def last_observation(self):
        return self._last_observation.copy()

    def seed(self, seed=None):
        self.env.reset(seed=seed)
        self.action_space.seed(seed)

        return seed

    @staticmethod
    def _unravel(step_result):
        unravel_result = [step_result[0]['sensor_data']['base_camera']['rgb'][0]]
        unravel_result += [x[0] if hasattr(x, '__len__') else x for x in step_result[1:-1]]
        info = {key: value[0] if hasattr(value, '__len__') else value for key, value in step_result[-1].items()}
        if 'success' in info:
            info['is_success'] = info['success']
        unravel_result.append(info)

        return unravel_result

    def reset(self):
        self._last_observation = self._unravel(self.env.reset())[0].numpy()
        return self.last_observation()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._unravel(self.env.step(action))
        assert not truncated, 'Cannot have time limit in unwrapped ManiSkill environment!'

        self._last_observation = obs.numpy()
        return self.last_observation(), float(reward.item()), bool(terminated.item()), {k: v.item() for k, v in info.items()}

    def render(self, mode=None):
        return self.last_observation()
