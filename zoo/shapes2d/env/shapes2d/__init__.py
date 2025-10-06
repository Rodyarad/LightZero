from gym.envs.registration import register

register(
    'Navigation5x5-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'return_state': False,
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Navigation5x5Interaction-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'return_state': False,
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'ternary_interactions': True,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'RandomWalk-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'static_goals': False,
        'width': 5,
        'render_scale': 11,
        'channel_wise': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Navigation10x10-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'return_state': False,
        'n_boxes': 8,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 10,
        'render_scale': 10,
        'channel_wise': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Pushing7x7-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 7,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': True,
        'do_reward_push_only': False,
    },
)

register(
    'PushingNoAgent5x5-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': False,
        'do_reward_push_only': True,
    },
)

register(
    'Navigation4x4B3G1Circle-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'circle',
        'border_walls': True,
        'n_boxes': 3,
        'n_goals': 1,
        'n_static_boxes': 0,
        'use_random_shapes': False,
        'static_goals': True,
        'width': 4,
        'render_scale': 16,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'PushingNoAgent4x4B3G1Circle-v0',
    entry_point='zoo.shapes2d.env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'circle',
        'border_walls': True,
        'n_boxes': 3,
        'n_goals': 1,
        'n_static_boxes': 0,
        'use_random_shapes': False,
        'static_goals': True,
        'width': 4,
        'render_scale': 16,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': False,
        'do_reward_push_only': True,
    },
)