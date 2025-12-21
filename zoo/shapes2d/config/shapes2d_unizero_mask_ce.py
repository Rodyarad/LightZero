from easydict import EasyDict
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map
import comet_ml


def main(env_id='Navigation5x5-v0', seed=0):
    action_space_size = shapes2d_env_action_space_map[env_id]
    num_objects = action_space_size // 4

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    game_segment_length = 20
    evaluator_env_num = 30
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25

    # TODO: only for debug
    # collector_env_num = 2
    # game_segment_length = 20
    # evaluator_env_num = 2
    # num_simulations = 2
    # max_env_step = int(5e5)
    # batch_size = 10
    # num_unroll_steps = 5
    # infer_context_length = 2
    # num_layers = 1
    # replay_ratio = 0.1
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    shapes2d_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        run_id_comet_ml=None,
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1e6, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                # Mirror num_objects at the top model level so that MCTS config
                # (which reads cfg.model.num_objects) can access it.
                num_objects=num_objects,
                # Hard threshold for converting object mask logits to binary mask in MCTS.
                # Used in both Python policy (root) and C++ MCTS tree for per-node masking.
                # mcts_mask_thres=0.5,
                world_model_cfg=dict(
                    policy_entropy_weight=1e-4,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    rotary_emb=False,
                    # ===== Object-level action abstraction & mask head =====
                    # Enable separate object-mask head in the world model.
                    use_action_absraction=True,
                    # Number of objects for object-level masking; actions are split into
                    # contiguous groups via actions_per_obj = action_space_size // num_objects.
                    num_objects=num_objects,
                    mask_loss_type='ce',
                    mask_loss_weight=1.0,
                ),
            ),
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            num_simulations=num_simulations,
            train_start_after_envsteps=2000,
            # train_start_after_envsteps=0, # TODO: only for debug
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # Global train_iter step from which object-based masking starts to influence
            # MCTS branching and target formation. Before this step, MCTS ignores the
            # learned object mask and relies only on env_action_mask.
            causal_puct_start_step=25_000,
        ),
    )
    shapes2d_unizero_config = EasyDict(shapes2d_unizero_config)
    main_config = shapes2d_unizero_config

    shapes2d_unizero_create_config = dict(
        env=dict(
            type='shapes2d_lightzero',
            import_names=['zoo.shapes2d.env.shapes2d_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero_action_abstraction'],
        ),
    )
    shapes2d_unizero_create_config = EasyDict(shapes2d_unizero_create_config)
    create_config = shapes2d_unizero_create_config

    main_config.exp_name = f'data_lz/data_unizero/shapes2d_uz_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str, help='The environment to use', default='Navigation5x5-v0')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)

