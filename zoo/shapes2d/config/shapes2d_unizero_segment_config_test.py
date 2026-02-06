from easydict import EasyDict
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map
import comet_ml

def main(env_id, seed):
    action_space_size = shapes2d_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 2
    num_segments = 2
    evaluator_env_num = 2

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 5
    batch_size = 2
    replay_ratio = 0.25

    num_layers = 2
    norm_type = "LN"

    max_env_step = int(5e5)

    # Reanalyze settings
    buffer_reanalyze_freq = 1/5000000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
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
        ),
        policy=dict(
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    latent_recon_loss_weight=0.0,
                    perceptual_loss_weight=0.0,
                    norm_type=norm_type,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    game_segment_length=game_segment_length,
                    device='cuda',
                    use_priority=True,
                ),
            ),
            # Learning settings
            learning_rate=0.0001,
            weight_decay=1e-2,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            num_unroll_steps=num_unroll_steps,
            num_segments=num_segments,
            game_segment_length=game_segment_length,
            num_simulations=num_simulations,

            # Priority settings
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(1e4),
            replay_buffer_size=int(5e5),
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
            import_names=['lzero.policy.unizero'],
        ),
    )
    shapes2d_unizero_create_config = EasyDict(shapes2d_unizero_create_config)
    create_config = shapes2d_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name = f'data_unizero/{env_id[3:-3]}/{env_id[3:-3]}_uz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    train_unizero_segment([main_config, create_config], seed=seed, model_path=None, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='Navigation5x5-v0')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)
