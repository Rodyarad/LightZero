from easydict import EasyDict
import comet_ml

def main(env_id, seed):
    action_space_size = 4

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    evaluator_env_num = 30

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 50
    batch_size = 128
    replay_ratio = 0.25

    num_layers = 2
    norm_type = "LN"

    max_env_step = int(5e5)

    # Reanalyze settings
    buffer_reanalyze_freq = 1/5000000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    
    num_slots = 6
    slot_dim = 192
    ocr_config_path = 'zoo/ocr/slate/config/slate_ocrl.yaml'
    checkpoint_path = 'zoo/ocr/slate_weights/slate_ocrl.pth'

    tokens_per_block = num_slots * 2
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    ocrl_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, context='spawn'),
            collect_max_episode_steps=int(100),
            eval_max_episode_steps=int(100),
            oc_model=True,
            ocr_config_path=ocr_config_path,
            checkpoint_path=checkpoint_path,
            num_slots=num_slots,
            slot_dim=slot_dim,
            warp_frame=True,
            scale=False,
        ),
        run_id_comet_ml=None,
        policy=dict(
            #store_obs_int8=True,
            model=dict(
                observation_shape=(num_slots, slot_dim),
                model_type='slot',
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    model_type='slot',
                    tokens_per_block=tokens_per_block,
                    latent_recon_loss_weight=0.0,
                    perceptual_loss_weight=0.0,
                    norm_type=norm_type,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    max_blocks=num_unroll_steps,
                    max_tokens=tokens_per_block * num_unroll_steps,
                    context_length=tokens_per_block * infer_context_length,
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=slot_dim,
                    num_slots=num_slots,
                    obs_type='slot',
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
            eval_freq=int(20e3),
            replay_buffer_size=int(5e5),
        ),
    )
    ocrl_unizero_config = EasyDict(ocrl_unizero_config)
    main_config = ocrl_unizero_config

    ocrl_unizero_create_config = dict(
        env=dict(
            type='ocrl_lightzero',
            import_names=['zoo.ocrl.env.ocrl_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    ocrl_unizero_create_config = EasyDict(ocrl_unizero_create_config)
    create_config = ocrl_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name = f'data_unizero/{env_id[3:-3]}/{env_id[3:-3]}_uz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    train_unizero_segment([main_config, create_config], seed=seed, model_path=None, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='TargetEnv-v0')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)
