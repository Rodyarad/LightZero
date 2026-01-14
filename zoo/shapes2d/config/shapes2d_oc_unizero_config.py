from easydict import EasyDict
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map
import comet_ml


def main(env_id='Navigation5x5-v0', seed=0):
    action_space_size = shapes2d_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4
    evaluator_env_num = 30
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 128
    replay_ratio = 0.25
    num_layers = 2
    buffer_reanalyze_freq = 1 / 5000000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    norm_type = "LN"

    num_slots = 6
    slot_dim = 64
    ocr_config_path = 'zoo/ocr/slate/config/navigation5x5.yaml'
    checkpoint_path = 'zoo/ocr/slate_weights/navigation5x5.pth'

    tokens_per_block = num_slots + 1

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
            frame_skip=1,
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
            ocr_config_path=ocr_config_path,
            checkpoint_path=checkpoint_path,
            num_slots=num_slots,
            slot_dim=slot_dim,
            warp_frame=True,
            scale=False,
        ),
        run_id_comet_ml=None,
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1e6, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(num_slots, slot_dim),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    model_type='slot',
                    tokens_per_block=tokens_per_block,
                    use_new_cache_manager=False,
                    norm_type=norm_type,
                    final_norm_option_in_obs_head=None,
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    analysis_dormant_ratio_weight_rank=False,
                    dormant_threshold=0.025,
                    task_embed_option=None,
                    use_task_embed=False,
                    use_shared_projection=False,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=tokens_per_block * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=tokens_per_block * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=slot_dim,
                    num_slots=num_slots,
                    obs_type='slot',
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    game_segment_length=game_segment_length,
                    use_priority=True,
                    rotary_emb=False,
                    encoder_type='resnet',
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                    lora_r=0,
                    lora_alpha=1,
                    lora_dropout=0.0,
                    optim_type='AdamW_mix_lr_wdecay',
                    # ==================== Policy Stability Fixes ====================
                    use_policy_logits_clip=True,
                    policy_logits_clip_min=-10.0,
                    policy_logits_clip_max=10.0,
                    use_target_policy_resmooth=False,
                    target_policy_resmooth_eps=0.05,
                    use_policy_loss_temperature=True,
                    policy_loss_temperature=1.5,
                    use_continuous_label_smoothing=True,
                    continuous_ls_eps=0.05,
                ),
            ),
            optim_type='AdamW_mix_lr_wdecay',
            weight_decay=1e-2,
            learning_rate=0.0001,
            model_path=None,
            # Adaptive entropy weight
            use_adaptive_entropy_weight=True,
            adaptive_entropy_alpha_lr=1e-3,
            target_entropy_start_ratio=0.98,
            target_entropy_end_ratio=0.05,
            target_entropy_decay_steps=500000,
            # Encoder clip annealing
            use_encoder_clip_annealing=True,
            encoder_clip_anneal_type='cosine',
            encoder_clip_start_value=30.0,
            encoder_clip_end_value=10.0,
            encoder_clip_anneal_steps=100000,
            # Head clip
            use_head_clip=True,
            head_clip_config=dict(
                enabled=True,
                enabled_heads=['policy'],
                head_configs=dict(
                    policy=dict(
                        use_annealing=True,
                        anneal_type='cosine',
                        start_value=30.0,
                        end_value=10.0,
                        anneal_steps=650000,
                    ),
                ),
                monitor_freq=1,
                log_freq=1000,
            ),
            # Label smoothing
            policy_ls_eps_start=0.05,
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000,
            label_smoothing_eps=0.1,
            # Monitoring
            use_enhanced_policy_monitoring=True,
            monitor_norm_freq=5000,
            use_augmentation=True,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5,
            target_update_theta=0.05,
            train_start_after_envsteps=0,
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(5e5),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
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

    main_config.exp_name = f'data_lz/data_unizero/shapes2d_uz_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_unizero_segment
    train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str, help='The environment to use', default='Navigation5x5-v0')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)

