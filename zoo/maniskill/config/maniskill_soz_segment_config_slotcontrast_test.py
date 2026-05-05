from easydict import EasyDict
import comet_ml
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================


def main(seed):
    action_space_size = 8

    continuous_action_space = True
    K = 20  # num_of_sampled_actions
    collector_env_num = 2
    n_episode = 2
    num_segments = 2
    game_segment_length = 100
    evaluator_env_num = 2
    num_simulations = 5
    replay_ratio = 0.1
    max_env_step = int(5e5)
    batch_size = 2
    num_layers = 2
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1 / 100000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75

    num_slots = 3
    slot_dim = 64
    ocr_config_path = 'zoo/ocr/slotcontrast/configs/slotcontrast_maniskill.yaml'
    checkpoint_path = 'zoo/ocr/slotcontrast_weights/slotcontrast_maniskill.ckpt'

    tokens_per_block = num_slots * 2

    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    maniskill_pixels_cont_sampled_unizero_config = dict(
        env=dict(
            from_pixels=True,
            observation_shape=(3, 336, 336),
            continuous=True,
            gray_scale=False,
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False,),
            oc_model=True,
            oc_model_type='SlotContrast',
            ocr_config_path=ocr_config_path,
            checkpoint_path=checkpoint_path,
            num_slots=num_slots,
            slot_dim=slot_dim,
            warp_frame=True,
            scale=False,
        ),
        run_id_comet_ml=None,
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1e6,),),),  # default is 10000
            model=dict(
                observation_shape=(num_slots, slot_dim),
                model_type='slot',
                action_space_size=action_space_size,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                world_model_cfg=dict(
                    model_type='slot',
                    tokens_per_block=tokens_per_block,
                    policy_loss_type='kl',
                    obs_type='slot',
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_weight=5e-2,
                    continuous_action_space=continuous_action_space,
                    num_of_sampled_actions=K,
                    sigma_type='conditioned',
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    norm_type=norm_type,
                    max_blocks=num_unroll_steps,
                    max_tokens=tokens_per_block * num_unroll_steps,  # NOTE: each timestep has tokens_per_block tokens per timestep
                    context_length=tokens_per_block * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    num_slots=num_slots,
                    embed_dim=slot_dim,
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            cuda=True,
            use_root_value=False,
            use_augmentation=False,
            use_priority=False,
            env_type='not_board_games',
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            discount_factor=0.925,
            td_steps=5,
            piecewise_decay_lr_scheduler=False,
            learning_rate=1e-4,
            grad_clip_value=5,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            cos_lr_scheduler=True,
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            num_simulations=num_simulations,
            reanalyze_ratio=0,
            n_episode=n_episode,
            eval_freq=int(5e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for ReZero =============
            buffer_reanalyze_freq=buffer_reanalyze_freq,  # 1 means reanalyze one times per epoch, 2 means reanalyze one times each two epoch
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )

    maniskill_pixels_cont_sampled_unizero_config = EasyDict(maniskill_pixels_cont_sampled_unizero_config)
    main_config = maniskill_pixels_cont_sampled_unizero_config

    maniskill_pixels_cont_sampled_unizero_create_config = dict(
        env=dict(
            type='maniskill_lightzero',
            import_names=['zoo.maniskill.env.maniskill_lightzero_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(
            type='sampled_unizero',
            import_names=['lzero.policy.sampled_unizero'],
        ),
    )
    maniskill_pixels_cont_sampled_unizero_create_config = EasyDict(maniskill_pixels_cont_sampled_unizero_create_config)
    create_config = maniskill_pixels_cont_sampled_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name = f'data_sampled_unizero/maniskill_Push_slotcontrast_brf{buffer_reanalyze_freq}_image_cont_suz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_K{K}_ns{num_simulations}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_{norm_type}_seed{seed}_learnsigma'
    train_unizero_segment([main_config, create_config], model_path=main_config.policy.model_path, seed=seed, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')

    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.seed)
