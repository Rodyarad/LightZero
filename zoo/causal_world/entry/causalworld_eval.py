import numpy as np
from easydict import EasyDict

from lzero.entry import eval_muzero


def build_config():
    action_space_size = 3
    continuous_action_space = True
    num_of_sampled_actions = 20

    collector_env_num = 8
    evaluator_env_num = 30
    num_segments = 8

    game_segment_length = 100
    num_unroll_steps = 5
    infer_context_length = 2

    num_simulations = 50
    batch_size = 64
    replay_ratio = 0.1

    num_layers = 2
    norm_type = "LN"

    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    num_slots = 10
    slot_dim = 192
    ocr_config_path = "zoo/ocr/slate/config/slate_3d.yaml"
    checkpoint_path = "zoo/ocr/slate_weights/slate_3d.pth"

    tokens_per_block = num_slots * 2

    main_config = EasyDict(
        dict(
            env=dict(
                stop_value=int(1e6),
                env_config_path="zoo/causal_world/env/causal_world/cw_envs/config/reaching-hard_orig.yaml",
                from_pixels=True,
                observation_shape=(3, 64, 64),
                continuous=True,
                gray_scale=False,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False, context="spawn"),
                collect_max_episode_steps=int(125),
                eval_max_episode_steps=int(125),
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
                learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=int(1e6)))),
                model=dict(
                    observation_shape=(num_slots, slot_dim),
                    model_type="slot",
                    action_space_size=action_space_size,
                    continuous_action_space=continuous_action_space,
                    num_of_sampled_actions=num_of_sampled_actions,
                    world_model_cfg=dict(
                        model_type="slot",
                        tokens_per_block=tokens_per_block,
                        policy_loss_type="kl",
                        obs_type="slot",
                        num_unroll_steps=num_unroll_steps,
                        policy_entropy_weight=5e-2,
                        continuous_action_space=continuous_action_space,
                        num_of_sampled_actions=num_of_sampled_actions,
                        sigma_type="conditioned",
                        fixed_sigma_value=0.5,
                        bound_type=None,
                        norm_type=norm_type,
                        max_blocks=num_unroll_steps,
                        max_tokens=tokens_per_block * num_unroll_steps,
                        context_length=tokens_per_block * infer_context_length,
                        device="cuda",
                        action_space_size=action_space_size,
                        num_layers=num_layers,
                        num_heads=8,
                        num_slots=num_slots,
                        embed_dim=slot_dim,
                        env_num=max(collector_env_num, evaluator_env_num),
                        game_segment_length=game_segment_length,
                        num_simulations=num_simulations,
                    ),
                ),
                model_path=None,
                cuda=True,
                learning_rate=1e-4,
                batch_size=batch_size,
                replay_ratio=replay_ratio,
                num_unroll_steps=num_unroll_steps,
                use_root_value=False,
                use_augmentation=False,
                num_segments=num_segments,
                game_segment_length=game_segment_length,
                num_simulations=num_simulations,
                use_priority=False,
                env_type='not_board_games',
                discount_factor=0.99,
                td_steps=5,
                piecewise_decay_lr_scheduler=False,
                grad_clip_value=5,
                manual_temperature_decay=True,
                threshold_training_steps_for_final_temperature=int(2.5e4),
                cos_lr_scheduler=True,
                train_start_after_envsteps=2000,
                reanalyze_ratio=0,
                n_episode=8,
                buffer_reanalyze_freq=buffer_reanalyze_freq,
                reanalyze_batch_size=reanalyze_batch_size,
                reanalyze_partition=reanalyze_partition,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                use_adaptive_entropy_weight=False,
                eval_freq=int(5e3),
                replay_buffer_size=int(1e6),
                log_causality_probs=True,
                causality_log_dir='./visuals/oz_policy_log',
                log_unizero_slots=True,
                unizero_slots_dir='./visuals/oz_policy_log',
                log_eval_actions=True,
                eval_actions_dir='./visuals/oz_policy_log',
            ),
        )
    )

    create_config = EasyDict(
        dict(
            env=dict(
                type="causalworld_lightzero",
                import_names=["zoo.causal_world.env.causalworld_lightzero_env"],
            ),
            env_manager=dict(type="subprocess"),
            policy=dict(
                type="sampled_unizero",
                import_names=["lzero.policy.sampled_unizero"],
            ),
        )
    )

    return main_config, create_config


if __name__ == "__main__":
    main_config, create_config = build_config()

    model_path = '/home/rodya-rad/Desktop/work/LightZero/oc_agents_weights/oz_stica_cw_slate_seed7.pth.tar'
    seeds = [142]
    num_episodes_each_seed = 1

    create_config.env_manager.type = "base"
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    main_config.env.save_replay = True
    main_config.env.replay_path = "./visuals/video"
    main_config.env.save_replay_gif = False
    main_config.env.replay_path_gif = "./visuals"
    main_config.env.eval_max_episode_steps = int(125)
    main_config.policy.log_eval_actions = True
    main_config.policy.eval_actions_dir = "./visuals/oz_policy_log"

    returns_mean_seeds = []
    returns_seeds = []

    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=False,
            model_path=model_path,
        )
        print(returns_mean, returns)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)
