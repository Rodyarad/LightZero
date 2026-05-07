import argparse

import numpy as np
from easydict import EasyDict

from lzero.entry import eval_muzero


def build_config(replay_image_size: int = 672):
    action_space_size = 8
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

    num_slots = 3
    slot_dim = 64
    ocr_config_path = "zoo/ocr/slotcontrast/configs/slotcontrast_maniskill.yaml"
    checkpoint_path = "zoo/ocr/slotcontrast_weights/slotcontrast_maniskill.ckpt"

    tokens_per_block = num_slots * 2

    main_config = EasyDict(
        dict(
            env=dict(
                from_pixels=True,
                observation_shape=(3, 336, 336),
                replay_image_size=int(replay_image_size),
                continuous=True,
                gray_scale=False,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False),
                collect_max_episode_steps=int(50),
                eval_max_episode_steps=int(50),
                oc_model=True,
                oc_model_type="SlotContrast",
                ocr_config_path=ocr_config_path,
                checkpoint_path=checkpoint_path,
                num_slots=num_slots,
                slot_dim=slot_dim,
                warp_frame=True,
                scale=False,
            ),
            run_id_comet_ml=None,
            policy=dict(
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
                num_unroll_steps=num_unroll_steps,
                cuda=True,
                use_root_value=False,
                use_augmentation=False,
                use_priority=False,
                env_type="not_board_games",
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
                n_episode=8,
                eval_freq=int(5e3),
                replay_buffer_size=int(1e6),
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                buffer_reanalyze_freq=buffer_reanalyze_freq,
                reanalyze_batch_size=reanalyze_batch_size,
                reanalyze_partition=reanalyze_partition,
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
                type="maniskill_lightzero",
                import_names=["zoo.maniskill.env.maniskill_lightzero_env"],
            ),
            env_manager=dict(type="subprocess"),
            policy=dict(
                type="sampled_unizero",
                import_names=["lzero.policy.sampled_unizero"],
            ),
        )
    )

    return main_config, create_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SlotContrast-based policy on ManiSkill.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/rodya-rad/Desktop/work/LightZero/oc_agents_weights/oz_stica_maniskill_seed0.pth.tar",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[35])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=50)
    parser.add_argument("--replay-path", type=str, default="./visuals/video")
    parser.add_argument(
        "--save-replay-frames",
        action="store_true",
        help="Save replay as per-step JPG frames instead of RecordVideo mp4.",
    )
    parser.add_argument(
        "--replay-frames-path",
        type=str,
        default="./visuals/video_frames",
        help="Directory for per-step replay frames when --save-replay-frames is set.",
    )
    parser.add_argument(
        "--replay-image-size",
        type=int,
        default=672,
        help="Base render size for saved replay (frames/video). Model observations stay at 336 via WarpFrame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_config, create_config = build_config(replay_image_size=args.replay_image_size)

    create_config.env_manager.type = "base"
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    main_config.env.render_mode_human = False
    if args.save_replay_frames:
        main_config.env.save_replay_frames = True
        main_config.env.replay_frames_path = args.replay_frames_path
        main_config.env.save_replay = False
    else:
        main_config.env.save_replay = True
        main_config.env.replay_path = args.replay_path
    main_config.env.eval_max_episode_steps = int(args.eval_max_steps)

    returns_mean_seeds = []
    returns_seeds = []
    for seed in args.seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=args.episodes_per_seed,
            print_seed_details=False,
            model_path=args.model_path,
        )
        print(returns_mean, returns)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)
    print("=" * 20)
    print(
        f"We evaluated a total of {len(args.seeds)} seeds. "
        f"For each seed, we evaluated {args.episodes_per_seed} episode(s)."
    )
    print(
        f"For seeds {args.seeds}, the mean returns are {returns_mean_seeds}, "
        f"and the returns are {returns_seeds}."
    )
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)
