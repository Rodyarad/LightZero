import argparse

import numpy as np
from easydict import EasyDict

from lzero.entry import eval_muzero


def build_config(env_id: str = "VizdoomDefendLine-v0"):
    action_space_size = 4

    collector_env_num = 8
    evaluator_env_num = 30
    num_segments = 8

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 50
    batch_size = 128
    replay_ratio = 0.25

    num_layers = 2
    norm_type = "LN"

    buffer_reanalyze_freq = 1 / 5000000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    num_slots = 7
    slot_dim = 64
    ocr_config_path = "zoo/ocr/slotcontrast/configs/vizdoom_sc.yaml"
    checkpoint_path = "zoo/ocr/slotcontrast_weights/slotcontrast_vizdoom.ckpt"
    tokens_per_block = num_slots * 2

    main_config = EasyDict(
        dict(
            env=dict(
                stop_value=int(1e6),
                env_id=env_id,
                observation_shape=(3, 336, 336),
                gray_scale=False,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False, context="spawn"),
                collect_max_episode_steps=int(40),
                eval_max_episode_steps=int(40),
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
                    reward_support_range=(-300.0, 301.0, 1.0),
                    value_support_range=(-300.0, 301.0, 1.0),
                    norm_type=norm_type,
                    num_res_blocks=2,
                    num_channels=128,
                    world_model_cfg=dict(
                        model_type="slot",
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
                        obs_type="slot",
                        env_num=max(collector_env_num, evaluator_env_num),
                        num_simulations=num_simulations,
                        game_segment_length=game_segment_length,
                        device="cuda",
                        use_priority=True,
                    ),
                ),
                cuda=True,
                learning_rate=0.0001,
                weight_decay=1e-2,
                batch_size=batch_size,
                replay_ratio=replay_ratio,
                num_unroll_steps=num_unroll_steps,
                num_segments=num_segments,
                game_segment_length=game_segment_length,
                num_simulations=num_simulations,
                use_priority=True,
                priority_prob_alpha=1,
                priority_prob_beta=1,
                buffer_reanalyze_freq=buffer_reanalyze_freq,
                reanalyze_batch_size=reanalyze_batch_size,
                reanalyze_partition=reanalyze_partition,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                eval_freq=int(20e3),
                replay_buffer_size=int(5e5),
                log_causality_probs=True,
                causality_log_dir="./visuals/oz_policy_log",
                log_unizero_slots=True,
                unizero_slots_dir="./visuals/oz_policy_log",
                log_eval_actions=True,
                eval_actions_dir="./visuals/oz_policy_log",
            ),
        )
    )

    create_config = EasyDict(
        dict(
            env=dict(
                type="vizdoom_lightzero",
                import_names=["zoo.vizdoom.env.vizdoom_lightzero_env"],
            ),
            env_manager=dict(type="subprocess", context="spawn"),
            policy=dict(
                type="unizero",
                import_names=["lzero.policy.unizero"],
            ),
        )
    )

    return main_config, create_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SlotContrast-based policy on Vizdoom.")
    parser.add_argument("--env-id", type=str, default="VizdoomDefendLine-v0")
    parser.add_argument("--model-path", type=str, default='/home/rodya-rad/Desktop/work/LightZero/oc_agents_weights/oz_stica_vizdoom_seed0.pth.tar')
    parser.add_argument("--seeds", type=int, nargs="+", default=[777])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--replay-path", type=str, default="./visuals/video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_config, create_config = build_config(env_id=args.env_id)

    create_config.env_manager.type = "base"
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    main_config.env.render_mode_human = False
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
