import argparse

import numpy as np
from easydict import EasyDict

from lzero.entry import eval_muzero
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map


def build_config(env_id='Navigation5x5-v0'):
    action_space_size = shapes2d_env_action_space_map[env_id]

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

    buffer_reanalyze_freq = 1 / 5000000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    num_slots = 6
    slot_dim = 64
    ocr_config_path = 'zoo/ocr/slate/config/navigation5x5.yaml'
    checkpoint_path = 'zoo/ocr/slate_weights/navigation5x5.pth'

    tokens_per_block = num_slots * 2

    main_config = EasyDict(dict(
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
            eval_freq=int(5e3),
            replay_buffer_size=int(5e5),
        ),
    ))

    create_config = EasyDict(dict(
        env=dict(
            type='shapes2d_lightzero',
            import_names=['zoo.shapes2d.env.shapes2d_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    ))

    return main_config, create_config


if __name__ == "__main__":
    """
    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, pointing to the ckpt file of the pretrained model. 
          The path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
        - seeds (:obj:`List[int]`): List of seeds to use for the evaluations.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to evaluate for each seed.
        - total_test_episodes (:obj:`int`): Total number of test episodes, calculated as num_episodes_each_seed * len(seeds).
        - returns_mean_seeds (:obj:`np.array`): Array of mean return values for each seed.
        - returns_seeds (:obj:`np.array`): Array of all return values for each seed.
    """
    main_config, create_config = build_config()

    # model_path is the path to the trained MuZero model checkpoint.
    # If no path is provided, the script will use the default model.
    model_path = None
    # seeds is a list of seed values for the random number generator, used to initialize the environment.
    seeds = [0, 7, 42]
    # num_episodes_each_seed is the number of episodes to run for each seed.
    num_episodes_each_seed = 1
    # total_test_episodes is the total number of test episodes, calculated as the product of the number of seeds and the number of episodes per seed
    total_test_episodes = num_episodes_each_seed * len(seeds)

    # Setting the type of the environment manager to 'base' for the visualization purposes.
    create_config.env_manager.type = 'base'
    # The number of environments to evaluate concurrently. Set to 1 for visualization purposes.
    main_config.env.evaluator_env_num = 1
    # The total number of evaluation episodes that should be run.
    main_config.env.n_evaluator_episode = 1
    # A boolean flag indicating whether to render the environments in real-time.
    main_config.env.render_mode_human = False

    # A boolean flag indicating whether to save the video of the environment.
    main_config.env.save_replay = True
    # The path where the recorded video will be saved.
    main_config.env.replay_path = './video'
    # The maximum number of steps for each episode during evaluation. This may need to be adjusted based on the specific characteristics of the environment.
    main_config.env.eval_max_episode_steps = int(100)

    # These lists will store the mean and total rewards for each seed.
    returns_mean_seeds = []
    returns_seeds = []

    # The main evaluation loop. For each seed, the MuZero model is evaluated and the mean and total rewards are recorded.
    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=False,
            model_path=model_path
        )
        print(returns_mean, returns)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    # Convert the list of mean and total rewards into numpy arrays for easier statistical analysis.
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Printing the evaluation results. The average reward and the total reward for each seed are displayed, followed by the mean reward across all seeds.
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)