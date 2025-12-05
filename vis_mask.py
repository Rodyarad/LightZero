import os
from easydict import EasyDict
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map
from lzero.entry.eval_muzero import eval_muzero


def main(env_id='Navigation5x5-v0', seed=0, model_path=None):
    """
    Run inference for unizero action abstraction model.
    
    Args:
        env_id: Environment ID
        seed: Random seed
        model_path: Path to the model checkpoint
    """
    action_space_size = shapes2d_env_action_space_map[env_id]
    num_objects = action_space_size // 4

    # Config parameters (matching training config)
    collector_env_num = 8
    game_segment_length = 20
    evaluator_env_num = 1  # For inference, we only need 1 environment
    num_simulations = 50
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25

    # Main config
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
        run_id_comet_ml=None,
        policy=dict(
            cuda=True,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1e6, ), ), ),
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                num_objects=num_objects,
                mcts_mask_thres=0.5,
                world_model_cfg=dict(
                    policy_entropy_weight=1e-4,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    rotary_emb=False,
                    use_action_absraction=True,
                    num_objects=num_objects,
                    mask_temp=1.0,
                    mask_thres=0.5,
                    mask_l1_weight=0,
                ),
            ),
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            num_simulations=num_simulations,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            causal_puct_start_step=0,
        ),
    )
    shapes2d_unizero_config = EasyDict(shapes2d_unizero_config)
    main_config = shapes2d_unizero_config

    # Create config
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

    main_config.exp_name = f'inference_unizero_action_abstraction_{env_id}_seed{seed}'

    # Default model path if not provided
    if model_path is None:
        model_path = 'weights/ckpt_best_mask_unizero.pth.tar'
    
    # Convert to absolute path if relative
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Running inference for 1 episode in environment: {env_id}")
    
    # Create visualization directory
    visualize_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    
    # Run inference with visualization
    mean_return, returns = eval_muzero(
        input_cfg=[main_config, create_config],
        seed=seed,
        model_path=model_path,
        num_episodes_each_seed=1,
        print_seed_details=True,
        visualize=True,
        visualize_dir=visualize_dir,
    )
    
    print(f"\n{'='*50}")
    print(f"Inference completed!")
    print(f"Mean return: {mean_return}")
    print(f"Returns: {returns}")
    print(f"Visualizations saved to: {visualize_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run inference for unizero action abstraction')
    parser.add_argument('--env', type=str, help='The environment to use', default='Navigation5x5-v0')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint', default=None)
    args = parser.parse_args()

    main(args.env, args.seed, args.model_path)

