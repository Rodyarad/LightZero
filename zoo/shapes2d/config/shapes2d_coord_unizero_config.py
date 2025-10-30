from easydict import EasyDict
from zoo.shapes2d.config.shapes2d_env_action_space_map import shapes2d_env_action_space_map
import comet_ml

env_id = 'Navigation5x5-coord-v0'  # You can specify any Shapes2d game here
action_space_size = shapes2d_env_action_space_map[env_id]
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 30
num_simulations = 50
update_per_collect = None
replay_ratio = 0.5
max_env_step = int(5e5)
batch_size = 256
num_unroll_steps = 5
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
shapes2d_coord_unizero_config = dict(
    exp_name=f'data_unizero/shapes2d_coord_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        from_pixels=False,
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=[40],
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000, ), ), ),
        run_id_comet_ml=None,
        model=dict(
            observation_shape=40,
            action_space_size=action_space_size,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='LN',
            model_type='mlp',
            world_model_cfg=dict(
                final_norm_option_in_obs_head='LayerNorm',
                final_norm_option_in_encoder='LayerNorm',
                predict_latent_loss_type='mse',
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                device='cuda',
                action_space_size=action_space_size,
                num_layers=3,
                num_heads=4,
                embed_dim=128,
                env_num=max(collector_env_num, evaluator_env_num),
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                obs_type='vector',
                norm_type='LN',
                rotary_emb=True,
            ),
        ),
        use_wandb=False,
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=50,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-4,
        target_update_freq=50,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

shapes2d_coord_unizero_config = EasyDict(shapes2d_coord_unizero_config)
main_config = shapes2d_coord_unizero_config

shapes2d_coord_unizero_create_config = dict(
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
shapes2d_coord_unizero_create_config = EasyDict(shapes2d_coord_unizero_create_config)
create_config = shapes2d_coord_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
