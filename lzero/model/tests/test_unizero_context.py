"""Single-task UniZero context contracts, without environments or OCR checkpoints."""
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ding.utils import deep_merge_dicts
from easydict import EasyDict

from lzero.model.sampled_unizero_model import SampledUniZeroModel
from lzero.model.unizero_model import UniZeroModel
from lzero.model.unizero_world_models.tokenizer import Tokenizer
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.mcts.tree_search.search_path import SearchPathTracker
from lzero.policy.sampled_unizero import SampledUniZeroPolicy
from lzero.policy.unizero import UniZeroPolicy


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def world_config(continuous=False, **overrides):
    cfg = copy.deepcopy(UniZeroPolicy.default_config().model.world_model_cfg)
    cfg.update(
        model_type='slot', obs_type='slot', tokens_per_block=4,
        max_blocks=5, max_tokens=20, train_context_length=5,
        context_length=8, infer_context_length=None,
        embed_dim=16, num_heads=4, num_layers=2, norm_type='LN',
        device='cpu', env_num=8, num_slots=2, continuous_action_space=continuous,
        action_space_size=3, support_size=7, embed_pdrop=0., attn_pdrop=0., resid_pdrop=0.,
        sigma_type='conditioned', bound_type=None, fixed_sigma_value=0.5,
        policy_loss_type='kl',
    )
    cfg.update(overrides)
    return cfg


def make_world(continuous=False, **overrides):
    cfg = world_config(continuous, **overrides)
    model = WorldModel(cfg, Tokenizer(torch.nn.Identity(), None, obs_type='slot')).eval()
    # Zero-initialized heads must not hide incorrect context dependencies.
    for head in (model.head_observations, model.head_rewards, model.head_value):
        for layer in reversed(head.head_module):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.03)
                break
    return model


def observe(model, values, actions, ids=None, mode='collect'):
    obs = torch.tensor(values, dtype=torch.float32)[:, None, None].expand(-1, 2, 16)
    return model.wm_forward_for_initial_infererence(
        obs, actions, obs, env_ids=ids, context_mode=mode
    )


def observation_ids(model, env_id, mode='collect'):
    return [int(s[0, 0]) for s in model.env_history[mode][env_id]['obs']]


def action(continuous, value=1):
    return np.full(3, value, dtype=np.float32) if continuous else value


@pytest.mark.parametrize('continuous', [False, True])
def test_env_id_histories_survive_reorder_shrink_and_partial_reset(continuous):
    model = make_world(continuous, infer_context_length=3)
    observe(model, [10, 20], [-1, -1], [7, 3])
    observe(model, [21, 11], [action(continuous), action(continuous)], [3, 7])
    observe(model, [22], [action(continuous)], [3])
    assert observation_ids(model, 7) == [10, 11]
    assert observation_ids(model, 3) == [20, 21, 22]
    model.reset_env_history([7])
    observe(model, [12, 23], [-1, action(continuous)], [7, 3])
    assert observation_ids(model, 7) == [12]
    assert observation_ids(model, 3) == [21, 22, 23]
    assert [len(states) for states, _ in model.root_context] == [0, 2]


@pytest.mark.parametrize('continuous', [False, True])
def test_mixed_episode_start_is_per_element(continuous):
    model = make_world(continuous, infer_context_length=3)
    observe(model, [1, 10], [-1, -1])
    observe(model, [2, 20], [action(continuous), -1])
    assert observation_ids(model, 0) == [1, 2]
    assert observation_ids(model, 1) == [20]


def test_collect_eval_histories_are_isolated():
    model = make_world()
    observe(model, [1], [-1], [0])
    observe(model, [10], [-1], [0], 'eval')
    observe(model, [2], [1], [0])
    assert observation_ids(model, 0) == [1, 2]
    assert observation_ids(model, 0, 'eval') == [10]
    model.reset_env_history(context_mode='eval')
    assert observation_ids(model, 0) == [1, 2]
    assert model.env_history['eval'] == {}


@pytest.mark.parametrize('continuous', [False, True])
def test_grouped_roots_match_independent_env_histories(continuous):
    batched = make_world(continuous, infer_context_length=3)
    separate = [make_world(continuous, infer_context_length=3) for _ in range(2)]
    for model in separate:
        model.load_state_dict(batched.state_dict())
    observe(batched, [10, 20], [-1, -1], [7, 3])
    observe(separate[0], [10], [-1], [7])
    observe(separate[1], [20], [-1], [3])
    observe(batched, [21], [action(continuous)], [3])
    observe(separate[1], [21], [action(continuous)], [3])
    batched.reset_env_history([7])
    separate[0].reset_env_history([7])
    outputs = observe(batched, [22, 11], [action(continuous), -1], [3, 7])
    for b, model, value, env_id, previous in (
            (0, separate[1], 22, 3, action(continuous)),
            (1, separate[0], 11, 7, -1)):
        single = observe(model, [value], [previous], [env_id])
        torch.testing.assert_close(outputs.logits_policy[b:b + 1], single.logits_policy)
        torch.testing.assert_close(outputs.logits_value[b:b + 1], single.logits_value)
        torch.testing.assert_close(outputs.output_sequence[b:b + 1], single.output_sequence)


def test_future_observations_and_current_action_do_not_leak_into_policy():
    model = make_world()
    obs, acts = torch.randn(2, 5, 2, 16), torch.randint(0, 3, (2, 5, 1))
    original = model.forward({'obs_embeddings_and_act_tokens': (obs, acts)})
    changed_obs, changed_acts = obs.clone(), acts.clone()
    changed_obs[:, 3:] = torch.randn_like(changed_obs[:, 3:])
    changed_acts[:, 2:] = (changed_acts[:, 2:] + 1) % 3
    changed = model.forward({'obs_embeddings_and_act_tokens': (changed_obs, changed_acts)})
    torch.testing.assert_close(original.logits_policy[:, :3], changed.logits_policy[:, :3])
    torch.testing.assert_close(original.logits_value[:, :3], changed.logits_value[:, :3])
    # Next-state predictions may depend on the current action, but not on later timesteps.
    changed_acts[:, 2] = acts[:, 2]
    changed = model.forward({'obs_embeddings_and_act_tokens': (changed_obs, changed_acts)})
    torch.testing.assert_close(original.logits_observations[:, :6], changed.logits_observations[:, :6])


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('window', [1, 2, 4])
def test_real_and_tree_windows_use_infer_length_not_capacity(continuous, window):
    model = make_world(continuous, infer_context_length=window, max_blocks=8, max_tokens=32)
    for t in range(7):
        observe(model, [t], [-1 if t == 0 else action(continuous)])
    assert observation_ids(model, 0) == list(range(7 - window, 7))
    assert len(model.root_context[0][0]) == window - 1
    paths = [([torch.randn(2, 16) for _ in range(7)],
              [action(continuous) for _ in range(7)])]
    groups = model._build_recurrent_context_groups(paths)
    indices, obs, act = groups[window]
    assert obs.shape[1] == act.shape[1] == window
    calls = []
    original = model.forward

    def record(data, **kwargs):
        calls.append(data)
        return original(data, **kwargs)

    model.forward = record
    result = model.forward_recurrent_inference({'search_paths': paths})
    obs2, act2 = calls[1]['last_obs_embeddings_act_tokens_and_current_obs']
    assert obs2.shape[1] == window
    assert act2.shape[1] == window - 1
    assert result[0].shape == (1, 2, 16)
    assert all(torch.isfinite(x).all() for x in result)


@pytest.mark.parametrize('continuous', [False, True])
def test_recurrent_context_uses_real_pairs_in_ready_order(continuous):
    model = make_world(continuous, infer_context_length=4)
    observe(model, [10, 20], [-1, -1], [9, 5])
    observe(model, [21, 11], [action(continuous, 1), action(continuous, 2)], [5, 9])
    paths = [([torch.full((2, 16), float(s))], [action(continuous, 0)]) for s in (21, 11)]
    _, obs, acts = model._build_recurrent_context_groups(paths)[2]
    assert obs[:, :, 0, 0].tolist() == [[20, 21], [10, 11]]
    assert acts[:, 0, 0].tolist() == [1, 2]


@pytest.mark.parametrize('continuous', [False, True])
def test_grouped_recurrent_matches_individual_results(continuous):
    model = make_world(continuous, infer_context_length=4)
    paths = [
        ([torch.randn(2, 16) for _ in range(depth)],
         [action(continuous) for _ in range(depth)])
        for depth in (1, 3, 7, 2)
    ]
    batched = model.forward_recurrent_inference({'search_paths': paths})
    assert batched[0].shape == (4, 2, 16)
    for i, path in enumerate(paths):
        single = model.forward_recurrent_inference({'search_paths': [path]})
        for batch_tensor, single_tensor in zip(batched, single):
            torch.testing.assert_close(batch_tensor[i:i + 1], single_tensor, atol=2e-6, rtol=2e-5)


def test_legacy_context_conversion_and_explicit_override():
    assert make_world(context_length=12).infer_context_length == 3
    assert make_world(context_length=12, infer_context_length=1).infer_context_length == 1


@pytest.mark.parametrize('overrides, message', [
    ({'context_length': 7}, 'multiple'),
    ({'infer_context_length': 5}, 'train_context_length - 1'),
    ({'infer_context_length': 0}, 'positive integer'),
    ({'infer_context_length': 1.5}, 'positive integer'),
    ({'train_context_length': 1}, '2 <='),
    ({'train_context_length': 6}, 'max_blocks'),
])
def test_invalid_context_configuration_fails_fast(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_world(**overrides)


def test_infer_window_changes_latent_predictions_without_changing_weights():
    torch.manual_seed(42)
    short = make_world(infer_context_length=1)
    long = make_world(infer_context_length=3)
    long.load_state_dict(short.state_dict())
    paths = [([torch.randn(2, 16) for _ in range(3)], [0, 1, 2])]
    out_short = short.forward_recurrent_inference({'search_paths': paths})[1]
    out_long = long.forward_recurrent_inference({'search_paths': paths})[1]
    assert not torch.allclose(out_short, out_long)


def test_training_length_and_supervised_latent_horizon():
    model = make_world(infer_context_length=4, max_blocks=8, max_tokens=32)
    B, H = 2, 5
    batch = {
        'observations': torch.randn(B, H, 2, 16),
        'actions': torch.randint(0, 3, (B, H)),
        'rewards': torch.softmax(torch.randn(B, H, 7), dim=-1),
        'target_value': torch.softmax(torch.randn(B, H, 7), dim=-1),
        'target_policy': torch.softmax(torch.randn(B, H, 3), dim=-1),
        'mask_padding': torch.ones(B, H, dtype=torch.bool),
        'ends': torch.zeros(B, H, dtype=torch.long),
    }
    predictions = []
    original = model.forward

    def retain_predictions(data, **kwargs):
        outputs = original(data, **kwargs)
        outputs.logits_observations.retain_grad()
        predictions.append(outputs.logits_observations)
        return outputs

    model.forward = retain_predictions
    losses = model.compute_loss(batch, model.tokenizer)
    assert torch.isfinite(losses.loss_total)
    losses.loss_total.backward()
    gradient = predictions[0].grad
    assert gradient[:, :-2].abs().sum() > 0
    assert gradient[:, -2:].abs().sum() == 0
    assert model.infer_context_length <= H - 1
    bad = dict(batch, observations=batch['observations'][:, :-1])
    with pytest.raises(ValueError, match='Training observations'):
        model.compute_loss(bad, model.tokenizer)


@pytest.mark.parametrize('continuous', [False, True])
def test_tracker_random_tree_and_batch_permutations(continuous):
    rng, B = np.random.default_rng(123), 3
    tracker = SearchPathTracker(B)
    states = [rng.normal(size=(B, 2, 16))]
    references = [[([], []) for _ in range(B)]]
    for simulation in range(30):
        parents = rng.integers(0, simulation + 1, size=B)
        order = rng.permutation(B)
        actions = rng.normal(size=(B, 3)) if continuous else rng.integers(0, 3, size=B)
        paths = tracker.build_paths(states, parents[order], order, actions)
        for b, (obs, act) in enumerate(paths):
            prior_obs, prior_act = references[parents[b]][b]
            np.testing.assert_allclose(obs, prior_obs + [states[parents[b]][b]])
            np.testing.assert_allclose(act, prior_act + [actions[b]])
        references.append([(list(s), list(a)) for s, a in paths])
        states.append(rng.normal(size=(B, 2, 16)))


def make_policy(continuous=False, enable_field=('collect', 'eval'), device='cpu'):
    cfg = UniZeroPolicy.default_config()
    if continuous:
        cfg = EasyDict(deep_merge_dicts(cfg, SampledUniZeroPolicy.default_config()))
    cfg.update(cuda=device == 'cuda', device=device, num_unroll_steps=5, batch_size=2,
               num_simulations=4, collect_num_simulations=4, eval_num_simulations=4,
               collector_env_num=3, evaluator_env_num=3)
    cfg.model.update(
        model_type='slot', observation_shape=(2, 16), action_space_size=3,
        continuous_action_space=continuous, num_of_sampled_actions=4,
        reward_support_range=(-3., 4., 1.), value_support_range=(-3., 4., 1.),
        world_model_cfg=world_config(continuous, device=device),
    )
    cls = SampledUniZeroPolicy if continuous else UniZeroPolicy
    policy = cls(cfg, enable_field=list(enable_field))
    # Usually installed by learn initialization; these tests intentionally enable inference only.
    policy.value_inverse_scalar_transform_handle = policy._mcts_eval.value_inverse_scalar_transform_handle
    return policy


@pytest.mark.parametrize('continuous', [False, True])
def test_actual_policy_and_cpp_mcts_dynamic_env_ids(continuous):
    policy = make_policy(continuous)
    world = policy._model.world_model

    def collect(ids, values):
        data = torch.tensor(values, dtype=torch.float32)[:, None, None].expand(-1, 2, 16)
        return policy._forward_collect(
            data, action_mask=np.ones((len(ids), 3)), ready_env_id=ids,
            to_play=[-1] * len(ids), timestep=[0] * len(ids)
        )

    collect([7, 3], [10, 20])
    collect([3], [21])
    assert observation_ids(world, 7) == [10]
    assert observation_ids(world, 3) == [20, 21]
    policy._reset_collect([7])
    assert 3 in policy._context_actions['collect']
    collect([7, 3], [11, 22])
    assert observation_ids(world, 7) == [11]
    assert observation_ids(world, 3) == [21, 22]
    policy._reset_collect([3], current_steps=100, reset_init_data=False)
    assert observation_ids(world, 3) == [21, 22]

    data = torch.full((1, 2, 16), 100.)
    policy._forward_eval(data, action_mask=np.ones((1, 3)), ready_env_id=[3],
                         to_play=[-1], timestep=[0])
    assert observation_ids(world, 3, 'eval') == [100]
    assert observation_ids(world, 3) == [21, 22]
    assert 3 in policy._context_actions['collect']
    policy._reset_eval([3])
    assert not world.env_history['eval']
    assert observation_ids(world, 3) == [21, 22]
    policy._reset_collect()
    assert not world.env_history['collect']


@pytest.mark.parametrize('continuous', [False, True])
def test_policy_sets_train_length_independently_of_capacity(continuous):
    policy = make_policy(continuous)
    assert policy._cfg.model.world_model_cfg.train_context_length == 5
    assert policy._model.world_model.train_context_length == 5


@pytest.mark.parametrize('model_cls', [UniZeroModel, SampledUniZeroModel])
def test_existing_state_dicts_remain_compatible(model_cls):
    model = model_cls(observation_shape=(2, 16), action_space_size=3,
                      norm_type='LN', world_model_cfg=world_config())
    state = copy.deepcopy(model.state_dict())
    model.initial_inference(torch.randn(1, 2, 16), [-1], torch.randn(1, 2, 16),
                            env_ids=[7], context_mode='eval')
    model.load_state_dict(state, strict=True)
    assert set(model.state_dict()) == set(state)


@pytest.mark.parametrize('continuous', [False, True])
def test_policy_prepares_training_batch_and_updates_parameters(continuous):
    policy = make_policy(continuous, enable_field=('learn', 'collect', 'eval'))
    B, H, K, E, A = 2, 5, 2, 16, 3
    obs = np.random.default_rng(42).normal(size=(B, H + 1, K, E)).astype(np.float32)
    actions = np.zeros((B, H, A), dtype=np.float32) if continuous else np.zeros((B, H), dtype=np.int64)
    sampled_actions = np.zeros((B, H + 1, 4, A), dtype=np.float32)
    mask = np.ones((B, H + 1), dtype=np.float32)
    current = [obs, actions]
    if continuous:
        current.append(sampled_actions)
    current += [actions, mask, np.arange(B), np.ones(B, dtype=np.float32),
                np.zeros(B), np.zeros((B, H), dtype=np.int64)]
    target = [np.zeros((B, H + 1), dtype=np.float32)] * 2 + [
        np.full((B, H + 1, 4 if continuous else A),
                1. / (4 if continuous else A), dtype=np.float32)]
    world = policy._model.world_model
    prepared = []
    original = world.compute_loss

    def capture(batch, *args, **kwargs):
        prepared.append(batch)
        return original(batch, *args, **kwargs)

    world.compute_loss = capture
    before = [p.detach().clone() for p in world.parameters()]
    policy._forward_learn([current, target, 1])
    assert prepared[0]['observations'].shape == (B, H, K, E)
    assert prepared[0]['actions'].shape[1] == H
    assert all(torch.isfinite(p).all() for p in world.parameters())
    assert any(not torch.equal(old, p) for old, p in zip(before, world.parameters()))


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_cuda_policy_root_and_recurrent_context(continuous):
    policy = make_policy(continuous, device='cuda')
    for ids in ([2, 8], [8], [8, 2]):
        policy._forward_collect(
            torch.randn(len(ids), 2, 16, device='cuda'),
            action_mask=np.ones((len(ids), 3)), ready_env_id=ids,
            to_play=[-1] * len(ids), timestep=[0] * len(ids)
        )
    world = policy._model.world_model
    assert len(world.env_history['collect'][2]['obs']) == 2
    assert all(s.device.type == 'cuda' for s in world.env_history['collect'][8]['obs'])


@pytest.mark.parametrize('module_name, args, expected_train, expected_infer', [
    ('zoo.robosuite.config.robosuite_soz_segment_config_slotcontrast', (0,), 5, 2),
    ('zoo.robosuite.config.robosuite_soz_segment_config_slotcontrast_test', (0,), 5, 2),
    ('zoo.ocrl.config.ocrl_objectzero_segment_config', ('odd-one-out-N4C2S2S1', 0), 10, 4),
    ('zoo.vizdoom.config.vizdoom_objectzero_segment_config', ('VizdoomDefendLine-v0', 0), 10, 4),
])
def test_existing_experiment_configs_use_their_requested_infer_window(
        monkeypatch, module_name, args, expected_train, expected_infer):
    import lzero.entry

    captured = []
    monkeypatch.setattr(lzero.entry, 'train_unizero_segment',
                        lambda configs, **kwargs: captured.append(configs[0]))
    importlib.import_module(module_name).main(*args)
    policy_cfg = captured[0].policy
    world_cfg = policy_cfg.model.world_model_cfg
    context = SimpleNamespace(
        config=world_cfg, train_context_length=None,
        max_blocks=world_cfg.max_blocks, context_length=world_cfg.context_length,
        tokens_per_block=world_cfg.tokens_per_block,
    )
    world_cfg.train_context_length = policy_cfg.num_unroll_steps
    WorldModel._initialize_context_lengths(context)
    assert (context.train_context_length, context.infer_context_length) == (expected_train, expected_infer)
