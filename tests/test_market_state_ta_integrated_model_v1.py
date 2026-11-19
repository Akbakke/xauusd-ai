"""M5/MTF direction uses one native encode and excludes M1 model input."""
import pytest
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import (
    IntegratedTAMarketStateModel, masked_structure_hit_sums,
)
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests.test_market_state_time_query_dedicated_m5_side_v1 import WIDTH, _backbone
from tests.test_market_state_time_query_native_F_model import features


def _model(monkeypatch, calls):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        assert kwargs.pop('research_representation_only') is True
        assert kwargs.pop('research_timeframe_representations') is True
        calls.append(1)
        rows = seq_x.shape[0]
        hidden = (seq_x[:, :1, :1] + torch.arange(5, dtype=torch.float32)[None, :, None])
        return {'entry_q_timeframe_hidden': hidden.expand(rows, 5, WIDTH).clone()}
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', synthetic_forward)
    state = FactorizedTimeQueryMarketStateModel(
        _backbone(), dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True)
    return IntegratedTAMarketStateModel(NativeFutureDualModel(state))


def _inputs(batch=2):
    return dict(features=features(batch, ta_only=True),
                query_k=torch.tensor([NATIVE_K] * batch, dtype=torch.int64))


def test_direct_m5_mtf_heads_preserve_dual_outputs_and_use_one_backbone(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    out = model(**args)
    assert calls == [1]
    calls.clear()
    old = model.dual(**args['features'], query_k=args['query_k'])
    assert calls == [1]
    for key in old:
        assert torch.equal(out[key], old[key])
    assert out['m5_structure_hit_logits'].shape == (2, 2)
    assert model.research_objective['M5_structure_truth_as_forward_input'] is False


def test_m5_input_reaches_side_and_structure_gradients(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    args['features']['seq_x'].requires_grad_()
    out = model(**args)
    (out['state_activity_side_logits'][:, :, 1].sum()
     + out['future_activity_side_logits'][:, :, 1].sum()
     + out['m5_structure_hit_logits'].sum()).backward()
    assert torch.count_nonzero(args['features']['seq_x'].grad) > 0
    assert torch.count_nonzero(model.m5_structure_head.weight.grad) > 0
    assert model.dual.state_model.m5_side_head.weight.grad is not None
    for head in model.dual.state_model.non_m5_side_heads.values():
        assert head.weight.grad is not None
    assert calls == [1]


def test_m1_inputs_and_legacy_checkpoint_are_rejected(monkeypatch):
    model = _model(monkeypatch, [])
    args = _inputs()
    for name, value in (
        ('m1_path_bps', torch.zeros(2, 5, 4)),
        ('m1_observed', torch.ones(2, 5, dtype=torch.bool)),
    ):
        with pytest.raises(TypeError, match=name):
            model(**args, **{name: value})
    assert not any('m1' in name for name, _ in model.named_parameters())
    assert model.research_objective['forecast_input_timeframes'] == [
        'M5', 'M15', 'H1', 'H4', 'D1']
    legacy = dict(model.state_dict())
    legacy['m1_tf_gate.weight'] = torch.zeros(5 * WIDTH, 25)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        model.load_state_dict(legacy)
    with torch.no_grad():
        model.integrated_objective_sha256.zero_()
    with pytest.raises(ValueError, match='MODEL_BINDING'):
        model(**args)


def test_structure_mask_has_zero_unknown_gradient():
    logits = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    hits = torch.tensor([[True, False], [False, True]])
    observed = torch.tensor([[True, False], [False, True]])
    report = masked_structure_hit_sums(logits, hits, observed)
    assert report['observed_counts'].tolist() == [1, 1]
    report['loss_sums'].sum().backward()
    assert logits.grad[0, 1] == logits.grad[1, 0] == 0
    assert logits.grad[0, 0] != 0 and logits.grad[1, 1] != 0


def test_paired_native_and_sampled_queries_share_one_encoder(monkeypatch):
    calls = []
    # Isolate decoder initialization from preceding tests' random draws.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _model(monkeypatch, calls)
    args = _inputs()
    sampled = torch.tensor([[1, 1, 2, 4, 24]] * 2, dtype=torch.int64)
    native = torch.tensor([NATIVE_K] * 2, dtype=torch.int64)
    paired = model(**{**args, 'query_k': torch.stack((sampled, native), dim=1)})
    assert calls == [1]
    assert paired['state_native_activity_side_logits'].shape == (2, 5, 2)
    assert paired['state_native_log_probs'].shape == (2, 5, 3)
    calls.clear()
    old_sampled = model(**{**args, 'query_k': sampled})
    old_native = model(**{**args, 'query_k': native})
    assert calls == [1, 1]
    assert torch.equal(paired['state_activity_side_logits'], old_sampled['state_activity_side_logits'])
    torch.testing.assert_close(paired['state_native_activity_side_logits'],
                               old_native['state_activity_side_logits'], atol=1e-5, rtol=1e-6)
    assert torch.equal(paired['future_activity_side_logits'], old_native['future_activity_side_logits'])
    assert torch.equal(paired['m5_structure_hit_logits'], old_native['m5_structure_hit_logits'])


def _native_model_binding(normalization):
    from gx1.contracts.market_state_ta_integrated_model_v1 import _seal_native
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    profile = compile_full_ta_entry_execution_profile()
    return _seal_native({
        'schema_version': 'gx1_native_m5_bar_direction_inputs_v1',
        'source_bar_seconds': 300, 'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
        'source_clock_contract': 'native_m5_and_closed_higher_no_m1_v1',
        'feature_provider_contract_sha256': normalization['feature_provider_contract_sha256'],
        'source_axes_sha256': normalization['source_axes_sha256'],
        'input_normalization_sha256': normalization['contract_sha256'],
        'execution_profile': {key: profile[key] for key in ('name', 'contract_sha256')},
        'warmstart_used': False,
    })


def test_native_bar_factory_accepts_fresh_normalization_and_binds_checkpoint(monkeypatch):
    from gx1.contracts.market_state_ta_integrated_model_v1 import (
        build_native_bar_direction_model, NativeM5BarDirectionModel,
    )
    from tests.test_entry_model_native_input_normalization import native_m5_normalization_fixture
    normalization, raw, cats = native_m5_normalization_fixture()
    architecture = {'dropout': .05, 'multi_tf_num_layers': 2, 'multi_tf_scale': .5,
                    'specialist_num_layers': 1, 'specialist_fusion_scale': .25,
                    'cross_family_fusion_scale': .25}
    model = build_native_bar_direction_model(
        input_normalization=normalization, input_binding=_native_model_binding(normalization),
        architecture=architecture, device=torch.device('cpu'))
    assert type(model) is NativeM5BarDirectionModel
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    profile = compile_full_ta_entry_execution_profile()
    pairs = sorted(zip(profile['temporal_alias_signal_indices'],
                       profile['temporal_alias_ctx_cont_indices']))
    assert len(pairs) == 132
    assert pairs == [(a['signal_index'], a['ctx_cont_index'])
                     for a in normalization['temporal_aliases']]
    assert model.backbone.temporal_alias_signal_idx.tolist() == [si for si, _ in pairs]
    assert model.backbone.temporal_alias_ctx_cont_idx.tolist() == [ci for _, ci in pairs]
    assert all(profile['physical_field_names']['signal'][si] ==
               'ctx_cont.' + profile['physical_field_names']['ctx_cont'][ci] for si, ci in pairs)
    for family in profile['local_specialists']:
        assert getattr(model.backbone, 'specialist_idx_' + family).tolist() == list(
            profile['local_signal_indices_by_specialist'][family])
        assert getattr(model.backbone, 'multi_tf_specialist_idx_' + family).tolist() == list(
            profile['mtf_indices_by_specialist'][family])
    assert not hasattr(model, 'dual')
    assert not hasattr(model.backbone, 'dynamic_m5_query_horizons')
    assert model.backbone.cfg.dynamic_m5_query_minutes is None
    assert model.research_objective['class_order'] == [-1, 0, 1]
    stack_index = profile['physical_field_names']['signal'].index('ctx_cont.d1_ema_stack_aligned_v2')
    consumer_input = torch.from_numpy(raw['signal'][:3].copy()).requires_grad_()
    normalized = model.backbone._normalize_input_surface(consumer_input, surface='signal')
    assert torch.unique(normalized[:, stack_index]).numel() == 3
    model.backbone.seq_proj(normalized).square().sum().backward()
    assert bool((consumer_input.grad[:, stack_index].abs() > 0.).all())
    binary_index = profile['physical_field_names']['signal'].index('smc_bos_up')
    assert torch.equal(normalized[:, binary_index], consumer_input[:, binary_index])
    assert bool((consumer_input.grad[:, binary_index].abs() > 0.).all())
    with pytest.raises(RuntimeError, match='NATIVE_BAR_DIRECTION_REPRESENTATION_ONLY'):
        model.backbone(None, None, ctx_cat=None, ctx_cont=None,
                       seq_m5=None, seq_m15=None, seq_h1=None, seq_h4=None, seq_d1=None)
    model.load_state_dict(model.state_dict())
    legacy = dict(model.state_dict())
    legacy['native_bar_objective_sha256'] = torch.zeros(32, dtype=torch.uint8)
    before = model.open_heads['M5'].weight.detach().clone()
    with pytest.raises(ValueError, match='NATIVE_CHECKPOINT_IDENTITY'):
        model.load_state_dict(legacy)
    assert torch.equal(before, model.open_heads['M5'].weight)

    calls = []
    def synthetic_encode(self, **values):
        assert values.pop('research_representation_only') is True
        assert values.pop('research_timeframe_representations') is True
        calls.append(set(values))
        width = self.cfg.d_model
        source = values['seq_x'][:, -1, :width]
        hidden = self.entry_q_joint_in(torch.cat((source,) * 4, dim=1))
        return {'entry_q_timeframe_hidden': hidden[:, None].expand(-1, 5, -1)}
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', synthetic_encode)
    values = {'seq_x': torch.from_numpy(raw['signal'][None]).requires_grad_(),
              'snap_x': torch.from_numpy(raw['signal'][-1:]),
              'ctx_cont': torch.from_numpy(raw['ctx_cont'][-1:]),
              'ctx_cat': torch.from_numpy(cats[-1:])}
    values.update({'seq_' + tf: torch.from_numpy(raw['mtf_' + tf][None])
                   for tf in ('m5', 'm15', 'h1', 'h4', 'd1')})
    out = model(values)
    assert set(out) == {'open_logits', 'close_logits'}
    assert out['open_logits'].shape == out['close_logits'].shape == (1, 5, 3)
    sum(value.sum() for value in out.values()).backward()
    assert torch.count_nonzero(values['seq_x'].grad)
    assert all(head.weight.grad is not None for head in model.open_heads.values())
    assert all(head.weight.grad is not None for head in model.close_heads.values())
    assert len(calls) == 1
    with pytest.raises(ValueError, match='NATIVE_FEATURE_KEYS'):
        model({**values, 'future_validity_ns': torch.zeros(1, 5)})
    with pytest.raises(TypeError, match='query_k'):
        model(values, query_k=torch.ones(1, 5))


def test_native_ta_fresh_process_firewall_covers_import_fit_factory_and_forward():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent('''
        from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
        guard = install_native_m5_direction_io_firewall()
        import json
        from pathlib import Path
        import numpy as np
        import torch
        from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
        from gx1.contracts.market_state_native_m5_feature_inputs_v1 import native_m5_feature_provider_contract
        from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
        from tests.test_entry_model_native_input_normalization import native_m5_normalization_fixture
        from tests.test_market_state_ta_integrated_model_v1 import _native_model_binding
        torch.set_num_threads(1)
        torch.manual_seed(41)
        normalization, raw, cats = native_m5_normalization_fixture()
        profile = compile_full_ta_entry_execution_profile()
        provider = native_m5_feature_provider_contract()
        architecture = {'dropout': .05, 'multi_tf_num_layers': 2, 'multi_tf_scale': .5,
            'specialist_num_layers': 1, 'specialist_fusion_scale': .25, 'cross_family_fusion_scale': .25}
        model = build_native_bar_direction_model(input_normalization=normalization,
            input_binding=_native_model_binding(normalization), architecture=architecture,
            device=torch.device('cpu')).eval()
        geometry = {'m5':16, 'm15':64, 'h1':96, 'h4':96, 'd1':252}
        sequence = raw['signal'][np.arange(96) % 32]
        batch = {'seq_x':torch.from_numpy(sequence[None]), 'snap_x':torch.from_numpy(sequence[-1:]),
            'ctx_cont':torch.from_numpy(raw['ctx_cont'][-1:]), 'ctx_cat':torch.from_numpy(cats[-1:])}
        batch.update({'seq_'+tf:torch.from_numpy(raw['mtf_'+tf][np.arange(length)%32][None])
                      for tf,length in geometry.items()})
        with torch.inference_mode():
            output = model(batch)
        assert all(value.shape==(1,5,3) and torch.isfinite(value).all() for value in output.values())
        serialized = json.dumps([normalization,profile,provider,model.research_objective])
        assert 'NATIVE_GC_CONTEXT_FIELDS_20261018_049' not in serialized
        assert '3e444777099a848cd8795688b57a19122d1ac5454f42902a3495e64aad73a757' not in serialized
        assert guard['denied_attempts']==0
        for operation in (
            lambda:Path('/tmp/NATIVE_GC_CONTEXT_FIELDS_20261018_049.json').read_bytes(),
            lambda:Path('/tmp/XAU_M1_NATIVE/MANIFEST.json').stat(),
            lambda:np.load('/tmp/inputs_M1.npz',allow_pickle=False),
            lambda:torch.load('/tmp/checkpoint_M1.pt',weights_only=True),
        ):
            try:
                operation()
            except PermissionError as error:
                assert 'DIRECTION_M1_IO_FORBIDDEN' in str(error)
            else:
                raise AssertionError('forbidden IO reached')
        assert guard['denied_attempts']==4
        print('NATIVE_FIREWALL_PASS')
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith('NATIVE_FIREWALL_PASS')
