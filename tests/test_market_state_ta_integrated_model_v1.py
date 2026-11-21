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


def native_gc_model_fixture():
    """Fresh synthetic full-GC domains; never a physical source qualification."""
    import numpy as np
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    from gx1.features import entry_specialist_feature_groups_v1 as groups
    from gx1.features.gc_native_m5_features_v1 import native_m5_field_catalog
    from gx1.contracts.market_state_native_m5_gc_feature_inputs_v1 import native_m5_gc_feature_provider_contract
    from gx1.contracts.market_state_ta_integrated_model_v1 import _seal_native
    from tests.test_entry_model_native_input_normalization import native_m5_normalization_fixture
    from tests.test_entry_gc_mtf_history_v1 import native_synthetic_history
    ta,raw,cats=native_m5_normalization_fixture()
    gc_history,gc_surfaces=native_synthetic_history(unknown_d1=False)
    profile=groups.compile_native_m5_full_gc_entry_execution_profile(gc_history)
    catalog=native_m5_field_catalog();fields=profile['physical_field_names']
    base=[item for item in catalog['fields'] if item['availability_field']==item['name']+'_observed']
    columns=[]
    for item in base:
        tf=item['timeframe'];surface=gc_surfaces[tf];names=surface['field_names']
        columns.extend([surface['features'][:,names.index(item['name'])],
                        surface['features'][:,names.index(item['availability_field'])]])
    local=np.column_stack(columns).astype(np.float32)
    raw['signal']=np.column_stack((raw['signal'],local)).astype(np.float32)
    raw['ctx_cont']=np.column_stack((raw['ctx_cont'],local)).astype(np.float32)
    surfaces=dict(ta['surfaces'])
    for key in ('signal','ctx_cont'):
        surfaces[key]=norm.fit_native_m5_gc_surface_normalization(raw[key],surface=key,
            field_names=fields[key],semantic_categorical_domains=(norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if key=='signal' else {}))
    aliases=[{'signal_index':si,'ctx_cont_index':ci,'signal_field':fields['signal'][si],
        'ctx_cont_field':fields['ctx_cont'][ci],'specialist':next(f for f,ids in profile['ctx_cont_indices_by_specialist'].items() if ci in ids)}
        for si,ci in zip(profile['temporal_alias_signal_indices'],profile['temporal_alias_ctx_cont_indices'])]
    surfaces['ctx_cont']=norm.share_temporal_alias_stats_from_signal(surfaces['ctx_cont'],surfaces['signal'],
        temporal_aliases=aliases,ctx_cont_values=raw['ctx_cont'])
    population=dict(ta['fit_population'])
    population['first_issue_ns']=1719792000000000000
    population['last_issue_ns']=1719801300000000000
    population['per_tf']={tf:{'rows':item['rows'],'source_rows_sha256':item['native_row_indices_sha256'],
        'available_ns_sha256':item['available_ns_sha256']}
        for tf,item in gc_history['normalization']['provenance']['row_union'].items()}
    provider=native_m5_gc_feature_provider_contract(gc_history)
    fitted=norm.build_native_m5_full_gc_input_normalization_contract(native_m5_manifest=ta['native_m5_manifest'],
        feature_provider_contract_sha256=provider['contract_sha256'],source_axes_sha256='7'*64,
        fit_population=population,surfaces=surfaces,ctx_cat=ta['ctx_cat'],temporal_aliases=aliases,
        gc_mtf_history=gc_history,native_gc_source_binding_sha256='9'*64)
    binding=_native_model_binding(fitted)
    binding.update(schema_version='gx1_native_m5_full_gc_bar_direction_inputs_v1',
        execution_profile={key:profile[key] for key in ('name','contract_sha256')},
        native_gc_source_binding_sha256='9'*64,gc_layout_sha256=gc_history['layout']['contract_sha256'],
        gc_normalization_sha256=gc_history['normalization']['contract_sha256'])
    binding=_seal_native({k:v for k,v in binding.items() if k!='contract_sha256'})
    return fitted,binding,gc_history,raw,cats,gc_surfaces


def _assert_native_full_gc_family_projection_bar_heads_and_unknown_d1(monkeypatch):
    import numpy as np
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.features import entry_specialist_feature_groups_v1 as groups
    fitted,binding,history,raw,cats,gc_surfaces=native_gc_model_fixture()
    architecture={'dropout':.05,'multi_tf_num_layers':2,'multi_tf_scale':.5,
        'specialist_num_layers':1,'specialist_fusion_scale':.25,'cross_family_fusion_scale':.25}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model=build_native_bar_direction_model(input_normalization=fitted,input_binding=binding,
            gc_mtf_history=history,architecture=architecture,device=torch.device('cpu')).eval()
    assert not hasattr(model.backbone,'dynamic_m5_query_horizons')
    assert len(model.research_objective['forward_inputs'])==14
    profile=groups.compile_native_m5_full_gc_entry_execution_profile(history)
    assert len(profile['local_specialists'])==len(profile['mtf_specialists'])==8
    assert len(profile['macro_context_fields'])==61
    for tf in history['layout']['timeframes']:
        ids=[i for values in history['layout']['family_indices'][tf].values() for i in values]
        assert sorted(ids)==list(range(history['layout']['widths'][tf]))
        for family,indices in history['layout']['family_indices'][tf].items():
            if indices:
                layer=model.backbone.gc_mtf_family_proj[tf.lower()+'__'+family]
                assert layer.in_features==len(indices)
                assert getattr(model.backbone,'gc_mtf_specialist_idx_'+tf.lower()+'__'+family).tolist()==indices
        for side in ('buy','sell'):
            name='gc_'+tf.lower()+'_observed_'+side+'_max_executed_trade_contracts'
            index=history['layout']['field_names'][tf].index(name)
            assert index in history['layout']['family_indices'][tf]['momentum_flow_encoder']
    sequence=raw['signal'][np.arange(96)%32]
    batch={'seq_x':torch.from_numpy(sequence[None]),'snap_x':torch.from_numpy(sequence[-1:]),
        'ctx_cont':torch.from_numpy(raw['ctx_cont'][-1:]),'ctx_cat':torch.from_numpy(cats[-1:])}
    lengths={'m5':16,'m15':64,'h1':96,'h4':96,'d1':252}
    for tf,length in lengths.items():
        batch['seq_'+tf]=torch.from_numpy(raw['mtf_'+tf][np.arange(length)%32][None])
        batch['gc_seq_'+tf]=torch.from_numpy(gc_surfaces[tf.upper()]['features'][np.arange(length)%32][None]).requires_grad_()
    output=model(batch)
    assert all(value.shape==(1,5,3) and torch.isfinite(value).all() for value in output.values())
    loss=sum(torch.nn.functional.cross_entropy(value.reshape(-1,3),torch.ones(5,dtype=torch.int64)) for value in output.values())
    loss.backward()
    for route,layer in model.backbone.gc_mtf_family_proj.items():
        assert layer.weight.grad is not None and bool((layer.weight.grad!=0.).any()),route
    for tf in history['layout']['timeframes']:
        tensor=batch['gc_seq_'+tf.lower()];names=history['layout']['field_names'][tf]
        known_market_gradient=0.
        for vi,mi in history['layout']['value_mask_pairs'][tf]:
            missing=tensor[...,mi]==0.
            assert torch.count_nonzero(tensor.grad[...,vi][missing])==0
            if names[vi] not in history['layout']['binary_fields'][tf]:
                known_market_gradient+=float(tensor.grad[...,vi][~missing].abs().sum())
        assert known_market_gradient>0.,tf
        for family,indices in history['layout']['family_indices'][tf].items():
            market=[vi for vi,mi in history['layout']['value_mask_pairs'][tf]
                    if vi in indices and names[vi] not in history['layout']['binary_fields'][tf]]
            if market:
                assert float(tensor.grad[...,market].abs().sum())>0.,(tf,family)
    assert all(head.weight.grad is not None for heads in (model.open_heads,model.close_heads) for head in heads.values())
    missing={key:value.detach().clone() for key,value in batch.items()}
    missing['gc_seq_d1'].zero_()
    d1fields=set(history['layout']['field_names']['D1'])
    for j,name in enumerate(profile['physical_field_names']['signal']):
        if name.removeprefix('ctx_cont.') in d1fields:
            missing['seq_x'][...,j]=0.;missing['snap_x'][...,j]=0.
    for j,name in enumerate(profile['physical_field_names']['ctx_cont']):
        if name in d1fields:missing['ctx_cont'][...,j]=0.
    with torch.inference_mode():
        assert all(value.shape==(1,5,3) and torch.isfinite(value).all() for value in model(missing).values())
    corrupted=dict(model.state_dict());corrupted['backbone.gc_mtf_layout_sha256']=torch.zeros(32,dtype=torch.uint8)
    before=model.open_heads['M5'].weight.detach().clone()
    with pytest.raises(ValueError,match='NATIVE_CHECKPOINT_IDENTITY'):
        model.load_state_dict(corrupted)
    assert torch.equal(before,model.open_heads['M5'].weight)
    with pytest.raises(ValueError,match='NATIVE_FEATURE_KEYS'):
        model({key:value for key,value in batch.items() if key!='gc_seq_d1'})
    from gx1.contracts import market_state_ta_integrated_model_v1 as factory_owner
    constructor=factory_owner.EntryV10CtxHybridTransformer
    def wrong_context_width(**kwargs):
        kwargs['ctx_cont_dim']=len(profile['physical_field_names']['ctx_cont'])-1
        return constructor(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(factory_owner,'EntryV10CtxHybridTransformer',wrong_context_width)
        with pytest.raises(RuntimeError,match='ARCHITECTURE_MISMATCH.*shared_surface.ctx_cont_dim'):
            factory_owner.build_native_bar_direction_model(input_normalization=fitted,input_binding=binding,
                gc_mtf_history=history,architecture=architecture,device=torch.device('cpu'))


@pytest.fixture
def native_gc_single_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_native_gc_counts_preserve_existing_ta_observation_masks(native_gc_single_thread):
    from copy import deepcopy
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    fitted,_,_,_,_,_=native_gc_model_fixture()
    stats=fitted['surfaces']['signal'];names=stats['field_names']
    value=names.index('ctx_cont.close_distance_below_high_range_fraction')
    assert stats['fit_row_count']==32 and stats['observed_fit_counts'][value]==16
    norm.require_surface_normalization(stats,surface='signal',field_names=names)
    observed_values={vi for vi,mi in (*norm.native_gc_masking_pairs(names),
        *norm._micro_context.range_observation_masking_pairs(names))}
    unmasked=next(i for i in range(len(names)) if i not in observed_values)
    bad=deepcopy(stats);bad['observed_fit_counts'][unmasked]=16
    with pytest.raises(RuntimeError,match='NATIVE_GC_UNMASKED_COUNTS'):
        norm.require_surface_normalization(bad,surface='signal',field_names=names)


def test_native_full_gc_uses_existing_family_projection_and_bar_heads_with_unknown_d1():
    """A warmed legacy registry must never conceal native source access."""
    import subprocess
    import sys
    from pathlib import Path
    code = r"""
import importlib.util
spec=importlib.util.spec_from_file_location('native_direction_guard','gx1/contracts/native_m5_direction_io_firewall_v1.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.install_native_m5_direction_io_firewall()
import pytest,torch
from tests.test_market_state_ta_integrated_model_v1 import _assert_native_full_gc_family_projection_bar_heads_and_unknown_d1
previous=torch.get_num_threads();torch.set_num_threads(1);patch=pytest.MonkeyPatch()
try:
    _assert_native_full_gc_family_projection_bar_heads_and_unknown_d1(patch)
finally:
    patch.undo();torch.set_num_threads(previous)
assert state['denied_attempts']==0,state
print('FRESH_GC_FULL_FORWARD_BACKWARD_PASS 0 {}')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,(result.stdout+result.stderr)[-12000:]
    assert result.stdout.strip().endswith('FRESH_GC_FULL_FORWARD_BACKWARD_PASS 0 {}')


def _native_ta_prior_packet():
    import numpy as np
    from tests.test_entry_model_native_input_normalization import native_m5_normalization_fixture
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_bar_train_prior
    normalization,raw,cats=native_m5_normalization_fixture()
    issues=normalization['fit_population']['first_issue_ns']+np.arange(32,dtype=np.int64)*300_000_000_000
    shape=(32,5);known=np.zeros(shape,np.bool_);known[:,-1]=True
    signs=np.broadcast_to((np.arange(32)%3-1).astype(np.int8)[:,None],shape).copy()
    truth={'issue_ns':issues,'target_row':np.broadcast_to(np.arange(1,33,dtype=np.int64)[:,None],shape).copy(),
        'validity_ns':np.broadcast_to((issues+300_000_000_000)[:,None],shape).copy(),
        'target_present':np.ones(shape,np.bool_),'observed_previous':np.ones(shape,np.bool_),
        'mature':np.ones(shape,np.bool_),'eligible':np.ones(shape,np.bool_),
        'open_known_at_issue':known,'open_sign':signs,'close_sign':signs.copy()}
    prior=fit_native_bar_train_prior(truth=truth,fold_upper_ns=normalization['fit_population']['cutoff_ns'],
        target_contract_sha256=target_contract()['contract_sha256'])
    return normalization,raw,cats,truth,prior


def test_native_ta_prior_heads_are_exact_input_independent_and_enable_backbone_after_first_step(monkeypatch,native_gc_single_thread):
    import copy
    import numpy as np
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model,NativeM5BarDirectionModel
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits,native_bar_direction_loss,fit_native_bar_train_prior
    normalization,raw,cats,truth,prior=_native_ta_prior_packet();binding=_native_model_binding(normalization)
    architecture={'dropout':.05,'multi_tf_num_layers':2,'multi_tf_scale':.5,
        'specialist_num_layers':1,'specialist_fusion_scale':.25,'cross_family_fusion_scale':.25}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model=build_native_bar_direction_model(input_normalization=normalization,input_binding=binding,
            architecture=architecture,device=torch.device('cpu'),train_prior=prior)
    support=model.require_prior_initialization_state()['per_head_support']
    assert support['open']['D1']['predictive_train_cells']==0
    assert support['open']['D1']['train_observations_available']is False
    assert support['open']['D1']['role'].startswith('UNFITTED_uniform')
    assert all(parameter.requires_grad for parameter in model.parameters())
    def synthetic_encode(self,**values):
        assert values.pop('research_representation_only')is True
        assert values.pop('research_timeframe_representations')is True
        source=values['seq_x'][:,-1,:self.cfg.d_model]
        hidden=self.entry_q_joint_in(torch.cat((source,)*4,dim=1))
        return {'entry_q_timeframe_hidden':hidden[:,None].expand(-1,5,-1)}
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',synthetic_encode)
    sequence=np.stack((raw['signal'],raw['signal'][::-1]))
    values={'seq_x':torch.from_numpy(sequence),'snap_x':torch.from_numpy(sequence[:,-1]),
        'ctx_cont':torch.from_numpy(np.stack((raw['ctx_cont'][-1],raw['ctx_cont'][0]))),
        'ctx_cat':torch.from_numpy(np.stack((cats[-1],cats[0])))}
    values.update({'seq_'+tf:torch.from_numpy(np.stack((raw['mtf_'+tf],raw['mtf_'+tf][::-1])))
        for tf in ('m5','m15','h1','h4','d1')})
    expected=native_bar_train_prior_logits(prior,rows=2);output=model(values)
    for key in output:assert torch.equal(output[key],torch.from_numpy(expected[key]))
    assert torch.equal(torch.softmax(output['open_logits'][:,-1],dim=-1),torch.full((2,3),1/3))
    selected={key:value[:2].copy()for key,value in truth.items()}
    kwargs={'truth':selected,'fold_upper_ns':normalization['fit_population']['cutoff_ns'],
        'target_contract_sha256':prior['target_contract_sha256']}
    loss=native_bar_direction_loss(model_output=output,**kwargs)['loss'];loss.backward()
    assert torch.count_nonzero(model.backbone.entry_q_joint_in.weight.grad)==0
    assert torch.count_nonzero(model.close_heads['M5'].weight.grad)>0
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    native_bar_direction_loss(model_output=model(values),**kwargs)['loss'].backward()
    assert torch.count_nonzero(model.backbone.entry_q_joint_in.weight.grad)>0
    snapshot=copy.deepcopy(model.state_dict())
    restored=NativeM5BarDirectionModel(model.backbone,input_binding=binding,train_prior=prior)
    restored.load_state_dict(snapshot,strict=True)
    assert torch.equal(restored.close_heads['M5'].weight,model.close_heads['M5'].weight)
    assert torch.count_nonzero(restored.close_heads['M5'].weight)>0
    forged=dict(snapshot);forged['native_bar_train_prior_sha256']=torch.zeros(32,dtype=torch.uint8)
    before=restored.close_heads['M5'].weight.detach().clone()
    with pytest.raises(ValueError,match='NATIVE_CHECKPOINT_IDENTITY'):restored.load_state_dict(forged,strict=True)
    assert torch.equal(before,restored.close_heads['M5'].weight)
    with pytest.raises(ValueError,match='NATIVE_FRESH_PRIOR_HEAD_STATE'):restored.require_prior_initialization_state()
    different=copy.deepcopy(truth);different['close_sign'][0]=1
    other=fit_native_bar_train_prior(truth=different,fold_upper_ns=kwargs['fold_upper_ns'],
        target_contract_sha256=kwargs['target_contract_sha256'])
    wrong=NativeM5BarDirectionModel(model.backbone,input_binding=binding,train_prior=other)
    before=wrong.close_heads['M5'].weight.detach().clone()
    with pytest.raises(ValueError,match='NATIVE_CHECKPOINT_IDENTITY'):wrong.load_state_dict(snapshot,strict=True)
    assert torch.equal(before,wrong.close_heads['M5'].weight)
    later=copy.deepcopy(truth)
    shift=normalization['fit_population']['cutoff_ns']-int(later['issue_ns'][0])
    later['issue_ns']+=shift;later['validity_ns']+=shift
    future=fit_native_bar_train_prior(truth=later,
        fold_upper_ns=int(later['validity_ns'].max())+300_000_000_000,
        target_contract_sha256=kwargs['target_contract_sha256'])
    earlier=copy.deepcopy(truth);earlier['issue_ns']-=300_000_000_000;earlier['validity_ns']-=300_000_000_000
    outside=fit_native_bar_train_prior(truth=earlier,fold_upper_ns=kwargs['fold_upper_ns'],
        target_contract_sha256=kwargs['target_contract_sha256'])
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn,'Linear',lambda *args,**keys:pytest.fail('constructed head before prior-fold validation'))
        for invalid in (future,outside):
            with pytest.raises(ValueError,match='NATIVE_PRIOR_TRAIN_POPULATION'):
                NativeM5BarDirectionModel(model.backbone,input_binding=binding,train_prior=invalid)


def test_native_random_default_retains_linear_initialization_rng_contract_and_state(native_gc_single_thread):
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    normalization,_,_,_,_=_native_ta_prior_packet();binding=_native_model_binding(normalization)
    architecture={'dropout':.05,'multi_tf_num_layers':2,'multi_tf_scale':.5,
        'specialist_num_layers':1,'specialist_fusion_scale':.25,'cross_family_fusion_scale':.25}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        parent=owner.build_native_bar_direction_model(input_normalization=normalization,input_binding=binding,
            architecture=architecture,device=torch.device('cpu'))
        torch.manual_seed(53)
        model=owner.NativeM5BarDirectionModel(parent.backbone,input_binding=binding)
        after=torch.get_rng_state().clone()
        torch.manual_seed(53)
        expected=[torch.nn.Linear(parent.backbone.cfg.d_model,3,dtype=torch.float32)for _ in range(10)]
        assert torch.equal(after,torch.get_rng_state())
    for head,reference in zip([*model.open_heads.values(),*model.close_heads.values()],expected):
        assert torch.equal(head.weight,reference.weight)and torch.equal(head.bias,reference.bias)
    assert model.research_objective['schema_version']=='gx1_native_m5_bar_direction_model_v1'
    assert model.research_objective['initialization']=='fresh_random_no_checkpoint'
    assert owner.native_bar_model_contract(binding)==owner.native_bar_model_contract(binding,train_prior=None)
    assert not any(key in model.state_dict()for key in ('native_bar_initialization_sha256','native_bar_train_prior_sha256'))


def test_native_ta_m5_primary_policy_preserves_forward_heads_and_strict_state(monkeypatch,native_gc_single_thread):
    import copy
    import numpy as np
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss,native_bar_train_prior_logits
    normalization,raw,cats,truth,prior=_native_ta_prior_packet();binding=_native_model_binding(normalization)
    architecture={'dropout':.05,'multi_tf_num_layers':2,'multi_tf_scale':.5,
        'specialist_num_layers':1,'specialist_fusion_scale':.25,'cross_family_fusion_scale':.25}
    policy=owner.M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY
    models=[]
    with torch.random.fork_rng(devices=[]):
        for choice in (None,policy):
            torch.manual_seed(61)
            models.append(owner.build_native_bar_direction_model(input_normalization=normalization,
                input_binding=binding,architecture=architecture,device=torch.device('cpu'),
                train_prior=prior,backbone_gradient_policy=choice).eval())
            if choice is None:rng=torch.get_rng_state().clone()
            else:assert torch.equal(rng,torch.get_rng_state())
    default,routed=models
    assert all(torch.equal(a,b) for a,b in zip(default.parameters(),routed.parameters()))
    assert all(p.requires_grad for model in models for p in model.parameters())
    excluded={'native_bar_objective_sha256','native_bar_backbone_gradient_policy_sha256'}
    assert {k for k in default.state_dict()}=={k for k in routed.state_dict() if k not in excluded}|{'native_bar_objective_sha256'}
    assert all(torch.equal(value,routed.state_dict()[key]) for key,value in default.state_dict().items() if key not in excluded)
    assert 'backbone_gradient_policy' not in default.research_objective
    assert owner.native_bar_model_contract(binding,train_prior=prior)==owner.native_bar_model_contract(
        binding,train_prior=prior,backbone_gradient_policy=None)
    descriptor=routed.research_objective['backbone_gradient_policy']
    assert descriptor['attached_heads']==['close.M5'] and len(descriptor['detached_heads'])==9
    def synthetic_encode(self,**values):
        assert values.pop('research_representation_only')is True
        assert values.pop('research_timeframe_representations')is True
        source=values['seq_x'][:,-1,:self.cfg.d_model]
        hidden=self.entry_q_joint_in(torch.cat((source,)*4,dim=1))
        return {'entry_q_timeframe_hidden':hidden[:,None].expand(-1,5,-1)}
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',synthetic_encode)
    sequence=np.stack((raw['signal'],raw['signal'][::-1]))
    values={'seq_x':torch.from_numpy(sequence),'snap_x':torch.from_numpy(sequence[:,-1]),
        'ctx_cont':torch.from_numpy(np.stack((raw['ctx_cont'][-1],raw['ctx_cont'][0]))),
        'ctx_cat':torch.from_numpy(np.stack((cats[-1],cats[0])))}
    values.update({'seq_'+tf:torch.from_numpy(np.stack((raw['mtf_'+tf],raw['mtf_'+tf][::-1])))
        for tf in ('m5','m15','h1','h4','d1')})
    kwargs={'truth':{key:value[:2].copy() for key,value in truth.items()},
        'fold_upper_ns':normalization['fit_population']['cutoff_ns'],
        'target_contract_sha256':prior['target_contract_sha256']}
    expected=native_bar_train_prior_logits(prior,rows=2)
    for model in models:
        output=model(values)
        assert all(torch.equal(output[key],torch.from_numpy(expected[key])) for key in output)
        loss=native_bar_direction_loss(model_output=output,**kwargs)['loss']
        shared,head=torch.autograd.grad(loss,(model.backbone.entry_q_joint_in.weight,model.close_heads['M5'].weight))
        assert torch.count_nonzero(shared)==0 and torch.count_nonzero(head)>0
    # Exercise routing after the zero-head initialization, when hidden gradients are real.
    with torch.no_grad():
        for model in models:
            for index,head in enumerate([*model.open_heads.values(),*model.close_heads.values()]):
                source=torch.arange(head.weight.numel(),dtype=torch.float32).reshape_as(head.weight)
                head.weight.copy_(torch.sin(source*.13+index)*.02)
    outputs=[model(values) for model in models]
    assert all(torch.equal(outputs[0][key],outputs[1][key]) for key in outputs[0])
    gradients=[]
    for model,output in zip(models,outputs):
        result=native_bar_direction_loss(model_output=output,**kwargs)
        n=result['counts']['open'].sum()+result['counts']['close'].sum()
        primary=result['loss_sums']['close'][0]/n
        auxiliary=(result['loss_sums']['open'].sum()+result['loss_sums']['close'][1:].sum())/n
        shared=model.backbone.entry_q_joint_in.weight
        primary_heads=list(model.close_heads['M5'].parameters())
        auxiliary_heads=[p for tf in owner.TIMEFRAMES for p in model.open_heads[tf].parameters()]
        auxiliary_heads += [p for tf in owner.TIMEFRAMES if tf!='M5' for p in model.close_heads[tf].parameters()]
        primary_grad=torch.autograd.grad(primary,[shared,*primary_heads],retain_graph=True)
        auxiliary_grad=torch.autograd.grad(auxiliary,[shared,*auxiliary_heads],allow_unused=True,retain_graph=True)
        total_shared=torch.autograd.grad(result['loss'],shared)[0]
        gradients.append((primary_grad,auxiliary_grad,total_shared))
    left,right=gradients
    assert all(torch.equal(a,b) for a,b in zip(left[0],right[0]))
    assert all(torch.count_nonzero(value)>0 for value in right[0])
    assert left[1][0]is not None and torch.count_nonzero(left[1][0])>0
    # The close-logit stack can retain a zero derivative through its M5 cell.
    assert right[1][0]is None or torch.count_nonzero(right[1][0])==0
    assert all(torch.equal(a,b) for a,b in zip(left[1][1:],right[1][1:]))
    assert any(torch.count_nonzero(value)>0 for value in right[1][1:])
    assert torch.equal(right[2],right[0][0])
    assert all(p.grad is None for model in models for p in model.parameters())
    snapshot=copy.deepcopy(routed.state_dict())
    restored=owner.NativeM5BarDirectionModel(copy.deepcopy(routed.backbone),input_binding=binding,
        train_prior=prior,backbone_gradient_policy=policy).eval()
    restored.load_state_dict(snapshot,strict=True)
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in snapshot.items())
    assert all(torch.equal(restored(values)[key],outputs[1][key]) for key in outputs[1])
    with pytest.raises(ValueError,match='NATIVE_FRESH_PRIOR_HEAD_STATE'):
        restored.require_prior_initialization_state()
    for recipient,state in ((default,snapshot),(routed,default.state_dict()),
        (routed,{**snapshot,'native_bar_backbone_gradient_policy_sha256':torch.zeros(32,dtype=torch.uint8)})):
        before=copy.deepcopy(recipient.state_dict())
        with pytest.raises(ValueError,match='NATIVE_CHECKPOINT_IDENTITY'):
            recipient.load_state_dict(state,strict=True)
        assert all(torch.equal(value,recipient.state_dict()[key]) for key,value in before.items())
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn,'Linear',lambda *args,**keys:pytest.fail('head created before policy validation'))
        with pytest.raises(ValueError,match='NATIVE_BACKBONE_GRADIENT_POLICY_TA_PRIOR_REQUIRED'):
            owner.NativeM5BarDirectionModel(default.backbone,input_binding=binding,backbone_gradient_policy=policy)
        with pytest.raises(ValueError,match='NATIVE_BACKBONE_GRADIENT_POLICY'):
            owner.NativeM5BarDirectionModel(default.backbone,input_binding=binding,train_prior=prior,backbone_gradient_policy='unknown')


def test_native_noise_model_binds_new_prior_classes_and_rejects_equal_checkpoint_before_copy(monkeypatch,native_gc_single_thread):
    import copy
    import numpy as np
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    normalization,raw,cats,truth,raw_prior=_native_ta_prior_packet();binding=_native_model_binding(normalization)
    spec=noise_margin_dataset_contract(.3);shape=truth['target_row'].shape
    truth.update(m5_count=np.ones(shape,np.int64),source_gap=np.zeros(shape,np.bool_),
        noise_atr14_asof=np.ones(shape,np.float64),noise_margin_price=np.full(shape,.3,np.float64),
        noise_atr_asof_row=np.broadcast_to(np.arange(32,dtype=np.int64)[:,None],shape).copy(),
        noise_atr_asof_available_ns=np.broadcast_to(truth['issue_ns'][:,None],shape).copy(),
        noise_margin_contract=spec['noise_margin_contract'])
    for task in ('open','close'):
        truth['noise_'+task+'_delta']=truth[task+'_sign'].astype(np.float64)*.2
        truth['noise_'+task+'_sign']=np.zeros(shape,np.int8)
        truth['noise_'+task+'_known']=truth['eligible'].copy()
        truth['noise_'+task+'_eligible']=truth['eligible'].copy()if task=='close'else truth['eligible']&~truth['open_known_at_issue']
    prior=objective.fit_native_bar_train_prior(truth=truth,fold_upper_ns=normalization['fit_population']['cutoff_ns'],
        target_contract_sha256=spec['contract_sha256'],target_contract=spec)
    architecture={'dropout':.05,'multi_tf_num_layers':2,'multi_tf_scale':.5,
        'specialist_num_layers':1,'specialist_fusion_scale':.25,'cross_family_fusion_scale':.25}
    policy=owner.M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(67)
        model=owner.build_native_bar_direction_model(input_normalization=normalization,input_binding=binding,
            architecture=architecture,device=torch.device('cpu'),train_prior=prior,
            backbone_gradient_policy=policy,target_contract=spec).eval()
    metadata=model.research_objective
    assert metadata['schema_version']=='gx1_native_m5_bar_noise_margin_model_v1'
    assert metadata['class_names']==['DOWN','NEUTRAL_NET','UP']
    assert metadata['class_names_sha256']==prior['class_names_sha256']
    assert metadata['target_contract_sha256']==spec['contract_sha256']
    assert metadata['backbone_gradient_policy']['attached_heads']==['close.M5']
    assert bytes(model.native_bar_target_contract_sha256.tolist()).hex()==spec['contract_sha256']
    assert all(p.requires_grad for p in model.parameters())
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    profile=compile_full_ta_entry_execution_profile()
    assert len(profile['local_specialists'])==8
    for family,indices in profile['local_signal_indices_by_specialist'].items():
        assert getattr(model.backbone,'specialist_idx_'+family).tolist()==list(indices)
    assert model.require_prior_initialization_state()['per_head_support']['open']['D1']['role'].startswith('UNFITTED_')
    def synthetic_encode(self,**values):
        assert values.pop('research_representation_only')is True
        assert values.pop('research_timeframe_representations')is True
        source=values['seq_x'][:,-1,:self.cfg.d_model]
        hidden=self.entry_q_joint_in(torch.cat((source,)*4,dim=1))
        return {'entry_q_timeframe_hidden':hidden[:,None].expand(-1,5,-1)}
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',synthetic_encode)
    sequence=np.stack((raw['signal'],raw['signal'][::-1]))
    values={'seq_x':torch.from_numpy(sequence),'snap_x':torch.from_numpy(sequence[:,-1]),
        'ctx_cont':torch.from_numpy(np.stack((raw['ctx_cont'][-1],raw['ctx_cont'][0]))),
        'ctx_cat':torch.from_numpy(np.stack((cats[-1],cats[0])))}
    values.update({'seq_'+tf:torch.from_numpy(np.stack((raw['mtf_'+tf],raw['mtf_'+tf][::-1])))
        for tf in ('m5','m15','h1','h4','d1')})
    expected=objective.native_bar_train_prior_logits(prior,rows=2,target_contract=spec)
    output=model(values)
    assert all(value.shape==(2,5,3)and torch.equal(value,torch.from_numpy(expected[key]))for key,value in output.items())
    with pytest.raises(ValueError,match='NATIVE_FEATURE_KEYS'):model({**values,'target_contract':spec})
    old=owner.NativeM5BarDirectionModel(copy.deepcopy(model.backbone),input_binding=binding,
        train_prior=raw_prior,backbone_gradient_policy=policy).eval()
    assert 'target_contract'not in old.research_objective
    assert 'native_bar_target_contract_sha256'not in old.state_dict()
    for recipient,state in ((model,old.state_dict()),(old,model.state_dict()),
        (model,{**model.state_dict(),'native_bar_target_contract_sha256':torch.zeros(32,dtype=torch.uint8)})):
        before=copy.deepcopy(recipient.state_dict())
        with pytest.raises(ValueError,match='NATIVE_CHECKPOINT_IDENTITY'):recipient.load_state_dict(state,strict=True)
        assert all(torch.equal(value,recipient.state_dict()[key])for key,value in before.items())
    with torch.no_grad():model.close_heads['M5'].weight.fill_(.01)
    snapshot=copy.deepcopy(model.state_dict())
    restored=owner.NativeM5BarDirectionModel(copy.deepcopy(model.backbone),input_binding=binding,
        train_prior=prior,backbone_gradient_policy=policy,target_contract=spec).eval()
    restored.load_state_dict(snapshot,strict=True)
    assert torch.equal(restored.close_heads['M5'].weight,model.close_heads['M5'].weight)
    assert torch.count_nonzero(restored.close_heads['M5'].weight)>0
    assert all(torch.equal(restored(values)[key],model(values)[key])for key in output)
    with pytest.raises(ValueError,match='NATIVE_FRESH_PRIOR_HEAD_STATE'):restored.require_prior_initialization_state()
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn,'Linear',lambda *a,**k:pytest.fail('head constructed before target/prior check'))
        for init_prior,init_target in ((raw_prior,spec),(prior,None)):
            with pytest.raises(ValueError,match='NATIVE_BAR_TRAIN_PRIOR_SCHEMA'):
                owner.NativeM5BarDirectionModel(model.backbone,input_binding=binding,
                    train_prior=init_prior,backbone_gradient_policy=policy,target_contract=init_target)



def _synthetic_native_binding_guard(monkeypatch):
    """Only exercise metadata guards and ten tiny Linear heads, no backbone."""
    from copy import deepcopy
    from types import SimpleNamespace
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    binding = {'schema_version': 'gx1_native_m5_bar_direction_inputs_v1',
        'execution_profile': {'name': 'synthetic', 'contract_sha256': '1' * 64},
        'source_axes_sha256': '2' * 64,
        'feature_provider_contract_sha256': '3' * 64,
        'input_normalization_sha256': '4' * 64}
    calls = {'binding': 0, 'normalization': 0, 'objective': 0}
    def validate(value):
        calls['binding'] += 1
        return deepcopy(value)
    def normalize():
        calls['normalization'] += 1
    def objective(value, **kwargs):
        calls['objective'] += 1
        assert value == binding
        return {'contract_sha256': '5' * 64}
    monkeypatch.setattr(owner, 'require_native_bar_input_binding', validate)
    monkeypatch.setattr(owner, '_native_bar_model_contract_from_binding', objective)
    backbone = SimpleNamespace(entry_execution_profile=deepcopy(binding['execution_profile']),
        _native_bar_direction_only=True,
        cfg=SimpleNamespace(dynamic_m5_query_minutes=None, d_model=1),
        _input_normalization_contract={'schema_version': 'gx1_native_m5_input_normalization_v2',
            'source_axes_sha256': binding['source_axes_sha256'],
            'feature_provider_contract_sha256': binding['feature_provider_contract_sha256']},
        input_norm_contract_sha256=torch.tensor([0x44] * 32, dtype=torch.uint8),
        require_input_normalization_state=normalize,
        entry_q_joint_in=SimpleNamespace(weight=torch.zeros(1)))
    model = SimpleNamespace(input_binding=deepcopy(binding), backbone=backbone,
        _train_prior=None, _backbone_gradient_policy=None, _target_contract=None,
        _gc_binding_matches=lambda value: True,
        native_bar_objective_sha256=torch.tensor([0x55] * 32, dtype=torch.uint8),
        open_heads=torch.nn.ModuleDict({tf: torch.nn.Linear(1, 3) for tf in owner.TIMEFRAMES}),
        close_heads=torch.nn.ModuleDict({tf: torch.nn.Linear(1, 3) for tf in owner.TIMEFRAMES}))
    return owner, model, calls


def test_native_bar_binding_reuses_only_one_local_validation(monkeypatch):
    owner, model, calls = _synthetic_native_binding_guard(monkeypatch)
    for _ in range(2):
        owner.NativeM5BarDirectionModel._require_binding(model)
    assert calls == {'binding': 2, 'normalization': 2, 'objective': 2}


def test_native_bar_binding_rejects_mutation_during_local_reuse(monkeypatch):
    owner, model, calls = _synthetic_native_binding_guard(monkeypatch)
    def mutate():
        model.input_binding['source_axes_sha256'] = '6' * 64
    model.backbone.require_input_normalization_state = mutate
    with pytest.raises(ValueError, match='NATIVE_INPUT_BINDING_MUTATED_DURING_VALIDATION'):
        owner.NativeM5BarDirectionModel._require_binding(model)
    assert calls['binding'] == 1 and calls['objective'] == 0


@pytest.mark.parametrize('field', ['normalization', 'objective', 'head'])
def test_native_bar_binding_rechecks_mutable_state_on_every_call(monkeypatch, field):
    owner, model, calls = _synthetic_native_binding_guard(monkeypatch)
    owner.NativeM5BarDirectionModel._require_binding(model)
    if field == 'normalization':
        model.backbone.input_norm_contract_sha256[0] = 0
        reason = 'NATIVE_BACKBONE_IDENTITY'
    elif field == 'objective':
        model.native_bar_objective_sha256[0] = 0
        reason = 'NATIVE_MODEL_BINDING'
    else:
        with torch.no_grad():
            model.open_heads['M5'].weight.fill_(float('nan'))
        reason = 'NATIVE_HEAD_BINDING'
    with pytest.raises(ValueError, match=reason):
        owner.NativeM5BarDirectionModel._require_binding(model)
    assert calls['binding'] == 2


def test_native_bar_public_contract_always_validates_input(monkeypatch):
    owner, model, calls = _synthetic_native_binding_guard(monkeypatch)
    for _ in range(2):
        assert owner.native_bar_model_contract(model.input_binding) == {'contract_sha256': '5' * 64}
    assert calls['binding'] == calls['objective'] == 2
    def reject(value):
        raise ValueError('synthetic_invalid_binding')
    monkeypatch.setattr(owner, 'require_native_bar_input_binding', reject)
    with pytest.raises(ValueError, match='synthetic_invalid_binding'):
        owner.native_bar_model_contract(model.input_binding)


def _future_readout_inputs():
    from tests.test_market_state_ta_integrated_objective_v1 import _future_state_inputs
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior
    kwargs = _future_state_inputs()
    kwargs.pop('model_output')
    prior = fit_native_future_state_train_prior(**kwargs)
    hidden = torch.arange(1, 61, dtype=torch.float32).reshape(3, 5, 4)
    return kwargs, prior, hidden


def _future_readout(kwargs, prior, d_model=4):
    from gx1.contracts.market_state_ta_integrated_model_v1 import NativeFutureStateReadout
    return NativeFutureStateReadout(d_model=d_model,
        target_contract=kwargs['target_contract'], train_prior=prior)


def test_native_future_state_readout_exact_prior_initialization_and_probability_loss_parity():
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    kwargs, prior, hidden = _future_readout_inputs()
    rng = torch.random.get_rng_state().clone()
    model = _future_readout(kwargs, prior)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert set(dict(model.named_children())) == {'activity_heads', 'side_heads'}
    assert len(list(model.parameters())) == 20
    reference = objective.native_future_state_train_prior_logits(prior, rows=3,
        target_contract=kwargs['target_contract'])
    output = model(hidden)
    assert set(output) == {'activity_logits', 'side_logits'}
    for key, logits in output.items():
        assert logits.dtype == torch.float32 and logits.shape == (3, 5)
        assert torch.equal(logits, reference[key])
    for heads in (model.activity_heads, model.side_heads):
        assert set(heads) == set(prior['timeframes'])
        assert all(torch.count_nonzero(head.weight) == 0 for head in heads.values())
    logp = objective.native_future_state_log_probabilities(model_output=output,
        target_contract=kwargs['target_contract'])
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3, 5), rtol=0, atol=1e-7)
    loss = objective.native_future_state_loss(model_output=output, **kwargs)
    baseline = objective.native_future_state_loss(model_output=reference, **kwargs)
    assert torch.equal(loss['loss'], baseline['loss'])
    scores = objective.native_future_state_score_sums(model_output=output, **kwargs)
    assert scores == objective.native_future_state_score_sums(model_output=reference, **kwargs)
    assert sum(cell['partial_log_loss_sum'] for cell in scores['per_tf'].values()) == pytest.approx(
        float(loss['loss']) * loss['count'], abs=1e-6)
    spec = model.readout_contract
    assert spec['production_training_admitted'] is False
    assert spec['feature_backbone_forward_qualified'] is False
    spec['d_model'] = 99
    assert model.readout_contract['d_model'] == 4


def test_native_future_state_readout_unknown_side_free_parameters_are_exactly_zero_gradient():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    kwargs['truth']['activity'][:] = True
    kwargs['truth']['side_known'][:] = False
    kwargs['truth']['side_up'][:] = False
    result = native_future_state_loss(model_output=model(hidden), **kwargs)
    result['loss'].backward()
    for head in model.side_heads.values():
        assert torch.equal(head.weight.grad, torch.zeros_like(head.weight))
        assert torch.equal(head.bias.grad, torch.zeros_like(head.bias))
    for head in model.activity_heads.values():
        assert torch.count_nonzero(head.weight.grad) == head.weight.numel()
        assert torch.count_nonzero(head.bias.grad) == head.bias.numel()


def test_native_future_state_readout_all_unknown_skips_and_all_parameter_gradients_are_zero():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    for key in ('activity_known', 'activity', 'side_known', 'side_up'):
        kwargs['truth'][key][:] = False
    result = native_future_state_loss(model_output=model(hidden), **kwargs)
    assert result['skip_update'] is True and float(result['loss']) == 0
    result['loss'].backward()
    assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in model.parameters())


@pytest.mark.parametrize('case', ['dtype', 'shape', 'empty', 'nonfinite', 'device', 'parameter_dtype',
                                  'parameter_nonfinite', 'live_identity'])
def test_native_future_state_readout_rejects_invalid_hidden_parameters_and_binding(case):
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    if case == 'dtype': hidden = hidden.double()
    elif case == 'shape': hidden = hidden[:, :, :3]
    elif case == 'empty': hidden = hidden[:0]
    elif case == 'nonfinite': hidden[0, 0, 0] = float('nan')
    elif case == 'device': hidden = hidden.to('meta')
    elif case == 'parameter_dtype': model.double()
    elif case == 'parameter_nonfinite':
        with torch.no_grad(): model.side_heads['D1'].bias.fill_(float('inf'))
    elif case == 'live_identity': model.future_state_target_sha256[0] ^= 1
    with pytest.raises(ValueError, match='FUTURE_READOUT'):
        model(hidden)


@pytest.mark.parametrize('case', ['target_identity', 'prior_identity', 'head_identity',
                                  'shape', 'dtype', 'nonfinite', 'missing', 'nonstrict'])
def test_native_future_state_readout_checks_entire_state_before_any_parameter_copy(case):
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() for key, value in before.items()}
    state['activity_heads.M5.weight'].fill_(42)
    if case == 'target_identity': state['future_state_target_sha256'][0] ^= 1
    elif case == 'prior_identity': state['future_state_prior_sha256'][0] ^= 1
    elif case == 'head_identity': state['future_state_readout_sha256'][0] ^= 1
    elif case == 'shape': state['side_heads.D1.weight'] = torch.zeros(1, 3)
    elif case == 'dtype': state['side_heads.D1.weight'] = state['side_heads.D1.weight'].double()
    elif case == 'nonfinite': state['side_heads.D1.bias'].fill_(float('nan'))
    elif case == 'missing': del state['side_heads.D1.bias']
    with pytest.raises(ValueError, match='FUTURE_READOUT'):
        model.load_state_dict(state, strict=(case != 'nonstrict'))
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_native_future_state_readout_roundtrips_and_rejects_another_valid_frozen_prior():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    with torch.no_grad(): model.side_heads['M15'].weight.fill_(.25)
    restored = _future_readout(kwargs, prior)
    restored.load_state_dict(model.state_dict())
    for key, value in model(hidden).items():
        assert torch.equal(restored(hidden)[key], value)
    kwargs['truth']['side_up'][2, 0] = False
    changed_prior = fit_native_future_state_train_prior(**kwargs)
    another = _future_readout(kwargs, changed_prior)
    before = {key: value.clone() for key, value in another.state_dict().items()}
    with pytest.raises(ValueError, match='STATE_IDENTITY'):
        another.load_state_dict(model.state_dict())
    assert all(torch.equal(value, before[key]) for key, value in another.state_dict().items())


@pytest.mark.parametrize('width', [0, -1, True, 4.0])
def test_native_future_state_readout_requires_explicit_positive_integer_width(width):
    kwargs, prior, hidden = _future_readout_inputs()
    with pytest.raises(ValueError, match='FUTURE_READOUT_WIDTH'):
        _future_readout(kwargs, prior, d_model=width)


def test_native_future_state_readout_requires_validated_frozen_train_prior():
    kwargs, prior, hidden = _future_readout_inputs()
    prior['fit_population']['scope'] = 'CONTROL'
    with pytest.raises(ValueError, match='PRIOR_POPULATION'):
        _future_readout(kwargs, prior)


def test_native_future_state_readout_nested_restore_checks_identity_before_child_parameter_copy():
    kwargs, prior, hidden = _future_readout_inputs()
    model = _future_readout(kwargs, prior)
    parent = torch.nn.Module()
    parent.add_module('head', model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() for key, value in parent.state_dict().items()}
    state['head.activity_heads.M5.weight'].fill_(42)
    state['head.future_state_target_sha256'][0] ^= 1
    with pytest.raises(ValueError, match='STATE_IDENTITY'):
        parent.load_state_dict(state)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def _future_native_full_packet():
    import copy
    import hashlib
    import numpy as np
    from gx1.contracts import entry_model_native_input_normalization_v1 as normalizer
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior
    from tests.test_entry_model_native_input_normalization import native_m5_normalization_fixture
    norm, raw, cats = native_m5_normalization_fixture()
    spec = fully_future_state_contract()
    first = norm['fit_population']['first_issue_ns']
    rows = norm['fit_population']['issue_rows']
    issues = first + np.arange(rows, dtype=np.int64) * 300_000_000_000
    duration = np.asarray([spec['tf_duration_ns'][tf] for tf in spec['timeframes']], np.int64)
    origin = np.asarray([spec['tf_origin_ns'][tf] for tf in spec['timeframes']], np.int64)
    starts = ((issues[:, None] - origin + duration - 1) // duration) * duration + origin
    shape = (rows, 5)
    complete = np.ones(shape, np.bool_)
    active = np.broadcast_to((np.arange(rows) % 3 != 0)[:, None], shape).copy()
    up = active & np.broadcast_to((np.arange(rows) % 2 == 0)[:, None], shape)
    truth = {'issue_ns': issues, 'target_start_ns': starts, 'validity_ns': starts + duration,
        'target_present': complete.copy(), 'mature': complete.copy(), 'nominal_complete': complete.copy(),
        'state_window_complete': complete.copy(), 'partial_range_active': np.zeros(shape, np.bool_),
        'activity_known': complete.copy(), 'activity': active, 'side_known': active.copy(), 'side_up': up}
    population = copy.deepcopy(norm['fit_population'])
    population['selected_issue_ns_sha256'] = hashlib.sha256(issues.tobytes()).hexdigest()
    norm = normalizer.build_native_m5_input_normalization_contract(
        **{key: norm[key] for key in ('native_m5_manifest', 'feature_provider_contract_sha256',
            'source_axes_sha256', 'surfaces', 'ctx_cat', 'temporal_aliases')}, fit_population=population)
    prior = fit_native_future_state_train_prior(truth=truth,
        fold_upper_ns=population['cutoff_ns'], target_contract=spec)
    architecture = {'dropout': .05, 'multi_tf_num_layers': 2, 'multi_tf_scale': .5,
        'specialist_num_layers': 1, 'specialist_fusion_scale': .25, 'cross_family_fusion_scale': .25}
    args = dict(input_normalization=norm, input_binding=_native_model_binding(norm),
        architecture=architecture, device=torch.device('cpu'), train_prior=prior, target_contract=spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)  # Existing native factory fixture seed.
        model = owner.build_native_future_state_model(**args).eval()
    geometry = owner.native_bar_backbone_geometry()
    sequence = raw['signal'][np.arange(geometry['local']) % len(raw['signal'])]
    features = {'seq_x': torch.from_numpy(sequence[None]), 'snap_x': torch.from_numpy(sequence[-1:]),
        'ctx_cont': torch.from_numpy(raw['ctx_cont'][-1:]), 'ctx_cat': torch.from_numpy(cats[-1:])}
    features.update({'seq_' + tf.lower(): torch.from_numpy(raw['mtf_' + tf.lower()][
        np.arange(geometry[tf]) % len(raw['mtf_' + tf.lower()])][None]) for tf in spec['timeframes']})
    return model, args, features, truth


@pytest.fixture(scope='module')
def future_native_full_packet():
    return _future_native_full_packet()


def test_native_future_state_model_real_factory_one_encode_exact_prior_no_raw_heads(
        future_native_full_packet, native_gc_single_thread):
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    model, args, features, truth = future_native_full_packet
    assert set(dict(model.named_children())) == {'backbone', 'readout'}
    assert model.backbone._native_bar_direction_only is True
    assert not hasattr(model, 'open_heads') and not hasattr(model, 'close_heads')
    assert not any('open_heads.' in name or 'close_heads.' in name for name in model.state_dict())
    calls = []
    def capture(module, inputs, output):
        hidden = output['entry_q_timeframe_hidden']
        calls.append(hidden.detach().clone())
    hook = model.backbone.register_forward_hook(capture)
    try:
        with torch.inference_mode(): output = model(**features)
    finally:
        hook.remove()
    assert len(calls) == 1
    assert calls[0].dtype == torch.float32 and calls[0].shape == (1, 5, model.readout.d_model)
    assert bool(torch.isfinite(calls[0]).all())
    reference = objective.native_future_state_train_prior_logits(args['train_prior'], rows=1,
        target_contract=args['target_contract'])
    assert all(torch.equal(output[key], value) for key, value in reference.items())
    one = {key: value[:1] for key, value in truth.items()}
    kwargs = dict(truth=one, fold_upper_ns=args['train_prior']['fit_population']['cutoff_ns'],
                  target_contract=args['target_contract'])
    assert objective.native_future_state_score_sums(model_output=output, **kwargs) == \
           objective.native_future_state_score_sums(model_output=reference, **kwargs)
    assert torch.equal(objective.native_future_state_loss(model_output=output, **kwargs)['loss'],
                       objective.native_future_state_loss(model_output=reference, **kwargs)['loss'])
    assert model.research_objective['production_training_admitted'] is False


def test_native_future_state_model_rejects_target_metadata_and_nontensors_before_encode(
        future_native_full_packet, monkeypatch):
    from gx1.contracts.market_state_ta_integrated_model_v1 import EntryV10CtxHybridTransformer
    model, args, features, truth = future_native_full_packet
    def forbidden(*args, **kwargs): raise AssertionError('encoder reached')
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', forbidden)
    with pytest.raises(ValueError, match='FEATURE_KEYS'):
        model(**{**features, 'validity_ns': torch.ones(1, 5)})
    with pytest.raises(ValueError, match='FEATURE_TENSORS'):
        model(**{**features, 'seq_x': None})


@pytest.mark.parametrize('name', ['future_state_model_sha256', 'backbone.input_norm_contract_sha256',
    'backbone.input_norm_signal_center', 'readout.future_state_target_sha256',
    'readout.future_state_objective_sha256', 'readout.future_state_prior_sha256',
    'readout.future_state_readout_sha256'])
def test_native_future_state_model_rejects_identity_before_backbone_parameter_copy(
        future_native_full_packet, name):
    model, args, features, truth = future_native_full_packet
    before = model.backbone.entry_q_joint_in.weight.detach().clone()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state['backbone.entry_q_joint_in.weight'].fill_(42)
    value = state[name]
    value.reshape(-1)[0] += 1
    with pytest.raises(ValueError, match='STATE_IDENTITY'):
        model.load_state_dict(state)
    assert torch.equal(model.backbone.entry_q_joint_in.weight, before)


def test_native_future_state_model_roundtrip_and_strict_header_guards(future_native_full_packet):
    model, args, features, truth = future_native_full_packet
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert 'backbone.entry_signal_projection_idx' in dict(model.named_buffers())
    assert 'backbone.entry_signal_projection_idx' not in state
    assert not model.load_state_dict(state).missing_keys
    with pytest.raises(ValueError, match='STRICT_STATE'):
        model.load_state_dict(state, strict=False)
    state['backbone.entry_q_joint_in.weight'] = state['backbone.entry_q_joint_in.weight'].double()
    with pytest.raises(ValueError, match='STATE_TENSOR'):
        model.load_state_dict(state)


def test_native_future_state_model_rejects_changed_prior_normalization_population(future_native_full_packet):
    import copy
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    model, args, features, truth = future_native_full_packet
    other = copy.deepcopy(args['train_prior'])
    other['fit_population']['issue_ns_sha256'] = '9' * 64
    other = objective._native_prior_seal({key: value for key, value in other.items() if key != 'contract_sha256'})
    with pytest.raises(ValueError, match='PRIOR_NORMALIZATION_POPULATION'):
        owner.NativeFutureStateModel(model.backbone, input_binding=args['input_binding'],
            architecture=args['architecture'], train_prior=other, target_contract=args['target_contract'])


def test_native_future_state_factory_backbone_only_rejects_raw_arguments(future_native_full_packet):
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    model, args, features, truth = future_native_full_packet
    base = {key: args[key] for key in ('input_normalization', 'input_binding', 'architecture', 'device')}
    for extra in ({'train_prior': args['train_prior']}, {'target_contract': args['target_contract']},
                  {'backbone_gradient_policy': 'anything'}):
        with pytest.raises(ValueError, match='BACKBONE_ONLY_RAW_ARGUMENTS'):
            owner.build_native_bar_direction_model(**base, return_backbone_only=True, **extra)
    with pytest.raises(ValueError, match='BACKBONE_ONLY_FLAG'):
        owner.build_native_bar_direction_model(**base, return_backbone_only=1)


def test_native_future_state_factory_legacy_default_seeded_state_and_rng_parity(
        future_native_full_packet, native_gc_single_thread):
    import ast
    import inspect
    from gx1.contracts import market_state_ta_integrated_model_v1 as owner
    model, args, features, truth = future_native_full_packet
    node = ast.parse(inspect.getsource(owner.build_native_bar_direction_model)).body[0]
    assert node.args.kwonlyargs[-1].arg == 'return_backbone_only'
    node.args.kwonlyargs.pop(); node.args.kw_defaults.pop()
    node.body = [n for n in node.body if not (
        isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'return_backbone_only')
        and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith('NATIVE_BACKBONE_ONLY_') for a in n.value.args))]
    namespace = dict(vars(owner))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<legacy-native-factory-body>', 'exec'), namespace)
    base = {key: args[key] for key in ('input_normalization', 'input_binding', 'architecture', 'device')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        legacy = namespace['build_native_bar_direction_model'](**base)
        legacy_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(47)
        current = owner.build_native_bar_direction_model(**base)
        assert torch.equal(legacy_rng, torch.random.get_rng_state())
    assert all(torch.equal(value, legacy.state_dict()[key]) for key, value in current.state_dict().items())
