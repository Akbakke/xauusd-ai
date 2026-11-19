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
