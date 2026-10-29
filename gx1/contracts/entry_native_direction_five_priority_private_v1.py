"""Exact metadata checks for the M5 five-priority private qualification.

This validator grants no training authority. An outer, independently bound
one-shot plan, guarded launch and physical reboot are still required.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
from gx1.contracts import local_random_access_campaign_v2 as campaign

STAGE = 'dynamic_m5_direction_five_priority_private_20261029_001'
DESIGN = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_M5_FIVE_PRIORITY_DESIGN_20261029.json',
          'sha256':'b4a06aaa2d723941c1b64a58fc5ec34ec6456642f71296a7bbac6788df98581b'}
BASELINE_INITIAL_REPORT = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_PRICE_ONLY_ONEPASS_FIT_20261029_001/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json',
                           'sha256':'a468c941d3d6246581e94c6ae4d728fc2689d51eee4d29e28f6302ac460d7005'}
BASELINE_RECIPE = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_M5_PRICE_ONLY_ONEPASS_FIT_RECIPE_20261029.json',
                   'sha256':'14ba328acbaff66dadd1e6e6b40fda628badf0767ff414bde49fefa113c1eb71'}
SOURCE_REVIEW_PATH = '/home/andre2/src/GX1_CURRENT/docs/DIRECTION_M5_FIVE_PRIORITY_SOURCE_REVIEW_20261029.json'
EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_five_priority_v1.py',
                      'gx1/contracts/entry_native_direction_m5_fullfit_v1.py',
                      'gx1/contracts/entry_native_direction_five_priority_private_v1.py')


def _require(value, why):
    if not value: raise campaign.RandomAccessCampaignError('M5_FIVE_PRIORITY_PRIVATE_' + why)


def _read(ref, label):
    checked=campaign.require_binding(ref,label=label,verify_file=True)
    return campaign.read_bound_json(Path(checked['path']),checked['sha256'])

BASELINE_SOURCE_COMMIT = 'ac0a17671ebb0d66ea5e90da3f3c501e2725b60e'
BASELINE_FIT_COMMIT = '6b199953e0f178c073a064d60dd3b978b5f77b1c'
_ALLOWED_SOURCE_CHANGES = {
    'gx1/contracts/entry_model_native_train_recipe_v1.py': {'functions':('native_direction_research_recipe','require_native_direction_research_recipe')},
    'gx1/contracts/entry_native_direction_dense_v1.py': {'class_name':'DenseNativeEntryDirectionResearch', 'methods':('__init__','research_objective','_require_binding'), 'new_methods':('five_minute_priority',)},
    'gx1/contracts/entry_native_direction_feature_view_v1.py': {'functions':('_require_view','bind_native_direction_feature_view','build_native_direction_research_model'), 'new_assignments':('FIVE_PRIORITY_SOURCE_REF_KEYS',)},
    'gx1/contracts/entry_native_direction_research_campaign_v1.py': {'functions':('require_research_launch_recipe',)},
    'gx1/scripts/run_entry_native_direction_dense_qualification_v1.py': {'functions':('run_dense_qualification',), 'new_functions':('require_five_priority_initial_parity',)},
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': {'functions':('_prepare','_train_batch')},
    'gx1/scripts/run_entry_native_direction_research_window_v1.py': {'functions':('_step',)},
}


def _unchanged_ast(source, specification, *, original):
    """Remove only explicitly declared definitions, requiring each to exist."""
    import ast
    import copy
    tree=ast.parse(source)
    functions=set(specification.get('functions',())); added_functions=set(specification.get('new_functions',()))
    assignments=set(specification.get('new_assignments',()))
    seen_functions=set();seen_added=set();seen_assignments=set();seen_class=False;output=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in functions:
            seen_functions.add(node.name);continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in added_functions:
            seen_added.add(node.name);continue
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in assignments:
            seen_assignments.add(node.targets[0].id);continue
        if isinstance(node,ast.ClassDef) and node.name==specification.get('class_name'):
            _require(not seen_class,'DUPLICATE_CLASS');seen_class=True
            clone=copy.deepcopy(node);methods=set(specification.get('methods',()));added_methods=set(specification.get('new_methods',()));seen_methods=set();seen_new_methods=set();body=[]
            for member in clone.body:
                if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name in methods:
                    seen_methods.add(member.name);continue
                if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name in added_methods:
                    seen_new_methods.add(member.name);continue
                body.append(member)
            _require(seen_methods==methods and seen_new_methods==(set() if original else added_methods),'CLASS_METHOD_SET')
            clone.body=body;node=clone
        output.append(ast.dump(node,include_attributes=False))
    _require(seen_functions==functions and seen_added==(set() if original else added_functions)
        and seen_assignments==(set() if original else assignments)
        and seen_class==bool(specification.get('class_name')),'MODULE_NODE_SET')
    return output


def require_source_transition(baseline, owners, *, source_repo, source_commit):
    """Check all old fit owners and limit source edits to reviewed definitions."""
    import hashlib
    import subprocess
    old=baseline.get('source_owners')
    _require(isinstance(old,Mapping) and set(old).issubset(owners),'BASELINE_SOURCE_SET')
    for before,after in ((BASELINE_FIT_COMMIT,BASELINE_SOURCE_COMMIT),(BASELINE_SOURCE_COMMIT,source_commit)):
        result=subprocess.run(['git','merge-base','--is-ancestor',before,after],cwd=source_repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        _require(result.returncode==0,'SOURCE_ANCESTRY')
    seen=set()
    for name,original_ref in old.items():
        current_ref=owners[name]
        if name.startswith('.venv/'):
            _require(current_ref==original_ref,'PACKAGE_OWNER:'+name);continue
        path=source_repo/name
        before=subprocess.check_output(['git','show',BASELINE_SOURCE_COMMIT+':'+name],cwd=source_repo)
        # The existing capped runner changed after the baseline fit for a
        # separate producer. Confirm the historical fit blob and preserve the
        # complete runner used before this priority work, including all guards.
        if name=='scripts/gx1_capped_run.sh':
            historical=subprocess.check_output(['git','show',BASELINE_FIT_COMMIT+':'+name],cwd=source_repo)
            _require(hashlib.sha256(historical).hexdigest()==original_ref['sha256']
                and hashlib.sha256(before).hexdigest()==current_ref['sha256'],'RUNNER_UNCHANGED')
        else:
            _require(hashlib.sha256(before).hexdigest()==original_ref['sha256'],'BASELINE_OWNER:'+name)
        current=path.read_bytes()
        _require(hashlib.sha256(current).hexdigest()==current_ref['sha256'],'CURRENT_OWNER:'+name)
        if before!=current:
            _require(name in _ALLOWED_SOURCE_CHANGES,'UNREVIEWED_SOURCE:'+name)
            spec=_ALLOWED_SOURCE_CHANGES[name]
            _require(_unchanged_ast(before,spec,original=True)==_unchanged_ast(current,spec,original=False),'UNREVIEWED_AST:'+name)
            seen.add(name)
    _require(seen==set(_ALLOWED_SOURCE_CHANGES),'EXPECTED_CHANGED_OWNERS')
    return {'baseline_owner_count':len(old),'reviewed_changed_owners':sorted(seen),'dense_forward_ast_unchanged':True}


def require_private_recipe(recipe, owners, *, source_repo, source_commit):
    """Bind the one private M5 hypothesis, source review, and old initial state."""
    from gx1.contracts.entry_native_direction_m5_fullfit_v1 import INITIAL
    _require(isinstance(recipe,Mapping) and isinstance(owners,Mapping)
        and recipe.get('admission_stage')==STAGE and recipe.get('candidate_mode')=='price_only'
        and 'candidate_m1_refs' not in recipe and recipe.get('priority_design')==DESIGN
        and recipe.get('priority_baseline_initial_report')==BASELINE_INITIAL_REPORT
        and recipe.get('numerical_recipe',{}).get('five_minute_priority') is True,
        'RECIPE_SCOPE')
    _require(source_repo==Path('/home/andre2/src/GX1_CURRENT')
        and isinstance(source_commit,str) and campaign._COMMIT.fullmatch(source_commit) is not None,
        'SOURCE_REPO')
    design=_read(DESIGN,'five priority design')
    initial=_read(BASELINE_INITIAL_REPORT,'five priority baseline initial report')
    baseline=_read(BASELINE_RECIPE,'five priority baseline fit recipe')
    ref=recipe.get('priority_source_review')
    _require(isinstance(ref,Mapping) and ref.get('path')==SOURCE_REVIEW_PATH,'SOURCE_REVIEW_PATH')
    review=_read(ref,'five priority source review')
    _require(design.get('schema_version')=='gx1_m5_five_priority_candidate_design_v1'
        and design.get('status')=='PREDECLARED_NO_EXECUTION_AUTHORITY'
        and design.get('candidate',{}).get('five_minute_weight')==10
        and design.get('candidate',{}).get('other_horizon_weight')==1
        and design.get('candidate',{}).get('horizon_minutes')==list(range(1,61))
        and design.get('baseline',{}).get('recipe')==BASELINE_RECIPE
        and design.get('test_data_used') is False,'DESIGN')
    _require(initial.get('role')=='INITIAL_TRAIN' and initial.get('test_data_used') is False
        and initial.get('fold')==recipe.get('fold')==baseline.get('fold')
        and initial.get('predictions',{}).get('sha256')==INITIAL['sha256']
        and initial.get('preservation',{}).get('model')
        and initial.get('preservation',{}).get('all_buffers'), 'INITIAL_REPORT')
    _require(review.get('schema_version')=='gx1_m5_five_priority_source_review_v1'
        and review.get('status')=='SOURCE_REVIEWED_NO_EXECUTION'
        and review.get('baseline_recipe')==BASELINE_RECIPE
        and review.get('candidate_design')==DESIGN
        and review.get('baseline_source_commit')==BASELINE_SOURCE_COMMIT
        and review.get('candidate_source_owners')==dict(owners)
        and review.get('declared_source_changes_reviewed') is True
        and review.get('dense_forward_ast_unchanged') is True
        and review.get('all_unmodified_baseline_owners_verified') is True
        and review.get('test_data_used') is False,'SOURCE_REVIEW')
    _require(all(name in owners and owners[name].get('path')==str(source_repo/name)
        for name in EXTRA_SOURCE_PATHS),'EXTRA_OWNERS')
    transition=require_source_transition(baseline,owners,source_repo=source_repo,source_commit=source_commit)
    _require(review.get('transition_summary')==transition,'SOURCE_TRANSITION_SUMMARY')
    return {'design':design,'baseline_initial_report':initial,'source_review':review}
