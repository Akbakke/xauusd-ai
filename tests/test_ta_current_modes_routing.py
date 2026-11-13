"""Exact one-shot current mode command boundaries; no model or market data."""
import hashlib
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("suffix,constant", [("current_modes","TA_CURRENT_MODES_PRODUCER_MODULE"),("context_drift","TA_CONTEXT_DRIFT_PRODUCER_MODULE"),("head_input","TA_HEAD_INPUT_PRODUCER_MODULE")])
def test_current_mode_cap_accepts_only_bound_command_and_optional_preflight(tmp_path,suffix,constant):
    repo=Path(__file__).resolve().parents[1]
    source=(repo/'scripts/gx1_capped_run.sh').read_text()
    start=source.index('matches_bound_market_state_ta_'+suffix+'() {')
    function=source[start:source.index('\n}\n',start)+3]
    plan=tmp_path/'PLAN.json';plan.write_text('{}\n')
    digest=hashlib.sha256(plan.read_bytes()).hexdigest()
    python=str(repo/'.venv/bin/python')
    module='gx1.audit.observe_market_state_ta_'+suffix+'_v1'
    flag='--market-state-ta-'+suffix.replace('_','-')
    args=[python,'-m',module,flag+'-plan',str(plan),
          flag+'-plan-sha256',digest,'--device','cuda','--out-dir',str(tmp_path)]
    script='CANONICAL_TRAINER_PYTHON="$1"\n'+constant+'="$2"\nshift 2\n'+function+'\nmatches_bound_market_state_ta_'+suffix+' "$@"\n'
    def accepted(values):
        return subprocess.run(['bash','-c',script,'bash',python,module,str(plan),str(tmp_path),*values],capture_output=True).returncode==0
    assert accepted(args) and accepted(args+['--preflight'])
    assert not accepted(args+['--preflight','--preflight'])
    assert not accepted(args+['--optimizer-updates','1'])
    assert not accepted(args[:6]+['f'*64]+args[7:])
    assert not accepted(args[:-1]+[str(tmp_path/'other')])
    assert not accepted(args[:2]+['gx1.audit.observe_market_state_ta_current_updates_v1']+args[3:])
    assert not accepted(args[:8]+['cpu']+args[9:])
    assert not accepted(args[:4]+[str(tmp_path/'different-plan')]+args[5:])
