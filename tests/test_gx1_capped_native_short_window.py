"""Execute the real native cap timeout gate only; no launcher, guard or model."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[1] / 'scripts/gx1_capped_run.sh'
RESEARCH = 'gx1_native_direction_research_window_policy_v1'
SHORT = '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002/RECIPE.json'
PAIRED = '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_PAIRED_BATCH_ORDER_20261110_002/RECIPE.json'


def actual_gate(tmp_path, *, seconds, recipe=SHORT, schema=RESEARCH):
    lines = RUNNER.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('native_seconds=$('))
    end = next(i for i in range(start, len(lines))
               if lines[i].strip() == 'TRAINER_MODEL_MAX_WALL_SECONDS=$TRAINER_MAX_WALL_SECONDS')
    # Test the embedded JSON validator and its real shell mapping, not a copied predicate.
    block = '\n'.join(lines[start:end + 1])
    window = tmp_path / 'WINDOW.json'
    window.write_text(json.dumps(dict(schema_version=schema,
        max_invocation_seconds=seconds, recipe={'path': recipe})))
    script = '''set -eu
CANONICAL_TRAINER_PYTHON="$1"
target_args=("" "" "" "" "$2")
''' + block + '''
printf '%s %s %s\\n' "$native_seconds" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS"
'''
    return subprocess.run(['bash', '-c', script, 'native-cap-gate-test', sys.executable, str(window)],
                          capture_output=True, text=True, timeout=10, check=False)


def test_exact_replacement_has_3600_window_and_3900_guard(tmp_path):
    result = actual_gate(tmp_path, seconds=3600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '3600 3900 3900'


@pytest.mark.parametrize('seconds,recipe,schema', [
    (3600, SHORT.replace('_002/', '_001/'), RESEARCH),
    (3600, SHORT.replace('_002/', '_003/'), RESEARCH),
    (3600, '/tmp/RECIPE.json', RESEARCH),
    (3600, SHORT, 'gx1_native_direction_prefix_fit_window_policy_v1'),
    (3599, SHORT, RESEARCH),
    (3601, SHORT, RESEARCH),
    (3600.0, SHORT, RESEARCH),
])
def test_consumed_or_unbound_short_window_remains_closed(tmp_path, seconds, recipe, schema):
    result = actual_gate(tmp_path, seconds=seconds, recipe=recipe, schema=schema)
    assert result.returncode == 75
    assert 'AssertionError' in result.stderr and result.stdout == ''


@pytest.mark.parametrize('seconds,recipe,expected', [
    (3000, PAIRED, '3000 3300 3300'),
    (12000, '/fixture/existing-native-recipe.json', '12000 13800 13800'),
])
def test_existing_native_timeout_cases_are_unchanged(tmp_path, seconds, recipe, expected):
    result = actual_gate(tmp_path, seconds=seconds, recipe=recipe)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_short_path_does_not_inherit_paired_3000_exception(tmp_path):
    result = actual_gate(tmp_path, seconds=3000)
    assert result.returncode == 75 and 'AssertionError' in result.stderr
