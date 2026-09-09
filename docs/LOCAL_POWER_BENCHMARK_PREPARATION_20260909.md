# Local 200 W continuation — successor preparation

The matched 160/200 W smoke comparison selected 200 W and the shared Exit MTF
family scan. The selected configuration measured 71.882425555 seconds for 504
TRAIN rows (7.0114495 rows/s), with exact FP32 loss, gradient, model and VAL
parity. Its projected full TRAIN epoch is 12.8275 hours versus the earlier
14.634-hour projection. This is a measured 12.3% reduction without removing any
feature, head, objective, data row or validation work.

The next operation is an exact-state continuation from the preserved epoch-1
checkpoint at batch 19072 of 39175. The successor path changes only the session
contract digest; all learning and progress components must retain their exact
hashes, and a real CPU next-batch comparison must pass before CUDA execution.

Power scope v2 keeps the completed short comparison and adds one narrow candidate
continuation mode. Candidate mode requires the exact full candidate geometry,
immutable gate and invocation-budget digests, a single-use operator token, at
most 6000 seconds of authorization, continuous signed telemetry, 65 C core,
80 C memory-junction, 12 GiB VRAM and target+10 W draw stops. Each invocation
stops after at most 5400 seconds or after the first completed VAL epoch, restores
the 160 W baseline, and is followed by a PC reboot. TEST, promotion and permanent
power changes remain unavailable.

The optional Windows keeper scope is now wired in source. Its default config
still accepts at most 160 W. A benchmark requires both explicit scope-file and
SHA arguments, continuous mode, an idle GPU, exact 160 W baseline, canonical
scope directory, and durable exclusive scope/operator tokens. Restarted or
reused tokens close rather than rearm. UTC and monotonic expiry, caller closure,
changed scope/token bytes, a blocker, sample/treatment failures and recovery
return to baseline. Polling must be at most five seconds; normal 900-second
power reapplication cannot delay scope expiry restoration. Active and closed
receipts bind source/recipe/scope identities; closure records actual restoration.

Actual native PowerShell controller tests passed 20 scenarios using real
Windows temporary token/receipt files and mocked GPU/service owners. They found
and fixed a restoration UUID mismatch and PowerShell's null-string conversion
at atomic File.Replace. No physical setting or installed service changed.
The module, keeper and test script also passed native PowerShell syntax parsing.
Evidence: /var/tmp/gx1-local-efficiency-20260908/power_heartbeat_matched_reference_checks/verification.json.
Earlier parser/transition/restoration function tests are separate historical
source-bound evidence, not a replacement for this integrated controller test.

Both the 160 W reference and 200 W treatment now use the same scoped token,
receipt, heartbeat and closure path. Active keeper receipts renew every sample.
The new standard-library Linux admission/receipt owner passed 54 synthetic
file/geometry/identity checks, including native Windows timestamp compatibility.
It is now connected in preparation source to the canonical launcher, capped
runner and runtime watchdog. Both measured arms use that same optional path.
A claimed scope is single-use and owned by the exact Linux guard PID/start time;
each heartbeat checks the receipt, keeper identity, UTC and kernel-uptime expiry.
Exit requests closure after terminating the owned trainer group. Signed telemetry
requires the exact configured treatment in addition to existing draw/temperature
stops. This wiring has not yet passed an actual canonical launch.

The updated owner passed the same 54 metadata checks, 36 CLI/file lifecycle
checks with explicitly mocked process/clocks/cgroup boundaries, and five retained
repository unittest regressions on Mac Python 3.10. Actual WSL checks verified
atomic token publication/no replacement on the Windows mount and rejection of
an unowned parent and uncapped process (four checks). These are not a substitute
for the pending capped integration and full repository regression suite.
The existing cgroup owner now has a trainer-specific proof for the runner's
existing 128-task ceiling; audit/producer defaults remain 64. Recipe closure
includes the optional Windows keeper and scope source files.

Remaining: full keeper-loop/default regressions and failure-path review;
actual Linux guard/launcher integration and default-path regressions;
installed Windows source-byte verification; complete tests/commit/recipe; reviewed operator
installation/start/restore commands and then the actual matched 160/200 W run.
The optional keeper parameters alone do not provide Linux launch authority.


The Windows main loop now closes an active scope in a finally block when an
uncaught ordinary loop error exits. Ten native PowerShell scenarios passed with
source-identical control flow, an in-memory module loader, real temporary JSON
files and mocked hardware/service functions: default Once/default continuous,
caller closure, expiry, sample failure, finally cleanup, periodic treatment
reapplication, recovery only after baseline restoration, matched160 and rejected
incomplete scope. The existing default idle policy self-test also passed.
Forced OS process termination is not covered by finally or this test.

Linux inspect/claim/heartbeat now verify the two installed Windows owner file
hashes against recipe source bindings. The operator procedure must additionally
bind the receipt PID to the actual newly started keeper command/start time;
matching files on disk alone do not prove which code an older process loaded.
The six retained stdlib lifecycle tests passed on Mac Python3.10. New real-shell
regressions cover matching160/200, failed claim, wrong initial treatment, lost
heartbeat, SIGTERM child cleanup and closure failure; these await the next capped
CPU verification window. Two cgroup regressions cover trainer128 and unchanged
producer64 admission. No higher-power or installed-service mutation occurred.
