#!/usr/bin/env python3
"""
GX1 PreToolUse guard — XAU/EURUSD isolation, DEFENSE-IN-DEPTH at the command layer.
Exit 2 = block, 0 = allow.

*** READ THIS — L3 is the CHEAP EARLY NET, not the guarantee ***
A command-STRING guard CANNOT be a 100% isolation guarantee. Two red-teams (2026-06-20)
confirmed it: env-vars ($GXDATA), command-substitution ($(...)), base64-assembled paths,
neutral wrapper scripts / Makefiles, and interpreter-internal path construction
(perl `do "$ENV{HOME}/..."`, node `require(os.homedir()+...)`) build the real target
AFTER this guard sees the command — unresolvable here. THE LOAD-BEARING
never-run-on-the-wrong-data guarantee is therefore (1) OS-level (L1): GX1_DATA and
EURUSD_DATA owned/permissioned so neither project's process can read the other's
(kernel-enforced), and (2) code-level (L2): each project's central data-root resolver
asserts its root at access time and raises on the foreign root. THIS guard is the
cheap early net that catches the obvious + the resolvable obfuscations.

Hardening 2026-06-20 (v2, after a 2nd red-team found 64 bypasses of the v1 substring/
collapse logic — see /tmp/redteam_exec.py). Membership is now realpath() OR component-wise
fnmatch of the GLOB PATTERN against the fixed roots — so {a,b,X}, [!X], [F-H], char-classes,
braces with the live path as a NON-first alternative, `..`/`.`/`//` traversal, and
not-yet-existing EUR paths are all caught WITHOUT relying on the path existing on disk.
Tokenization strips redirect operators (`>foo`, `2>foo`, `&>foo`, `<<<`), splits on
background `&`, isolates process-substitution `<(...)`, and splits colon/comma path-lists.
`find -exec`, bundled `bash -c '...'`/`sh -c`/`env -S`/`eval` strings, and
reader-argv0-shielded execs no longer pass. Git catastrophe-floor (rule 6) is parsed
structurally (global-flag-adjacency-proof): force push (--force / --force-with-lease /
+refspec / -f cluster), --amend, reset --hard, filter-branch / filter-repo, rebase
-i/--root/--autosquash, inline `-c alias.<x>=…`, and add/commit of .env/credentials.

KNOWN RESIDUAL LIMITS (= exactly why L1+L2 exist; a string guard cannot close these):
  - runtime-constructed paths: $(...), backticks, $VAR sourced from a file/earlier export,
    base64-decoded, neutral wrapper-script/Makefile recipes, interpreter-internal building.
  - implicit-stage secrets: `git add -u` / `git add -A` / `git commit -a` can stage an
    already-tracked .env with no filename token (blocking all of these would break the
    normal workflow); and `cp .env x && git add x` renames the secret. Not catchable by string.

Model: context = a `.PROJECT_EURUSD` marker walked up from session cwd (=> EURUSD), else
XAU/neutral. DATA never crosses (membership under the two data roots, read OR write). CODE
may be READ across but never EXECUTED or WRITTEN across. One command touching BOTH data
roots => block. Always: force-push / amend / history-rewrite / secrets.
"""

import fnmatch
import json
import os
import re
import shlex
import sys
from pathlib import Path

HOME = "/home/andre2"
XAU_DATA_ROOT = os.path.realpath(f"{HOME}/GX1_DATA")
EUR_DATA_ROOT = os.path.realpath(f"{HOME}/EURUSD/DATA")
XAU_CODE_ROOT = os.path.realpath(f"{HOME}/src/GX1_ENGINE")
EUR_CODE_ROOT = os.path.realpath(f"{HOME}/EURUSD/ENGINE")
EUR_HANDOVER = os.path.realpath(f"{HOME}/EURUSD/HANDOVER")

READERS = {
    "cat", "bat", "less", "more", "head", "tail", "grep", "egrep", "fgrep", "rg",
    "ag", "view", "nl", "wc", "diff", "ls", "find", "stat", "file", "realpath",
    "readlink", "basename", "dirname", "tree", "du", "column", "cut", "comm",
    "cmp", "md5sum", "sha256sum", "xxd", "od", "strings", "echo", "printf",
}
GIT_RO = {"log", "show", "diff", "blame", "cat-file", "ls-files", "status", "rev-parse"}

# git global options that consume the NEXT token as a value (so we skip both when finding the subcommand)
_GIT_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
# options whose VALUE is a commit message / file, not a pathspec (skip when scanning for secrets)
_GIT_MSG_OPTS = {"-m", "--message", "-F", "--file", "-C", "--reuse-message", "-c", "--reedit-message"}


# ───────────────────────── git catastrophe-floor (rule 6) ─────────────────────────
def _git_block(seg: str):
    """Return a reason string if this segment is a forbidden git/secrets op, else None.
    Parses the subcommand PAST global flags so `git -c x`/`git -C d`/`git --no-pager`
    can't shift the anchor."""
    try:
        toks = shlex.split(seg)
    except Exception:
        toks = seg.split()
    gi = None
    for i, t in enumerate(toks):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            continue
        if os.path.basename(t) == "git":
            gi = i
        break
    if gi is None:
        return None
    rest = toks[gi + 1:]

    # inline alias definition pointing at git (`-c alias.x=push`) = floor-evasion smell
    for i, t in enumerate(rest):
        if t == "-c" and i + 1 < len(rest) and rest[i + 1].startswith("alias."):
            return "inline git alias definition (-c alias.…) — evasion of the git floor"
        if t.startswith("-c") and "alias." in t:
            return "inline git alias definition (-c alias.…) — evasion of the git floor"

    # locate the subcommand, skipping global flags (and their values)
    j = 0
    sub = None
    while j < len(rest):
        t = rest[j]
        if t in _GIT_OPTS_WITH_VALUE:
            j += 2
            continue
        if t.startswith("-"):
            j += 1
            continue
        sub = t
        break
    after = rest[j + 1:] if sub is not None else []

    if sub == "push":
        if any(f in ("--force", "--force-with-lease", "--force-if-includes") for f in after):
            return "force push (--force) is forbidden"
        if any(re.match(r"^-[A-Za-z]*f", t) for t in after):
            return "force push (-f) is forbidden"
        if any(re.match(r"^\+", t) for t in after):
            return "force push (+refspec) is forbidden"
    if sub == "commit" and "--amend" in after:
        return "git commit --amend is forbidden (never amend live commits)"
    if sub in ("filter-branch", "filter-repo"):
        return f"git {sub} rewrites history — forbidden"
    if sub == "rebase" and any(f in after for f in ("-i", "--interactive", "--root", "--autosquash")):
        return "interactive/root rebase rewrites history — forbidden"
    if sub == "reset" and "--hard" in after:
        return "git reset --hard is forbidden on shared history"
    if sub in ("add", "commit"):
        scan, k = [], 0
        while k < len(after):
            t = after[k]
            if t in _GIT_MSG_OPTS:
                k += 2
                continue
            if t.startswith(("-m", "-F")) or t.startswith("--message=") or t.startswith("--file="):
                k += 1
                continue
            scan.append(t)
            k += 1
        if re.search(r"(\.env\b|credentials)", " ".join(scan)):
            return "staging a secret (.env/credentials) is forbidden"
    return None


# ───────────────────────── path membership (realpath + fnmatch) ─────────────────────────
def _brace_expand(s: str):
    """Expand {a,b,c} (incl. empty alts and multiple/nested braces) to all combinations."""
    m = re.search(r"\{([^{}]*)\}", s)
    if not m:
        return [s]
    pre, body, post = s[: m.start()], m.group(1), s[m.end():]
    out = []
    for alt in body.split(","):
        out.extend(_brace_expand(pre + alt + post))
    return out


_REDIR = re.compile(r"^\s*[0-9]*&?(?:>>|&>|>|<<<|<<|<)\s*")


def _fragments(tok: str):
    """All path-ish fragments a raw shell token might carry: strips quotes + leading redirect
    operators, splits on whitespace / ':' / ',' (PATH/PYTHONPATH/list joins and bundled
    `-c '…'` strings), peels procsub/grouping punctuation, and exposes =values. Braces stay
    INTACT (expanded later in _touches)."""
    base = tok.replace("'", "").replace('"', "")
    base = _REDIR.sub("", base)
    raw = {base}
    # split on whitespace / ':' / ',' AND code punctuation '()' so a path glued inside a
    # function call (e.g. copytree("/foreign/x",...)) or embedded in a -c program is exposed.
    for part in re.split(r"[\s,:()]+", base):
        if part:
            raw.add(part)
    frags = set()
    for p in raw:
        if not p:
            continue
        frags.add(p)
        st = p.lstrip("<(>").rstrip(");")
        if st and st != p:
            frags.add(st)
        if "=" in p and not p.startswith("="):
            v = p.split("=", 1)[1]
            if v:
                frags.add(v)
                for vv in re.split(r"[\s,:]+", v):
                    if vv:
                        frags.add(vv)
    return frags


def _pattern_under_root(frag: str, cwd: str, root: str) -> bool:
    """True if the GLOB PATTERN `frag` could reference a path under `root`, by component-wise
    fnmatch — filesystem-independent (catches not-yet-existing EUR dirs and ALL glob/brace/
    char-class/range/negation forms, incl. the live path as a non-first brace alternative)."""
    root_parts = [x for x in root.split("/") if x != ""]
    for alt in _brace_expand(frag):
        a = os.path.expanduser(alt)
        if not os.path.isabs(a):
            a = os.path.join(cwd, a)
        a = os.path.normpath(a)
        parts = [x for x in a.split("/") if x != ""]
        if len(parts) < len(root_parts):
            continue
        try:
            if all(fnmatch.fnmatch(root_parts[i], parts[i]) for i in range(len(root_parts))):
                return True
        except Exception:
            continue
    return False


def _realpath_under(frag: str, cwd: str, root: str) -> bool:
    """True if `frag` realpath-resolves (following symlinks, via literal or glob) under `root`.
    Complements the fnmatch check for the symlink-to-foreign-root case."""
    import glob as _glob
    for alt in _brace_expand(frag):
        a = os.path.expanduser(alt)
        cands = [a]
        if not os.path.isabs(a):
            cands.append(os.path.join(cwd, a))
        for c in list(cands):
            try:
                g = _glob.glob(c)
            except Exception:
                g = []
            cands.extend(g)
        for c in cands:
            try:
                rp = os.path.realpath(c)
                if os.path.commonpath([rp, root]) == root:
                    return True
            except Exception:
                pass
    return False


def _touches(frag: str, cwd: str, root: str) -> bool:
    return _pattern_under_root(frag, cwd, root) or _realpath_under(frag, cwd, root)


# ───────────────────────── segmentation & reader classification ─────────────────────────
def _segments(cmd: str):
    # split on && || ; newline | ; then split each piece on a lone background & (not && / &>)
    out = []
    for p in re.split(r"&&|\|\||;|\n|\|", cmd):
        out.extend(re.split(r"&(?!>)", p))
    return [s for s in out if s.strip()]


def _argv0_basename(seg: str):
    try:
        toks = shlex.split(seg)
    except Exception:
        toks = seg.split()
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
        i += 1
    if i >= len(toks):
        return None, toks
    return os.path.basename(toks[i]), toks


def _is_reader(seg: str) -> bool:
    # a segment that EXECUTES (find -exec, process substitution) is never a "reader"
    if re.search(r"(?:^|\s)-(?:exec|execdir|ok)\b", seg) or "<(" in seg or ">(" in seg:
        return False
    a0, toks = _argv0_basename(seg)
    if a0 is None:
        return True
    if a0 in READERS:
        return True
    if a0 == "git":
        sub, skip = None, False
        for t in toks[1:]:
            if skip:
                skip = False
                continue
            if t in _GIT_OPTS_WITH_VALUE:
                skip = True
                continue
            if t.startswith("-"):
                continue
            sub = t
            break
        return sub in GIT_RO
    if a0 == "sed":
        return "-i" not in toks and any(t.startswith("-n") for t in toks)
    if a0 == "awk":
        return not any(">" in t or "system" in t for t in toks)
    return False


def _write_targets(seg: str):
    """Fragments that are REDIRECT WRITE targets (`>x`, `>>x`, `&>x`, `1>x`) — writing into a
    foreign tree (data OR code) is forbidden even for a reader argv0."""
    out = set()
    for m in re.finditer(r"(?:[0-9]*&?(?:>>|&>|>))\s*([^\s;|&]+)", seg):
        tgt = m.group(1).replace("'", "").replace('"', "")
        if tgt and not tgt.startswith(("&", "-")):  # skip fd-dup like >&2
            out |= _fragments(tgt)
    return out


def is_eurusd_context(cwd: str) -> bool:
    try:
        p = Path(cwd).resolve()
    except Exception:
        return False
    for d in [p, *p.parents]:
        try:
            if (d / ".PROJECT_EURUSD").exists():
                return True
            if (d / ".PROJECT_XAU").exists():
                return False
        except Exception:
            break
    return False


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = event.get("tool_input", {}).get("command", "")
    if not cmd:
        return 0
    cwd = event.get("cwd") or os.getcwd()

    # git catastrophe-floor (context-independent) — check every segment
    for seg in _segments(cmd):
        reason = _git_block(seg)
        if reason:
            print(f"BLOCKED by GX1 guard: {reason}\n  command: {cmd}", file=sys.stderr)
            return 2

    eur_ctx = is_eurusd_context(cwd)
    foreign_data = XAU_DATA_ROOT if eur_ctx else EUR_DATA_ROOT
    foreign_code_roots = [XAU_CODE_ROOT] if eur_ctx else [EUR_CODE_ROOT, EUR_HANDOVER]
    other = "XAU" if eur_ctx else "EURUSD"

    try:
        if os.path.commonpath([os.path.realpath(cwd), foreign_data]) == foreign_data:
            print(f"BLOCKED by GX1 guard: cwd is inside the {other} DATA tree\n  command: {cmd}", file=sys.stderr)
            return 2
    except Exception:
        pass

    touches_xau_data = touches_eur_data = False
    for seg in _segments(cmd):
        try:
            toks = shlex.split(seg)
        except Exception:
            toks = seg.split()
        frags = set()
        for t in toks:
            frags |= _fragments(t)
        writes = _write_targets(seg)
        reader = _is_reader(seg)

        seg_foreign_data = seg_foreign_code_read = seg_foreign_code_write = False
        for f in frags:
            if _touches(f, cwd, XAU_DATA_ROOT):
                touches_xau_data = True
            if _touches(f, cwd, EUR_DATA_ROOT):
                touches_eur_data = True
            if _touches(f, cwd, foreign_data):
                seg_foreign_data = True
            if any(_touches(f, cwd, r) for r in foreign_code_roots):
                seg_foreign_code_read = True
                if f in writes:
                    seg_foreign_code_write = True

        if seg_foreign_data:
            print(f"BLOCKED by GX1 guard: reference to {other} DATA — data never crosses "
                  f"(prevents running on the wrong dataset)\n  command: {cmd}", file=sys.stderr)
            return 2
        if seg_foreign_code_write:
            print(f"BLOCKED by GX1 guard: writing into the {other} tree — code may be READ "
                  f"across but never WRITTEN across\n  command: {cmd}", file=sys.stderr)
            return 2
        if seg_foreign_code_read and not reader:
            print(f"BLOCKED by GX1 guard: executing against the {other} tree — code may be "
                  f"READ across but never RUN across\n  command: {cmd}", file=sys.stderr)
            return 2

    if touches_xau_data and touches_eur_data:
        print(f"BLOCKED by GX1 guard: one command touches BOTH data roots\n  command: {cmd}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
