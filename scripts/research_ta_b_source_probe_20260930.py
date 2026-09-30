#!/usr/bin/env python3
"""One-shot B source audit. Executes frozen HTTP helpers, never research fits."""
import ast
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

manifest_path, expected, owner_path = map(str, sys.argv[1:])
digest = lambda data: hashlib.sha256(data).hexdigest()
manifest_raw = Path(manifest_path).read_bytes()
assert digest(manifest_raw) == expected, "MANIFEST_HASH"
spec = json.loads(manifest_raw)
assert digest(Path(__file__).read_bytes()) == spec["probe_sha256"], "PROBE_HASH"
assert socket.gethostname() == spec["transport_hostname"], "TRANSPORT_HOST"
owner_raw = Path(owner_path).read_bytes()
assert digest(owner_raw) == spec["owner_sha256"], "OWNER_HASH"
names = {"sha", "write_json", "AlfredDownloadForm", "alfred_form_body", "fetch_alfred"}
nodes = [node for node in ast.parse(owner_raw).body
         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
assert {node.name for node in nodes} == names
# Execute only these frozen stdlib HTTP helpers. No model module imports.
exec(compile(ast.Module(body=nodes, type_ignores=[]), owner_path, "exec"), globals())
out = Path(spec["relay_output_directory"])
out.mkdir(parents=True, exist_ok=False)
started = datetime.now(timezone.utc).isoformat()
records = []
for entry in spec["archive_metadata_requests"]:
    record = {"id": entry["id"], "url": entry["url"],
              "started_utc": datetime.now(timezone.utc).isoformat()}
    begin = time.monotonic()
    try:
        req = Request(entry["url"], headers={"User-Agent": "GX1 offline research source audit"})
        try:
            response = urlopen(req, timeout=spec["timeout_seconds"])
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(spec["maximum_metadata_bytes"] + 1)
            record.update(http_status=response.code, final_url=response.url,
                          content_type=response.headers.get("Content-Type"))
        path = out / (entry["id"] + ".response")
        path.write_bytes(raw)
        record.update(response_sha256=sha(path), response_bytes=len(raw))
        if len(raw) > spec["maximum_metadata_bytes"]:
            raise ValueError("METADATA_SIZE_LIMIT")
        if record["http_status"] != 200:
            raise ValueError("HTTP_NON_200")
        decoded = json.loads(raw)
        record.update(status="METADATA_RECEIVED_NOT_ADMITTED",
                      json_type=type(decoded).__name__)
    except Exception as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"] = time.monotonic() - begin
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    records.append(record)
    write_json(out / (entry["id"] + ".receipt.json"), record)
    print(entry["id"], record["status"], flush=True)
alfred = dict(spec["alfred_probe"])
alfred["output_directory"] = str(out / "ALFRED")
fetch_alfred(alfred, Path(manifest_path))
files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
result = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
          "manifest_sha256": expected, "probe_sha256": spec["probe_sha256"],
          "owner_sha256": spec["owner_sha256"], "transport_hostname": socket.gethostname(),
          "archive_requests": records, "files": files, "predictors_admitted": [],
          "fits_run": False, "market_outcomes_read": False, "test_accessed": False,
          "status": "COMPLETE_SOURCE_TRANSPORT_AUDIT"}
write_json(out / "RESULT.json", result)
write_json(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json")})
print(json.dumps({"result": str(out / "RESULT.json"), "sha256": sha(out / "RESULT.json")}), flush=True)
