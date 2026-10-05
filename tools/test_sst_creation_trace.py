#!/usr/bin/env python3
"""Check opt-in trace validity, paired timing, failure flags, and concurrent TLS.

Run: python3 tools/test_sst_creation_trace.py --work-dir <workspace scratch dir>
Uses no RocksDB/HDFS server; full engine compilation is a separate required gate.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = root / "util/sst_creation_trace_test.cc"
    obj = args.work_dir / "trace_writer.o"
    exe = args.work_dir / "trace_test"
    flags = [args.cxx, "-std=c++17", "-O2", "-pthread", "-Wall", "-Wextra",
             "-Werror", "-I" + str(root), "-I" + str(root / "include")]
    subprocess.run(flags + ["-DSST_TRACE_TEST_WRITER", "-c", str(source),
                            "-o", str(obj)], check=True)
    subprocess.run(flags + [str(source), str(obj), "-o", str(exe)], check=True)
    env = os.environ.copy()
    env.pop("SST_CREATION_TRACE", None)
    subprocess.run([str(exe)], check=True, env=env)
    trace_path = args.work_dir / "trace_test.jsonl"
    trace_path.unlink(missing_ok=True)
    env["SST_CREATION_TRACE"] = str(trace_path)
    subprocess.run([str(exe)], check=True, env=env)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events = collections.defaultdict(list)
    for row in rows:
        assert row["schema"] == 1 and row["wall_us"] > 0
        events[row["event"]].append(row)
    assert len(events["file_open"]) == len(events["file_close"]) == 169
    assert len(events["build_result"]) == 169
    assert len(events["scope_end"]) == 163
    assert len(events["ownership_rpc"]) == 169  # 168 owns + separate failed release
    closes = {r["path"]: r for r in events["file_close"]}
    opens = {r["path"]: r for r in events["file_open"]}
    scopes = {r["scope_id"]: r for r in events["scope_end"]}
    assert len(closes) == len(opens) == 169 and len(scopes) == 163
    for path, row in closes.items():
        assert row["scope_id"] == opens[path]["scope_id"] != 0
        assert row["origin"] == scopes[row["scope_id"]]["origin"]
        assert row["open_to_close_ns"] == row["close_end_ns"] - row["open_start_ns"]
        assert row["open_to_close_ns"] >= row["hdfs_open_ns"] >= 0
        assert row["open_to_close_ns"] >= row["own_rpc_ns"] >= 0
        assert row["start_wall_us"] >= opens[path]["wall_us"]
    first = closes['/quoted"\\\n/first.sst']
    assert first["own_rpc_ns"] >= 1_000_000
    assert first["bytes"] == 192 and first["own_ok"]
    assert first["scope_id"] == closes["/second.sst"]["scope_id"]
    assert first["scope_id"] != closes["/nested.sst"]["scope_id"]
    assert first["first_output_preopen_ns"] is not None
    assert closes["/second.sst"]["first_output_preopen_ns"] is None
    assert not closes["/open_failed.sst"]["open_ok"]
    assert closes["/open_failed.sst"]["close_kind"] == "open_failed"
    assert not closes["/io_failed.sst"]["io_ok"]
    assert not closes["/own_failed.sst"]["own_ok"]
    assert not closes["/close_failed.sst"]["close_ok"]
    assert closes["/destructor.sst"]["close_kind"] == "destructor"
    assert closes["/empty.sst"]["bytes"] == 0
    assert sum(r["files_opened"] for r in scopes.values()) == len(closes)
    assert all(r["success_known"] for r in scopes.values())
    print(json.dumps({"status": "PASS", "files": len(closes),
                      "scopes": len(scopes), "rows": len(rows),
                      "trace": str(trace_path)}))


if __name__ == "__main__":
    main()
