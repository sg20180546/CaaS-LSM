#!/usr/bin/env python3
"""Build/run durable CP registry tests without RocksDB, gRPC, or HDFS deps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "db/compaction/remote_compaction/storage_ownership_registry_test.cc"
    header = source.with_name("storage_ownership_registry.h")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise RuntimeError("C++ compiler unavailable: " + args.cxx)
    binary = work / "storage_ownership_registry_test"
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
               str(source), "-o", str(binary)]
    started = time.monotonic()
    build = subprocess.run(command, text=True, capture_output=True)
    (work / "compile.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    with tempfile.TemporaryDirectory(prefix="registry-cases-", dir=work) as scratch:
        result = subprocess.run([str(binary), scratch], text=True, capture_output=True)
    (work / "test.log").write_text(result.stdout + result.stderr)
    receipt = {
        "valid": result.returncode == 0, "compile_command": command,
        "return_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "cases": [line[len("PASS "):] for line in result.stdout.splitlines()
                  if line.startswith("PASS ") and not line.startswith("PASS all ")],
        "scope": "standalone POSIX durability/state/concurrency tests; no gRPC/HDFS integration or throughput claim",
    }
    (work / "registry_test_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    result.check_returncode()


if __name__ == "__main__":
    main()
