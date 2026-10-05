# Per-SST creation tracing

Set `SST_CREATION_TRACE=/local/run/sst_creation_<process>.jsonl` on both the
LSM server and CSA processes. Use a different local path per process and run.
Unset or empty disables the trace, including its clocks and pathname copies.
This instrumentation does not change ownership RPC behavior or error handling.

## Measurement boundary

`file_close.open_to_close_ns` starts immediately before `hdfsOpenFile` and ends
immediately after `hdfsCloseFile`. It includes open, the existing synchronous
initial `NotifyCreate`, merge iterator work performed while this writer is open,
encoding/compression, writes, Sync and Close. It is elapsed wall time, including
any scheduling/I/O waits. It is not just time inside HDFS write calls.

Iterator construction and initial seeks before file open are outside this
interval. A `scope_end` diagnostic records the entire BuildTable invocation or
local/remote-worker subcompaction execution. The first file also records
`first_output_preopen_ns`. These diagnostics include trace emission overhead and
must not be presented as pure merge CPU time or divided by output file count.
Scopes exclude preceding flush scheduling and remote-compaction queue/RPC time.

The main interval ends at physical close, before output verification, MANIFEST
installation, and any later remote-output rename. A successful `build_result`
validates that the builder finished successfully. It does not certify that a
later job-wide installation succeeded. A known failed scope should be excluded.

Initial ownership RPC wait is paired with the file in `own_rpc_ns`; the fraction
`own_rpc_ns / open_to_close_ns` is the measured initial-own wait share with
ownership enabled. It is not total control-plane overhead or an on/off throughput
comparison. Request serialization outside the synchronous stub invocation and
later rename-own/release calls are not in that numerator.

## JSONL schema 1

Every row has `schema`, `event`, `pid`, `tid`, and `wall_us` (Unix UTC microseconds
at emission). Integer durations use monotonic nanoseconds. Match records within
the process trace by `(pid, scope_id, path)`; scope IDs distinguish repeated
paths or job IDs. A run/node/process trace path provides the external identity.

| Event | Fields |
| --- | --- |
| `file_open` | `path`, `origin`, `job_id`, `scope_id` |
| `file_close` | Same identity; `start_wall_us`, `open_start_ns`, `close_end_ns`, `open_to_close_ns`, `hdfs_open_ns`, `bytes`, `own_attempted`, `own_ok`, `own_rpc_ns`, `open_ok`, `io_ok`, `close_ok`, `close_kind`, `first_output_preopen_ns` |
| `build_result` | Identity; `bytes`, `success`, `empty` |
| `scope_end` | `origin`, `job_id`, `scope_id`, `duration_ns`, `files_opened`, `success_known`, `success` |
| `ownership_rpc` | `op`, `path`, `duration_ns`, `success`, `files` |

`origin` is `flush`, `compaction`, `build_table_other`, or `unknown`.
`close_kind` is `explicit`, `destructor`, or `open_failed`. `bytes` on file close
counts bytes in successful appends; on build result it is the builder file size.
`first_output_preopen_ns` is null for later outputs in the same scope.

The opening intent is emitted **before** the measured clock starts; close and
builder records are emitted **after** it ends. An open without close is censored
or interrupted, not a zero-duration sample. For complete files, `start_wall_us`
is the authoritative start-cohort timestamp. The intent's `wall_us` is an
approximate earlier timestamp for unmatched opens. Trace output is synchronized
per process, not in the per-key merge/write path. Initial-own `ownership_rpc`
emission is deferred until file close, so its `wall_us` is not the RPC end time.
Other RPC records are emitted after their synchronous call returns.

For a successful per-file CDF, require a matching successful, nonempty
`build_result`, `open_ok && io_ok && close_ok`, `close_kind == "explicit"`,
and no known failed scope. For an ownership-enabled experiment additionally
check `own_attempted && own_ok`. Report excluded failures, unmatched opens,
and incomplete build records. Select a start cohort and drain it to completion
to avoid silently dropping its slowest files. Do not include preload in a
steady-state workload sample. Give every physical SST equal weight.

## Standalone validation

```sh
python3 tools/test_sst_creation_trace.py --work-dir /workspace/trace-test
```

This compiles two C++ translation units and checks shared TLS/sink state,
eight concurrent writers, JSON escaping and atomic records, own/file timing,
nested scopes, and unsuccessful/empty/destructor paths. Full engine build and
an ownership-enabled HDFS smoke run remain integration gates.

`CSA_ADDR=host:port` optionally overrides the CSA's default `:8010` address so
an experiment can use an isolated endpoint without stopping another CSA.
