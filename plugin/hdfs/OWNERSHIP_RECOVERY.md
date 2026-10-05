# Ownership publication and recovery

New SST files become owned after their final path and contents are ready and
before their MANIFEST edit is attempted. A failed MANIFEST write or sync has an
uncertain outcome: it can have persisted despite the error. Such files keep
their ownership until recovery resolves the MANIFEST. Do not release them from
the error path.

The CN uses `NotifyCreateBatch` for the final paths of a flush, compaction,
WAL-recovery flush, ingestion, or import. `NewWritableFile` and SST rename do
not send own/release RPCs. A completed CSA result is renamed to its final CN
path, verified, then registered with the CN's shard identity. Relink uses
`PrepareReferences` before the destination MANIFEST; it never registers a
second physical birth. A terminal registration failure stops publication.

Ordinary obsolete-file deletion releases only paths known to have been
published by this process or verified from its recovered MANIFEST. Unknown
paths and uncertain commits are kept for offline resolution. SST targets are
never overwritten by rename, and ownership-enabled clients never perform
physical SST deletion, including on RPC failure.

## Deployment and durability

Set `STORAGE_CP_ADDR` and a stable, unique `STORAGE_CP_SHARD` per owning DB
process. One shard identity must not be shared by independently live DBs.
Set `STORAGE_CP_STATE_DIR` on the CP to an absolute local directory on durable
storage. The CP acquires an exclusive process lock, replays its checksummed
append journal, and fsyncs state changes before ACK. Reuse this directory for
CP restarts; do not clear it while any corresponding SST namespace exists.
Corrupt or truncated journals stop startup instead of silently losing owners.
Back up this directory together with the deployment's namespace/identity map.

Unsetting `STORAGE_CP_ADDR` retains the baseline HDFS path. A CP without
`STORAGE_CP_STATE_DIR` still serves remote compaction, but rejects all ownership
requests and disables storage GC. Enable ownership only with matching new CN,
CSA, and CP binaries. The new publication RPCs deliberately have no fallback to
an older volatile CP. Existing databases with no durable ownership journal
cannot be silently adopted: recovered live references fail validation.

Batch calls retry transport failures with the same operation ID and a bounded
deadline (default 5 seconds per attempt, up to 3 attempts;
`STORAGE_CP_RPC_TIMEOUT_MS` accepts 1–60000). Immutable path tombstones prevent
a late own from resurrecting a released file. CP GC durably claims each retired
path before HDFS I/O and retries interrupted claims after restart.

The current release API identifies a reference by path and shard, without a
reference generation token. To prevent a delayed old release from consuming a
new reference, the CP rejects reacquisition of the same physical path by a
shard that already released it. A different new path can be owned normally;
repeated migration back to a previously released shard needs a future
generation-aware ownership/release API. Such a request fails publication and
keeps the data rather than weakening the stale-release fence.

The registry currently serializes journal commits under a mutex and fsyncs
each mutation batch. Journal records, idempotency records, and tombstones are
retained; this implementation does not yet compact them. Consequently the old
in-memory ownership capacity measurements are **not** a throughput guarantee
for this durable implementation. Rebenchmark it before quoting capacity or
latency. The SST trace now separates file open-to-close from final publication
RPCs and records rename/publication evidence and individual retry attempts.

On normal read/write DB open, the engine recovers the MANIFEST, gathers all live
SST paths (including relink `external_path` values), and asks `RecoverReferences`
to validate their durable CP owner records. This runs before WAL replay and
obsolete-file cleanup. Missing or retired CP state fails the open; a single DB
cannot reconstruct every other shard's references. Restore the CP journal from
its authoritative backup or perform an explicitly coordinated full-registry
reconstruction. Do not replace this failure with `NotifyCreate`.

SSTs produced by WAL recovery are also registered as a final-path batch before
the recovery MANIFEST edit. A successful edit marks them published locally; an
uncertain edit preserves them. RPC waits run outside the DB mutex.

## Offline orphan cleanup

An SST's absence from the MANIFEST does not prove it is safe to delete: a live
flush or CSA task can still be writing or preparing it. The current HDFS
`LockFile` is a no-op. Therefore normal DB open does **not** sweep staging
directories or infer an abort from a file's age.

Use `storage_cp_recover` for explicit offline maintenance after a crash. It
handles both never-owned temporary files and owned final files whose MANIFEST
edit did not commit. All physical SST deletion remains in the CP.

Prerequisites:

1. Stop and fence every CN and CSA that can create, rename, publish, or reference
   the affected files, including destination shards using relink. Disable their
   restart controllers for the duration. `--producers-stopped` records this
   operator precondition; the utility does not establish a distributed fence.
2. Keep the authoritative CP and its durable journal available. The CP must
   use the same HDFS namespace selected by `--fs-uri`.
3. Use the correct owning shard ID and canonical absolute **bare HDFS paths**
   (for example `/experiment/s0/000123.sst`). The CP serves one HDFS namespace;
   `hdfs://...` keys, repeated slashes, and dot segments are rejected so two
   registry keys cannot alias one physical file. Use the same spelling in DB
   and CF paths. Immutable registry keys must not be recycled.
4. Supply an explicit local candidate list, one absolute `.sst` path per line.
   Include abandoned CSA job paths and final outputs identified during the
   incident. Candidates must be under this DB's configured data directories.

Build with the engine's usual HDFS configuration:

```sh
cmake --build build --target storage_cp_recover storage_cp_recovery_plan_test -j32
```

Preview, using the same HDFS endpoint and shard identity as the affected DB:

```sh
build/tools/storage_cp_recover \
  --db /experiment/s0 --fs-uri hdfs://namenode:9000 \
  --cp ownership-host:18120 --shard 0 \
  --candidates /data/maintenance/s0-candidates.txt
```

The default is a dry run. `KEEP` paths are in the recovered live MANIFEST.
`DISCARD_OWNER` paths are absent from it; this label does not authorize deletion
until the producer-fencing precondition is satisfied.

After producers are fenced, repeat the same command with
`--apply --producers-stopped`. The utility:

1. Loads the actual OPTIONS, requires every live column family to be described,
   opens the DB read-only, and obtains its live SST paths including external
   references. An incomplete or unreadable snapshot aborts the operation.
2. Checks that CURRENT and the entire active MANIFEST have not changed during
   inspection. It validates live paths with `RecoverReferences` before any
   discard, then checks the snapshot again.
3. Sends `AbortUnpublished` only for candidate paths absent from the live set.
   The CP durably fences these paths, releases this owner's uncommitted
   sole references, and performs physical GC itself. Linked or foreign
   ownership makes the abort batch fail closed without changing those records;
   resolve the other shard's references before attempting further cleanup.

The snapshot checks detect accidental concurrent MANIFEST changes; they do not
replace stopping producers. Partial RPC success is safe to retry: ownership
validation and abort are idempotent, and retired paths cannot be re-owned by a
late request. `DONE` means the CP acknowledged the operations, not that its
asynchronous HDFS deletion has already finished.

The tool intentionally does not guess candidate paths by age, delete whole job
directories, infer missing owners, rewrite MANIFESTs, or erase files directly.
It cleans the explicitly listed orphan candidates; omitted candidates remain
safe storage leaks until a later maintenance pass.

## Verification

`storage_cp_recovery_plan_test` exercises crashes before publication, an owned
but uncommitted final output, a never-owned staging file, uncertain MANIFEST
commit resolving as live, missing producer fencing, a changing MANIFEST,
missing CP state, shared references, lost-ACK retry, and path-scope validation.
The CP registry tests separately exercise the durable abort and ownership
semantics. The planner test can also run without Hadoop or gRPC:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -I . \
  plugin/hdfs/storage_cp_recovery_plan_test.cc -o recovery_plan_test
./recovery_plan_test
```
