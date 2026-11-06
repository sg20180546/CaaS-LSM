// [ft 2026-11-04] Explicit release of this shard's Storage-CP references.
//
// The destination of an uncommitted migration rolls back by (1) unregistering the transferred files it published in
// its MANIFEST (their obsolete-file purge releases them) and (2) releasing the references it prepared at the
// Storage-CP but never published -- those files are in no MANIFEST of this shard, so only an explicit release can
// drop them. A path this shard does not reference is ignored by the Storage-CP (idempotent).
// No-op when the Env is not backed by the HDFS file system or STORAGE_CP_ADDR is unset.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "rocksdb/env.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
Status StorageCpReleaseReferences(Env* env, const std::vector<std::string>& paths);
// [ft 2026-11-05] While the counter is > 0 every compaction of this process runs locally instead of on the remote
// compaction service (whose DB open has no compaction filter): a migration's PREPARE uses it to discard the stale
// remnants of a returning key range through the OwnershipFilter (CompactFiles, intra-L0). delta = +1 / -1.
void StorageCpForceLocalCompactions(int delta);
int StorageCpForceLocalCompactionsCount();
// [ownership 2026-11-05] The host registers a provider of its retained key ranges ("lo:hi;..." over 8-byte big-endian keys);
// every remote compaction request carries the current value and the CSA drops keys outside the ranges (the host's
// OwnershipFilter, evaluated remotely). Unset / empty string = the remote compaction keeps every key (stock behaviour).
void SetOwnershipRetainedRangesProvider(std::function<std::string()> provider);
std::string OwnershipRetainedRanges();
}  // namespace ROCKSDB_NAMESPACE
