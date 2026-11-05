// [ft 2026-11-04] Explicit release of this shard's Storage-CP references.
//
// The destination of an uncommitted migration rolls back by (1) unregistering the transferred files it published in
// its MANIFEST (their obsolete-file purge releases them) and (2) releasing the references it prepared at the
// Storage-CP but never published -- those files are in no MANIFEST of this shard, so only an explicit release can
// drop them. A path this shard does not reference is ignored by the Storage-CP (idempotent).
// No-op when the Env is not backed by the HDFS file system or STORAGE_CP_ADDR is unset.
#pragma once

#include <string>
#include <vector>

#include "rocksdb/env.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
Status StorageCpReleaseReferences(Env* env, const std::vector<std::string>& paths);
}  // namespace ROCKSDB_NAMESPACE
