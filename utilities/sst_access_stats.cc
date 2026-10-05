//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [sst access stats 2026-10-04] Free-function readers for the per-SST
// foreground block-cache counters and the LRUCache per-shard pool counters.
// Why: both are consumed by the relink block-cache transfer (the source
// describes each migrated file by a hit RATE, the destination plans
// admission from per-shard room) and neither may touch a PUBLIC vtable --
// csa/procp/userclient link against librocksdb.so without being rebuilt.
// Everything here reaches internal classes through the root DB and the
// internal TableReader, which is why this lives in the library and not in the
// driver.

#include "rocksdb/utilities/sst_access_stats.h"

#include <cstring>
#include <string>

#include "cache/lru_cache.h"
#include "db/column_family.h"
#include "db/db_impl/db_impl.h"
#include "db/table_cache.h"
#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/system_clock.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/table_reader.h"

namespace ROCKSDB_NAMESPACE {

Status GetSstAccessStats(DB* db, ColumnFamilyHandle* column_family,
                         std::vector<SstAccessStats>* out) {
  if (db == nullptr) {
    return Status::InvalidArgument("db cannot be null");
  }
  if (out == nullptr) {
    return Status::InvalidArgument("out cannot be null");
  }
  out->clear();

  // Same unwrapping as utilities/debug.cc: a StackableDB forwards
  // DefaultColumnFamily() to the DBImpl underneath, so the handle is always a
  // ColumnFamilyHandleImpl once the root DB is reached.
  DB* root_db = db->GetRootDB();
  if (column_family == nullptr) {
    column_family = root_db->DefaultColumnFamily();
  }
  if (column_family == nullptr) {
    return Status::InvalidArgument("column family handle cannot be null");
  }
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(column_family)->cfd();
  if (cfd == nullptr || cfd->table_cache() == nullptr) {
    return Status::InvalidArgument("column family has no table cache");
  }
  // The TableCache filters by key size and deleter: the blob file cache shares
  // the same Cache and its values are not TableReaders.
  cfd->table_cache()->GetSstAccessStatsOfResidentTables(out);
  return Status::OK();
}

void SetSstAccessStatsCounting(bool on) {
  g_sst_access_stats_counting.store(on, std::memory_order_relaxed);
}

bool SstAccessStatsCounting() {
  return g_sst_access_stats_counting.load(std::memory_order_relaxed);
}

uint64_t SstAccessStatsNowMicros(DB* db) {
  if (db == nullptr) {
    return 0;
  }
  // ImmutableDBOptions::clock is the object every ImmutableOptions (and so
  // every BlockBasedTable::Rep::ioptions) copies its clock pointer from, so
  // this is the same clock that stamped open_time_micros.
  DBImpl* impl = static_cast<DBImpl*>(db->GetRootDB());
  return impl->immutable_db_options().clock->NowMicros();
}

Status GetLRUCacheShardPoolStats(Cache* cache,
                                 std::vector<LRUCacheShardPoolStats>* out) {
  if (cache == nullptr) {
    return Status::InvalidArgument("cache cannot be null");
  }
  if (out == nullptr) {
    return Status::InvalidArgument("out cannot be null");
  }
  out->clear();
  // No RTTI in release builds, so the class is recognised by its Name(). A
  // ChargedCache wrapper (Name() == "ChargedCache", used when a cache is
  // shared with a CacheReservationManager) is rejected here rather than
  // unwrapped: unwrapping would need a virtual on Cache, which is public ABI.
  // The driver constructs its block cache with NewLRUCache directly, so the
  // plain case is the one that matters. The experimental FastLRUCache
  // (NewFastLRUCache only) also answers "LRUCache"; nothing in this project
  // constructs one.
  if (std::strcmp(cache->Name(), LRUCache::kClassName()) != 0) {
    return Status::NotSupported(
        "per-shard pool stats are only available for LRUCache, not " +
        std::string(cache->Name()));
  }
  static_cast<LRUCache*>(cache)->GetShardPoolStats(out);
  return Status::OK();
}

}  // namespace ROCKSDB_NAMESPACE
