//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [sst access stats 2026-10-04] Per-SST foreground block-cache access counters,
// read without touching any Cache / DB vtable (free functions only, so binaries
// linked against an older librocksdb keep working).
//
// Why: the relink block-cache transfer has to decide, per migrated SST, whether
// the blocks the source holds in its block cache are worth more at the
// destination than the destination's own resident blocks. A block's LRU class
// (HIGH / BOTTOM) is a rank inside ONE node's cache and cannot be compared
// across nodes. A hit RATE can: each BlockBasedTable counts the foreground
// (non-compaction) data-block lookups it issues, split into block-cache hits
// and misses. hits / seconds-open / resident-bytes-of-the-file is then the mean
// reference rate of one resident byte of that file -- the same unit on both
// nodes. RocksDB already keeps a sampled per-SST read counter
// (FileMetaData::stats.num_reads_sampled, 1/1024 sampling, statistics only);
// these counters are exact, per open TableReader, and split hit/miss.
//
// Counting is OFF by default and switched on process-wide with
// SetSstAccessStatsCounting(true): exact per-file counters make every reader
// thread of a hot SST increment one shared cache line (RocksDB's own per-file
// read counter is sampled 1/1024 for that reason), which could change the
// throughput of a process that never reads them. Off, a lookup pays one relaxed
// load of a read-mostly flag; on, one relaxed increment of its reader's
// counter. Neither changes what the DB does.
//
// Exactness: hits are exact for synchronous Get, MultiGet and iterators.
// Misses are exact for synchronous Get and iterators. Known gaps: an async_io
// iterator can count one miss twice (the lookup is repeated after TryAgain), a
// MultiGet with fill_cache=false (non-mmap) does not count its misses, a
// secondary-cache handle that is still pending counts as a hit, and
// paranoid_file_checks verification reads and BlockBasedTable::Prefetch count
// as foreground. The relink block-cache transfer ranks files by hits only; the
// table-cache install (DBImpl::InstallExternalTableCacheEntries rate mode)
// ranks readers by hits + misses, so the miss gaps can shift a reader's rate
// there (none of them occurs in the driver's Get / iterator workloads).

#pragma once

#include <cstdint>
#include <vector>

#include "rocksdb/cache.h"
#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {

class ColumnFamilyHandle;
class DB;

struct SstAccessStats {
  // SST file number (the number in the file name / MANIFEST).
  uint64_t file_number = 0;
  // OffsetableCacheKey::CommonPrefixSlice() of this file: the first 8 bytes of
  // every block-cache key of this file. Lets a caller map a cached block (key
  // bytes [0,8)) to its file without consulting the MANIFEST.
  char cache_key_prefix[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // Foreground (not for_compaction) data-block lookups served by the block
  // cache since the TableReader was opened.
  uint64_t fg_data_block_hits = 0;
  // Foreground data-block lookups that missed the block cache (the block was
  // read from the file) since the TableReader was opened.
  uint64_t fg_data_block_misses = 0;
  // ioptions.clock->NowMicros() when the TableReader was opened.
  uint64_t open_time_micros = 0;
};

// Stats of every TableReader currently open in the DB's table cache.
// `column_family` (nullptr = the default one) only selects the DB: the table
// cache is shared by all column families, so the result covers every column
// family's open readers. A file whose reader was evicted from the table cache
// has no entry: it has not been read since, and a block-cache lookup needs an
// open reader, so "absent" means "no recent foreground block traffic". Only
// BlockBasedTable readers report; other entries (blob file readers share the
// cache) are skipped. Non-virtual on purpose (no ABI change).
Status GetSstAccessStats(DB* db, ColumnFamilyHandle* column_family,
                         std::vector<SstAccessStats>* out);

// Process-wide switch for the counters above (default off, see the top of this
// file). Readers opened while it is off keep counting from zero once it is
// switched on; open_time_micros is always recorded.
void SetSstAccessStatsCounting(bool on);
bool SstAccessStatsCounting();

// The clock used for SstAccessStats::open_time_micros, read now.
uint64_t SstAccessStatsNowMicros(DB* db);

// [block hit count 2026-10-06] Process-wide switch for the per-ENTRY counters
// of every LRUCache (default off): with it on, each cache entry records the
// lookups that hit it since it was inserted and the engine clock at that
// insert (one plain increment under the shard mutex the lookup already holds;
// one clock read per insert). A warm-up probe (LookupForCacheWarmup) and a
// warm-up insert's own admission do not count. The values are exposed per
// entry by the warm-up traversal (Cache::ApplyToAllEntriesForCacheWarmup) and
// per cataloged block by CacheDumper::CatalogWarmupDataBlocksForPull
// (CacheWarmupPulledBlock::hits / insert_time_us). Why: the relink cache
// hand-off's per-block selection compares a block's hits over its time in the
// source cache against the destination's own hit density. Entries inserted
// while the switch was off report insert_time 0 (hits do count once it is on).
void SetBlockCacheHitCounting(bool on);
bool BlockCacheHitCounting();

// [recency 2026-10-06] One linked block-cache entry as the LRU sees it.
struct LRUCacheEntryRecency {
  uint64_t last_ref_us = 0;  // engine clock at its last counted hit (its insert
                             // until then; 0 if inserted while counting was off)
  size_t charge = 0;         // what the cache charges for it
  uint8_t pool = 2;          // Cache::Priority of the pool it is LINKED in: 0 HIGH, 1 LOW, 2 BOTTOM
                             // (the pools are contiguous segments of the list, in that order
                             // from the oldest end)
  bool evictable = false;    // linked and unpinned: a possible victim
};

// Every linked entry of every shard of an LRUCache, one vector per shard
// (index = Cache::GetCacheWarmupShardIndex of the keys mapping to it), each in
// the order the cache evicts: bottom pool oldest first, then low, then high.
// Each shard is read under its own mutex. NotSupported for other caches.
// Why: the relink cache hand-off's per-block recency selection
// (MIG_CACHE_SELECT=age) simulates the merge of the incoming blocks into this
// LRU order to decide which blocks to ask for and how many bytes they displace.
Status GetLRUCacheShardEntryRecency(
    Cache* cache, std::vector<std::vector<LRUCacheEntryRecency>>* out);

// InsertForCacheWarmup with the victims widened by recency: an entry may be
// evicted for the incoming block if its class is strictly lower (as
// InsertForCacheWarmup) or its class is the incoming's and it was last
// referenced before `last_ref_us` (unknown recency = never); victims are taken
// from the LRU end, oldest position first, through the pools at or below the
// incoming's. All-or-nothing; the inserted entry carries `last_ref_us` but
// links at its pool's newest position (its arrival counts as a reference from
// then on). Needs SetBlockCacheHitCounting(true). Ownership as
// InsertForCacheWarmup. NotSupported for other caches. Non-virtual on purpose
// (no Cache vtable change).
Status LRUCacheInsertForCacheWarmupByRecency(
    Cache* cache, const Slice& key, void* value, size_t charge,
    Cache::DeleterFn deleter, Cache::Priority priority, uint64_t last_ref_us,
    Cache::CacheWarmupInsertResult* result);

// [sst access stats 2026-10-05] A lookup rate the caller measured for one of
// this DB's resident TableReaders (file_number = the file it reads).
struct SstReaderRate {
  uint64_t file_number = 0;
  double lookups_per_sec = 0.0;
};

// Rates for the NEXT InstallExternalTableCacheEntries call on this DB: in rate
// mode that call ranks a resident reader by the rate given here instead of by
// its lifetime counters ((lookups - 1 + 0.5) / age), so a caller that keeps a
// windowed history of the counters (e.g. hits over the last block-cache
// turnover, the same window it measures the incoming files over) can make the
// two sides comparable. Readers not listed keep the lifetime estimate. The
// list is consumed (cleared) by that call, in any mode; calling again
// replaces an unconsumed list.
Status SetResidentReaderRatesForWarmup(DB* db, std::vector<SstReaderRate> rates);

// Per-shard LRU pool accounting of an LRUCache, the facts a caller needs to
// decide admission without walking any LRU list:
//   capacity            shard capacity in bytes
//   usage               bytes of all entries (in the LRU list or referenced)
//   lru_usage           bytes of entries linked in the LRU list (unreferenced,
//                       or pinned by a warm-up lease: see warmup_pinned_usage)
//   high_pri_pool_usage bytes of LRU-list entries in the high-priority pool
//   low_pri_pool_usage  bytes of LRU-list entries in the low-priority pool
//   warmup_pinned_usage bytes of LRU-list entries pinned by a cache-warmup
//                       lease (linked, but not evictable while pinned)
// So the bytes a HIGH-priority warmup insert may evict are at least
//   lru_usage - high_pri_pool_usage - warmup_pinned_usage
// and a BOTTOM-priority warmup insert may evict nothing (free space only).
struct LRUCacheShardPoolStats {
  size_t capacity = 0;
  size_t usage = 0;
  size_t lru_usage = 0;
  size_t high_pri_pool_usage = 0;
  size_t low_pri_pool_usage = 0;
  size_t warmup_pinned_usage = 0;
};

// One entry per shard, shard index = the index Cache::GetCacheWarmupShardIndex
// reports for a key. NotSupported when `cache` is not an LRUCache
// (cache->Name() != "LRUCache"). Each shard's numbers are read under that
// shard's mutex (consistent per shard, not across shards).
Status GetLRUCacheShardPoolStats(Cache* cache,
                                 std::vector<LRUCacheShardPoolStats>* out);

}  // namespace ROCKSDB_NAMESPACE
