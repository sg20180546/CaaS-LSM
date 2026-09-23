//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once
#ifndef ROCKSDB_LITE

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "rocksdb/cache.h"
#include "rocksdb/env.h"
#include "rocksdb/file_system.h"
#include "rocksdb/io_status.h"
#include "rocksdb/memory_allocator.h"
#include "rocksdb/secondary_cache.h"
#include "rocksdb/table.h"
#include "rocksdb/table_properties.h"

namespace ROCKSDB_NAMESPACE {

// The classes and functions in this header file is used for dumping out the
// blocks in a block cache, storing or transfering the blocks to another
// destination host, and load these blocks to the secondary cache at destination
// host.
// NOTE that: The classes, functions, and data structures are EXPERIMENTAL! They
// my be changed in the future when the development continues.

// The major and minor version number of the data format to be stored/trandfered
// via CacheDumpWriter and read out via CacheDumpReader
static const int kCacheDumpMajorVersion = 0;
static const int kCacheDumpMinorVersion = 1;

// NOTE that: this class is EXPERIMENTAL! May be changed in the future!
// This is an abstract class to write or transfer the data that is created by
// CacheDumper. We pack one block with its block type, dump time, block key in
// the block cache, block len, block crc32c checksum and block itself as a unit
// and it is stored via WritePacket. Before we call WritePacket, we must call
// WriteMetadata once, which stores the sequence number, block unit checksum,
// and block unit size.
// We provide file based CacheDumpWriter to store the metadata and its package
// sequentially in a file as the defualt implementation. Users can implement
// their own CacheDumpWriter to store/transfer the data. For example, user can
// create a subclass which transfer the metadata and package on the fly.
class CacheDumpWriter {
 public:
  virtual ~CacheDumpWriter() = default;

  // Called ONCE before the calls to WritePacket
  virtual IOStatus WriteMetadata(const Slice& metadata) = 0;
  virtual IOStatus WritePacket(const Slice& data) = 0;
  virtual IOStatus Close() = 0;
};

// NOTE that: this class is EXPERIMENTAL! May be changed in the future!
// This is an abstract class to read or receive the data that is stored
// or transfered by CacheDumpWriter. Note that, ReadMetadata must be called
// once before we call a ReadPacket.
class CacheDumpReader {
 public:
  virtual ~CacheDumpReader() = default;
  // Called ONCE before the calls to ReadPacket
  virtual IOStatus ReadMetadata(std::string* metadata) = 0;
  // Sets data to empty string on EOF
  virtual IOStatus ReadPacket(std::string* data) = 0;
  // (Close not needed)
};

// CacheDumpOptions is the option for CacheDumper and CacheDumpedLoader. Any
// dump or load process related control variables can be added here.
struct CacheDumpOptions {
  SystemClock* clock;
};

// Options for the relink warmup stream. Unlike the stock cache dump format,
// the warmup stream is deliberately data-block-only and preserves the cache
// entry's effective LRU priority.
constexpr size_t kDefaultCacheWarmupMaxEntryBytes = 8 * 1024 * 1024;

struct CacheWarmupOptions {
  // Skip a source entry larger than this many bytes. Must be nonzero; this
  // bounds the one owned staging buffer held while copying an entry.
  size_t max_entry_bytes = kDefaultCacheWarmupMaxEntryBytes;

  // Aggregate limits for one versioned warmup stream. These default to
  // unlimited for API compatibility; network users should always provide
  // finite values. The payload budget counts decoded data-block bytes.
  uint64_t max_entries = std::numeric_limits<uint64_t>::max();
  size_t max_total_bytes = std::numeric_limits<size_t>::max();

  // Monotonic per-call processing deadline. Zero preserves the historical
  // unlimited API behavior. Network transports should additionally enforce
  // their absolute deadline around socket reads and writes.
  uint64_t max_transfer_duration_micros = 0;
};

// Per-priority counters. Bytes are data payload bytes, excluding the cache-dump
// frame headers and checksums, so they remain meaningful for arbitrary writers.
struct CacheWarmupPriorityTransferStats {
  uint64_t cataloged_entries = 0;
  uint64_t data_candidates = 0;
  uint64_t entries_staged = 0;
  uint64_t entries_written = 0;
  uint64_t entries_received = 0;
  uint64_t entries_inserted = 0;
  uint64_t entries_duplicate = 0;
  uint64_t entries_rejected_no_space = 0;
  uint64_t payload_bytes = 0;
};

// Best-effort accounting for one warmup dump or restore call. The dump side
// resets this structure before cataloging; the restore side resets it before
// reading the warmup header. Entries skipped because the concurrent cache
// changed are normal, not a stream failure.
struct CacheWarmupTransferStats {
  uint64_t cataloged_entries = 0;
  uint64_t data_candidates = 0;
  uint64_t entries_staged = 0;
  uint64_t entries_written = 0;
  uint64_t payload_bytes = 0;
  uint64_t skipped_disappeared = 0;
  uint64_t skipped_replaced = 0;
  uint64_t skipped_type_changed = 0;
  uint64_t skipped_unsupported = 0;
  uint64_t skipped_too_large = 0;
  uint64_t priority_changed_after_catalog = 0;

  uint64_t entries_received = 0;
  uint64_t entries_inserted = 0;
  uint64_t entries_duplicate = 0;
  uint64_t entries_rejected_no_space = 0;
  uint64_t skipped_invalid = 0;

  CacheWarmupPriorityTransferStats high;
  CacheWarmupPriorityTransferStats low;
  CacheWarmupPriorityTransferStats bottom;
};

// Transitional spelling retained for callers written against the first
// warmup vertical-slice drop.
using CacheWarmupStats = CacheWarmupTransferStats;

// [relink cache handoff, RDMA pull] One resident data block exposed for a
// destination-driven, one-sided pull. The streamed (TCP push) transport costs
// the SOURCE ~15 us of CPU per 4 KiB block (lookup + lease + copy + CRC + send)
// and tops out at 200-270 MB/s on a 100 GbE link, while the source is the
// node that is already overloaded. With a pull transport the source only
// catalogs {key, priority, payload address, length} and keeps a no-touch
// lease on each block until the destination has read the payload straight
// out of this process's heap (RDMA READ against an implicit-ODP region). No
// per-block copy, CRC or socket send happens on the source.
struct CacheWarmupPulledBlock {
  std::string key;          // exactly kCacheKeySize (16) bytes
  Cache::Priority priority;  // effective LRU class at catalog time
  const void* data;          // Block::data() of the cached value, valid
                             // while the lease is held by the catalog
  size_t size;               // Block::size(); what the dump path would write
};

// Owns the warmup leases taken by CacheDumper::CatalogWarmupDataBlocksForPull.
// Every block in blocks() stays resident (eviction skips a leased entry) and
// its `data` pointer stays valid until Release() runs. Release() is
// idempotent and the destructor calls it. The caller must Release() (or
// destroy the catalog) BEFORE closing the DB / dropping the last reference to
// the block cache: the implementation keeps the cache alive via shared_ptr,
// but a leased Block pinned across DB close would keep that memory resident
// and the LRU shard would report the pins as leaked handles.
class CacheWarmupPullCatalog {
 public:
  virtual ~CacheWarmupPullCatalog() = default;
  virtual const std::vector<CacheWarmupPulledBlock>& blocks() const = 0;
  virtual size_t payload_bytes() const = 0;
  virtual void Release() = 0;
};

// NOTE that: this class is EXPERIMENTAL! May be changed in the future!
// This the class to dump out the block in the block cache, store/transfer them
// via CacheDumpWriter. In order to dump out the blocks belonging to a certain
// DB or a list of DB (block cache can be shared by many DB), user needs to call
// SetDumpFilter to specify a list of DB to filter out the blocks that do not
// belong to those DB.
// A typical use case is: when we migrate a DB instance from host A to host B.
// We need to reopen the DB at host B after all the files are copied to host B.
// At this moment, the block cache at host B does not have any block from this
// migrated DB. Therefore, the read performance can be low due to cache warm up.
// By using CacheDumper before we shut down the DB at host A and using
// CacheDumpedLoader at host B before we reopen the DB, we can warmup the cache
// ahead. This function can be used in other use cases also.
class CacheDumper {
 public:
  virtual ~CacheDumper() = default;
  // Only dump the blocks in the block cache that belong to the DBs in this list
  virtual Status SetDumpFilter(std::vector<DB*> db_list) {
    (void)db_list;
    return Status::NotSupported("SetDumpFilter is not supported");
  }
  // [relink cache handoff] Like SetDumpFilter, but restrict the dump to the given
  // SST files of one DB (matched against GetPropertiesOfAllTables paths, full path
  // or basename). Lets a key-group migration ship ONLY the moved files' hot blocks.
  virtual Status SetDumpFilterFiles(DB* db,
                                    const std::vector<std::string>& sst_paths) {
    (void)db;
    (void)sst_paths;
    return Status::NotSupported("SetDumpFilterFiles is not supported");
  }
  // [relink cache handoff] Variant taking a PRE-CAPTURED properties collection.
  // Needed when the files have already been dropped from the dumper DB's version
  // by the time the dump runs (relink: the src unregisters the moved files at
  // cutover) — capture GetPropertiesOfAllTables BEFORE the drop, filter later.
  virtual Status SetDumpFilterFiles(const TablePropertiesCollection& ptc,
                                    const std::vector<std::string>& sst_paths) {
    (void)ptc;
    (void)sst_paths;
    return Status::NotSupported("SetDumpFilterFiles is not supported");
  }
  // Restricts a warmup dump using stable 8-byte block-cache key prefixes
  // already derived from in-memory Version metadata. This keeps block-cache
  // transfer independent from table-cache residency.
  virtual Status SetDumpFilterPrefixes(
      const std::vector<std::string>& prefixes) {
    (void)prefixes;
    return Status::NotSupported("SetDumpFilterPrefixes is not supported");
  }
  // The main function to dump out all the blocks that satisfy the filter
  // condition from block cache to a certain CacheDumpWriter in one shot. This
  // process may take some time.
  virtual IOStatus DumpCacheEntriesToWriter() {
    return IOStatus::NotSupported("DumpCacheEntriesToWriter is not supported");
  }
  // [relink cache handoff] A separately versioned, data-block-only stream for
  // asynchronous cache warmup. It first catalogs matching resident keys and
  // their effective priority, then pins and copies only one entry at a time
  // into an owned encoded record. It releases the entry before CRC, framing,
  // or writer I/O. This preserves the stock DumpCacheEntriesToWriter behavior
  // and avoids doing I/O under a cache shard lock.
  virtual IOStatus DumpWarmupCacheEntriesToWriter(
      const CacheWarmupOptions& warmup_options,
      CacheWarmupTransferStats* warmup_stats = nullptr) {
    (void)warmup_options;
    (void)warmup_stats;
    return IOStatus::NotSupported(
        "DumpWarmupCacheEntriesToWriter is not supported");
  }
  // [relink cache handoff, RDMA pull] Same selection as
  // DumpWarmupCacheEntriesToWriter (prefix filter from SetDumpFilterPrefixes,
  // kDataBlock entries only, HIGH -> LOW -> BOTTOM order, options.max_entries
  // / max_total_bytes / max_entry_bytes honoured, identical skipped_* and
  // per-class accounting), but instead of copying and writing each block it
  // takes ONE no-touch lease per block and records {key, priority,
  // Block::data(), Block::size()} in *catalog. The leases live until
  // catalog->Release(). No I/O, no copy, no cache-key re-derivation. A block
  // whose lease fails or whose effective class/charge/type changed since the
  // catalog pass is skipped exactly as the dump path skips it. stats->
  // entries_staged counts leased blocks; entries_written stays 0 because
  // nothing is written here (the transport reports what was pulled).
  virtual Status CatalogWarmupDataBlocksForPull(
      const CacheWarmupOptions& options,
      std::unique_ptr<CacheWarmupPullCatalog>* catalog,
      CacheWarmupTransferStats* stats) {
    (void)options;
    (void)stats;
    if (catalog != nullptr) {
      catalog->reset();
    }
    return Status::NotSupported(
        "CatalogWarmupDataBlocksForPull is not supported");
  }
  virtual const CacheWarmupTransferStats& GetCacheWarmupTransferStats() const {
    static const CacheWarmupTransferStats kEmptyStats;
    return kEmptyStats;
  }
};

// NOTE that: this class is EXPERIMENTAL! May be changed in the future!
// This is the class to load the dumped blocks to the destination cache. For now
// we only load the blocks to the SecondaryCache. In the future, we may plan to
// support loading to the block cache.
class CacheDumpedLoader {
 public:
  virtual ~CacheDumpedLoader() = default;
  virtual IOStatus RestoreCacheEntriesToSecondaryCache() {
    return IOStatus::NotSupported(
        "RestoreCacheEntriesToSecondaryCache is not supported");
  }
  // [relink cache handoff] Insert the dumped blocks directly into a PRIMARY block
  // cache. Valid when the receiving DB reads the SAME physical files as the dumper
  // (cache keys are derived from file-embedded ids, so they match across processes)
  // — exactly the relink situation. The stock loader only targets SecondaryCache.
  virtual IOStatus RestoreCacheEntriesToPrimaryCache() {
    return IOStatus::NotSupported(
        "RestoreCacheEntriesToPrimaryCache is not supported");
  }
  // [relink cache handoff] Restore the separately versioned warmup stream to
  // a primary cache. The wire priority is used for admission. A duplicate key
  // or no suitable lower-priority space is a normal per-entry skip, not a
  // stream error.
  virtual IOStatus RestoreWarmupCacheEntriesToPrimaryCache(
      const CacheWarmupOptions& warmup_options,
      CacheWarmupTransferStats* warmup_stats = nullptr) {
    (void)warmup_options;
    (void)warmup_stats;
    return IOStatus::NotSupported(
        "RestoreWarmupCacheEntriesToPrimaryCache is not supported");
  }
  IOStatus RestoreWarmupCacheEntriesToPrimaryCache(
      CacheWarmupTransferStats* warmup_stats = nullptr) {
    return RestoreWarmupCacheEntriesToPrimaryCache(CacheWarmupOptions{},
                                                   warmup_stats);
  }
  // [relink cache handoff, RDMA pull] Exactly one unit of
  // RestoreWarmupCacheEntriesToPrimaryCache, for a payload that arrived by
  // some transport other than the framed stream (e.g. an RDMA READ into a
  // pinned receive buffer): validate the 16-byte key and the default
  // CacheWarmupOptions size limits, copy `size` bytes from `data` into a
  // fresh CacheAllocationPtr, build the Block, and admit it with the
  // priority-aware InsertForCacheWarmup. Counts entries_received / inserted /
  // duplicate / rejected_no_space / payload_bytes per class in *stats (or in
  // the loader's own stats when null) exactly like the streamed path, which
  // is implemented on top of this same function so both transports admit
  // identically. A non-OK return means the unit was invalid (skipped_invalid
  // / skipped_too_large already bumped); the caller decides whether to go on.
  //
  // Thread safety (2026-09-23, both InsertWarmupDataBlock entry points): the
  // admission routine behind them touches only the primary cache (whose
  // InsertForCacheWarmup takes the per-shard mutex), the const table options,
  // a function-local static cache helper, and *stats. It is therefore safe to
  // call concurrently from several threads PROVIDED each thread passes its own
  // non-null `stats`; the recommended shape is one loader instance per thread
  // (NewDefaultCacheDumpedLoaderToPrimary is one small heap allocation) with
  // its own CacheWarmupTransferStats. A null `stats` falls back to the
  // loader's own counters, which is the one case that must NOT be shared
  // across threads.
  virtual IOStatus InsertWarmupDataBlock(const Slice& key,
                                         Cache::Priority priority,
                                         const char* data, size_t size,
                                         CacheWarmupTransferStats* stats) {
    (void)key;
    (void)priority;
    (void)data;
    (void)size;
    (void)stats;
    return IOStatus::NotSupported("InsertWarmupDataBlock is not supported");
  }
  // [relink cache handoff, zero-copy landing, 2026-09-23] Same checks,
  // counters and admission outcome as InsertWarmupDataBlock, but the caller
  // supplies the already-filled buffer (e.g. an arena slot an RDMA READ or
  // WRITE landed in) and no copy is made. On kInserted the cache owns `buf`.
  // On kDuplicate / kRejectedNoSpace / skipped_unsupported / any non-OK
  // return, `buf` has been released through its deleter (the allocator it
  // came from) exactly once BEFORE this returns; the caller's CacheAllocationPtr
  // is null afterwards in every case. Thread safety: as InsertWarmupDataBlock.
  virtual IOStatus InsertWarmupDataBlockOwned(const Slice& key,
                                              Cache::Priority priority,
                                              CacheAllocationPtr&& buf,
                                              size_t size,
                                              CacheWarmupTransferStats* stats) {
    (void)key;
    (void)priority;
    (void)size;
    (void)stats;
    buf.reset();  // honour the release-before-return contract
    return IOStatus::NotSupported(
        "InsertWarmupDataBlockOwned is not supported");
  }
  // [relink cache handoff, 2026-09-23] Optional hook on the streamed restore
  // (RestoreWarmupCacheEntriesToPrimaryCache). When a sink is set, each data
  // unit read from the reader is validated against the per-unit and aggregate
  // limits in the same order as the inline path (key size, max_entry_bytes,
  // max_entries, max_total_bytes, null payload), then its payload is copied
  // into a buffer from AllocateBlock(size, allocator) (allocator may be null
  // = new char[]) and handed to the sink INSTEAD of being admitted inline.
  // The sink owns the buffer and normally admits it later, from any thread,
  // via InsertWarmupDataBlockOwned on its own loader; a non-OK sink return
  // ends the stream with that status.
  // Accounting in sink mode: the stats returned by the streamed call carry
  // only the reader-side gate outcomes (skipped_invalid / skipped_too_large;
  // the aggregate caps are enforced on reader-local counters), NOT
  // entries_received / inserted / duplicate / rejected_no_space /
  // payload_bytes — those are produced by whoever performs the owned inserts,
  // so a caller adds loader stats + inserter stats without double counting.
  // No sink set: byte-identical to the historical inline behaviour.
  using WarmupUnitSink = std::function<IOStatus(
      const Slice& key, Cache::Priority priority, CacheAllocationPtr&& buf,
      size_t size)>;
  virtual void SetWarmupUnitSink(WarmupUnitSink sink,
                                 MemoryAllocator* allocator) {
    (void)sink;
    (void)allocator;
  }
  virtual const CacheWarmupTransferStats& GetCacheWarmupTransferStats() const {
    static const CacheWarmupTransferStats kEmptyStats;
    return kEmptyStats;
  }
};

// Get the writer which stores all the metadata and data sequentially to a file
IOStatus NewToFileCacheDumpWriter(const std::shared_ptr<FileSystem>& fs,
                                  const FileOptions& file_opts,
                                  const std::string& file_name,
                                  std::unique_ptr<CacheDumpWriter>* writer);

// Get the reader which read out the metadata and data sequentially from a file
IOStatus NewFromFileCacheDumpReader(const std::shared_ptr<FileSystem>& fs,
                                    const FileOptions& file_opts,
                                    const std::string& file_name,
                                    std::unique_ptr<CacheDumpReader>* reader);

// Get the default cache dumper
Status NewDefaultCacheDumper(const CacheDumpOptions& dump_options,
                             const std::shared_ptr<Cache>& cache,
                             std::unique_ptr<CacheDumpWriter>&& writer,
                             std::unique_ptr<CacheDumper>* cache_dumper);

// [relink cache handoff] loader variant that installs into a primary block cache
// (see CacheDumpedLoader::RestoreCacheEntriesToPrimaryCache). toptions must outlive
// the loader (it is held by reference).
Status NewDefaultCacheDumpedLoaderToPrimary(
    const CacheDumpOptions& dump_options, const BlockBasedTableOptions& toptions,
    const std::shared_ptr<Cache>& primary_cache,
    std::unique_ptr<CacheDumpReader>&& reader,
    std::unique_ptr<CacheDumpedLoader>* cache_dump_loader);

// Get the default cache dump loader
Status NewDefaultCacheDumpedLoader(
    const CacheDumpOptions& dump_options,
    const BlockBasedTableOptions& toptions,
    const std::shared_ptr<SecondaryCache>& secondary_cache,
    std::unique_ptr<CacheDumpReader>&& reader,
    std::unique_ptr<CacheDumpedLoader>* cache_dump_loader);

}  // namespace ROCKSDB_NAMESPACE
#endif  // ROCKSDB_LITE
