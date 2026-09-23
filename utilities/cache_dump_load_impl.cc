//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/cache_key.h"
#include "table/block_based/block_based_table_reader.h"
#ifndef ROCKSDB_LITE

#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <unordered_set>

#include "cache/cache_entry_roles.h"
#include "file/writable_file_writer.h"
#include "port/lang.h"
#include "rocksdb/env.h"
#include "rocksdb/file_system.h"
#include "rocksdb/utilities/ldb_cmd.h"
#include "table/format.h"
#include "util/crc32c.h"
#include "utilities/cache_dump_load_impl.h"

namespace ROCKSDB_NAMESPACE {

namespace {

static_assert(OffsetableCacheKey::kCommonPrefixSize == sizeof(uint64_t),
              "warmup prefix filter assumes the stable cache prefix is u64");

uint64_t CacheWarmupPrefixAsUint64(const Slice& key) {
  uint64_t prefix = 0;
  std::memcpy(&prefix, key.data(), sizeof(prefix));
  return prefix;
}

CacheWarmupPriorityTransferStats* CacheWarmupStatsForPriority(
    CacheWarmupTransferStats* stats, Cache::Priority priority) {
  assert(stats != nullptr);
  switch (priority) {
    case Cache::Priority::HIGH:
      return &stats->high;
    case Cache::Priority::LOW:
      return &stats->low;
    case Cache::Priority::BOTTOM:
      return &stats->bottom;
  }
  return &stats->low;  // Defensive fallback for a future invalid enum value.
}

// A source entry must never remain warmup-pinned if staging throws. This small
// guard also keeps every validation/early-return path visibly no-touch.
class ScopedWarmupLease {
 public:
  ScopedWarmupLease(Cache* cache, Cache::Handle* handle,
                    Cache::Priority priority)
      : cache_(cache), handle_(handle), priority_(priority) {}
  ~ScopedWarmupLease() { Release(); }

  ScopedWarmupLease(const ScopedWarmupLease&) = delete;
  ScopedWarmupLease& operator=(const ScopedWarmupLease&) = delete;

  void Release() {
    if (handle_ != nullptr) {
      cache_->ReleaseForCacheWarmup(handle_, priority_);
      handle_ = nullptr;
    }
  }

  // Give the lease up without releasing it; the caller now owns the handle
  // (used to move a lease into a CacheWarmupPullCatalog).
  void Detach() { handle_ = nullptr; }

 private:
  Cache* cache_;
  Cache::Handle* handle_;
  Cache::Priority priority_;
};

}  // namespace

// Set the dump filter with a list of DBs. Block cache may be shared by multipe
// DBs and we may only want to dump out the blocks belonging to certain DB(s).
// Therefore, a filter is need to decide if the key of the block satisfy the
// requirement.
Status CacheDumperImpl::SetDumpFilter(std::vector<DB*> db_list) {
  Status s = Status::OK();
  for (size_t i = 0; i < db_list.size(); i++) {
    assert(i < db_list.size());
    TablePropertiesCollection ptc;
    assert(db_list[i] != nullptr);
    s = db_list[i]->GetPropertiesOfAllTables(&ptc);
    if (!s.ok()) {
      return s;
    }
    for (auto id = ptc.begin(); id != ptc.end(); id++) {
      OffsetableCacheKey base;
      // We only want to save cache entries that are portable to another
      // DB::Open, so only save entries with stable keys.
      bool is_stable;
      BlockBasedTable::SetupBaseCacheKey(id->second.get(),
                                         /*cur_db_session_id*/ "",
                                         /*cur_file_num*/ 0, &base, &is_stable);
      if (is_stable) {
        Slice prefix_slice = base.CommonPrefixSlice();
        assert(prefix_slice.size() == OffsetableCacheKey::kCommonPrefixSize);
        prefix_filter_.insert(prefix_slice.ToString());
      }
    }
  }
  return s;
}

// [relink cache handoff] Restrict the dump to specific SST files of one DB.
// Matching is by full path first, then basename — relink external paths can
// carry a different directory prefix than the properties-collection keys.
Status CacheDumperImpl::SetDumpFilterFiles(
    DB* db, const std::vector<std::string>& sst_paths) {
  if (db == nullptr) {
    return Status::InvalidArgument("db is null");
  }
  TablePropertiesCollection ptc;
  Status s = db->GetPropertiesOfAllTables(&ptc);
  if (!s.ok()) {
    return s;
  }
  return SetDumpFilterFiles(ptc, sst_paths);
}

Status CacheDumperImpl::SetDumpFilterFiles(
    const TablePropertiesCollection& ptc,
    const std::vector<std::string>& sst_paths) {
  auto base_name = [](const std::string& p) {
    size_t q = p.find_last_of('/');
    return q == std::string::npos ? p : p.substr(q + 1);
  };
  std::set<std::string> want_full(sst_paths.begin(), sst_paths.end());
  std::set<std::string> want_base;
  for (const auto& p : sst_paths) {
    want_base.insert(base_name(p));
  }
  size_t matched = 0;
  for (auto id = ptc.begin(); id != ptc.end(); id++) {
    if (want_full.find(id->first) == want_full.end() &&
        want_base.find(base_name(id->first)) == want_base.end()) {
      continue;
    }
    OffsetableCacheKey base;
    bool is_stable;
    BlockBasedTable::SetupBaseCacheKey(id->second.get(),
                                       /*cur_db_session_id*/ "",
                                       /*cur_file_num*/ 0, &base, &is_stable);
    if (is_stable) {
      prefix_filter_.insert(base.CommonPrefixSlice().ToString());
      matched++;
    }
  }
  return matched > 0 ? Status::OK()
                     : Status::NotFound("no matching tables for dump filter");
}

Status CacheDumperImpl::SetDumpFilterPrefixes(
    const std::vector<std::string>& prefixes) {
  size_t accepted = 0;
  for (const auto& prefix : prefixes) {
    if (prefix.size() != OffsetableCacheKey::kCommonPrefixSize) {
      return Status::InvalidArgument(
          "block-cache warmup prefix must be exactly 8 bytes");
    }
    accepted += prefix_filter_.insert(prefix).second ? 1 : 0;
  }
  return accepted > 0
             ? Status::OK()
             : Status::NotFound("no stable block-cache prefixes supplied");
}

// This is the main function to dump out the cache block entries to the writer.
// The writer may create a file or write to other systems. Currently, we will
// iterate the whole block cache, get the blocks, and write them to the writer
IOStatus CacheDumperImpl::DumpCacheEntriesToWriter() {
  // Prepare stage, check the parameters.
  if (cache_ == nullptr) {
    return IOStatus::InvalidArgument("Cache is null");
  }
  if (writer_ == nullptr) {
    return IOStatus::InvalidArgument("CacheDumpWriter is null");
  }
  // Set the system clock
  if (options_.clock == nullptr) {
    return IOStatus::InvalidArgument("System clock is null");
  }
  clock_ = options_.clock;
  // We copy the Cache Deleter Role Map as its member.
  role_map_ = CopyCacheDeleterRoleMap();
  // Set the sequence number
  sequence_num_ = 0;

  // Dump stage, first, we write the hader
  IOStatus io_s = WriteHeader();
  if (!io_s.ok()) {
    return io_s;
  }

  // Then, we iterate the block cache and dump out the blocks that are not
  // filtered out.
  cache_->ApplyToAllEntries(DumpOneBlockCallBack(), {});

  // Finally, write the footer
  io_s = WriteFooter();
  if (!io_s.ok()) {
    return io_s;
  }
  io_s = writer_->Close();
  return io_s;
}

IOStatus CacheDumperImpl::ValidateWarmupOptions(
    const CacheWarmupOptions& warmup_options) {
  if (cache_ == nullptr) {
    return IOStatus::InvalidArgument("Cache is null");
  }
  if (warmup_options.max_entry_bytes == 0) {
    return IOStatus::InvalidArgument(
        "cache warmup max_entry_bytes must be nonzero");
  }
  if (warmup_options.max_entries == 0 ||
      warmup_options.max_total_bytes == 0) {
    return IOStatus::InvalidArgument(
        "cache warmup aggregate limits must be nonzero");
  }
  return IOStatus::OK();
}

// Phase one of both warmup transports: catalog data-entry metadata while a
// cache shard lock is held. Nothing is pinned or copied here.
IOStatus CacheDumperImpl::CatalogWarmupCandidates(
    const WarmupCallDeadline& deadline,
    std::vector<CacheWarmupCandidate>* high_candidates,
    std::vector<CacheWarmupCandidate>* low_candidates,
    std::vector<CacheWarmupCandidate>* bottom_candidates) {
  // Materialize the stable 8-byte prefixes once. The catalog callback does no
  // std::string allocation or tree lookup to decide whether an entry belongs
  // to the migrated file set.
  std::unordered_set<uint64_t> warmup_prefix_filter;
  warmup_prefix_filter.reserve(prefix_filter_.size());
  for (const auto& prefix : prefix_filter_) {
    if (prefix.size() == OffsetableCacheKey::kCommonPrefixSize) {
      warmup_prefix_filter.insert(CacheWarmupPrefixAsUint64(Slice(prefix)));
    }
  }

  // Bucketing during catalog is O(N), preserves catalog order within each
  // class, and gives the wire stream its required HIGH -> LOW -> BOTTOM order
  // without an O(N log N) sort.
  auto candidate_bucket =
      [&](Cache::Priority priority) -> std::vector<CacheWarmupCandidate>* {
    switch (priority) {
      case Cache::Priority::HIGH:
        return high_candidates;
      case Cache::Priority::LOW:
        return low_candidates;
      case Cache::Priority::BOTTOM:
        return bottom_candidates;
    }
    return nullptr;
  };

  Cache::ApplyToAllEntriesOptions catalog_options;
  // Iterate a moderate number of hash buckets per lock acquisition. The
  // callback only does fixed-key metadata work; batching avoids one mutex
  // round trip per bucket on large caches without creating long lock holds.
  catalog_options.average_entries_per_lock = 64;
  bool catalog_deadline_expired = false;
  Status catalog_status = cache_->ApplyToAllEntriesForCacheWarmup(
      [&](const Slice& key, size_t charge, Cache::DeleterFn deleter,
          Cache::Priority effective_priority) {
        if (deadline.Expired()) {
          catalog_deadline_expired = true;
          return;
        }
        if (key.size() < OffsetableCacheKey::kCommonPrefixSize ||
            warmup_prefix_filter.find(CacheWarmupPrefixAsUint64(key)) ==
                warmup_prefix_filter.end()) {
          return;
        }
        if (key.size() != kCacheKeySize) {
          ++warmup_stats_.skipped_unsupported;
          return;
        }
        const auto role_it = role_map_.find(deleter);
        if (role_it == role_map_.end()) {
          ++warmup_stats_.skipped_unsupported;
          return;
        }
        if (role_it->second != CacheEntryRole::kDataBlock) {
          // We intentionally do not copy a full key for index/filter/etc.
          ++warmup_stats_.skipped_unsupported;
          return;
        }
        std::vector<CacheWarmupCandidate>* bucket =
            candidate_bucket(effective_priority);
        if (bucket == nullptr) {
          ++warmup_stats_.skipped_unsupported;
          return;
        }
        CacheWarmupPriorityTransferStats* priority_stats =
            CacheWarmupStatsForPriority(&warmup_stats_, effective_priority);
        ++warmup_stats_.cataloged_entries;
        ++warmup_stats_.data_candidates;
        ++priority_stats->cataloged_entries;
        ++priority_stats->data_candidates;
        CacheWarmupCandidate candidate{};
        std::memcpy(candidate.key.data(), key.data(), kCacheKeySize);
        candidate.charge = charge;
        candidate.priority = effective_priority;
        bucket->push_back(candidate);
      },
      catalog_options);
  if (!catalog_status.ok()) {
    return status_to_io_status(std::move(catalog_status));
  }
  if (catalog_deadline_expired || deadline.Expired()) {
    return IOStatus::TimedOut("cache warmup catalog deadline");
  }
  return IOStatus::OK();
}

// Phase two, per candidate: lease and re-validate. The pull transport keeps
// the returned lease for the whole transfer; the streamed transport releases
// it right after its one copy. Both apply exactly these rechecks and limits.
IOStatus CacheDumperImpl::LeaseWarmupDataBlock(
    const CacheWarmupCandidate& candidate,
    const CacheWarmupOptions& warmup_options, Cache::Handle** handle,
    Cache::Priority* effective_priority, const char** data, size_t* size) {
  assert(handle != nullptr);
  assert(effective_priority != nullptr);
  assert(data != nullptr);
  assert(size != nullptr);
  *handle = nullptr;
  *data = nullptr;
  *size = 0;

  Cache::Handle* leased = nullptr;
  Cache::Priority current_priority = candidate.priority;
  Status lookup_status = cache_->LookupForCacheWarmup(
      Slice(candidate.key.data(), candidate.key.size()), &leased,
      &current_priority);
  if (!lookup_status.ok()) {
    return status_to_io_status(std::move(lookup_status));
  }
  if (leased == nullptr) {
    ++warmup_stats_.skipped_disappeared;
    return IOStatus::OK();
  }
  ScopedWarmupLease lease(cache_.get(), leased, current_priority);

  // Every skip below releases this one pinned handle via the scoped lease.
  const size_t current_charge = cache_->GetCharge(leased);
  const Cache::DeleterFn current_deleter = cache_->GetDeleter(leased);
  void* const current_value = cache_->Value(leased);
  if (current_priority != candidate.priority) {
    ++warmup_stats_.priority_changed_after_catalog;
    // Keeping the catalog class here would mislabel the entry, while using
    // the new class could violate the stream's HIGH -> LOW -> BOTTOM order.
    // Skip this concurrently changed entry and preserve both invariants.
    return IOStatus::OK();
  }
  if (current_charge != candidate.charge) {
    ++warmup_stats_.skipped_replaced;
    return IOStatus::OK();
  }
  const auto role_it = role_map_.find(current_deleter);
  if (role_it == role_map_.end()) {
    ++warmup_stats_.skipped_unsupported;
    return IOStatus::OK();
  }
  if (role_it->second != CacheEntryRole::kDataBlock) {
    ++warmup_stats_.skipped_type_changed;
    return IOStatus::OK();
  }
  if (current_value == nullptr) {
    ++warmup_stats_.skipped_unsupported;
    return IOStatus::OK();
  }

  const Block* block = static_cast<const Block*>(current_value);
  const char* const block_data = block->data();
  const size_t block_size = block->size();
  if (block_size > warmup_options.max_entry_bytes ||
      block_size > std::numeric_limits<uint32_t>::max()) {
    ++warmup_stats_.skipped_too_large;
    return IOStatus::OK();
  }
  if (block_size != 0 && block_data == nullptr) {
    ++warmup_stats_.skipped_unsupported;
    return IOStatus::OK();
  }
  if (warmup_stats_.payload_bytes > warmup_options.max_total_bytes ||
      block_size > warmup_options.max_total_bytes -
                       static_cast<size_t>(warmup_stats_.payload_bytes)) {
    ++warmup_stats_.skipped_too_large;
    return IOStatus::OK();
  }

  // Hand the lease to the caller. Detach it from the scoped guard by
  // reconstructing: the guard's Release() is a no-op once handle_ is null,
  // so transfer ownership explicitly instead of letting the guard fire.
  lease.Detach();
  *handle = leased;
  *effective_priority = current_priority;
  *data = block_data;
  *size = block_size;
  return IOStatus::OK();
}

// A warmup dump has a deliberately separate two-phase flow from the stock
// dump. The first phase only catalogs data-entry metadata while a cache shard
// lock is held. The second phase pins one entry, copies it directly into one
// owned encoded record, releases it, and only then performs CRC/framing/writer
// I/O. In particular, no network writer call happens under a cache shard lock
// or while a cache entry is pinned.
IOStatus CacheDumperImpl::DumpWarmupCacheEntriesToWriter(
    const CacheWarmupOptions& warmup_options,
    CacheWarmupTransferStats* warmup_stats) {
  warmup_stats_ = CacheWarmupTransferStats{};
  auto finish = [&](IOStatus status) {
    if (warmup_stats != nullptr) {
      *warmup_stats = warmup_stats_;
    }
    return status;
  };
  IOStatus io_s = ValidateWarmupOptions(warmup_options);
  if (!io_s.ok()) {
    return finish(io_s);
  }
  if (writer_ == nullptr) {
    return finish(IOStatus::InvalidArgument("CacheDumpWriter is null"));
  }
  WarmupCallDeadline deadline(warmup_options.max_transfer_duration_micros);
  role_map_ = CopyCacheDeleterRoleMap();
  sequence_num_ = 0;

  std::vector<CacheWarmupCandidate> high_candidates;
  std::vector<CacheWarmupCandidate> low_candidates;
  std::vector<CacheWarmupCandidate> bottom_candidates;
  io_s = CatalogWarmupCandidates(deadline, &high_candidates, &low_candidates,
                                 &bottom_candidates);
  if (!io_s.ok()) {
    return finish(io_s);
  }

  io_s = WriteWarmupHeader();
  if (!io_s.ok()) {
    return finish(io_s);
  }
  const std::vector<CacheWarmupCandidate>* const candidate_buckets[] = {
      &high_candidates, &low_candidates, &bottom_candidates};
  bool aggregate_limit_reached = false;
  for (const auto* candidates : candidate_buckets) {
    for (const auto& candidate : *candidates) {
      if (deadline.Expired()) {
        return finish(IOStatus::TimedOut("cache warmup dump deadline"));
      }
      if (warmup_stats_.entries_written >= warmup_options.max_entries) {
        aggregate_limit_reached = true;
        break;
      }
      Cache::Handle* handle = nullptr;
      Cache::Priority current_priority = candidate.priority;
      const char* block_data = nullptr;
      size_t block_size = 0;
      io_s = LeaseWarmupDataBlock(candidate, warmup_options, &handle,
                                  &current_priority, &block_data, &block_size);
      if (!io_s.ok()) {
        return finish(io_s);
      }
      if (handle == nullptr) {
        continue;  // skipped; counters already updated
      }
      ScopedWarmupLease lease(cache_.get(), handle, current_priority);

      // Encode directly into the reusable owned staging slab while this one
      // cache handle is pinned. This is the only payload copy; CRC and all
      // writer framing happen only after the release below.
      CacheWarmupDumpUnit staged_unit;
      staged_unit.type = CacheWarmupDumpUnitType::kData;
      staged_unit.priority = candidate.priority;
      staged_unit.key = Slice(candidate.key.data(), candidate.key.size());
      staged_unit.value_len = block_size;
      staged_unit.value = const_cast<char*>(block_size == 0 ? "" : block_data);
      warmup_encoded_data_.clear();
      Status staging_status = CacheDumperHelper::EncodeWarmupDumpUnit(
          staged_unit, &warmup_encoded_data_);
      lease.Release();
      if (!staging_status.ok()) {
        return finish(status_to_io_status(std::move(staging_status)));
      }
      if (deadline.Expired()) {
        return finish(IOStatus::TimedOut("cache warmup dump deadline"));
      }

      CacheWarmupPriorityTransferStats* priority_stats =
          CacheWarmupStatsForPriority(&warmup_stats_, candidate.priority);
      ++warmup_stats_.entries_staged;
      ++priority_stats->entries_staged;
      io_s = WriteWarmupEncodedUnit();
      if (!io_s.ok()) {
        return finish(io_s);
      }
      ++warmup_stats_.entries_written;
      warmup_stats_.payload_bytes += block_size;
      ++priority_stats->entries_written;
      priority_stats->payload_bytes += block_size;
    }
    if (aggregate_limit_reached) {
      break;
    }
  }

  if (deadline.Expired()) {
    return finish(IOStatus::TimedOut("cache warmup dump deadline"));
  }
  io_s = WriteWarmupFooter();
  if (!io_s.ok()) {
    return finish(io_s);
  }
  return finish(writer_->Close());
}

// [relink cache handoff, RDMA pull] The pull transport's source half. Same
// catalog pass and per-block lease/recheck as the dump above, but the lease
// is KEPT (moved into the returned catalog) instead of being released after a
// copy: the destination reads Block::data() out of this heap with one-sided
// RDMA READs (measured 10.5 GB/s for 4 KiB reads over one RC QP, versus the
// 200-270 MB/s and ~15 us/block of source CPU of the streamed push), and the
// lease is what guarantees the address stays valid and the bytes unchanged
// until Release(). No writer, no copy, no CRC here.
Status CacheDumperImpl::CatalogWarmupDataBlocksForPull(
    const CacheWarmupOptions& warmup_options,
    std::unique_ptr<CacheWarmupPullCatalog>* catalog,
    CacheWarmupTransferStats* warmup_stats) {
  warmup_stats_ = CacheWarmupTransferStats{};
  auto finish = [&](Status status) {
    if (warmup_stats != nullptr) {
      *warmup_stats = warmup_stats_;
    }
    return status;
  };
  if (catalog == nullptr) {
    return finish(Status::InvalidArgument("catalog output is null"));
  }
  catalog->reset();
  IOStatus io_s = ValidateWarmupOptions(warmup_options);
  if (!io_s.ok()) {
    return finish(io_s);
  }
  WarmupCallDeadline deadline(warmup_options.max_transfer_duration_micros);
  role_map_ = CopyCacheDeleterRoleMap();

  std::vector<CacheWarmupCandidate> high_candidates;
  std::vector<CacheWarmupCandidate> low_candidates;
  std::vector<CacheWarmupCandidate> bottom_candidates;
  io_s = CatalogWarmupCandidates(deadline, &high_candidates, &low_candidates,
                                 &bottom_candidates);
  if (!io_s.ok()) {
    return finish(io_s);
  }

  // Leases taken so far are owned by this local object; on any early return
  // its destructor releases them, so a failed catalog never leaves a source
  // block pinned.
  std::unique_ptr<CacheWarmupPullCatalogImpl> pull_catalog(
      new CacheWarmupPullCatalogImpl(cache_));
  const std::vector<CacheWarmupCandidate>* const candidate_buckets[] = {
      &high_candidates, &low_candidates, &bottom_candidates};
  bool aggregate_limit_reached = false;
  for (const auto* candidates : candidate_buckets) {
    for (const auto& candidate : *candidates) {
      if (deadline.Expired()) {
        return finish(Status::TimedOut("cache warmup catalog deadline"));
      }
      // entries_staged == blocks leased here; the dump path bounds
      // entries_written the same way, and nothing is written on this path.
      if (warmup_stats_.entries_staged >= warmup_options.max_entries) {
        aggregate_limit_reached = true;
        break;
      }
      Cache::Handle* handle = nullptr;
      Cache::Priority current_priority = candidate.priority;
      const char* block_data = nullptr;
      size_t block_size = 0;
      io_s = LeaseWarmupDataBlock(candidate, warmup_options, &handle,
                                  &current_priority, &block_data, &block_size);
      if (!io_s.ok()) {
        return finish(io_s);
      }
      if (handle == nullptr) {
        continue;  // skipped; counters already updated
      }
      pull_catalog->Add(handle, current_priority,
                        Slice(candidate.key.data(), candidate.key.size()),
                        block_data, block_size);
      CacheWarmupPriorityTransferStats* priority_stats =
          CacheWarmupStatsForPriority(&warmup_stats_, candidate.priority);
      ++warmup_stats_.entries_staged;
      ++priority_stats->entries_staged;
      // payload_bytes is the byte total the dump path would have written for
      // the same selection; it also feeds the max_total_bytes check in
      // LeaseWarmupDataBlock for the next candidate.
      warmup_stats_.payload_bytes += block_size;
      priority_stats->payload_bytes += block_size;
    }
    if (aggregate_limit_reached) {
      break;
    }
  }
  if (deadline.Expired()) {
    return finish(Status::TimedOut("cache warmup catalog deadline"));
  }
  *catalog = std::move(pull_catalog);
  return finish(Status::OK());
}

void CacheWarmupPullCatalogImpl::Add(Cache::Handle* handle,
                                     Cache::Priority effective_priority,
                                     const Slice& key, const char* data,
                                     size_t size) {
  assert(handle != nullptr);
  CacheWarmupPulledBlock block;
  block.key = key.ToString();
  block.priority = effective_priority;
  block.data = data;
  block.size = size;
  blocks_.push_back(std::move(block));
  leases_.push_back(Lease{handle, effective_priority});
  payload_bytes_ += size;
}

void CacheWarmupPullCatalogImpl::Release() {
  // Idempotent: the destructor calls this too, and a transport that already
  // released on its success path must not double-release on teardown. The
  // block descriptors stay readable (for logging) but their data pointers
  // are dangling from here on.
  if (leases_.empty()) {
    return;
  }
  if (cache_ != nullptr) {
    for (const Lease& lease : leases_) {
      cache_->ReleaseForCacheWarmup(lease.handle, lease.priority);
    }
  }
  leases_.clear();
  leases_.shrink_to_fit();
  for (CacheWarmupPulledBlock& block : blocks_) {
    block.data = nullptr;
  }
}

// Check if we need to filter out the block based on its key
bool CacheDumperImpl::ShouldFilterOut(const Slice& key) {
  if (key.size() < OffsetableCacheKey::kCommonPrefixSize) {
    return /*filter out*/ true;
  }
  Slice key_prefix(key.data(), OffsetableCacheKey::kCommonPrefixSize);
  std::string prefix = key_prefix.ToString();
  // Filter out if not found
  return prefix_filter_.find(prefix) == prefix_filter_.end();
}

// This is the callback function which will be applied to
// Cache::ApplyToAllEntries. In this callback function, we will get the block
// type, decide if the block needs to be dumped based on the filter, and write
// the block through the provided writer.
std::function<void(const Slice&, void*, size_t, Cache::DeleterFn)>
CacheDumperImpl::DumpOneBlockCallBack() {
  return [&](const Slice& key, void* value, size_t /*charge*/,
             Cache::DeleterFn deleter) {
    // Step 1: get the type of the block from role_map_
    auto e = role_map_.find(deleter);
    CacheEntryRole role;
    CacheDumpUnitType type = CacheDumpUnitType::kBlockTypeMax;
    if (e == role_map_.end()) {
      role = CacheEntryRole::kMisc;
    } else {
      role = e->second;
    }
    bool filter_out = false;

    // Step 2: based on the key prefix, check if the block should be filter out.
    if (ShouldFilterOut(key)) {
      filter_out = true;
    }

    // Step 3: based on the block type, get the block raw pointer and length.
    const char* block_start = nullptr;
    size_t block_len = 0;
    switch (role) {
      case CacheEntryRole::kDataBlock:
        type = CacheDumpUnitType::kData;
        block_start = (static_cast<Block*>(value))->data();
        block_len = (static_cast<Block*>(value))->size();
        break;
      case CacheEntryRole::kFilterBlock:
        type = CacheDumpUnitType::kFilter;
        block_start = (static_cast<ParsedFullFilterBlock*>(value))
                          ->GetBlockContentsData()
                          .data();
        block_len = (static_cast<ParsedFullFilterBlock*>(value))
                        ->GetBlockContentsData()
                        .size();
        break;
      case CacheEntryRole::kFilterMetaBlock:
        type = CacheDumpUnitType::kFilterMetaBlock;
        block_start = (static_cast<Block*>(value))->data();
        block_len = (static_cast<Block*>(value))->size();
        break;
      case CacheEntryRole::kIndexBlock:
        type = CacheDumpUnitType::kIndex;
        block_start = (static_cast<Block*>(value))->data();
        block_len = (static_cast<Block*>(value))->size();
        break;
      case CacheEntryRole::kDeprecatedFilterBlock:
        // Obsolete
        filter_out = true;
        break;
      case CacheEntryRole::kMisc:
        filter_out = true;
        break;
      case CacheEntryRole::kOtherBlock:
        filter_out = true;
        break;
      case CacheEntryRole::kWriteBuffer:
        filter_out = true;
        break;
      default:
        filter_out = true;
    }

    // Step 4: if the block should not be filter out, write the block to the
    // CacheDumpWriter
    if (!filter_out && block_start != nullptr) {
      WriteBlock(type, key, Slice(block_start, block_len))
          .PermitUncheckedError();
    }
  };
}

// Write the block to the writer. It takes the timestamp of the
// block being copied from block cache, block type, key, block pointer,
// block size and block checksum as the input. When writing the dumper raw
// block, we first create the dump unit and encoude it to a string. Then,
// we calculate the checksum of the whole dump unit string and store it in
// the dump unit metadata.
// First, we write the metadata first, which is a fixed size string. Then, we
// Append the dump unit string to the writer.
IOStatus CacheDumperImpl::WriteBlock(CacheDumpUnitType type, const Slice& key,
                                     const Slice& value) {
  uint64_t timestamp = clock_->NowMicros();
  uint32_t value_checksum = crc32c::Value(value.data(), value.size());

  // First, serialize the block information in a string
  DumpUnit dump_unit;
  dump_unit.timestamp = timestamp;
  dump_unit.key = key;
  dump_unit.type = type;
  dump_unit.value_len = value.size();
  dump_unit.value = const_cast<char*>(value.data());
  dump_unit.value_checksum = value_checksum;
  std::string encoded_data;
  CacheDumperHelper::EncodeDumpUnit(dump_unit, &encoded_data);

  // Second, create the metadata, which contains a sequence number, the dump
  // unit string checksum and the string size. The sequence number monotonically
  // increases from 0.
  DumpUnitMeta unit_meta;
  unit_meta.sequence_num = sequence_num_;
  sequence_num_++;
  unit_meta.dump_unit_checksum =
      crc32c::Value(encoded_data.data(), encoded_data.size());
  unit_meta.dump_unit_size = encoded_data.size();
  std::string encoded_meta;
  CacheDumperHelper::EncodeDumpUnitMeta(unit_meta, &encoded_meta);

  // We write the metadata first.
  assert(writer_ != nullptr);
  IOStatus io_s = writer_->WriteMetadata(encoded_meta);
  if (!io_s.ok()) {
    return io_s;
  }
  // followed by the dump unit.
  return writer_->WritePacket(encoded_data);
}

// Before we write any block, we write the header first to store the cache dump
// format version, rocksdb version, and brief intro.
IOStatus CacheDumperImpl::WriteHeader() {
  std::string header_key = "header";
  std::ostringstream s;
  s << kTraceMagic << "\t"
    << "Cache dump format version: " << kCacheDumpMajorVersion << "."
    << kCacheDumpMinorVersion << "\t"
    << "RocksDB Version: " << kMajorVersion << "." << kMinorVersion << "\t"
    << "Format: dump_unit_metadata <sequence_number, dump_unit_checksum, "
       "dump_unit_size>, dump_unit <timestamp, key, block_type, "
       "block_size, block_data, block_checksum> cache_value\n";
  std::string header_value(s.str());
  CacheDumpUnitType type = CacheDumpUnitType::kHeader;
  return WriteBlock(type, header_key, header_value);
}

// Write the footer after all the blocks are stored to indicate the ending.
IOStatus CacheDumperImpl::WriteFooter() {
  std::string footer_key = "footer";
  std::string footer_value("cache dump completed");
  CacheDumpUnitType type = CacheDumpUnitType::kFooter;
  return WriteBlock(type, footer_key, footer_value);
}

IOStatus CacheDumperImpl::WriteWarmupUnit(CacheWarmupDumpUnitType type,
                                          Cache::Priority priority,
                                          const Slice& key,
                                          const Slice& value) {
  CacheWarmupDumpUnit unit;
  unit.type = type;
  unit.priority = priority;
  unit.key = key;
  unit.value_len = value.size();
  unit.value = const_cast<char*>(value.data());

  warmup_encoded_data_.clear();
  Status encode_status =
      CacheDumperHelper::EncodeWarmupDumpUnit(unit, &warmup_encoded_data_);
  if (!encode_status.ok()) {
    return status_to_io_status(std::move(encode_status));
  }
  return WriteWarmupEncodedUnit();
}

IOStatus CacheDumperImpl::WriteWarmupEncodedUnit() {
  DumpUnitMeta unit_meta;
  unit_meta.sequence_num = sequence_num_++;
  unit_meta.dump_unit_checksum =
      crc32c::Value(warmup_encoded_data_.data(), warmup_encoded_data_.size());
  unit_meta.dump_unit_size = warmup_encoded_data_.size();
  std::string encoded_meta;
  CacheDumperHelper::EncodeDumpUnitMeta(unit_meta, &encoded_meta);

  IOStatus io_s = writer_->WriteMetadata(encoded_meta);
  if (!io_s.ok()) {
    return io_s;
  }
  return writer_->WritePacket(warmup_encoded_data_);
}

IOStatus CacheDumperImpl::WriteWarmupHeader() {
  static const std::string kHeaderKey = "cache-warmup-header";
  static const std::string kHeaderValue =
      "versioned data-block-only priority-preserving cache warmup";
  return WriteWarmupUnit(CacheWarmupDumpUnitType::kHeader, Cache::Priority::LOW,
                         Slice(kHeaderKey), Slice(kHeaderValue));
}

IOStatus CacheDumperImpl::WriteWarmupFooter() {
  static const std::string kFooterKey = "cache-warmup-footer";
  static const std::string kFooterValue = "cache warmup completed";
  return WriteWarmupUnit(CacheWarmupDumpUnitType::kFooter, Cache::Priority::LOW,
                         Slice(kFooterKey), Slice(kFooterValue));
}

// This is the main function to restore the cache entries to secondary cache.
// First, we check if all the arguments are valid. Then, we read the block
// sequentially from the reader and insert them to the secondary cache.
IOStatus CacheDumpedLoaderImpl::RestoreCacheEntriesToSecondaryCache() {
  // TODO: remove this line when options are used in the loader
  (void)options_;
  // Step 1: we check if all the arguments are valid
  if (secondary_cache_ == nullptr) {
    return IOStatus::InvalidArgument("Secondary Cache is null");
  }
  if (reader_ == nullptr) {
    return IOStatus::InvalidArgument("CacheDumpReader is null");
  }
  // we copy the Cache Deleter Role Map as its member.
  role_map_ = CopyCacheDeleterRoleMap();

  // Step 2: read the header
  // TODO: we need to check the cache dump format version and RocksDB version
  // after the header is read out.
  IOStatus io_s;
  DumpUnit dump_unit;
  std::string data;
  io_s = ReadHeader(&data, &dump_unit);
  if (!io_s.ok()) {
    return io_s;
  }

  // Step 3: read out the rest of the blocks from the reader. The loop will stop
  // either I/O status is not ok or we reach to the the end.
  while (io_s.ok() && dump_unit.type != CacheDumpUnitType::kFooter) {
    dump_unit.reset();
    data.clear();
    // read the content and store in the dump_unit
    io_s = ReadCacheBlock(&data, &dump_unit);
    if (!io_s.ok()) {
      break;
    }
    // Create the uncompressed_block based on the information in the dump_unit
    // (There is no block trailer here compatible with block-based SST file.)
    BlockContents uncompressed_block(
        Slice(static_cast<char*>(dump_unit.value), dump_unit.value_len));
    Cache::CacheItemHelper* helper = nullptr;
    Statistics* statistics = nullptr;
    Status s = Status::OK();
    // according to the block type, get the helper callback function and create
    // the corresponding block
    switch (dump_unit.type) {
      case CacheDumpUnitType::kFilter: {
        helper = BlocklikeTraits<ParsedFullFilterBlock>::GetCacheItemHelper(
            BlockType::kFilter);
        std::unique_ptr<ParsedFullFilterBlock> block_holder;
        block_holder.reset(BlocklikeTraits<ParsedFullFilterBlock>::Create(
            std::move(uncompressed_block), toptions_.read_amp_bytes_per_bit,
            statistics, false, toptions_.filter_policy.get()));
        if (helper != nullptr) {
          s = secondary_cache_->Insert(dump_unit.key,
                                       (void*)(block_holder.get()), helper);
        }
        break;
      }
      case CacheDumpUnitType::kData: {
        helper = BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData);
        std::unique_ptr<Block> block_holder;
        block_holder.reset(BlocklikeTraits<Block>::Create(
            std::move(uncompressed_block), toptions_.read_amp_bytes_per_bit,
            statistics, false, toptions_.filter_policy.get()));
        if (helper != nullptr) {
          s = secondary_cache_->Insert(dump_unit.key,
                                       (void*)(block_holder.get()), helper);
        }
        break;
      }
      case CacheDumpUnitType::kIndex: {
        helper = BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kIndex);
        std::unique_ptr<Block> block_holder;
        block_holder.reset(BlocklikeTraits<Block>::Create(
            std::move(uncompressed_block), 0, statistics, false,
            toptions_.filter_policy.get()));
        if (helper != nullptr) {
          s = secondary_cache_->Insert(dump_unit.key,
                                       (void*)(block_holder.get()), helper);
        }
        break;
      }
      case CacheDumpUnitType::kFilterMetaBlock: {
        helper = BlocklikeTraits<Block>::GetCacheItemHelper(
            BlockType::kFilterPartitionIndex);
        std::unique_ptr<Block> block_holder;
        block_holder.reset(BlocklikeTraits<Block>::Create(
            std::move(uncompressed_block), toptions_.read_amp_bytes_per_bit,
            statistics, false, toptions_.filter_policy.get()));
        if (helper != nullptr) {
          s = secondary_cache_->Insert(dump_unit.key,
                                       (void*)(block_holder.get()), helper);
        }
        break;
      }
      case CacheDumpUnitType::kFooter:
        break;
      case CacheDumpUnitType::kDeprecatedFilterBlock:
        // Obsolete
        break;
      default:
        continue;
    }
    if (!s.ok()) {
      io_s = status_to_io_status(std::move(s));
    }
  }
  if (dump_unit.type == CacheDumpUnitType::kFooter) {
    return IOStatus::OK();
  } else {
    return io_s;
  }
}

// [relink cache handoff] Same read loop as the secondary-cache variant, but the
// blocks are installed straight into a primary block cache under the dumped keys.
// Correct only when this DB reads the SAME physical files the dumper's DB did
// (cache keys are derived from ids embedded in the file at creation) — i.e. the
// relinked-external-file situation. Only data blocks are installed: with
// cache_index_and_filter_blocks=false the receiver never looks index/filter
// blocks up in the block cache, so installing them would only waste capacity.
IOStatus CacheDumpedLoaderImpl::RestoreCacheEntriesToPrimaryCache() {
  if (primary_cache_ == nullptr) {
    return IOStatus::InvalidArgument("Primary cache is null");
  }
  if (reader_ == nullptr) {
    return IOStatus::InvalidArgument("CacheDumpReader is null");
  }
  IOStatus io_s;
  DumpUnit dump_unit;
  std::string data;
  io_s = ReadHeader(&data, &dump_unit);
  if (!io_s.ok()) {
    return io_s;
  }
  Cache::CacheItemHelper* helper =
      BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData);
  while (io_s.ok() && dump_unit.type != CacheDumpUnitType::kFooter) {
    dump_unit.reset();
    data.clear();
    io_s = ReadCacheBlock(&data, &dump_unit);
    if (!io_s.ok()) {
      break;
    }
    if (dump_unit.type != CacheDumpUnitType::kData || helper == nullptr) {
      continue;
    }
    CacheAllocationPtr buf = AllocateBlock(dump_unit.value_len, nullptr);
    memcpy(buf.get(), dump_unit.value, dump_unit.value_len);
    BlockContents contents(std::move(buf), dump_unit.value_len);
    std::unique_ptr<Block> block_holder;
    block_holder.reset(BlocklikeTraits<Block>::Create(
        std::move(contents), toptions_.read_amp_bytes_per_bit,
        /*statistics*/ nullptr, /*using_zstd*/ false,
        toptions_.filter_policy.get()));
    if (block_holder == nullptr) {
      continue;
    }
    size_t charge = block_holder->ApproximateMemoryUsage();
    Status s = primary_cache_->Insert(dump_unit.key, block_holder.get(), charge,
                                      helper->del_cb);
    if (s.ok()) {
      block_holder.release();  // owned by the cache now
    }
  }
  if (dump_unit.type == CacheDumpUnitType::kFooter) {
    return IOStatus::OK();
  }
  return io_s;
}

// Restore the separately versioned warmup stream. Unlike the legacy primary
// restore above, this path calls the cache's warmup admission primitive so a
// destination resident entry is never overwritten and a higher-priority
// resident set is not displaced merely to admit warmup data.
IOStatus CacheDumpedLoaderImpl::RestoreWarmupCacheEntriesToPrimaryCache(
    const CacheWarmupOptions& warmup_options,
    CacheWarmupTransferStats* warmup_stats) {
  warmup_stats_ = CacheWarmupTransferStats{};
  auto finish = [&](IOStatus status) {
    if (warmup_stats != nullptr) {
      *warmup_stats = warmup_stats_;
    }
    return status;
  };
  if (primary_cache_ == nullptr) {
    return finish(IOStatus::InvalidArgument("Primary cache is null"));
  }
  if (reader_ == nullptr) {
    return finish(IOStatus::InvalidArgument("CacheDumpReader is null"));
  }
  if (warmup_options.max_entry_bytes == 0) {
    return finish(IOStatus::InvalidArgument(
        "cache warmup max_entry_bytes must be nonzero"));
  }
  if (warmup_options.max_entries == 0 ||
      warmup_options.max_total_bytes == 0) {
    return finish(IOStatus::InvalidArgument(
        "cache warmup aggregate limits must be nonzero"));
  }
  WarmupCallDeadline deadline(warmup_options.max_transfer_duration_micros);

  std::string data;
  CacheWarmupDumpUnit unit;
  IOStatus io_s = ReadWarmupHeader(&data, &unit);
  if (!io_s.ok()) {
    return finish(io_s);
  }
  if (BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData) ==
      nullptr) {
    return finish(IOStatus::NotSupported("data block cache helper is null"));
  }

  // Sink mode (2026-09-23): the aggregate caps are enforced on these
  // reader-local counters because entries_received / payload_bytes are then
  // produced by whoever runs the owned inserts, not by this loader.
  uint64_t gate_received = 0;
  uint64_t gate_bytes = 0;

  while (true) {
    if (deadline.Expired()) {
      return finish(IOStatus::TimedOut("cache warmup restore deadline"));
    }
    unit = CacheWarmupDumpUnit{};
    data.clear();
    io_s = ReadWarmupCacheBlock(&data, &unit);
    if (!io_s.ok()) {
      return finish(io_s);
    }
    if (unit.type == CacheWarmupDumpUnitType::kFooter) {
      return finish(IOStatus::OK());
    }
    if (unit.type != CacheWarmupDumpUnitType::kData) {
      ++warmup_stats_.skipped_invalid;
      return finish(
          IOStatus::Corruption("non-data unit in cache warmup stream"));
    }
    if (unit_sink_) {
      // Same gate, same order, then copy out of the per-iteration `data`
      // string (unit.value points into it) into a buffer the sink owns.
      io_s = CheckWarmupUnitLimits(unit.key, unit.value_len,
                                   unit.value != nullptr, warmup_options,
                                   gate_received, gate_bytes, &warmup_stats_);
      if (!io_s.ok()) {
        return finish(io_s);
      }
      ++gate_received;
      gate_bytes += unit.value_len;
      CacheAllocationPtr buf = AllocateBlock(unit.value_len, sink_allocator_);
      if (unit.value_len != 0) {
        if (buf == nullptr) {
          return finish(IOStatus::IOError(
              "cache warmup sink allocator returned null"));
        }
        std::memcpy(buf.get(), unit.value, unit.value_len);
      }
      io_s = unit_sink_(unit.key, unit.priority, std::move(buf),
                        unit.value_len);
      if (!io_s.ok()) {
        return finish(io_s);
      }
      continue;
    }
    // One admission unit, shared with the pull receiver. A non-OK status is a
    // stream error here (the framed stream is trusted to be well-formed once
    // its CRC passed), exactly as before the factoring.
    io_s = InsertWarmupDataBlockWithLimits(
        unit.key, unit.priority, static_cast<const char*>(unit.value),
        unit.value_len, warmup_options, &warmup_stats_);
    if (!io_s.ok()) {
      return finish(io_s);
    }
  }
}

// The single admission unit behind every transport, split (2026-09-23) into
// the gate (CheckWarmupUnitLimits) and the admission (AdmitWarmupBlock) so
// the streamed sink path can gate before it allocates and the owned-buffer
// entry point can admit without a copy. The order of the checks and of the
// counter updates is the streamed path's original order, so a block admitted
// over RDMA is accounted and admitted exactly like the same block admitted
// from the TCP stream.
IOStatus CacheDumpedLoaderImpl::CheckWarmupUnitLimits(
    const Slice& key, size_t size, bool has_payload,
    const CacheWarmupOptions& warmup_options, uint64_t received_so_far,
    uint64_t bytes_so_far, CacheWarmupTransferStats* stats) {
  assert(stats != nullptr);
  if (primary_cache_ == nullptr) {
    return IOStatus::InvalidArgument("Primary cache is null");
  }
  if (BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData) ==
      nullptr) {
    return IOStatus::NotSupported("data block cache helper is null");
  }
  if (key.size() != kCacheKeySize) {
    ++stats->skipped_invalid;
    return IOStatus::Corruption(
        "cache warmup data unit has non-standard cache key size");
  }
  if (size > warmup_options.max_entry_bytes) {
    ++stats->skipped_too_large;
    return IOStatus::Corruption(
        "cache warmup data unit exceeds destination entry limit");
  }
  if (received_so_far >= warmup_options.max_entries) {
    ++stats->skipped_too_large;
    return IOStatus::Corruption(
        "cache warmup stream exceeds destination entry-count limit");
  }
  if (bytes_so_far > warmup_options.max_total_bytes ||
      size > warmup_options.max_total_bytes -
                 static_cast<size_t>(bytes_so_far)) {
    ++stats->skipped_too_large;
    return IOStatus::Corruption(
        "cache warmup stream exceeds destination aggregate byte limit");
  }
  if (size != 0 && !has_payload) {
    ++stats->skipped_invalid;
    return IOStatus::Corruption("cache warmup data unit has null payload");
  }
  return IOStatus::OK();
}

// Thread-safe with a per-thread *stats: touches only primary_cache_ (the
// shard mutex is inside InsertForCacheWarmup), the const toptions_, and a
// function-local static helper. `buf` is moved into the Block; on any
// outcome but kInserted the Block (and with it the buffer, through its
// deleter) is destroyed by block_holder before this returns.
IOStatus CacheDumpedLoaderImpl::AdmitWarmupBlock(
    const Slice& key, Cache::Priority priority, CacheAllocationPtr&& buf,
    size_t size, CacheWarmupTransferStats* stats) {
  assert(stats != nullptr);
  Cache::CacheItemHelper* helper =
      BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData);
  assert(helper != nullptr);  // CheckWarmupUnitLimits ran first
  CacheWarmupPriorityTransferStats* priority_stats =
      CacheWarmupStatsForPriority(stats, priority);
  ++stats->entries_received;
  ++priority_stats->entries_received;
  stats->payload_bytes += size;
  priority_stats->payload_bytes += size;

  BlockContents contents(std::move(buf), size);
  std::unique_ptr<Block> block_holder;
  block_holder.reset(BlocklikeTraits<Block>::Create(
      std::move(contents), toptions_.read_amp_bytes_per_bit,
      /*statistics=*/nullptr, /*using_zstd=*/false,
      toptions_.filter_policy.get()));
  if (block_holder == nullptr) {
    ++stats->skipped_unsupported;
    return IOStatus::OK();
  }

  const size_t charge = block_holder->ApproximateMemoryUsage();
  Cache::CacheWarmupInsertResult insert_result =
      Cache::CacheWarmupInsertResult::kRejectedNoSpace;
  Status insert_status = primary_cache_->InsertForCacheWarmup(
      key, block_holder.get(), charge, helper->del_cb, priority,
      &insert_result);
  if (!insert_status.ok()) {
    return status_to_io_status(std::move(insert_status));
  }
  switch (insert_result) {
    case Cache::CacheWarmupInsertResult::kInserted:
      block_holder.release();  // cache owns it only for this outcome
      ++stats->entries_inserted;
      ++priority_stats->entries_inserted;
      break;
    case Cache::CacheWarmupInsertResult::kDuplicate:
      ++stats->entries_duplicate;
      ++priority_stats->entries_duplicate;
      break;
    case Cache::CacheWarmupInsertResult::kRejectedNoSpace:
      ++stats->entries_rejected_no_space;
      ++priority_stats->entries_rejected_no_space;
      break;
    default:
      ++stats->skipped_invalid;
      return IOStatus::Corruption("unknown cache warmup admission outcome");
  }
  return IOStatus::OK();
}

// The historical inline unit: gate, copy into fresh cache-owned memory,
// admit. Unchanged behaviour (the received/payload counters are now bumped
// after the allocation instead of before it; unobservable).
IOStatus CacheDumpedLoaderImpl::InsertWarmupDataBlockWithLimits(
    const Slice& key, Cache::Priority priority, const char* data, size_t size,
    const CacheWarmupOptions& warmup_options,
    CacheWarmupTransferStats* stats) {
  assert(stats != nullptr);
  IOStatus io_s = CheckWarmupUnitLimits(
      key, size, data != nullptr, warmup_options, stats->entries_received,
      stats->payload_bytes, stats);
  if (!io_s.ok()) {
    return io_s;
  }
  CacheAllocationPtr buf = AllocateBlock(size, nullptr);
  if (size != 0) {
    std::memcpy(buf.get(), data, size);
  }
  return AdmitWarmupBlock(key, priority, std::move(buf), size, stats);
}

// [relink cache handoff, RDMA pull] Destination half of the pull transport:
// the receiver hands over each payload it RDMA-READ into its pinned pool and
// this admits it through the very same unit the streamed path uses. Default
// CacheWarmupOptions apply (8 MiB per entry, no aggregate cap): the source
// already enforced the aggregate budget when it built the catalog, and the
// receiver owns the wall-clock deadlines around its reads.
IOStatus CacheDumpedLoaderImpl::InsertWarmupDataBlock(
    const Slice& key, Cache::Priority priority, const char* data, size_t size,
    CacheWarmupTransferStats* stats) {
  return InsertWarmupDataBlockWithLimits(
      key, priority, data, size, CacheWarmupOptions{},
      stats != nullptr ? stats : &warmup_stats_);
}

// [relink cache handoff, zero-copy landing, 2026-09-23] Same gate and
// admission as InsertWarmupDataBlock, but the buffer is the caller's (an
// arena slot the RDMA bytes landed in) and is never copied. The gate runs
// before ownership moves; on a gate failure the buffer is released here so
// the caller's pointer is null on every return, as the header promises.
IOStatus CacheDumpedLoaderImpl::InsertWarmupDataBlockOwned(
    const Slice& key, Cache::Priority priority, CacheAllocationPtr&& buf,
    size_t size, CacheWarmupTransferStats* stats) {
  CacheWarmupTransferStats* s = stats != nullptr ? stats : &warmup_stats_;
  IOStatus io_s = CheckWarmupUnitLimits(
      key, size, buf != nullptr, CacheWarmupOptions{}, s->entries_received,
      s->payload_bytes, s);
  if (!io_s.ok()) {
    buf.reset();
    return io_s;
  }
  return AdmitWarmupBlock(key, priority, std::move(buf), size, s);
}

// Read and copy the dump unit metadata to std::string data, decode and create
// the unit metadata based on the string
IOStatus CacheDumpedLoaderImpl::ReadDumpUnitMeta(std::string* data,
                                                 DumpUnitMeta* unit_meta) {
  assert(reader_ != nullptr);
  assert(data != nullptr);
  assert(unit_meta != nullptr);
  IOStatus io_s = reader_->ReadMetadata(data);
  if (!io_s.ok()) {
    return io_s;
  }
  return status_to_io_status(
      CacheDumperHelper::DecodeDumpUnitMeta(*data, unit_meta));
}

// Read and copy the dump unit to std::string data, decode and create the unit
// based on the string
IOStatus CacheDumpedLoaderImpl::ReadDumpUnit(size_t len, std::string* data,
                                             DumpUnit* unit) {
  assert(reader_ != nullptr);
  assert(data != nullptr);
  assert(unit != nullptr);
  IOStatus io_s = reader_->ReadPacket(data);
  if (!io_s.ok()) {
    return io_s;
  }
  if (data->size() != len) {
    return IOStatus::Corruption(
        "The data being read out does not match the size stored in metadata!");
  }
  Slice block;
  return status_to_io_status(CacheDumperHelper::DecodeDumpUnit(*data, unit));
}

// Read the header
IOStatus CacheDumpedLoaderImpl::ReadHeader(std::string* data,
                                           DumpUnit* dump_unit) {
  DumpUnitMeta header_meta;
  header_meta.reset();
  std::string meta_string;
  IOStatus io_s = ReadDumpUnitMeta(&meta_string, &header_meta);
  if (!io_s.ok()) {
    return io_s;
  }

  io_s = ReadDumpUnit(header_meta.dump_unit_size, data, dump_unit);
  if (!io_s.ok()) {
    return io_s;
  }
  uint32_t unit_checksum = crc32c::Value(data->data(), data->size());
  if (unit_checksum != header_meta.dump_unit_checksum) {
    return IOStatus::Corruption("Read header unit corrupted!");
  }
  return io_s;
}

// Read the blocks after header is read out
IOStatus CacheDumpedLoaderImpl::ReadCacheBlock(std::string* data,
                                               DumpUnit* dump_unit) {
  // According to the write process, we read the dump_unit_metadata first
  DumpUnitMeta unit_meta;
  unit_meta.reset();
  std::string unit_string;
  IOStatus io_s = ReadDumpUnitMeta(&unit_string, &unit_meta);
  if (!io_s.ok()) {
    return io_s;
  }

  // Based on the information in the dump_unit_metadata, we read the dump_unit
  // and verify if its content is correct.
  io_s = ReadDumpUnit(unit_meta.dump_unit_size, data, dump_unit);
  if (!io_s.ok()) {
    return io_s;
  }
  uint32_t unit_checksum = crc32c::Value(data->data(), data->size());
  if (unit_checksum != unit_meta.dump_unit_checksum) {
    return IOStatus::Corruption(
        "Checksum does not match! Read dumped unit corrupted!");
  }
  return io_s;
}

IOStatus CacheDumpedLoaderImpl::ReadWarmupUnit(size_t len, std::string* data,
                                               CacheWarmupDumpUnit* unit) {
  assert(reader_ != nullptr);
  assert(data != nullptr);
  assert(unit != nullptr);
  IOStatus io_s = reader_->ReadPacket(data);
  if (!io_s.ok()) {
    return io_s;
  }
  if (data->size() != len) {
    return IOStatus::Corruption(
        "cache warmup packet size differs from its metadata");
  }
  return status_to_io_status(
      CacheDumperHelper::DecodeWarmupDumpUnit(*data, unit));
}

IOStatus CacheDumpedLoaderImpl::ReadWarmupHeader(std::string* data,
                                                 CacheWarmupDumpUnit* unit) {
  DumpUnitMeta header_meta;
  header_meta.reset();
  std::string metadata;
  IOStatus io_s = ReadDumpUnitMeta(&metadata, &header_meta);
  if (!io_s.ok()) {
    return io_s;
  }
  io_s = ReadWarmupUnit(header_meta.dump_unit_size, data, unit);
  if (!io_s.ok()) {
    return io_s;
  }
  if (crc32c::Value(data->data(), data->size()) !=
      header_meta.dump_unit_checksum) {
    return IOStatus::Corruption("cache warmup header checksum mismatch");
  }
  if (unit->type != CacheWarmupDumpUnitType::kHeader) {
    return IOStatus::Corruption(
        "cache warmup stream does not start with header");
  }
  return IOStatus::OK();
}

IOStatus CacheDumpedLoaderImpl::ReadWarmupCacheBlock(
    std::string* data, CacheWarmupDumpUnit* unit) {
  DumpUnitMeta metadata;
  metadata.reset();
  std::string metadata_string;
  IOStatus io_s = ReadDumpUnitMeta(&metadata_string, &metadata);
  if (!io_s.ok()) {
    return io_s;
  }
  io_s = ReadWarmupUnit(metadata.dump_unit_size, data, unit);
  if (!io_s.ok()) {
    return io_s;
  }
  if (crc32c::Value(data->data(), data->size()) !=
      metadata.dump_unit_checksum) {
    return IOStatus::Corruption("cache warmup unit checksum mismatch");
  }
  return IOStatus::OK();
}

}  // namespace ROCKSDB_NAMESPACE
#endif  // ROCKSDB_LITE
