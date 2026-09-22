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

class WarmupCallDeadline {
 public:
  explicit WarmupCallDeadline(uint64_t duration_micros)
      : duration_micros_(duration_micros),
        start_(std::chrono::steady_clock::now()) {}

  bool Expired() const {
    if (duration_micros_ == 0) {
      return false;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start_)
                             .count();
    return elapsed >= 0 && static_cast<uint64_t>(elapsed) >= duration_micros_;
  }

 private:
  uint64_t duration_micros_;
  std::chrono::steady_clock::time_point start_;
};

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
  if (cache_ == nullptr) {
    return finish(IOStatus::InvalidArgument("Cache is null"));
  }
  if (writer_ == nullptr) {
    return finish(IOStatus::InvalidArgument("CacheDumpWriter is null"));
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
  role_map_ = CopyCacheDeleterRoleMap();
  sequence_num_ = 0;

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

  struct WarmupCandidate {
    std::array<char, kCacheKeySize> key;
    size_t charge;
    Cache::Priority priority;
  };
  // Bucketing during catalog is O(N), preserves catalog order within each
  // class, and gives the wire stream its required HIGH -> LOW -> BOTTOM order
  // without an O(N log N) sort.
  std::vector<WarmupCandidate> high_candidates;
  std::vector<WarmupCandidate> low_candidates;
  std::vector<WarmupCandidate> bottom_candidates;
  auto candidate_bucket =
      [&](Cache::Priority priority) -> std::vector<WarmupCandidate>* {
    switch (priority) {
      case Cache::Priority::HIGH:
        return &high_candidates;
      case Cache::Priority::LOW:
        return &low_candidates;
      case Cache::Priority::BOTTOM:
        return &bottom_candidates;
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
        std::vector<WarmupCandidate>* bucket =
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
        WarmupCandidate candidate{};
        std::memcpy(candidate.key.data(), key.data(), kCacheKeySize);
        candidate.charge = charge;
        candidate.priority = effective_priority;
        bucket->push_back(candidate);
      },
      catalog_options);
  if (!catalog_status.ok()) {
    return finish(status_to_io_status(std::move(catalog_status)));
  }
  if (catalog_deadline_expired || deadline.Expired()) {
    return finish(IOStatus::TimedOut("cache warmup catalog deadline"));
  }

  IOStatus io_s = WriteWarmupHeader();
  if (!io_s.ok()) {
    return finish(io_s);
  }
  const std::vector<WarmupCandidate>* const candidate_buckets[] = {
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
      Status lookup_status = cache_->LookupForCacheWarmup(
          Slice(candidate.key.data(), candidate.key.size()), &handle,
          &current_priority);
      if (!lookup_status.ok()) {
        return finish(status_to_io_status(std::move(lookup_status)));
      }
      if (handle == nullptr) {
        ++warmup_stats_.skipped_disappeared;
        continue;
      }
      ScopedWarmupLease lease(cache_.get(), handle, current_priority);

      // Every path below releases this one-and-only pinned handle before CRC,
      // metadata framing, or a writer call.
      const size_t current_charge = cache_->GetCharge(handle);
      const Cache::DeleterFn current_deleter = cache_->GetDeleter(handle);
      void* const current_value = cache_->Value(handle);
      if (current_priority != candidate.priority) {
        ++warmup_stats_.priority_changed_after_catalog;
        // Keeping the catalog class here would mislabel the entry, while using
        // the new class could violate the stream's HIGH -> LOW -> BOTTOM order.
        // Skip this concurrently changed entry and preserve both invariants.
        continue;
      }
      if (current_charge != candidate.charge) {
        ++warmup_stats_.skipped_replaced;
        continue;
      }
      const auto role_it = role_map_.find(current_deleter);
      if (role_it == role_map_.end()) {
        ++warmup_stats_.skipped_unsupported;
        continue;
      }
      if (role_it->second != CacheEntryRole::kDataBlock) {
        ++warmup_stats_.skipped_type_changed;
        continue;
      }
      if (current_value == nullptr) {
        ++warmup_stats_.skipped_unsupported;
        continue;
      }

      const Block* block = static_cast<const Block*>(current_value);
      const char* const block_data = block->data();
      const size_t block_size = block->size();
      if (block_size > warmup_options.max_entry_bytes ||
          block_size > std::numeric_limits<uint32_t>::max()) {
        ++warmup_stats_.skipped_too_large;
        continue;
      }
      if (block_size != 0 && block_data == nullptr) {
        ++warmup_stats_.skipped_unsupported;
        continue;
      }
      if (warmup_stats_.payload_bytes > warmup_options.max_total_bytes ||
          block_size > warmup_options.max_total_bytes -
                           static_cast<size_t>(warmup_stats_.payload_bytes)) {
        ++warmup_stats_.skipped_too_large;
        continue;
      }

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
  Cache::CacheItemHelper* helper =
      BlocklikeTraits<Block>::GetCacheItemHelper(BlockType::kData);
  if (helper == nullptr) {
    return finish(IOStatus::NotSupported("data block cache helper is null"));
  }

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
    if (unit.key.size() != kCacheKeySize) {
      ++warmup_stats_.skipped_invalid;
      return finish(IOStatus::Corruption(
          "cache warmup data unit has non-standard cache key size"));
    }
    if (unit.value_len > warmup_options.max_entry_bytes) {
      ++warmup_stats_.skipped_too_large;
      return finish(IOStatus::Corruption(
          "cache warmup data unit exceeds destination entry limit"));
    }
    if (warmup_stats_.entries_received >= warmup_options.max_entries) {
      ++warmup_stats_.skipped_too_large;
      return finish(IOStatus::Corruption(
          "cache warmup stream exceeds destination entry-count limit"));
    }
    if (warmup_stats_.payload_bytes > warmup_options.max_total_bytes ||
        unit.value_len > warmup_options.max_total_bytes -
                             static_cast<size_t>(warmup_stats_.payload_bytes)) {
      ++warmup_stats_.skipped_too_large;
      return finish(IOStatus::Corruption(
          "cache warmup stream exceeds destination aggregate byte limit"));
    }

    CacheWarmupPriorityTransferStats* priority_stats =
        CacheWarmupStatsForPriority(&warmup_stats_, unit.priority);
    ++warmup_stats_.entries_received;
    ++priority_stats->entries_received;
    warmup_stats_.payload_bytes += unit.value_len;
    priority_stats->payload_bytes += unit.value_len;

    CacheAllocationPtr buf = AllocateBlock(unit.value_len, nullptr);
    if (unit.value_len != 0) {
      std::memcpy(buf.get(), unit.value, unit.value_len);
    }
    BlockContents contents(std::move(buf), unit.value_len);
    std::unique_ptr<Block> block_holder;
    block_holder.reset(BlocklikeTraits<Block>::Create(
        std::move(contents), toptions_.read_amp_bytes_per_bit,
        /*statistics=*/nullptr, /*using_zstd=*/false,
        toptions_.filter_policy.get()));
    if (block_holder == nullptr) {
      ++warmup_stats_.skipped_unsupported;
      continue;
    }

    const size_t charge = block_holder->ApproximateMemoryUsage();
    Cache::CacheWarmupInsertResult insert_result =
        Cache::CacheWarmupInsertResult::kRejectedNoSpace;
    Status insert_status = primary_cache_->InsertForCacheWarmup(
        unit.key, block_holder.get(), charge, helper->del_cb, unit.priority,
        &insert_result);
    if (!insert_status.ok()) {
      return finish(status_to_io_status(std::move(insert_status)));
    }
    switch (insert_result) {
      case Cache::CacheWarmupInsertResult::kInserted:
        block_holder.release();  // cache owns it only for this outcome
        ++warmup_stats_.entries_inserted;
        ++priority_stats->entries_inserted;
        break;
      case Cache::CacheWarmupInsertResult::kDuplicate:
        ++warmup_stats_.entries_duplicate;
        ++priority_stats->entries_duplicate;
        break;
      case Cache::CacheWarmupInsertResult::kRejectedNoSpace:
        ++warmup_stats_.entries_rejected_no_space;
        ++priority_stats->entries_rejected_no_space;
        break;
      default:
        ++warmup_stats_.skipped_invalid;
        return finish(
            IOStatus::Corruption("unknown cache warmup admission outcome"));
    }
  }
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
