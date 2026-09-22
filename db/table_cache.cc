//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/table_cache.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

#include "db/dbformat.h"
#include "db/range_tombstone_fragmenter.h"
#include "db/snapshot_impl.h"
#include "db/version_edit.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "file/random_access_file_reader.h"
#include "monitoring/perf_context_imp.h"
#include "rocksdb/advanced_options.h"
#include "rocksdb/statistics.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/get_context.h"
#include "table/internal_iterator.h"
#include "table/iterator_wrapper.h"
#include "table/multiget_context.h"
#include "table/table_builder.h"
#include "table/table_reader.h"
#include "test_util/sync_point.h"
#include "util/cast_util.h"
#include "util/coding.h"
#include "util/stop_watch.h"

namespace ROCKSDB_NAMESPACE {
namespace {
template <class T>
static void DeleteEntry(const Slice& /*key*/, void* value) {
  T* typed_value = reinterpret_cast<T*>(value);
  delete typed_value;
}
}  // anonymous namespace
}  // namespace ROCKSDB_NAMESPACE

// Generate the regular and coroutine versions of some methods by
// including table_cache_sync_and_async.h twice
// Macros in the header will expand differently based on whether
// WITH_COROUTINES or WITHOUT_COROUTINES is defined
// clang-format off
#define WITHOUT_COROUTINES
#include "db/table_cache_sync_and_async.h"
#undef WITHOUT_COROUTINES
#define WITH_COROUTINES
#include "db/table_cache_sync_and_async.h"
#undef WITH_COROUTINES
// clang-format on

namespace ROCKSDB_NAMESPACE {

namespace {

static void UnrefEntry(void* arg1, void* arg2) {
  Cache* cache = reinterpret_cast<Cache*>(arg1);
  Cache::Handle* h = reinterpret_cast<Cache::Handle*>(arg2);
  cache->Release(h);
}

class ScopedTableCacheWarmupLease {
 public:
  ScopedTableCacheWarmupLease(Cache* cache, Cache::Handle* handle,
                              Cache::Priority priority)
      : cache_(cache), handle_(handle), priority_(priority) {}
  ~ScopedTableCacheWarmupLease() { Release(); }

  ScopedTableCacheWarmupLease(const ScopedTableCacheWarmupLease&) = delete;
  ScopedTableCacheWarmupLease& operator=(
      const ScopedTableCacheWarmupLease&) = delete;

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

static Slice GetSliceForFileNumber(const uint64_t* file_number) {
  return Slice(reinterpret_cast<const char*>(file_number),
               sizeof(*file_number));
}

bool RangeContains(uint64_t range_offset, size_t range_size,
                   uint64_t read_offset, size_t read_size) {
  if (read_offset < range_offset || read_size > range_size) {
    return false;
  }
  return read_offset - range_offset <= range_size - read_size;
}

// Records exactly the bytes read while a TableReader is constructed. The
// wrapper is installed only when the experimental option is enabled, so the
// baseline path has no virtual wrapper, copies, or retained memory.
class TableCacheWarmupRecordingFile final
    : public FSRandomAccessFileOwnerWrapper {
 public:
  explicit TableCacheWarmupRecordingFile(
      std::unique_ptr<FSRandomAccessFile>&& target, size_t max_capture_bytes)
      : FSRandomAccessFileOwnerWrapper(std::move(target)),
        max_capture_bytes_(max_capture_bytes) {}

  IOStatus Read(uint64_t offset, size_t n, const IOOptions& options,
                Slice* result, char* scratch,
                IODebugContext* dbg) const override {
    IOStatus s = target()->Read(offset, n, options, result, scratch, dbg);
    if (s.ok() && !result->empty()) {
      Record(offset, *result);
    }
    return s;
  }

  IOStatus MultiRead(FSReadRequest* reqs, size_t num_reqs,
                     const IOOptions& options, IODebugContext* dbg) override {
    IOStatus s = target()->MultiRead(reqs, num_reqs, options, dbg);
    if (s.ok() && capture_.load(std::memory_order_acquire)) {
      for (size_t i = 0; i < num_reqs; ++i) {
        if (reqs[i].status.ok() && !reqs[i].result.empty()) {
          Record(reqs[i].offset, reqs[i].result);
        }
      }
    }
    return s;
  }

  std::shared_ptr<const TableCacheWarmupBundle> FinishCapture(
      uint64_t file_size) {
    std::vector<TableCacheWarmupRange> captured;
    {
      MutexLock lock(&mu_);
      capture_.store(false, std::memory_order_release);
      captured.swap(ranges_);
    }

    std::sort(
        captured.begin(), captured.end(),
        [](const TableCacheWarmupRange& lhs, const TableCacheWarmupRange& rhs) {
          return lhs.offset < rhs.offset;
        });
    std::vector<TableCacheWarmupRange> merged;
    merged.reserve(captured.size());
    for (auto& range : captured) {
      if (range.offset >= file_size || range.data.empty()) {
        continue;
      }
      const uint64_t available = file_size - range.offset;
      if (range.data.size() > available) {
        range.data.resize(static_cast<size_t>(available));
      }
      if (merged.empty()) {
        merged.push_back(std::move(range));
        continue;
      }
      auto& previous = merged.back();
      const uint64_t previous_end = previous.offset + previous.data.size();
      if (range.offset > previous_end) {
        merged.push_back(std::move(range));
        continue;
      }
      const size_t overlap = static_cast<size_t>(previous_end - range.offset);
      if (overlap < range.data.size()) {
        previous.data.append(range.data.data() + overlap,
                             range.data.size() - overlap);
      }
    }
    if (merged.empty()) {
      return nullptr;
    }
    auto bundle = std::make_shared<TableCacheWarmupBundle>();
    bundle->file_size = file_size;
    bundle->ranges = std::move(merged);
    return bundle;
  }

 private:
  void Record(uint64_t offset, const Slice& data) const {
    // The recording wrapper remains installed for the reader's lifetime. Keep
    // post-open data reads to one predictable atomic branch instead of taking
    // the capture mutex forever.
    if (!capture_.load(std::memory_order_acquire)) {
      return;
    }
    MutexLock lock(&mu_);
    if (!capture_.load(std::memory_order_relaxed)) {
      return;
    }
    if (data.size() > max_capture_bytes_ - captured_bytes_) {
      // Foreground opening still succeeds. This reader simply becomes
      // unavailable to the best-effort warmup plane.
      capture_.store(false, std::memory_order_release);
      ranges_.clear();
      captured_bytes_ = 0;
      return;
    }
    TableCacheWarmupRange range;
    range.offset = offset;
    range.data.assign(data.data(), data.size());
    ranges_.push_back(std::move(range));
    captured_bytes_ += data.size();
  }

  const size_t max_capture_bytes_;
  mutable port::Mutex mu_;
  mutable std::atomic<bool> capture_{true};
  mutable size_t captured_bytes_ = 0;
  mutable std::vector<TableCacheWarmupRange> ranges_;
};

// Serves captured metadata without opening the SST. Once TableReader::Open has
// parsed the bundle, DropBundle() releases the bytes; the first later data read
// lazily opens the real file. Thus table-cache warmup itself has no shared
// storage open or read.
class TableCacheWarmupRandomAccessFile final : public FSRandomAccessFile {
 public:
  TableCacheWarmupRandomAccessFile(
      std::shared_ptr<FileSystem> fs, std::string file_name,
      const FileOptions& file_options,
      std::shared_ptr<const TableCacheWarmupBundle> bundle)
      : fs_(std::move(fs)),
        file_name_(std::move(file_name)),
        file_options_(file_options),
        bundle_active_(bundle != nullptr),
        bundle_(std::move(bundle)) {}

  IOStatus Read(uint64_t offset, size_t n, const IOOptions& options,
                Slice* result, char* scratch,
                IODebugContext* dbg) const override {
    auto bundle = GetActiveBundle();
    if (bundle != nullptr) {
      return ReadFromBundle(*bundle, offset, n, result, scratch);
    }

    FSRandomAccessFile* target = nullptr;
    IOStatus s = EnsureTarget(&target);
    if (!s.ok()) {
      return s;
    }
    return target->Read(offset, n, options, result, scratch, dbg);
  }

  IOStatus MultiRead(FSReadRequest* reqs, size_t num_reqs,
                     const IOOptions& options, IODebugContext* dbg) override {
    auto bundle = GetActiveBundle();
    if (bundle != nullptr) {
      for (size_t i = 0; i < num_reqs; ++i) {
        reqs[i].status = ReadFromBundle(*bundle, reqs[i].offset, reqs[i].len,
                                        &reqs[i].result, reqs[i].scratch);
      }
      return IOStatus::OK();
    }
    FSRandomAccessFile* target = nullptr;
    IOStatus s = EnsureTarget(&target);
    return s.ok() ? target->MultiRead(reqs, num_reqs, options, dbg) : s;
  }

  IOStatus Prefetch(uint64_t offset, size_t n, const IOOptions& options,
                    IODebugContext* dbg) override {
    auto bundle = GetActiveBundle();
    if (bundle != nullptr) {
      for (const auto& range : bundle->ranges) {
        if (RangeContains(range.offset, range.data.size(), offset, n)) {
          return IOStatus::OK();
        }
      }
      return IOStatus::NotSupported(
          "warmup bundle intentionally disables broad prefetch");
    }
    FSRandomAccessFile* target = nullptr;
    IOStatus s = EnsureTarget(&target);
    return s.ok() ? target->Prefetch(offset, n, options, dbg) : s;
  }

  size_t GetUniqueId(char* id, size_t max_size) const override {
    if (GetActiveBundle() != nullptr) {
      return 0;
    }
    FSRandomAccessFile* target = nullptr;
    return EnsureTarget(&target).ok() ? target->GetUniqueId(id, max_size) : 0;
  }

  void Hint(AccessPattern pattern) override {
    if (!bundle_active_.load(std::memory_order_acquire)) {
      FSRandomAccessFile* target = nullptr;
      if (EnsureTarget(&target).ok()) {
        target->Hint(pattern);
      }
      return;
    }
    MutexLock lock(&mu_);
    if (target_ != nullptr) {
      target_->Hint(pattern);
    } else {
      pending_hint_ = pattern;
      has_pending_hint_ = true;
    }
  }

  bool use_direct_io() const override {
    if (GetActiveBundle() != nullptr) {
      return false;
    }
    FSRandomAccessFile* target = nullptr;
    return EnsureTarget(&target).ok() && target->use_direct_io();
  }

  size_t GetRequiredBufferAlignment() const override {
    if (GetActiveBundle() != nullptr) {
      return kDefaultPageSize;
    }
    FSRandomAccessFile* target = nullptr;
    return EnsureTarget(&target).ok() ? target->GetRequiredBufferAlignment()
                                      : kDefaultPageSize;
  }

  IOStatus InvalidateCache(size_t offset, size_t length) override {
    if (GetActiveBundle() != nullptr) {
      // The overlay is immutable private memory, so there is no underlying
      // filesystem cache to invalidate during strict replay.
      return IOStatus::OK();
    }
    FSRandomAccessFile* target = nullptr;
    IOStatus s = EnsureTarget(&target);
    return s.ok() ? target->InvalidateCache(offset, length) : s;
  }

  IOStatus ReadAsync(FSReadRequest& req, const IOOptions& opts,
                     std::function<void(const FSReadRequest&, void*)> cb,
                     void* cb_arg, void** io_handle, IOHandleDeleter* del_fn,
                     IODebugContext* dbg) override {
    auto bundle = GetActiveBundle();
    if (bundle != nullptr) {
      // Match FSRandomAccessFile's synchronous fallback semantics while
      // guaranteeing an active replay can never fall through to storage.
      req.status = ReadFromBundle(*bundle, req.offset, req.len, &req.result,
                                  req.scratch);
      cb(req, cb_arg);
      return IOStatus::OK();
    }
    FSRandomAccessFile* target = nullptr;
    IOStatus s = EnsureTarget(&target);
    return s.ok() ? target->ReadAsync(req, opts, std::move(cb), cb_arg,
                                      io_handle, del_fn, dbg)
                  : s;
  }

  Temperature GetTemperature() const override {
    if (GetActiveBundle() != nullptr) {
      return file_options_.temperature;
    }
    FSRandomAccessFile* target = nullptr;
    return EnsureTarget(&target).ok() ? target->GetTemperature()
                                      : Temperature::kUnknown;
  }

  void DropBundle() {
    {
      MutexLock lock(&mu_);
      bundle_.reset();
    }
    // Publishing this only after reset lets steady-state calls skip mu_ while
    // preserving the bundle's lifetime for any read that already copied it.
    bundle_active_.store(false, std::memory_order_release);
  }

 private:
  static IOStatus ReadFromBundle(const TableCacheWarmupBundle& bundle,
                                 uint64_t offset, size_t n, Slice* result,
                                 char* scratch) {
    for (const auto& range : bundle.ranges) {
      if (!RangeContains(range.offset, range.data.size(), offset, n)) {
        continue;
      }
      const size_t relative = static_cast<size_t>(offset - range.offset);
      if (scratch != nullptr) {
        std::memcpy(scratch, range.data.data() + relative, n);
        *result = Slice(scratch, n);
      } else {
        *result = Slice(range.data.data() + relative, n);
      }
      return IOStatus::OK();
    }
    return IOStatus::Corruption(
        "table-cache warmup bundle does not cover metadata read");
  }

  std::shared_ptr<const TableCacheWarmupBundle> GetActiveBundle() const {
    if (!bundle_active_.load(std::memory_order_acquire)) {
      return nullptr;
    }
    MutexLock lock(&mu_);
    return bundle_;
  }

  IOStatus EnsureTarget(FSRandomAccessFile** target) const {
    FSRandomAccessFile* current = target_raw_.load(std::memory_order_acquire);
    if (current != nullptr) {
      *target = current;
      return IOStatus::OK();
    }
    MutexLock lock(&mu_);
    if (target_ == nullptr) {
      IOStatus s = fs_->NewRandomAccessFile(file_name_, file_options_, &target_,
                                            nullptr);
      if (!s.ok()) {
        return s;
      }
      if (has_pending_hint_) {
        target_->Hint(pending_hint_);
      }
      target_raw_.store(target_.get(), std::memory_order_release);
    }
    *target = target_.get();
    return IOStatus::OK();
  }

  std::shared_ptr<FileSystem> fs_;
  std::string file_name_;
  FileOptions file_options_;
  std::atomic<bool> bundle_active_;
  mutable port::Mutex mu_;
  mutable std::unique_ptr<FSRandomAccessFile> target_;
  mutable std::atomic<FSRandomAccessFile*> target_raw_{nullptr};
  mutable std::shared_ptr<const TableCacheWarmupBundle> bundle_;
  AccessPattern pending_hint_ = kNormal;
  bool has_pending_hint_ = false;
};

#ifndef ROCKSDB_LITE

void AppendVarint64(IterKey* key, uint64_t v) {
  char buf[10];
  auto ptr = EncodeVarint64(buf, v);
  key->TrimAppend(key->Size(), buf, ptr - buf);
}

#endif  // ROCKSDB_LITE

}  // anonymous namespace

const int kLoadConcurency = 128;

TableCache::TableCache(const ImmutableOptions& ioptions,
                       const FileOptions* file_options, Cache* const cache,
                       BlockCacheTracer* const block_cache_tracer,
                       const std::shared_ptr<IOTracer>& io_tracer,
                       const std::string& db_session_id)
    : ioptions_(ioptions),
      file_options_(*file_options),
      cache_(cache),
      immortal_tables_(false),
      block_cache_tracer_(block_cache_tracer),
      loader_mutex_(kLoadConcurency, kGetSliceNPHash64UnseededFnPtr),
      io_tracer_(io_tracer),
      db_session_id_(db_session_id) {
  if (ioptions_.row_cache) {
    // If the same cache is shared by multiple instances, we need to
    // disambiguate its entries.
    PutVarint64(&row_cache_id_, ioptions_.row_cache->NewId());
  }
}

TableCache::~TableCache() {}

TableReader* TableCache::GetTableReaderFromHandle(Cache::Handle* handle) {
  return reinterpret_cast<TableReader*>(cache_->Value(handle));
}

void TableCache::ReleaseHandle(Cache::Handle* handle) {
  cache_->Release(handle);
}

Status TableCache::GetTableReader(
    const ReadOptions& ro, const FileOptions& file_options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta, bool sequential_mode, bool record_read_stats,
    HistogramImpl* file_read_hist, std::unique_ptr<TableReader>* table_reader,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    bool skip_filters, int level, bool prefetch_index_and_filter_in_cache,
    size_t max_file_size_for_l0_meta_pin, Temperature file_temperature,
    std::shared_ptr<const TableCacheWarmupBundle> warmup_bundle) {
  // [relink] If this file is an in-place external reference, open it directly
  // at its absolute HDFS path (no rename happened). Empty external_path =>
  // identical to baseline: derive path from cf_paths + file number.
  const bool is_external = !file_meta.fd.external_path.empty();
  std::string fname =
      is_external
          ? file_meta.fd.external_path
          : TableFileName(ioptions_.cf_paths, file_meta.fd.GetNumber(),
                          file_meta.fd.GetPathId());
  std::unique_ptr<FSRandomAccessFile> file;
  FileOptions fopts = file_options;
  fopts.temperature = file_temperature;
  TableCacheWarmupRandomAccessFile* warmup_file = nullptr;
  TableCacheWarmupRecordingFile* recording_file = nullptr;

  Status s;
  if (warmup_bundle != nullptr) {
    if (warmup_bundle->file_size != file_meta.fd.GetFileSize() ||
        warmup_bundle->ranges.empty()) {
      return Status::Corruption(
          "table-cache warmup bundle has invalid file identity or no ranges");
    }
    auto memory_file = std::make_unique<TableCacheWarmupRandomAccessFile>(
        ioptions_.fs, fname, fopts, warmup_bundle);
    warmup_file = memory_file.get();
    file = std::move(memory_file);
  } else {
    s = PrepareIOFromReadOptions(ro, ioptions_.clock, fopts.io_options);
    if (s.ok()) {
      s = ioptions_.fs->NewRandomAccessFile(fname, fopts, &file, nullptr);
    }
  }
  if (s.IsPathNotFound() && !is_external && warmup_bundle == nullptr) {
    // [relink] external (absolute) paths are exact; never apply the
    // Rocks2Level fallback rewrite to them.
    fname = Rocks2LevelTableFileName(fname);
    s = PrepareIOFromReadOptions(ro, ioptions_.clock, fopts.io_options);
    if (s.ok()) {
      s = ioptions_.fs->NewRandomAccessFile(fname, file_options, &file,
                                            nullptr);
    }
  }
  if (s.ok() && warmup_bundle == nullptr) {
    RecordTick(ioptions_.stats, NO_FILE_OPENS);
    if (ioptions_.table_factory->ShouldCaptureTableCacheWarmup()) {
      const size_t max_capture_bytes =
          ioptions_.table_factory->TableCacheWarmupMaxBytes();
      if (max_capture_bytes > 0) {
        auto recording = std::make_unique<TableCacheWarmupRecordingFile>(
            std::move(file), max_capture_bytes);
        recording_file = recording.get();
        file = std::move(recording);
      }
    }
  }

  if (s.ok()) {
    if (!sequential_mode && ioptions_.advise_random_on_open) {
      file->Hint(FSRandomAccessFile::kRandom);
    }
    StopWatch sw(ioptions_.clock, ioptions_.stats, TABLE_OPEN_IO_MICROS);
    std::unique_ptr<RandomAccessFileReader> file_reader(
        new RandomAccessFileReader(
            std::move(file), fname, ioptions_.clock, io_tracer_,
            record_read_stats && warmup_bundle == nullptr ? ioptions_.stats
                                                          : nullptr,
            SST_READ_MICROS,
            warmup_bundle == nullptr ? file_read_hist : nullptr,
            ioptions_.rate_limiter.get(), ioptions_.listeners, file_temperature,
            level == ioptions_.num_levels - 1));
    UniqueId64x2 expected_unique_id;
    if (warmup_bundle != nullptr ||
        ioptions_.verify_sst_unique_id_in_manifest) {
      expected_unique_id = file_meta.unique_id;
    } else {
      expected_unique_id = kNullUniqueId64x2;  // null ID == no verification
    }
    // A reader first opened by a filter-skipping caller must still capture the
    // filter metadata that destination replay (which builds a general-purpose
    // reader) will require.
    const bool table_open_skip_filters =
        recording_file != nullptr ? false : skip_filters;
    TableReaderOptions tro(
        ioptions_, prefix_extractor, file_options, internal_comparator,
        table_open_skip_filters, immortal_tables_,
        false /* force_direct_prefetch */, level, block_cache_tracer_,
        max_file_size_for_l0_meta_pin, db_session_id_, file_meta.fd.GetNumber(),
        expected_unique_id, file_meta.fd.largest_seqno);
    tro.global_seqno_override = file_meta.fd.global_seqno_override;  // [relink] inert when kDisable
    tro.skip_tail_prefetch =
        warmup_bundle != nullptr || recording_file != nullptr;
    tro.cache_warmup_replay = warmup_bundle != nullptr;
    tro.cache_warmup_capture = recording_file != nullptr;
    s = ioptions_.table_factory->NewTableReader(
        ro, tro, std::move(file_reader), file_meta.fd.GetFileSize(), table_reader,
        prefetch_index_and_filter_in_cache);
    if (s.ok() && table_reader->get() != nullptr) {
      if (recording_file != nullptr) {
        auto captured =
            recording_file->FinishCapture(file_meta.fd.GetFileSize());
        if (captured != nullptr) {
          // Capture is optional background-warmup state. A full reservation
          // cache must never turn an otherwise successful foreground table
          // open into an error; simply publish this reader without a bundle.
          (*table_reader)
              ->SetTableCacheWarmupBundle(std::move(captured))
              .PermitUncheckedError();
        }
      } else if (warmup_bundle != nullptr) {
        // Keep the received ranges with the reader so a later chained relink
        // can hand them on without touching shared storage.
        s = (*table_reader)->SetTableCacheWarmupBundle(warmup_bundle);
      }
      if (s.ok() && warmup_bundle != nullptr && record_read_stats) {
        (*table_reader)->SetFileReadStats(ioptions_.stats, file_read_hist);
      }
      if (s.ok() && warmup_file != nullptr) {
        // All metadata has now been parsed. Future data reads lazily open the
        // real SST; the duplicate overlay bytes are retained only by the
        // TableReader bundle above.
        warmup_file->DropBundle();
      }
      if (!s.ok()) {
        table_reader->reset();
      }
    }
    TEST_SYNC_POINT("TableCache::GetTableReader:0");
  }
  return s;
}

void TableCache::EraseHandle(const FileDescriptor& fd, Cache::Handle* handle) {
  ReleaseHandle(handle);
  uint64_t number = fd.GetNumber();
  Slice key = GetSliceForFileNumber(&number);
  cache_->Erase(key);
}

Status TableCache::FindTable(
    const ReadOptions& ro, const FileOptions& file_options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta, Cache::Handle** handle,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    const bool no_io, bool record_read_stats, HistogramImpl* file_read_hist,
    bool skip_filters, int level, bool prefetch_index_and_filter_in_cache,
    size_t max_file_size_for_l0_meta_pin, Temperature file_temperature) {
  PERF_TIMER_GUARD_WITH_CLOCK(find_table_nanos, ioptions_.clock);
  uint64_t number = file_meta.fd.GetNumber();
  Slice key = GetSliceForFileNumber(&number);
  *handle = cache_->Lookup(key);
  TEST_SYNC_POINT_CALLBACK("TableCache::FindTable:0",
                           const_cast<bool*>(&no_io));

  if (*handle == nullptr) {
    if (no_io) {
      return Status::Incomplete("Table not found in table_cache, no_io is set");
    }
    MutexLock load_lock(loader_mutex_.get(key));
    // We check the cache again under loading mutex
    *handle = cache_->Lookup(key);
    if (*handle != nullptr) {
      return Status::OK();
    }

    std::unique_ptr<TableReader> table_reader;
    Status s =
        GetTableReader(ro, file_options, internal_comparator, file_meta,
                       false /* sequential mode */, record_read_stats,
                       file_read_hist, &table_reader, prefix_extractor,
                       skip_filters, level, prefetch_index_and_filter_in_cache,
                       max_file_size_for_l0_meta_pin, file_temperature);
    if (!s.ok()) {
      assert(table_reader == nullptr);
      RecordTick(ioptions_.stats, NO_FILE_ERRORS);
      // We do not cache error results so that if the error is transient,
      // or somebody repairs the file, we recover automatically.
    } else {
      // [relink tail-preload 2026-09-20] Level-derived admission priority. A point
      // read probes EVERY L0 file (they overlap) but only one file per deeper
      // level, so per-file probe rate drops by roughly the level fanout; under a
      // finite max_open_files the shallow files are worth far more per slot.
      // Only two tiers are used on purpose: the table cache is built with
      // LRUCacheOptions defaults (db_impl.cc), where low_pri_pool_ratio is 0.0, so
      // LOW and BOTTOM are indistinguishable -- a three-tier mapping would be a
      // no-op without also setting that ratio. Gated off by default, in which case
      // this is the stock 4-argument Insert (priority LOW for everything).
      s = level_priority_.load(std::memory_order_relaxed)
              ? cache_->Insert(
                    key, table_reader.get(), 1, &DeleteEntry<TableReader>,
                    handle,
                    level == 0 ? Cache::Priority::HIGH : Cache::Priority::LOW)
              : cache_->Insert(key, table_reader.get(), 1,
                               &DeleteEntry<TableReader>, handle);
      if (s.ok()) {
        // Release ownership of table reader.
        table_reader.release();
      }
    }
    return s;
  }
  return Status::OK();
}

InternalIterator* TableCache::NewIterator(
    const ReadOptions& options, const FileOptions& file_options,
    const InternalKeyComparator& icomparator, const FileMetaData& file_meta,
    RangeDelAggregator* range_del_agg,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    TableReader** table_reader_ptr, HistogramImpl* file_read_hist,
    TableReaderCaller caller, Arena* arena, bool skip_filters, int level,
    size_t max_file_size_for_l0_meta_pin,
    const InternalKey* smallest_compaction_key,
    const InternalKey* largest_compaction_key, bool allow_unprepared_value,
    TruncatedRangeDelIterator** range_del_iter) {
  PERF_TIMER_GUARD(new_table_iterator_nanos);

  Status s;
  TableReader* table_reader = nullptr;
  Cache::Handle* handle = nullptr;
  if (table_reader_ptr != nullptr) {
    *table_reader_ptr = nullptr;
  }
  bool for_compaction = caller == TableReaderCaller::kCompaction;
  auto& fd = file_meta.fd;
  table_reader = fd.table_reader;
  if (table_reader == nullptr) {
    s = FindTable(
        options, file_options, icomparator, file_meta, &handle,
        prefix_extractor, options.read_tier == kBlockCacheTier /* no_io */,
        !for_compaction /* record_read_stats */, file_read_hist, skip_filters,
        level, true /* prefetch_index_and_filter_in_cache */,
        max_file_size_for_l0_meta_pin, file_meta.temperature);
    if (s.ok()) {
      table_reader = GetTableReaderFromHandle(handle);
    }
  }
  InternalIterator* result = nullptr;
  if (s.ok()) {
    if (options.table_filter &&
        !options.table_filter(*table_reader->GetTableProperties())) {
      result = NewEmptyInternalIterator<Slice>(arena);
    } else {
      result = table_reader->NewIterator(
          options, prefix_extractor.get(), arena, skip_filters, caller,
          file_options.compaction_readahead_size, allow_unprepared_value);
    }
    if (handle != nullptr) {
      result->RegisterCleanup(&UnrefEntry, cache_, handle);
      handle = nullptr;  // prevent from releasing below
    }

    if (for_compaction) {
      table_reader->SetupForCompaction();
    }
    if (table_reader_ptr != nullptr) {
      *table_reader_ptr = table_reader;
    }
  }
  if (s.ok() && !options.ignore_range_deletions) {
    if (range_del_iter != nullptr) {
      auto new_range_del_iter =
          table_reader->NewRangeTombstoneIterator(options);
      if (new_range_del_iter == nullptr || new_range_del_iter->empty()) {
        delete new_range_del_iter;
        *range_del_iter = nullptr;
      } else {
        *range_del_iter = new TruncatedRangeDelIterator(
            std::unique_ptr<FragmentedRangeTombstoneIterator>(
                new_range_del_iter),
            &icomparator, &file_meta.smallest, &file_meta.largest);
      }
    }
    if (range_del_agg != nullptr) {
      if (range_del_agg->AddFile(fd.GetNumber())) {
        std::unique_ptr<FragmentedRangeTombstoneIterator> new_range_del_iter(
            static_cast<FragmentedRangeTombstoneIterator*>(
                table_reader->NewRangeTombstoneIterator(options)));
        if (new_range_del_iter != nullptr) {
          s = new_range_del_iter->status();
        }
        if (s.ok()) {
          const InternalKey* smallest = &file_meta.smallest;
          const InternalKey* largest = &file_meta.largest;
          if (smallest_compaction_key != nullptr) {
            smallest = smallest_compaction_key;
          }
          if (largest_compaction_key != nullptr) {
            largest = largest_compaction_key;
          }
          range_del_agg->AddTombstones(std::move(new_range_del_iter), smallest,
                                       largest);
        }
      }
    }
  }

  if (handle != nullptr) {
    ReleaseHandle(handle);
  }
  if (!s.ok()) {
    assert(result == nullptr);
    result = NewErrorInternalIterator<Slice>(s, arena);
  }
  return result;
}

Status TableCache::GetRangeTombstoneIterator(
    const ReadOptions& options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta,
    std::unique_ptr<FragmentedRangeTombstoneIterator>* out_iter) {
  assert(out_iter);
  const FileDescriptor& fd = file_meta.fd;
  Status s;
  TableReader* t = fd.table_reader;
  Cache::Handle* handle = nullptr;
  if (t == nullptr) {
    s = FindTable(options, file_options_, internal_comparator, file_meta,
                  &handle);
    if (s.ok()) {
      t = GetTableReaderFromHandle(handle);
    }
  }
  if (s.ok()) {
    // Note: NewRangeTombstoneIterator could return nullptr
    out_iter->reset(t->NewRangeTombstoneIterator(options));
  }
  if (handle) {
    if (*out_iter) {
      (*out_iter)->RegisterCleanup(&UnrefEntry, cache_, handle);
    } else {
      ReleaseHandle(handle);
    }
  }
  return s;
}

#ifndef ROCKSDB_LITE
void TableCache::CreateRowCacheKeyPrefix(const ReadOptions& options,
                                         const FileDescriptor& fd,
                                         const Slice& internal_key,
                                         GetContext* get_context,
                                         IterKey& row_cache_key) {
  uint64_t fd_number = fd.GetNumber();
  // We use the user key as cache key instead of the internal key,
  // otherwise the whole cache would be invalidated every time the
  // sequence key increases. However, to support caching snapshot
  // reads, we append the sequence number (incremented by 1 to
  // distinguish from 0) only in this case.
  // If the snapshot is larger than the largest seqno in the file,
  // all data should be exposed to the snapshot, so we treat it
  // the same as there is no snapshot. The exception is that if
  // a seq-checking callback is registered, some internal keys
  // may still be filtered out.
  uint64_t seq_no = 0;
  // Maybe we can include the whole file ifsnapshot == fd.largest_seqno.
  if (options.snapshot != nullptr &&
      (get_context->has_callback() ||
       static_cast_with_check<const SnapshotImpl>(options.snapshot)
               ->GetSequenceNumber() <= fd.largest_seqno)) {
    // We should consider to use options.snapshot->GetSequenceNumber()
    // instead of GetInternalKeySeqno(k), which will make the code
    // easier to understand.
    seq_no = 1 + GetInternalKeySeqno(internal_key);
  }

  // Compute row cache key.
  row_cache_key.TrimAppend(row_cache_key.Size(), row_cache_id_.data(),
                           row_cache_id_.size());
  AppendVarint64(&row_cache_key, fd_number);
  AppendVarint64(&row_cache_key, seq_no);
}

bool TableCache::GetFromRowCache(const Slice& user_key, IterKey& row_cache_key,
                                 size_t prefix_size, GetContext* get_context) {
  bool found = false;

  row_cache_key.TrimAppend(prefix_size, user_key.data(), user_key.size());
  if (auto row_handle =
          ioptions_.row_cache->Lookup(row_cache_key.GetUserKey())) {
    // Cleanable routine to release the cache entry
    Cleanable value_pinner;
    auto release_cache_entry_func = [](void* cache_to_clean,
                                       void* cache_handle) {
      ((Cache*)cache_to_clean)->Release((Cache::Handle*)cache_handle);
    };
    auto found_row_cache_entry =
        static_cast<const std::string*>(ioptions_.row_cache->Value(row_handle));
    // If it comes here value is located on the cache.
    // found_row_cache_entry points to the value on cache,
    // and value_pinner has cleanup procedure for the cached entry.
    // After replayGetContextLog() returns, get_context.pinnable_slice_
    // will point to cache entry buffer (or a copy based on that) and
    // cleanup routine under value_pinner will be delegated to
    // get_context.pinnable_slice_. Cache entry is released when
    // get_context.pinnable_slice_ is reset.
    value_pinner.RegisterCleanup(release_cache_entry_func,
                                 ioptions_.row_cache.get(), row_handle);
    replayGetContextLog(*found_row_cache_entry, user_key, get_context,
                        &value_pinner);
    RecordTick(ioptions_.stats, ROW_CACHE_HIT);
    found = true;
  } else {
    RecordTick(ioptions_.stats, ROW_CACHE_MISS);
  }
  return found;
}
#endif  // ROCKSDB_LITE

Status TableCache::Get(
    const ReadOptions& options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta, const Slice& k, GetContext* get_context,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    HistogramImpl* file_read_hist, bool skip_filters, int level,
    size_t max_file_size_for_l0_meta_pin) {
  auto& fd = file_meta.fd;
  std::string* row_cache_entry = nullptr;
  bool done = false;
#ifndef ROCKSDB_LITE
  IterKey row_cache_key;
  std::string row_cache_entry_buffer;

  // Check row cache if enabled. Since row cache does not currently store
  // sequence numbers, we cannot use it if we need to fetch the sequence.
  if (ioptions_.row_cache && !get_context->NeedToReadSequence()) {
    auto user_key = ExtractUserKey(k);
    CreateRowCacheKeyPrefix(options, fd, k, get_context, row_cache_key);
    done = GetFromRowCache(user_key, row_cache_key, row_cache_key.Size(),
                           get_context);
    if (!done) {
      row_cache_entry = &row_cache_entry_buffer;
    }
  }
#endif  // ROCKSDB_LITE
  Status s;
  TableReader* t = fd.table_reader;
  Cache::Handle* handle = nullptr;
  if (!done) {
    assert(s.ok());
    if (t == nullptr) {
      s = FindTable(options, file_options_, internal_comparator, file_meta,
                    &handle, prefix_extractor,
                    options.read_tier == kBlockCacheTier /* no_io */,
                    true /* record_read_stats */, file_read_hist, skip_filters,
                    level, true /* prefetch_index_and_filter_in_cache */,
                    max_file_size_for_l0_meta_pin, file_meta.temperature);
      if (s.ok()) {
        t = GetTableReaderFromHandle(handle);
      }
    }
    SequenceNumber* max_covering_tombstone_seq =
        get_context->max_covering_tombstone_seq();
    if (s.ok() && max_covering_tombstone_seq != nullptr &&
        !options.ignore_range_deletions) {
      std::unique_ptr<FragmentedRangeTombstoneIterator> range_del_iter(
          t->NewRangeTombstoneIterator(options));
      if (range_del_iter != nullptr) {
        SequenceNumber seq =
            range_del_iter->MaxCoveringTombstoneSeqnum(ExtractUserKey(k));
        if (seq > *max_covering_tombstone_seq) {
          *max_covering_tombstone_seq = seq;
          if (get_context->NeedTimestamp()) {
            get_context->SetTimestampFromRangeTombstone(
                range_del_iter->timestamp());
          }
        }
      }
    }
    if (s.ok()) {
      get_context->SetReplayLog(row_cache_entry);  // nullptr if no cache.
      s = t->Get(options, k, get_context, prefix_extractor.get(), skip_filters);
      get_context->SetReplayLog(nullptr);
    } else if (options.read_tier == kBlockCacheTier && s.IsIncomplete()) {
      // Couldn't find Table in cache but treat as kFound if no_io set
      get_context->MarkKeyMayExist();
      s = Status::OK();
      done = true;
    }
  }

#ifndef ROCKSDB_LITE
  // Put the replay log in row cache only if something was found.
  if (!done && s.ok() && row_cache_entry && !row_cache_entry->empty()) {
    size_t charge = row_cache_entry->capacity() + sizeof(std::string);
    void* row_ptr = new std::string(std::move(*row_cache_entry));
    // If row cache is full, it's OK to continue.
    ioptions_.row_cache
        ->Insert(row_cache_key.GetUserKey(), row_ptr, charge,
                 &DeleteEntry<std::string>)
        .PermitUncheckedError();
  }
#endif  // ROCKSDB_LITE

  if (handle != nullptr) {
    ReleaseHandle(handle);
  }
  return s;
}

void TableCache::UpdateRangeTombstoneSeqnums(
    const ReadOptions& options, TableReader* t,
    MultiGetContext::Range& table_range) {
  std::unique_ptr<FragmentedRangeTombstoneIterator> range_del_iter(
      t->NewRangeTombstoneIterator(options));
  if (range_del_iter != nullptr) {
    for (auto iter = table_range.begin(); iter != table_range.end(); ++iter) {
      SequenceNumber* max_covering_tombstone_seq =
          iter->get_context->max_covering_tombstone_seq();
      SequenceNumber seq =
          range_del_iter->MaxCoveringTombstoneSeqnum(iter->ukey_with_ts);
      if (seq > *max_covering_tombstone_seq) {
        *max_covering_tombstone_seq = seq;
        if (iter->get_context->NeedTimestamp()) {
          iter->get_context->SetTimestampFromRangeTombstone(
              range_del_iter->timestamp());
        }
      }
    }
  }
}

Status TableCache::MultiGetFilter(
    const ReadOptions& options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    HistogramImpl* file_read_hist, int level,
    MultiGetContext::Range* mget_range, Cache::Handle** table_handle) {
  auto& fd = file_meta.fd;
#ifndef ROCKSDB_LITE
  IterKey row_cache_key;
  std::string row_cache_entry_buffer;

  // Check if we need to use the row cache. If yes, then we cannot do the
  // filtering here, since the filtering needs to happen after the row cache
  // lookup.
  KeyContext& first_key = *mget_range->begin();
  if (ioptions_.row_cache && !first_key.get_context->NeedToReadSequence()) {
    return Status::NotSupported();
  }
#endif  // ROCKSDB_LITE
  Status s;
  TableReader* t = fd.table_reader;
  Cache::Handle* handle = nullptr;
  MultiGetContext::Range tombstone_range(*mget_range, mget_range->begin(),
                                         mget_range->end());
  if (t == nullptr) {
    s = FindTable(
        options, file_options_, internal_comparator, file_meta, &handle,
        prefix_extractor, options.read_tier == kBlockCacheTier /* no_io */,
        true /* record_read_stats */, file_read_hist, /*skip_filters=*/false,
        level, true /* prefetch_index_and_filter_in_cache */,
        /*max_file_size_for_l0_meta_pin=*/0, file_meta.temperature);
    if (s.ok()) {
      t = GetTableReaderFromHandle(handle);
    }
    *table_handle = handle;
  }
  if (s.ok()) {
    s = t->MultiGetFilter(options, prefix_extractor.get(), mget_range);
  }
  if (s.ok() && !options.ignore_range_deletions) {
    // Update the range tombstone sequence numbers for the keys here
    // as TableCache::MultiGet may or may not be called, and even if it
    // is, it may be called with fewer keys in the rangedue to filtering.
    UpdateRangeTombstoneSeqnums(options, t, tombstone_range);
  }
  if (mget_range->empty() && handle) {
    ReleaseHandle(handle);
    *table_handle = nullptr;
  }

  return s;
}

Status TableCache::GetTableProperties(
    const FileOptions& file_options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta,
    std::shared_ptr<const TableProperties>* properties,
    const std::shared_ptr<const SliceTransform>& prefix_extractor, bool no_io) {
  auto table_reader = file_meta.fd.table_reader;
  // table already been pre-loaded?
  if (table_reader) {
    *properties = table_reader->GetTableProperties();

    return Status::OK();
  }

  Cache::Handle* table_handle = nullptr;
  Status s = FindTable(ReadOptions(), file_options, internal_comparator,
                       file_meta, &table_handle, prefix_extractor, no_io);
  if (!s.ok()) {
    return s;
  }
  assert(table_handle);
  auto table = GetTableReaderFromHandle(table_handle);
  *properties = table->GetTableProperties();
  ReleaseHandle(table_handle);
  return s;
}

Status TableCache::GetPropertiesOfResidentTables(
    std::unordered_map<uint64_t, std::shared_ptr<const TableProperties>>*
        properties) {
  if (properties == nullptr) {
    return Status::InvalidArgument("properties must not be null");
  }

  // Copy only the stable pieces needed after the shard lock is released. A
  // capacity-sized reserve keeps the callback allocation-free in the common
  // case while concurrent insertions remain harmless.
  std::vector<std::pair<uint64_t, std::shared_ptr<const TableProperties>>>
      resident;
  const size_t occupancy = cache_->GetOccupancyCount();
  if (occupancy != SIZE_MAX) {
    resident.reserve(occupancy);
  }
  Cache::ApplyToAllEntriesOptions opts;
  opts.average_entries_per_lock = 1;
  cache_->ApplyToAllEntries(
      [&](const Slice& key, void* value, size_t /*charge*/,
          Cache::DeleterFn deleter) {
        if (key.size() != sizeof(uint64_t) || value == nullptr ||
            deleter != &DeleteEntry<TableReader>) {
          return;
        }
        uint64_t file_number = 0;
        std::memcpy(&file_number, key.data(), sizeof(file_number));
        auto table_properties =
            static_cast<TableReader*>(value)->GetTableProperties();
        if (table_properties != nullptr) {
          resident.emplace_back(file_number, std::move(table_properties));
        }
      },
      opts);

  properties->clear();
  properties->reserve(resident.size());
  for (auto& entry : resident) {
    properties->emplace(entry.first, std::move(entry.second));
  }
  return Status::OK();
}

Status TableCache::StreamTableCacheWarmupEntries(
    const std::vector<std::pair<uint64_t, std::string>>& files,
    size_t max_entry_bytes, const TableCacheWarmupCallback& callback,
    TableCacheWarmupSourceStats* stats) {
  if (!callback) {
    return Status::InvalidArgument(
        "table-cache warmup callback must be provided");
  }
  if (max_entry_bytes == 0) {
    return Status::InvalidArgument(
        "table-cache warmup max entry bytes must be nonzero");
  }
  std::vector<TableCacheWarmupSnapshotInternal> snapshots;
  TableCacheWarmupSourceStats local_stats;
  Status s = SnapshotTableCacheWarmupEntries(files, max_entry_bytes, &snapshots,
                                             &local_stats);
  if (!s.ok()) {
    if (stats != nullptr) {
      *stats = local_stats;
    }
    return s;
  }

  // Snapshot acquisition above retained immutable bundle ownership and
  // released every cache handle. Payload copying and callback I/O therefore
  // remain safe even if all source entries are now evicted or unregistered.
  local_stats.copied = 0;
  local_stats.bytes = 0;
  for (const auto& snapshot : snapshots) {
    TableCacheWarmupBundle owned = *snapshot.bundle;
    s = callback(snapshot.external_file, std::move(owned));
    if (!s.ok()) {
      if (stats != nullptr) {
        *stats = local_stats;
      }
      return s;
    }
    ++local_stats.copied;
    local_stats.bytes += snapshot.payload_bytes;
  }

  if (stats != nullptr) {
    *stats = local_stats;
  }
  return Status::OK();
}

Status TableCache::SnapshotTableCacheWarmupEntries(
    const std::vector<std::pair<uint64_t, std::string>>& files,
    size_t max_entry_bytes,
    std::vector<TableCacheWarmupSnapshotInternal>* entries,
    TableCacheWarmupSourceStats* stats) {
  if (entries == nullptr) {
    return Status::InvalidArgument(
        "table-cache warmup snapshot output must not be null");
  }
  if (max_entry_bytes == 0) {
    return Status::InvalidArgument(
        "table-cache warmup max entry bytes must be nonzero");
  }
  entries->clear();
  entries->reserve(files.size());
  TableCacheWarmupSourceStats local_stats;
  local_stats.requested = files.size();

  for (const auto& file : files) {
    uint64_t file_number = file.first;
    Slice key = GetSliceForFileNumber(&file_number);
    Cache::Handle* handle = nullptr;
    Cache::Priority priority = Cache::Priority::BOTTOM;
    Status s = cache_->LookupForCacheWarmup(key, &handle, &priority);
    if (!s.ok()) {
      if (stats != nullptr) {
        *stats = local_stats;
      }
      return s;
    }
    if (handle == nullptr) {
      ++local_stats.skipped_busy;
      continue;
    }
    ScopedTableCacheWarmupLease lease(cache_, handle, priority);
    ++local_stats.resident;

    auto* reader = static_cast<TableReader*>(cache_->Value(handle));
    auto bundle =
        reader != nullptr ? reader->GetTableCacheWarmupBundle() : nullptr;
    if (bundle == nullptr || bundle->ranges.empty()) {
      ++local_stats.skipped_unavailable;
      continue;
    }

    size_t entry_bytes = 0;
    bool too_large = false;
    for (const auto& range : bundle->ranges) {
      if (entry_bytes > max_entry_bytes ||
          range.data.size() > max_entry_bytes - entry_bytes) {
        too_large = true;
        break;
      }
      entry_bytes += range.data.size();
    }
    if (too_large) {
      ++local_stats.skipped_too_large;
      continue;
    }

    entries->push_back(
        {file.first, file.second, std::move(bundle), entry_bytes});
    lease.Release();
    ++local_stats.copied;
    local_stats.bytes += entry_bytes;
  }

  if (stats != nullptr) {
    *stats = local_stats;
  }
  return Status::OK();
}

Status TableCache::PrepareTableReaderForWarmup(
    const FileOptions& file_options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    HistogramImpl* file_read_hist, int level,
    size_t max_file_size_for_l0_meta_pin,
    std::shared_ptr<const TableCacheWarmupBundle> bundle,
    std::unique_ptr<TableReader>* table_reader) {
  if (table_reader == nullptr || bundle == nullptr || bundle->ranges.empty()) {
    return Status::InvalidArgument(
        "table-cache warmup requires a nonempty bundle and reader output");
  }
  table_reader->reset();
  return GetTableReader(
      ReadOptions(), file_options, internal_comparator, file_meta,
      false /* sequential_mode */, true /* record_read_stats */, file_read_hist,
      table_reader, prefix_extractor, false /* skip_filters */, level,
      true /* prefetch_index_and_filter_in_cache */,
      max_file_size_for_l0_meta_pin, file_meta.temperature, std::move(bundle));
}

Status TableCache::InsertPreparedTableReaderForWarmup(
    uint64_t file_number, int level, std::unique_ptr<TableReader>* table_reader,
    Cache::CacheWarmupInsertResult* result) {
  if (table_reader == nullptr || table_reader->get() == nullptr ||
      result == nullptr) {
    return Status::InvalidArgument(
        "table-cache warmup insertion requires reader and result");
  }
  Slice key = GetSliceForFileNumber(&file_number);
  Status s = cache_->InsertForCacheWarmup(
      key, table_reader->get(), 1, &DeleteEntry<TableReader>,
      level == 0 ? Cache::Priority::HIGH : Cache::Priority::LOW, result);
  if (s.ok() && *result == Cache::CacheWarmupInsertResult::kInserted) {
    table_reader->release();
  }
  return s;
}

Status TableCache::InsertPreparedTableReaderForWarmupNoEvict(
    uint64_t file_number, int level, std::unique_ptr<TableReader>* table_reader,
    Cache::CacheWarmupInsertResult* result) {
  if (table_reader == nullptr || table_reader->get() == nullptr ||
      result == nullptr) {
    return Status::InvalidArgument(
        "table-cache warmup insertion requires reader and result");
  }
  Slice key = GetSliceForFileNumber(&file_number);
  Status s = cache_->InsertForCacheWarmupNoEvict(
      key, table_reader->get(), 1, &DeleteEntry<TableReader>,
      level == 0 ? Cache::Priority::HIGH : Cache::Priority::LOW, result);
  if (s.ok() && *result == Cache::CacheWarmupInsertResult::kInserted) {
    table_reader->release();
  }
  return s;
}

Status TableCache::ReplacePreparedTableReaderForWarmup(
    uint64_t victim_file_number, uint64_t file_number, int level,
    std::unique_ptr<TableReader>* table_reader,
    Cache::CacheWarmupInsertResult* result) {
  if (table_reader == nullptr || table_reader->get() == nullptr ||
      result == nullptr) {
    return Status::InvalidArgument(
        "table-cache warmup replacement requires reader and result");
  }
  Slice victim_key = GetSliceForFileNumber(&victim_file_number);
  Slice key = GetSliceForFileNumber(&file_number);
  Status s = cache_->ReplaceForCacheWarmup(
      victim_key, key, table_reader->get(), 1, &DeleteEntry<TableReader>,
      level == 0 ? Cache::Priority::HIGH : Cache::Priority::LOW, result);
  if (s.ok() && *result == Cache::CacheWarmupInsertResult::kInserted) {
    table_reader->release();
  }
  return s;
}

Status TableCache::PromoteFileForCacheWarmup(uint64_t file_number, int level) {
  Slice key = GetSliceForFileNumber(&file_number);
  return cache_->PromoteForCacheWarmup(
      key, level == 0 ? Cache::Priority::HIGH : Cache::Priority::LOW);
}

void TableCache::GetResidentFileNumbers(
    std::vector<uint64_t>* file_numbers) const {
  assert(file_numbers != nullptr);
  file_numbers->clear();
  const size_t occupancy = cache_->GetOccupancyCount();
  if (occupancy != SIZE_MAX) {
    file_numbers->reserve(occupancy);
  }
  Cache::ApplyToAllEntriesOptions opts;
  opts.average_entries_per_lock = 64;
  cache_->ApplyToAllEntries(
      [&](const Slice& key, void* value, size_t /*charge*/,
          Cache::DeleterFn deleter) {
        if (key.size() != sizeof(uint64_t) || value == nullptr ||
            deleter != &DeleteEntry<TableReader>) {
          return;
        }
        uint64_t file_number = 0;
        std::memcpy(&file_number, key.data(), sizeof(file_number));
        file_numbers->push_back(file_number);
      },
      opts);
}

size_t TableCache::GetCacheWarmupShardIndex(uint64_t file_number) const {
  Slice key = GetSliceForFileNumber(&file_number);
  return cache_->GetCacheWarmupShardIndex(key);
}

void TableCache::EraseFile(uint64_t file_number) {
  Slice key = GetSliceForFileNumber(&file_number);
  cache_->Erase(key);
}

Status TableCache::ApproximateKeyAnchors(
    const ReadOptions& ro, const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta, std::vector<TableReader::Anchor>& anchors) {
  Status s;
  TableReader* t = file_meta.fd.table_reader;
  Cache::Handle* handle = nullptr;
  if (t == nullptr) {
    s = FindTable(ro, file_options_, internal_comparator, file_meta, &handle);
    if (s.ok()) {
      t = GetTableReaderFromHandle(handle);
    }
  }
  if (s.ok() && t != nullptr) {
    s = t->ApproximateKeyAnchors(ro, anchors);
  }
  if (handle != nullptr) {
    ReleaseHandle(handle);
  }
  return s;
}

size_t TableCache::GetMemoryUsageByTableReader(
    const FileOptions& file_options,
    const InternalKeyComparator& internal_comparator,
    const FileMetaData& file_meta,
    const std::shared_ptr<const SliceTransform>& prefix_extractor) {
  auto table_reader = file_meta.fd.table_reader;
  // table already been pre-loaded?
  if (table_reader) {
    return table_reader->ApproximateMemoryUsage();
  }

  Cache::Handle* table_handle = nullptr;
  Status s = FindTable(ReadOptions(), file_options, internal_comparator,
                       file_meta, &table_handle, prefix_extractor, true);
  if (!s.ok()) {
    return 0;
  }
  assert(table_handle);
  auto table = GetTableReaderFromHandle(table_handle);
  auto ret = table->ApproximateMemoryUsage();
  ReleaseHandle(table_handle);
  return ret;
}

bool TableCache::HasEntry(Cache* cache, uint64_t file_number) {
  Cache::Handle* handle = cache->Lookup(GetSliceForFileNumber(&file_number));
  if (handle) {
    cache->Release(handle);
    return true;
  } else {
    return false;
  }
}

void TableCache::Evict(Cache* cache, uint64_t file_number) {
  cache->Erase(GetSliceForFileNumber(&file_number));
}

uint64_t TableCache::ApproximateOffsetOf(
    const Slice& key, const FileMetaData& file_meta, TableReaderCaller caller,
    const InternalKeyComparator& internal_comparator,
    const std::shared_ptr<const SliceTransform>& prefix_extractor) {
  uint64_t result = 0;
  TableReader* table_reader = file_meta.fd.table_reader;
  Cache::Handle* table_handle = nullptr;
  if (table_reader == nullptr) {
    const bool for_compaction = (caller == TableReaderCaller::kCompaction);
    Status s =
        FindTable(ReadOptions(), file_options_, internal_comparator, file_meta,
                  &table_handle, prefix_extractor, false /* no_io */,
                  !for_compaction /* record_read_stats */);
    if (s.ok()) {
      table_reader = GetTableReaderFromHandle(table_handle);
    }
  }

  if (table_reader != nullptr) {
    result = table_reader->ApproximateOffsetOf(key, caller);
  }
  if (table_handle != nullptr) {
    ReleaseHandle(table_handle);
  }

  return result;
}

uint64_t TableCache::ApproximateSize(
    const Slice& start, const Slice& end, const FileMetaData& file_meta,
    TableReaderCaller caller, const InternalKeyComparator& internal_comparator,
    const std::shared_ptr<const SliceTransform>& prefix_extractor) {
  uint64_t result = 0;
  TableReader* table_reader = file_meta.fd.table_reader;
  Cache::Handle* table_handle = nullptr;
  if (table_reader == nullptr) {
    const bool for_compaction = (caller == TableReaderCaller::kCompaction);
    Status s =
        FindTable(ReadOptions(), file_options_, internal_comparator, file_meta,
                  &table_handle, prefix_extractor, false /* no_io */,
                  !for_compaction /* record_read_stats */);
    if (s.ok()) {
      table_reader = GetTableReaderFromHandle(table_handle);
    }
  }

  if (table_reader != nullptr) {
    result = table_reader->ApproximateSize(start, end, caller);
  }
  if (table_handle != nullptr) {
    ReleaseHandle(table_handle);
  }

  return result;
}
}  // namespace ROCKSDB_NAMESPACE
