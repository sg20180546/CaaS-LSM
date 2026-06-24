//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/builder.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "db/blob/blob_file_builder.h"
#include "db/bucket_util.h"
#include "db/compaction/compaction_iterator.h"
#include "db/event_helpers.h"
#include "db/internal_stats.h"
#include "db/merge_helper.h"
#include "db/output_validator.h"
#include "db/range_del_aggregator.h"
#include "db/table_cache.h"
#include "db/version_edit.h"
#include "db/version_set.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "file/read_write_util.h"
#include "file/writable_file_writer.h"
#include "monitoring/iostats_context_imp.h"
#include "monitoring/thread_status_util.h"
#include "options/options_helper.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/iterator.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "table/block_based/block_based_table_builder.h"
#include "table/format.h"
#include "table/internal_iterator.h"
#include "table/unique_id_impl.h"
#include "test_util/sync_point.h"
#include "util/stop_watch.h"

namespace ROCKSDB_NAMESPACE {

class TableFactory;

TableBuilder* NewTableBuilder(const TableBuilderOptions& tboptions,
                              WritableFileWriter* file) {
  assert((tboptions.column_family_id ==
          TablePropertiesCollectorFactory::Context::kUnknownColumnFamily) ==
         tboptions.column_family_name.empty());
  return tboptions.ioptions.table_factory->NewTableBuilder(tboptions, file);
}

Status BuildTable(
    const std::string& dbname, VersionSet* versions,
    const ImmutableDBOptions& db_options, const TableBuilderOptions& tboptions,
    const FileOptions& file_options, TableCache* table_cache,
    InternalIterator* iter,
    std::vector<std::unique_ptr<FragmentedRangeTombstoneIterator>>
        range_del_iters,
    FileMetaData* meta, std::vector<BlobFileAddition>* blob_file_additions,
    std::vector<SequenceNumber> snapshots,
    SequenceNumber earliest_write_conflict_snapshot,
    SequenceNumber job_snapshot, SnapshotChecker* snapshot_checker,
    bool paranoid_file_checks, InternalStats* internal_stats,
    IOStatus* io_status, const std::shared_ptr<IOTracer>& io_tracer,
    BlobFileCreationReason blob_creation_reason,
    const SeqnoToTimeMapping& seqno_to_time_mapping, EventLogger* event_logger,
    int job_id, const Env::IOPriority io_priority,
    TableProperties* table_properties, Env::WriteLifeTimeHint write_hint,
    const std::string* full_history_ts_low,
    BlobFileCompletionCallback* blob_callback, uint64_t* num_input_entries,
    uint64_t* memtable_payload_bytes, uint64_t* memtable_garbage_bytes,
    std::vector<FileMetaData>* extra_metas) {
  assert((tboptions.column_family_id ==
          TablePropertiesCollectorFactory::Context::kUnknownColumnFamily) ==
         tboptions.column_family_name.empty());
  auto& mutable_cf_options = tboptions.moptions;
  auto& ioptions = tboptions.ioptions;
  // Reports the IOStats for flush for every following bytes.
  const size_t kReportFlushIOStatsEvery = 1048576;
  OutputValidator output_validator(
      tboptions.internal_comparator,
      /*enable_order_check=*/
      mutable_cf_options.check_flush_compaction_key_order,
      /*enable_hash=*/paranoid_file_checks);
  Status s;
  meta->fd.file_size = 0;
  iter->SeekToFirst();
  std::unique_ptr<CompactionRangeDelAggregator> range_del_agg(
      new CompactionRangeDelAggregator(&tboptions.internal_comparator,
                                       snapshots, full_history_ts_low));
  uint64_t num_unfragmented_tombstones = 0;
  uint64_t total_tombstone_payload_bytes = 0;
  for (auto& range_del_iter : range_del_iters) {
    num_unfragmented_tombstones +=
        range_del_iter->num_unfragmented_tombstones();
    total_tombstone_payload_bytes +=
        range_del_iter->total_tombstone_payload_bytes();
    range_del_agg->AddTombstones(std::move(range_del_iter));
  }

  std::string fname = TableFileName(ioptions.cf_paths, meta->fd.GetNumber(),
                                    meta->fd.GetPathId());
  std::vector<std::string> blob_file_paths;
  std::string file_checksum = kUnknownFileChecksum;
  std::string file_checksum_func_name = kUnknownFileChecksumFuncName;
#ifndef ROCKSDB_LITE
  EventHelpers::NotifyTableFileCreationStarted(ioptions.listeners, dbname,
                                               tboptions.column_family_name,
                                               fname, job_id, tboptions.reason);
#endif  // !ROCKSDB_LITE
  Env* env = db_options.env;
  assert(env);
  FileSystem* fs = db_options.fs.get();
  assert(fs);

  TableProperties tp;
  bool table_file_created = false;

  // [BucketLSM / relink — G5 only] BucketFlush: when enabled, a single flush is
  // split into N bucket-pure L0 SST files, cutting a new output file every time
  // BucketOf(user_key) changes. Gated strictly on extra_metas != nullptr (only
  // the normal flush path passes it) AND l0_bucket_count > 1. Off => the
  // original single-file code path below executes byte-for-byte unchanged.
  const bool bucketing = (extra_metas != nullptr) && (versions != nullptr) &&
                         ioptions.l0_bucket_count > 1;
  const uint64_t l0_bucket_count = ioptions.l0_bucket_count;
  const uint64_t l0_bucket_key_space = ioptions.l0_bucket_key_space;
  // All file names created (current `meta` + any extra bucket files), so that
  // every partial file is cleaned up on error.
  std::vector<std::string> created_fnames;

  if (iter->Valid() || !range_del_agg->IsEmpty()) {
    std::unique_ptr<CompactionFilter> compaction_filter;
    if (ioptions.compaction_filter_factory != nullptr &&
        ioptions.compaction_filter_factory->ShouldFilterTableFileCreation(
            tboptions.reason)) {
      CompactionFilter::Context context;
      context.is_full_compaction = false;
      context.is_manual_compaction = false;
      context.column_family_id = tboptions.column_family_id;
      context.reason = tboptions.reason;
      compaction_filter =
          ioptions.compaction_filter_factory->CreateCompactionFilter(context);
      if (compaction_filter != nullptr &&
          !compaction_filter->IgnoreSnapshots()) {
        s.PermitUncheckedError();
        return Status::NotSupported(
            "CompactionFilter::IgnoreSnapshots() = false is not supported "
            "anymore.");
      }
    }

    // ---- current output-file state (rotated by BucketFlush) ----
    TableBuilder* builder = nullptr;
    std::unique_ptr<WritableFileWriter> file_writer;
    FileMetaData* cur_meta = meta;  // first file is the caller's `meta`
    // Backing storage for the in-progress extra (2nd..Nth) bucket file. Its
    // contents are copied into `extra_metas` once finalized, then reused.
    FileMetaData extra_pending_meta;

    // Create a fresh on-disk table for `m` and install builder/file_writer/fname
    // for it. `m->fd` (file number / path id) must already be assigned. Returns
    // the create Status (and updates *io_status on failure, mirroring original).
    auto open_output_file = [&](FileMetaData* m) -> Status {
      Status open_s;
      fname =
          TableFileName(ioptions.cf_paths, m->fd.GetNumber(), m->fd.GetPathId());
      std::unique_ptr<FSWritableFile> file;
#ifndef NDEBUG
      bool use_direct_writes = file_options.use_direct_writes;
      TEST_SYNC_POINT_CALLBACK("BuildTable:create_file", &use_direct_writes);
#endif  // !NDEBUG
      IOStatus io_s = NewWritableFile(fs, fname, &file, file_options);
      open_s = io_s;
      if (io_status->ok()) {
        *io_status = io_s;
      }
      if (!open_s.ok()) {
        return open_s;
      }
      table_file_created = true;
      created_fnames.push_back(fname);
      FileTypeSet tmp_set = ioptions.checksum_handoff_file_types;
      file->SetIOPriority(io_priority);
      file->SetWriteLifeTimeHint(write_hint);
      file_writer.reset(new WritableFileWriter(
          std::move(file), fname, file_options, ioptions.clock, io_tracer,
          ioptions.stats, ioptions.listeners,
          ioptions.file_checksum_gen_factory.get(),
          tmp_set.Contains(FileType::kTableFile), false));
      if (m == meta) {
        // First file: identical builder construction to the original path.
        builder = NewTableBuilder(tboptions, file_writer.get());
      } else {
        // Extra bucket file: same options but with this file's own number so
        // table props (orig_file_number) and the block-cache key are correct.
        TableBuilderOptions extra_tbo(
            ioptions, mutable_cf_options, tboptions.internal_comparator,
            tboptions.int_tbl_prop_collector_factories,
            tboptions.compression_type, tboptions.compression_opts,
            tboptions.column_family_id, tboptions.column_family_name,
            tboptions.level_at_creation, tboptions.is_bottommost,
            tboptions.reason, tboptions.oldest_key_time,
            tboptions.file_creation_time, tboptions.db_id,
            tboptions.db_session_id, tboptions.target_file_size,
            m->fd.GetNumber());
        builder = NewTableBuilder(extra_tbo, file_writer.get());
      }
      return open_s;
    };

    s = open_output_file(meta);
    if (!s.ok()) {
      EventHelpers::LogAndNotifyTableFileCreationFinished(
          event_logger, ioptions.listeners, dbname,
          tboptions.column_family_name, fname, job_id, meta->fd,
          kInvalidBlobFileNumber, tp, tboptions.reason, s, file_checksum,
          file_checksum_func_name);
      return s;
    }

    MergeHelper merge(
        env, tboptions.internal_comparator.user_comparator(),
        ioptions.merge_operator.get(), compaction_filter.get(), ioptions.logger,
        true /* internal key corruption is not ok */,
        snapshots.empty() ? 0 : snapshots.back(), snapshot_checker);

    std::unique_ptr<BlobFileBuilder> blob_file_builder(
        (mutable_cf_options.enable_blob_files &&
         tboptions.level_at_creation >=
             mutable_cf_options.blob_file_starting_level &&
         blob_file_additions)
            ? new BlobFileBuilder(
                  versions, fs, &ioptions, &mutable_cf_options, &file_options,
                  tboptions.db_id, tboptions.db_session_id, job_id,
                  tboptions.column_family_id, tboptions.column_family_name,
                  io_priority, write_hint, io_tracer, blob_callback,
                  blob_creation_reason, &blob_file_paths, blob_file_additions)
            : nullptr);

    const std::atomic<bool> kManualCompactionCanceledFalse{false};
    CompactionIterator c_iter(
        iter, tboptions.internal_comparator.user_comparator(), &merge,
        kMaxSequenceNumber, &snapshots, earliest_write_conflict_snapshot,
        job_snapshot, snapshot_checker, env,
        ShouldReportDetailedTime(env, ioptions.stats),
        true /* internal key corruption is not ok */, range_del_agg.get(),
        blob_file_builder.get(), ioptions.allow_data_in_errors,
        ioptions.enforce_single_del_contracts,
        /*manual_compaction_canceled=*/kManualCompactionCanceledFalse,
        /*compaction=*/nullptr, compaction_filter.get(),
        /*shutting_down=*/nullptr, db_options.info_log, full_history_ts_low);

    // Accumulated raw key+value bytes written across all output files (for the
    // memtable garbage accounting, which is per-flush not per-file).
    uint64_t total_raw_key_value_written = 0;
    // Whether at least one output file was finalized non-empty (mirrors the
    // original `!empty` guard for the memtable accounting block).
    bool any_non_empty_output = false;

    // Finalize the current output file `m`: Finish the builder, fill `m`'s file
    // size / marked_for_compaction / table props, then Sync, Close, checksum and
    // unique_id, and validate via the table cache. Mirrors the original
    // single-file finalize sequence exactly (single-file path: m == meta). On an
    // empty builder, Abandon and leave m->fd.file_size == 0 (the caller skips
    // empty files). Sets `s`/`*io_status` like the original. Returns false on a
    // hard error so the caller can stop.
    auto finalize_output_file = [&](FileMetaData* m) -> void {
      TEST_SYNC_POINT("BuildTable:BeforeFinishBuildTable");
      const bool empty = builder->IsEmpty();
      if (!s.ok() || empty) {
        builder->Abandon();
      } else {
        std::string seqno_time_mapping_str;
        seqno_to_time_mapping.Encode(
            seqno_time_mapping_str, m->fd.smallest_seqno, m->fd.largest_seqno,
            m->file_creation_time);
        builder->SetSeqnoTimeTableProperties(
            seqno_time_mapping_str,
            ioptions.compaction_style == CompactionStyle::kCompactionStyleFIFO
                ? m->file_creation_time
                : m->oldest_ancester_time);
        s = builder->Finish();
      }
      if (io_status->ok()) {
        *io_status = builder->io_status();
      }

      if (s.ok() && !empty) {
        any_non_empty_output = true;
        uint64_t file_size = builder->FileSize();
        m->fd.file_size = file_size;
        m->marked_for_compaction = builder->NeedCompact();
        assert(m->fd.GetFileSize() > 0);
        tp = builder
                 ->GetTableProperties();  // refresh now that builder is finished
        total_raw_key_value_written += (tp.raw_key_size + tp.raw_value_size);
        if (table_properties) {
          // Reports the LAST finalized file's props (mirrors original for the
          // single-file path).
          *table_properties = tp;
        }
      }
      delete builder;
      builder = nullptr;

      // Finish and check for file errors
      TEST_SYNC_POINT("BuildTable:BeforeSyncTable");
      if (s.ok() && !empty) {
        StopWatch sw(ioptions.clock, ioptions.stats, TABLE_SYNC_MICROS);
        *io_status = file_writer->Sync(ioptions.use_fsync);
      }
      TEST_SYNC_POINT("BuildTable:BeforeCloseTableFile");
      if (s.ok() && io_status->ok() && !empty) {
        *io_status = file_writer->Close();
      }
      if (s.ok() && io_status->ok() && !empty) {
        // Add the checksum information to file metadata.
        m->file_checksum = file_writer->GetFileChecksum();
        m->file_checksum_func_name = file_writer->GetFileChecksumFuncName();
        file_checksum = m->file_checksum;
        file_checksum_func_name = m->file_checksum_func_name;
        // Set unique_id only if db_id and db_session_id exist
        if (!tboptions.db_id.empty() && !tboptions.db_session_id.empty()) {
          if (!GetSstInternalUniqueId(tboptions.db_id, tboptions.db_session_id,
                                      m->fd.GetNumber(), &(m->unique_id))
                   .ok()) {
            // if failed to get unique id, just set it Null
            m->unique_id = kNullUniqueId64x2;
          }
        }
      }

      if (s.ok()) {
        s = *io_status;
      }

      // TODO Also check the IO status when create the Iterator.

      TEST_SYNC_POINT("BuildTable:BeforeOutputValidation");
      if (s.ok() && !empty) {
        // Verify that the table is usable. (When bucketing, `output_validator`
        // is a rolling hash over ALL files' keys, so a per-file CompareValidator
        // cannot match; we still re-read the file to confirm it is usable but
        // only compare in the single-file path.)
        ReadOptions read_options;
        std::unique_ptr<InternalIterator> it(table_cache->NewIterator(
            read_options, file_options, tboptions.internal_comparator, *m,
            nullptr /* range_del_agg */, mutable_cf_options.prefix_extractor,
            nullptr,
            (internal_stats == nullptr) ? nullptr
                                        : internal_stats->GetFileReadHist(0),
            TableReaderCaller::kFlush, /*arena=*/nullptr,
            /*skip_filter=*/false, tboptions.level_at_creation,
            MaxFileSizeForL0MetaPin(mutable_cf_options),
            /*smallest_compaction_key=*/nullptr,
            /*largest_compaction_key*/ nullptr,
            /*allow_unprepared_value*/ false));
        s = it->status();
        if (s.ok() && paranoid_file_checks && !bucketing) {
          OutputValidator file_validator(tboptions.internal_comparator,
                                         /*enable_order_check=*/true,
                                         /*enable_hash=*/true);
          for (it->SeekToFirst(); it->Valid(); it->Next()) {
            // Generate a rolling 64-bit hash of the key and values
            file_validator.Add(it->key(), it->value()).PermitUncheckedError();
          }
          s = it->status();
          if (s.ok() && !output_validator.CompareValidator(file_validator)) {
            s = Status::Corruption("Paranoid checksums do not match");
          }
        }
      }
    };

    // Allocate the next output file's metadata, copying the per-flush fields
    // (creation/ancester time, temperature, etc.) from `meta` and assigning a
    // fresh file number from `versions`. Used only on the BucketFlush path.
    auto make_next_meta = [&]() -> FileMetaData {
      FileMetaData nm;
      nm.fd = FileDescriptor(versions->NewFileNumber(), meta->fd.GetPathId(),
                             0 /* file_size */);
      nm.oldest_ancester_time = meta->oldest_ancester_time;
      nm.file_creation_time = meta->file_creation_time;
      nm.temperature = meta->temperature;
      return nm;
    };

    c_iter.SeekToFirst();
    bool have_prev_user_key = false;
    uint64_t prev_bucket = 0;
    for (; c_iter.Valid(); c_iter.Next()) {
      const Slice& key = c_iter.key();
      const Slice& value = c_iter.value();
      const ParsedInternalKey& ikey = c_iter.ikey();
      // Generate a rolling 64-bit hash of the key and values
      // Note :
      // Here "key" integrates 'sequence_number'+'kType'+'user key'.
      s = output_validator.Add(key, value);
      if (!s.ok()) {
        break;
      }

      // [BucketLSM] Cut a new bucket-pure file when the bucket id of the user
      // key changes. The memtable iterator is in sorted user-key order, so a
      // bucket boundary is exactly BucketOf(prev) != BucketOf(cur).
      if (bucketing) {
        const uint64_t cur_bucket =
            BucketOf(ikey.user_key, l0_bucket_key_space, l0_bucket_count);
        if (have_prev_user_key && cur_bucket != prev_bucket) {
          // Finalize the current (non-empty) file and start a new one.
          finalize_output_file(cur_meta);
          if (!s.ok()) {
            break;
          }
          if (cur_meta != meta) {
            extra_metas->push_back(*cur_meta);
          }
          // Start the next file with a freshly allocated file number.
          extra_pending_meta = make_next_meta();
          cur_meta = &extra_pending_meta;
          s = open_output_file(cur_meta);
          if (!s.ok()) {
            break;
          }
        }
        prev_bucket = cur_bucket;
        have_prev_user_key = true;
      }

      builder->Add(key, value);

      s = cur_meta->UpdateBoundaries(key, value, ikey.sequence, ikey.type);
      if (!s.ok()) {
        break;
      }

      // TODO(noetzli): Update stats after flush, too.
      if (io_priority == Env::IO_HIGH &&
          IOSTATS(bytes_written) >= kReportFlushIOStatsEvery) {
        ThreadStatusUtil::SetThreadOperationProperty(
            ThreadStatus::FLUSH_BYTES_WRITTEN, IOSTATS(bytes_written));
      }
    }
    if (!s.ok()) {
      c_iter.status().PermitUncheckedError();
    } else if (!c_iter.status().ok()) {
      s = c_iter.status();
    }

    if (s.ok()) {
      auto range_del_it = range_del_agg->NewIterator();
      range_del_it->SeekToFirst();
      if (bucketing && range_del_it->Valid()) {
        // [BucketLSM] range tombstones are NOT bucket-partitioned. A tombstone spanning multiple
        // buckets would be written into only ONE bucket-pure file, while the point keys it must cover
        // live in OTHER bucket files -> those deleted keys would resurrect. Fail loudly rather than
        // silently corrupt. (Our G5 workload issues no DeleteRange; this guards future misuse. Correct
        // per-bucket range-del splitting is future work.)
        s = Status::NotSupported(
            "L0 bucketing (l0_bucket_count>1) does not support range tombstones");
      } else {
        for (; range_del_it->Valid(); range_del_it->Next()) {
          auto tombstone = range_del_it->Tombstone();
          auto kv = tombstone.Serialize();
          builder->Add(kv.first.Encode(), kv.second);
          cur_meta->UpdateBoundariesForRange(kv.first, tombstone.SerializeEndKey(),
                                             tombstone.seq_,
                                             tboptions.internal_comparator);
        }
      }
    }

    if (num_input_entries != nullptr) {
      *num_input_entries =
          c_iter.num_input_entry_scanned() + num_unfragmented_tombstones;
    }

    // Finalize the last (current) output file.
    finalize_output_file(cur_meta);
    if (s.ok() && cur_meta != meta && cur_meta->fd.GetFileSize() > 0) {
      extra_metas->push_back(*cur_meta);
    }

    // Per-flush memtable garbage accounting over the sum of all files written.
    // Guarded on any_non_empty_output to mirror the original `!empty` condition
    // (when nothing was written, leave the caller's counters untouched).
    if (s.ok() && any_non_empty_output && memtable_payload_bytes != nullptr &&
        memtable_garbage_bytes != nullptr) {
      const CompactionIterationStats& ci_stats = c_iter.iter_stats();
      uint64_t total_payload_bytes = ci_stats.total_input_raw_key_bytes +
                                     ci_stats.total_input_raw_value_bytes +
                                     total_tombstone_payload_bytes;
      uint64_t total_payload_bytes_written = total_raw_key_value_written;
      // Prevent underflow, which may still happen at this point
      // since we only support inserts, deletes, and deleteRanges.
      if (total_payload_bytes_written <= total_payload_bytes) {
        *memtable_payload_bytes = total_payload_bytes;
        *memtable_garbage_bytes =
            total_payload_bytes - total_payload_bytes_written;
      } else {
        *memtable_payload_bytes = 0;
        *memtable_garbage_bytes = 0;
      }
    }

    if (blob_file_builder) {
      if (s.ok()) {
        s = blob_file_builder->Finish();
      } else {
        blob_file_builder->Abandon(s);
      }
      blob_file_builder.reset();
    }
  }

  // Check for input iterator errors
  if (!iter->status().ok()) {
    s = iter->status();
  }

  if (bucketing) {
    // [BucketLSM] Multi-file cleanup. On a hard error, every file we created is
    // partial/uninstalled, so delete them all. The empty-first-file case
    // (meta->fd.GetFileSize() == 0) is handled by flush_job's has_output check
    // and the per-bucket empty handling; on success non-empty files are kept.
    if (!s.ok()) {
      TEST_SYNC_POINT("BuildTable:BeforeDeleteFile");
      constexpr IODebugContext* dbg = nullptr;
      for (const std::string& created : created_fnames) {
        Status ignored = fs->DeleteFile(created, IOOptions(), dbg);
        ignored.PermitUncheckedError();
      }
      extra_metas->clear();
      assert(blob_file_additions || blob_file_paths.empty());
      if (blob_file_additions) {
        for (const std::string& blob_file_path : blob_file_paths) {
          Status ignored = DeleteDBFile(&db_options, blob_file_path, dbname,
                                        /*force_bg=*/false, /*force_fg=*/false);
          ignored.PermitUncheckedError();
          TEST_SYNC_POINT("BuildTable::AfterDeleteFile");
        }
      }
    }
  } else if (!s.ok() || meta->fd.GetFileSize() == 0) {
    TEST_SYNC_POINT("BuildTable:BeforeDeleteFile");

    constexpr IODebugContext* dbg = nullptr;

    if (table_file_created) {
      Status ignored = fs->DeleteFile(fname, IOOptions(), dbg);
      ignored.PermitUncheckedError();
    }

    assert(blob_file_additions || blob_file_paths.empty());

    if (blob_file_additions) {
      for (const std::string& blob_file_path : blob_file_paths) {
        Status ignored = DeleteDBFile(&db_options, blob_file_path, dbname,
                                      /*force_bg=*/false, /*force_fg=*/false);
        ignored.PermitUncheckedError();
        TEST_SYNC_POINT("BuildTable::AfterDeleteFile");
      }
    }
  }

  // [BucketLSM] The final creation event reports `meta` (the first file); point
  // `fname` back at it so the event is coherent (extra bucket files do not get
  // their own LogAndNotify event — acceptable: G5 shards have no listeners).
  if (bucketing && meta->fd.GetFileSize() > 0) {
    fname = TableFileName(ioptions.cf_paths, meta->fd.GetNumber(),
                          meta->fd.GetPathId());
  }

  Status status_for_listener = s;
  if (meta->fd.GetFileSize() == 0) {
    fname = "(nil)";
    if (s.ok()) {
      status_for_listener = Status::Aborted("Empty SST file not kept");
    }
  }
  // Output to event logger and fire events.
  EventHelpers::LogAndNotifyTableFileCreationFinished(
      event_logger, ioptions.listeners, dbname, tboptions.column_family_name,
      fname, job_id, meta->fd, meta->oldest_blob_file_number, tp,
      tboptions.reason, status_for_listener, file_checksum,
      file_checksum_func_name);

  return s;
}

}  // namespace ROCKSDB_NAMESPACE
