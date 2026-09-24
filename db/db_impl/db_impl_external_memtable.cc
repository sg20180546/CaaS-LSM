//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] DBImpl::InstallExternalMemTable: install a
// caller-owned sorted KV-block (ExternalMemTableBlock) as an immutable
// memtable of a column family. See include/rocksdb/db.h for the contract and
// the preconditions; the design notes live in
// exp_out/_scratch/memtable_handoff_0923/PLAN.md (F1..F7).
//
// Sequence of operations (all under mutex_ + both write queues stopped):
//   validate -> build MemTable(SortedBlockMemTableRep) -> raise the DB's
//   last sequence to max(current, block.largest) -> SetNextLogNumber /
//   SetID (active memtable's id) -> imm()->Add -> new SuperVersion ->
//   (optionally) queue a flush request.
// On any non-OK return the block is untouched and block.release is NOT
// called; on success the MemTable owns the block and release() runs exactly
// once from ~MemTable after the flush result is in the MANIFEST and the last
// SuperVersion / iterator reference is gone.

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "db/column_family.h"
#include "db/db_impl/db_impl.h"
#include "db/dbformat.h"
#include "db/job_context.h"
#include "db/memtable.h"
#include "db/memtable_list.h"
#include "db/version_set.h"
#include "logging/logging.h"
#include "rocksdb/external_memtable.h"
#include "rocksdb/status.h"
#include "util/autovector.h"
#include "util/cast_util.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// Parse one memtable-encoded entry starting at p (bounded by limit).
bool ParseMemTableEntry(const char* p, const char* limit, Slice* internal_key,
                        Slice* value, const char** end) {
  uint32_t klen = 0;
  const char* q = GetVarint32Ptr(p, limit, &klen);
  if (q == nullptr || klen < kNumInternalBytes ||
      static_cast<uint64_t>(limit - q) < klen) {
    return false;
  }
  *internal_key = Slice(q, klen);
  q += klen;
  uint32_t vlen = 0;
  q = GetVarint32Ptr(q, limit, &vlen);
  if (q == nullptr || static_cast<uint64_t>(limit - q) < vlen) {
    return false;
  }
  *value = Slice(q, vlen);
  *end = q + vlen;
  return true;
}

// O(count) over 4-byte offsets plus two entry parses: cheap enough for the
// install window (~0.3 ms for 700k entries).
Status ValidateBlockCheap(const ExternalMemTableBlock& b) {
  if (b.data == nullptr || b.offsets == nullptr) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: block data/offsets must not be null");
  }
  if (b.count == 0 || b.data_size == 0) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: block is empty (count == 0)");
  }
  if (static_cast<uint64_t>(b.data_size) >= (0x1ull << 32)) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: block data_size must be < 4 GiB");
  }
  if (b.smallest_seqno == 0) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: smallest_seqno must be >= 1 (0 marks an "
        "empty memtable)");
  }
  if (b.smallest_seqno > b.largest_seqno ||
      b.largest_seqno > kMaxSequenceNumber) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: need 1 <= smallest_seqno <= largest_seqno "
        "<= kMaxSequenceNumber");
  }
  if (b.offsets[0] != 0) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: offsets[0] must be 0");
  }
  for (uint64_t i = 1; i < b.count; ++i) {
    if (b.offsets[i] <= b.offsets[i - 1]) {
      return Status::InvalidArgument(
          "InstallExternalMemTable: offsets must be strictly increasing "
          "(violation at index " +
          std::to_string(i) + ")");
    }
  }
  if (b.offsets[b.count - 1] >= b.data_size) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: last offset must be < data_size");
  }
  const char* limit = b.data + b.data_size;
  Slice ik, v;
  const char* end = nullptr;
  if (!ParseMemTableEntry(b.data + b.offsets[0], limit, &ik, &v, &end)) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: first entry is malformed");
  }
  if (!ParseMemTableEntry(b.data + b.offsets[b.count - 1], limit, &ik, &v,
                          &end) ||
      end > limit) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: last entry is malformed or exceeds "
        "data_size");
  }
  return Status::OK();
}

#ifndef NDEBUG
// Full entry walk (debug builds only: O(bytes), ~0.1 s for 780 MB): every
// entry parses inside its slot, carries an allowed type and a seqno inside
// the window, and the sequence is strictly increasing in
// InternalKeyComparator order.
Status ValidateBlockFull(const ExternalMemTableBlock& b,
                         const InternalKeyComparator& icmp) {
  const char* limit = b.data + b.data_size;
  Slice prev;
  for (uint64_t i = 0; i < b.count; ++i) {
    const char* slot_end =
        (i + 1 < b.count) ? b.data + b.offsets[i + 1] : limit;
    Slice ik, v;
    const char* end = nullptr;
    if (!ParseMemTableEntry(b.data + b.offsets[i], slot_end, &ik, &v, &end) ||
        end > slot_end) {
      return Status::InvalidArgument(
          "InstallExternalMemTable: entry " + std::to_string(i) +
          " does not fit its offset slot");
    }
    ParsedInternalKey pik;
    Status ps = ParseInternalKey(ik, &pik, false /* log_err_key */);
    if (!ps.ok()) {
      return Status::InvalidArgument("InstallExternalMemTable: entry " +
                                     std::to_string(i) +
                                     " has an invalid internal key");
    }
    if (pik.type != kTypeValue && pik.type != kTypeDeletion &&
        pik.type != kTypeSingleDeletion) {
      return Status::InvalidArgument("InstallExternalMemTable: entry " +
                                     std::to_string(i) +
                                     " has an unsupported value type");
    }
    if (pik.sequence < b.smallest_seqno || pik.sequence > b.largest_seqno) {
      return Status::InvalidArgument("InstallExternalMemTable: entry " +
                                     std::to_string(i) +
                                     " seqno outside [smallest, largest]");
    }
    if (i > 0 && icmp.Compare(prev, ik) >= 0) {
      return Status::InvalidArgument(
          "InstallExternalMemTable: entries not strictly increasing in "
          "InternalKeyComparator order at entry " +
          std::to_string(i));
    }
    prev = ik;
  }
  return Status::OK();
}
#endif  // NDEBUG

}  // namespace

Status DBImpl::InstallExternalMemTable(ColumnFamilyHandle* column_family,
                                       ExternalMemTableBlock&& block) {
  if (column_family == nullptr) {
    return Status::InvalidArgument(
        "InstallExternalMemTable: column_family must not be null");
  }
  // ---- 1. Validate WITHOUT consuming the block (no release() on failure).
  Status s = ValidateBlockCheap(block);
  if (!s.ok()) {
    return s;
  }
  if (immutable_db_options_.atomic_flush) {
    return Status::NotSupported(
        "InstallExternalMemTable: atomic_flush DBs are not supported");
  }
  auto* cfd =
      static_cast_with_check<ColumnFamilyHandleImpl>(column_family)->cfd();
#ifndef NDEBUG
  s = ValidateBlockFull(block, cfd->internal_comparator());
  if (!s.ok()) {
    return s;
  }
#endif
  const SequenceNumber smallest = block.smallest_seqno;
  const SequenceNumber largest = block.largest_seqno;
  const uint64_t count = block.count;
  const size_t bytes = block.data_size;
  const bool request_flush = block.request_flush;

  MemTable* mem = nullptr;
  uint64_t memtable_id = 0;
  int imm_unflushed = 0;
  SequenceNumber seq_before = 0;
  autovector<MemTable*> to_delete;
  SuperVersionContext sv_ctx(/*create_superversion=*/true);
  {
    InstrumentedMutexLock l(&mutex_);
    // ---- 2. Stop writes on both queues. Writers read and advance the last
    // sequence under write-thread leadership, not under mutex_, so the
    // sequence bump below must not race an in-flight write group (same
    // pattern as IngestExternalFiles / FlushMemTable).
    WriteThread::Writer w;
    write_thread_.EnterUnbatched(&w, &mutex_);
    WriteThread::Writer nonmem_w;
    if (two_write_queues_) {
      nonmem_write_thread_.EnterUnbatched(&nonmem_w, &mutex_);
    }
    WaitForPendingWrites();

    const MutableCFOptions* mopts = cfd->GetLatestMutableCFOptions();
    seq_before = versions_->LastSequence();
    if (cfd->IsDropped()) {
      s = Status::InvalidArgument(
          "InstallExternalMemTable: column family is dropped");
    } else if (versions_->GetColumnFamilySet()->NumberOfColumnFamilies() > 1) {
      // Single-CF only (review 2026-09-24): the block reuses the active
      // memtable's id and carries no WAL record; with several column
      // families the flush id fence (needs_to_sync_closed_wals) and the
      // per-CF WAL floor could both be defeated. The relink driver runs one
      // column family per shard DB.
      s = Status::NotSupported(
          "InstallExternalMemTable: DBs with more than one column family "
          "are not supported");
    } else if (mopts->memtable_protection_bytes_per_key != 0) {
      // MemTableIterator verifies a per-entry checksum the block does not
      // carry (db/memtable.cc VerifyEntryChecksum).
      s = Status::NotSupported(
          "InstallExternalMemTable: memtable_protection_bytes_per_key must "
          "be 0");
    } else if (smallest != largest) {
      // L0 seqno consistency rule (VersionBuilder::CheckConsistencyDetails,
      // active in release via force_consistency_checks): a multi-seqno L0
      // file must have a smallest seqno above every older non-external L0
      // file's. The block is flushed by its own flush job, so any older
      // unflushed write would later produce an L0 file that violates the
      // rule => only accept when nothing older is unflushed and the window
      // lies above everything the DB has handed out.
      const SequenceNumber db_seq =
          std::max({versions_->LastSequence(),
                    versions_->LastPublishedSequence(),
                    versions_->LastAllocatedSequence()});
      if (!cfd->mem()->IsEmpty() || cfd->imm()->NumNotFlushed() > 0) {
        s = Status::InvalidArgument(
            "InstallExternalMemTable: a multi-seqno window (smallest != "
            "largest) requires an empty active memtable and no unflushed "
            "immutable memtable (L0 seqno consistency rule); ship a width-1 "
            "window instead");
      } else if (smallest <= db_seq) {
        s = Status::InvalidArgument(
            "InstallExternalMemTable: a multi-seqno window must lie above "
            "the DB's latest sequence " +
            std::to_string(db_seq) + " (L0 seqno consistency rule)");
      }
    }

    if (s.ok()) {
      // ---- 3. Build the memtable (cheap: one arena block for the empty
      // range-deletion skiplist; the block bytes are referenced in place and
      // not charged to the write buffer manager).
      mem = MemTable::NewFromExternalSortedBlock(
          cfd->internal_comparator(), *cfd->ioptions(), *mopts,
          nullptr /* write_buffer_manager */, cfd->GetID(), std::move(block));
      // No range tombstones => no-op, but keeps the immutable-list invariant
      // (IsFragmentedRangeTombstonesConstructed) explicit like SwitchMemtable.
      mem->ConstructFragmentedRangeTombstones();

      // ---- 4. Raise the DB's sequence so every later write shadows the
      // block. Each setter asserts monotonicity, so bump each individually
      // (allocated -> published -> last; SetLastSequence asserts
      // <= allocated under two_write_queues).
      if (largest > versions_->LastAllocatedSequence()) {
        versions_->SetLastAllocatedSequence(largest);
      }
      if (largest > versions_->LastPublishedSequence()) {
        versions_->SetLastPublishedSequence(largest);
      }
      if (largest > versions_->LastSequence()) {
        versions_->SetLastSequence(largest);
      }

      // ---- 5. Bookkeeping the flush path relies on.
      // No WAL record exists for this memtable. Stamp it with the column
      // family's CURRENT log number, not the DB-global logfile_number_: a
      // block-only flush then leaves the CF's MANIFEST log number where it
      // is, so a WAL that still holds the (older) active memtable's records
      // cannot be declared obsolete (review 2026-09-24; only matters with a
      // WAL and several column families, which the driver never runs). A
      // co-flush with real memtables takes the max of their log numbers.
      mem->SetNextLogNumber(cfd->GetLogNumber());
      // ID = the ACTIVE memtable's id (mempurge precedent). IDs are only
      // minted by ColumnFamilyData::SetMemtable for the active memtable; a
      // fresh id larger than the active's would be skipped by
      // PickMemtablesToFlush (breaks on id > request max) and strand the
      // block. Duplicate ids are tolerated by every id comparison.
      memtable_id = cfd->mem()->GetID();
      mem->SetID(memtable_id);
      // The immutable list takes over this reference (MemTableList::Add
      // does not Ref).
      mem->Ref();
      cfd->imm()->Add(mem, &to_delete);  // push_front + MarkImmutable
      imm_unflushed = cfd->imm()->NumNotFlushed();

      // ---- 6. Publish: new SuperVersion so reads see the block.
      InstallSuperVersionAndScheduleWork(cfd, &sv_ctx, *mopts);

      // ---- 7. Flush scheduling. MemTableList::Add only marks the list;
      // nothing runs until a FlushRequest is queued (write-path pattern
      // db_impl_write.cc). With request_flush == false the block waits for
      // the next memtable switch and is flushed together with the active
      // memtable (single-CF flushes pick every immutable memtable).
      if (request_flush) {
        cfd->imm()->FlushRequested();
        FlushRequest req;
        GenerateFlushRequest({cfd}, &req);
        SchedulePendingFlush(req, FlushReason::kExternalMemTable);
        MaybeScheduleFlushOrCompaction();
      }
    }

    if (two_write_queues_) {
      nonmem_write_thread_.ExitUnbatched(&nonmem_w);
    }
    write_thread_.ExitUnbatched(&w);
  }
  // Outside mutex_: free the old SuperVersion and any history memtables the
  // list trimmed (never the block: it is the newest entry).
  sv_ctx.Clean();
  for (MemTable* m : to_delete) {
    delete m;
  }

  if (s.ok()) {
    ROCKS_LOG_INFO(immutable_db_options_.info_log,
                   "[%s] Installed external memtable id=%" PRIu64
                   " entries=%" PRIu64 " bytes=%" ROCKSDB_PRIszt
                   " seq=[%" PRIu64 ",%" PRIu64 "] db_seq_before=%" PRIu64
                   " db_seq_after=%" PRIu64
                   " imm_unflushed=%d flush_requested=%d",
                   cfd->GetName().c_str(), memtable_id, count, bytes, smallest,
                   largest, seq_before, versions_->LastSequence(),
                   imm_unflushed, request_flush ? 1 : 0);
  }
  return s;
}

}  // namespace ROCKSDB_NAMESPACE
