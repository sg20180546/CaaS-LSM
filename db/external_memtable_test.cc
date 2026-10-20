//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] Tests for ExternalMemTableBlockBuilder and
// DB::InstallExternalMemTable (sorted KV-block installed as an immutable
// memtable, flushed by the normal flush path).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "db/db_test_util.h"
#include "port/stack_trace.h"
#include "rocksdb/external_memtable.h"
#include "rocksdb/utilities/stackable_db.h"
#include "util/coding.h"

#ifndef NDEBUG  // TEST_* wait hooks exist in debug builds only (RocksDB test convention)
namespace ROCKSDB_NAMESPACE {

class ExternalMemTableTest : public DBTestBase {
 public:
  ExternalMemTableTest()
      : DBTestBase("external_memtable_test", /*env_do_fsync=*/false) {}

  struct Entry {
    std::string key;
    SequenceNumber seq;
    ValueType type;
    std::string value;
  };

  // Builds a block whose buffers live on the heap and are freed by
  // block.release, which also counts how often it ran.
  static ExternalMemTableBlock MakeBlock(const std::vector<Entry>& entries,
                                         std::atomic<int>* release_count,
                                         bool request_flush = true) {
    ExternalMemTableBlockBuilder b;
    for (const auto& e : entries) {
      b.Add(e.key, e.seq, e.type, e.value);
    }
    auto* data = new std::string;
    auto* offsets = new std::vector<uint32_t>;
    ExternalMemTableBlock blk;
    EXPECT_OK(b.Finish(data, offsets, &blk.smallest_seqno, &blk.largest_seqno,
                       &blk.count));
    blk.data = data->data();
    blk.data_size = data->size();
    blk.offsets = offsets->data();
    blk.request_flush = request_flush;
    blk.release = [data, offsets, release_count]() {
      delete data;
      delete offsets;
      release_count->fetch_add(1);
    };
    return blk;
  }

  static std::string Key(int i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "k%06d", i);
    return std::string(buf);
  }
  static std::string Val(int i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "v%06d", i);
    return std::string(buf);
  }

  static std::vector<Entry> SequentialEntries(int n, SequenceNumber first_seq,
                                              bool width_one = false) {
    std::vector<Entry> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) {
      v.push_back({Key(i), width_one ? first_seq : first_seq + i,
                   kExternalMemTableTypeValue, Val(i)});
    }
    return v;
  }

  void WaitForRelease(std::atomic<int>* release_count, int expected) {
    for (int tries = 0; tries < 100 && release_count->load() < expected;
         ++tries) {
      Env::Default()->SleepForMicroseconds(50 * 1000);
    }
  }
};

// (e) builder ordering / type / empty rejections.
TEST_F(ExternalMemTableTest, BuilderRejectsBadInput) {
  std::string data;
  std::vector<uint32_t> offsets;
  SequenceNumber smallest, largest;
  uint64_t count;
  {
    ExternalMemTableBlockBuilder b;
    b.Add("b", 5, kExternalMemTableTypeValue, "1");
    b.Add("a", 5, kExternalMemTableTypeValue, "2");  // user key goes backwards
    ASSERT_TRUE(b.Finish(&data, &offsets, &smallest, &largest, &count)
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlockBuilder b;
    b.Add("a", 5, kExternalMemTableTypeValue, "1");
    b.Add("a", 5, kExternalMemTableTypeValue, "2");  // duplicate (key, seq)
    ASSERT_TRUE(b.Finish(&data, &offsets, &smallest, &largest, &count)
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlockBuilder b;
    b.Add("a", 5, kExternalMemTableTypeValue, "1");
    b.Add("a", 6, kExternalMemTableTypeValue, "2");  // seq must descend
    ASSERT_TRUE(b.Finish(&data, &offsets, &smallest, &largest, &count)
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlockBuilder b;
    b.Add("a", 5, static_cast<ValueType>(0xF) /* range deletion */, "1");
    ASSERT_TRUE(b.Finish(&data, &offsets, &smallest, &largest, &count)
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlockBuilder b;  // nothing added
    ASSERT_TRUE(b.Finish(&data, &offsets, &smallest, &largest, &count)
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlockBuilder b(1 << 10);
    b.Add("a", 6, kExternalMemTableTypeValue, "new");
    b.Add("a", 5, kExternalMemTableTypeValue, "old");
    b.Add("b", 7, kExternalMemTableTypeDeletion, "");
    b.Add("c", 3, kExternalMemTableTypeSingleDeletion, "");
    ASSERT_OK(b.Finish(&data, &offsets, &smallest, &largest, &count));
    ASSERT_EQ(count, 4u);
    ASSERT_EQ(smallest, 3u);
    ASSERT_EQ(largest, 7u);
    ASSERT_EQ(offsets.size(), 4u);
    ASSERT_EQ(offsets[0], 0u);
    // varint(1+8)=1 | "a" | 8 | varint(3)=1 | "new" => 14 bytes
    ASSERT_EQ(offsets[1], 14u);
    ASSERT_LT(offsets[3], data.size());
    // Builder is reset after Finish.
    ASSERT_EQ(b.count(), 0u);
    ASSERT_EQ(b.data_size(), 0u);
  }
}

// (a) + (b) + (d) + (c, manual flush): 10k keys, reads, iterators, Seek
// semantics, precedence of later writes, absent keys, release() exactly once
// after the flush and not before.
TEST_F(ExternalMemTableTest, InstallReadsIteratorsAndManualFlush) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  const int kNum = 10000;
  const SequenceNumber s0 = db_->GetLatestSequenceNumber();
  std::atomic<int> released{0};
  // Multi-seqno window above the DB's sequence on a fresh (empty) CF.
  ExternalMemTableBlock blk = MakeBlock(SequentialEntries(kNum, s0 + 1),
                                        &released, /*request_flush=*/false);
  ASSERT_EQ(blk.count, static_cast<uint64_t>(kNum));
  ASSERT_EQ(blk.smallest_seqno, s0 + 1);
  ASSERT_EQ(blk.largest_seqno, s0 + kNum);

  const Snapshot* before = db_->GetSnapshot();
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  ASSERT_EQ(released.load(), 0);
  // The DB's sequence now covers the block.
  ASSERT_EQ(db_->GetLatestSequenceNumber(), s0 + kNum);
  ASSERT_EQ(NumTableFilesAtLevel(0), 0);

  // Point reads.
  for (int i = 0; i < kNum; ++i) {
    ASSERT_EQ(Get(Key(i)), Val(i));
  }
  // (d) absent keys: before, between and after the block's keys.
  ASSERT_EQ(Get("a"), "NOT_FOUND");
  ASSERT_EQ(Get("k0050005"), "NOT_FOUND");
  ASSERT_EQ(Get("zzz"), "NOT_FOUND");
  // A snapshot taken before the install does not see the block.
  ASSERT_EQ(Get(Key(0), before), "NOT_FOUND");
  db_->ReleaseSnapshot(before);

  // Iterators: forward, backward, Seek / SeekForPrev semantics.
  {
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    int n = 0;
    for (it->SeekToFirst(); it->Valid(); it->Next(), ++n) {
      ASSERT_EQ(it->key().ToString(), Key(n));
      ASSERT_EQ(it->value().ToString(), Val(n));
    }
    ASSERT_OK(it->status());
    ASSERT_EQ(n, kNum);
    n = kNum - 1;
    for (it->SeekToLast(); it->Valid(); it->Prev(), --n) {
      ASSERT_EQ(it->key().ToString(), Key(n));
    }
    ASSERT_OK(it->status());
    ASSERT_EQ(n, -1);

    it->Seek(Key(5000));
    ASSERT_TRUE(it->Valid());
    ASSERT_EQ(it->key().ToString(), Key(5000));
    it->Seek("k0050005");  // between k005000 and k005001
    ASSERT_TRUE(it->Valid());
    ASSERT_EQ(it->key().ToString(), Key(5001));
    it->Seek("zzz");
    ASSERT_FALSE(it->Valid());
    ASSERT_OK(it->status());
    it->SeekForPrev("k0050005");
    ASSERT_TRUE(it->Valid());
    ASSERT_EQ(it->key().ToString(), Key(5000));
    it->SeekForPrev(Key(0));
    ASSERT_TRUE(it->Valid());
    ASSERT_EQ(it->key().ToString(), Key(0));
    it->SeekForPrev("a");
    ASSERT_FALSE(it->Valid());
    ASSERT_OK(it->status());
    it->Seek(Key(9999));
    ASSERT_TRUE(it->Valid());
    it->Next();
    ASSERT_FALSE(it->Valid());
  }

  // (b) precedence: a later Put (higher seqno) shadows the block; a key only
  // in the block stays visible.
  ASSERT_OK(Put(Key(10), "new"));
  ASSERT_EQ(Get(Key(10)), "new");
  ASSERT_EQ(Get(Key(11)), Val(11));
  ASSERT_GT(db_->GetLatestSequenceNumber(), s0 + kNum);
  {
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    it->Seek(Key(10));
    ASSERT_TRUE(it->Valid());
    ASSERT_EQ(it->value().ToString(), "new");
    it->Next();
    ASSERT_EQ(it->key().ToString(), Key(11));
  }
  // Still in DRAM: nothing flushed, release not called.
  ASSERT_EQ(released.load(), 0);
  ASSERT_EQ(NumTableFilesAtLevel(0), 0);

  // (c) manual flush: block + active memtable go to L0; release exactly once
  // afterwards; reads unchanged.
  // [relink memtable 2026-09-27] allow_write_stall: a request_flush=false
  // block is an unflushed immutable memtable with NO queued flush, so with the
  // default max_write_buffer_number=2 DB::Flush's pre-wait
  // (WaitUntilFlushWouldNotStallWrites) sleeps on bg_cv_ forever (the test
  // hung before and after the 0927 WAL fix; lldb stack in
  // exp_out/_scratch/pushsrc_0927/crash/fixcheck). The relink driver installs
  // with request_flush=1.
  {
    FlushOptions fo;
    fo.allow_write_stall = true;
    ASSERT_OK(db_->Flush(fo));
  }
  { FlushOptions fo; fo.wait = true; ASSERT_OK(db_->Flush(fo)); }  // NDEBUG-safe wait (TEST_WaitForFlushMemTable is debug-only)
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_GE(NumTableFilesAtLevel(0), 1);
  {
    ColumnFamilyMetaData meta;
    db_->GetColumnFamilyMetaData(&meta);
    SequenceNumber lo = kMaxSequenceNumber, hi = 0;
    for (const auto& f : meta.levels[0].files) {
      lo = std::min(lo, f.smallest_seqno);
      hi = std::max(hi, f.largest_seqno);
    }
    ASSERT_EQ(lo, s0 + 1);           // block's smallest
    ASSERT_EQ(hi, s0 + kNum + 1);    // the Put after the install
  }
  for (int i = 0; i < kNum; ++i) {
    ASSERT_EQ(Get(Key(i)), i == 10 ? std::string("new") : Val(i));
  }
  ASSERT_EQ(Get("k0050005"), "NOT_FOUND");
  Reopen(options);
  ASSERT_EQ(Get(Key(10)), "new");
  ASSERT_EQ(Get(Key(4242)), Val(4242));
  ASSERT_EQ(released.load(), 1);
}

// (c) auto flush requested at install: L0 file with the block's seqno window,
// release exactly once after the flush; in-block versions / tombstones obey
// seqno precedence.
TEST_F(ExternalMemTableTest, AutoFlushOnInstallAndInBlockVersions) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  const SequenceNumber s0 = db_->GetLatestSequenceNumber();
  std::vector<Entry> entries = SequentialEntries(1000, s0 + 1);
  // Two versions of one key, a tombstone over a value, a single-deletion.
  entries.push_back({"x", s0 + 2000, kExternalMemTableTypeValue, "x-new"});
  entries.push_back({"x", s0 + 1500, kExternalMemTableTypeValue, "x-old"});
  entries.push_back({"y", s0 + 2000, kExternalMemTableTypeDeletion, ""});
  entries.push_back({"y", s0 + 1500, kExternalMemTableTypeValue, "y-old"});
  entries.push_back({"z", s0 + 2000, kExternalMemTableTypeSingleDeletion, ""});
  std::atomic<int> released{0};
  ExternalMemTableBlock blk = MakeBlock(entries, &released);
  ASSERT_EQ(blk.smallest_seqno, s0 + 1);
  ASSERT_EQ(blk.largest_seqno, s0 + 2000);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  ASSERT_EQ(db_->GetLatestSequenceNumber(), s0 + 2000);

  { FlushOptions fo; fo.wait = true; ASSERT_OK(db_->Flush(fo)); }  // NDEBUG-safe wait (TEST_WaitForFlushMemTable is debug-only)
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0), 1);
  {
    ColumnFamilyMetaData meta;
    db_->GetColumnFamilyMetaData(&meta);
    ASSERT_EQ(meta.levels[0].files.size(), 1u);
    ASSERT_EQ(meta.levels[0].files[0].smallest_seqno, s0 + 1);
    ASSERT_EQ(meta.levels[0].files[0].largest_seqno, s0 + 2000);
    // The flush's CompactionIterator may drop the versions hidden by a newer
    // entry in the same snapshot stripe (x@1500, y@1500); every visible
    // entry must be there.
    ASSERT_GE(meta.levels[0].files[0].num_entries, 1000u + 1u);
    ASSERT_LE(meta.levels[0].files[0].num_entries, entries.size());
  }
  for (int i = 0; i < 1000; ++i) {
    ASSERT_EQ(Get(Key(i)), Val(i));
  }
  ASSERT_EQ(Get("x"), "x-new");
  ASSERT_EQ(Get("y"), "NOT_FOUND");
  ASSERT_EQ(Get("z"), "NOT_FOUND");
  // Later writes still win after the flush.
  ASSERT_OK(Put("x", "x-newer"));
  ASSERT_EQ(Get("x"), "x-newer");
  Reopen(options);
  ASSERT_EQ(Get("x"), "x-newer");
  ASSERT_EQ(Get("y"), "NOT_FOUND");
  ASSERT_EQ(Get(Key(999)), Val(999));
}

// [relink memtable 2026-09-27] Regression: an EMPTY relink destination
// (fresh DB, nothing flushed yet) crashed in FindObsoleteFiles right after the
// block's flush (s1/s2 of the 0927 push-source cells; SIGSEGV in release,
// assert(alive_log_files_.size()) in debug). A DB that never committed a
// flush has column-family log number 0 (NewDB writes 0, no WAL recovered); the
// block used to carry that 0 into its flush's VersionEdit,
// PrecomputeMinLogNumberToKeepNon2PC turned it into UINT64_MAX (one CF), and
// FindObsoleteFiles popped the current WAL. The flush must also keep the WAL
// that holds the still-unflushed Put.
TEST_F(ExternalMemTableTest, BlockOnlyFlushOnNeverFlushedDB) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())->cfd();

  // WAL-backed write that stays in the active memtable: it lives only in the
  // current WAL, so that WAL must survive the block's flush.
  ASSERT_OK(Put("pre", "p"));
  ASSERT_EQ(cfd->GetLogNumber(), 0u);  // precondition: never flushed

  // Width-1 window above the DB's sequence with the default auto flush
  // request: the shape of the crashed destinations (seq 1, request_flush=1).
  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(SequentialEntries(100, g, /*width_one=*/true), &released);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  // Only the requested background flush runs. No db_->Flush(): it would
  // switch the active memtable and co-flush it with a real log number,
  // hiding the block-only case.
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0), 1);
  // The WAL floor is a real WAL number and never passes the current WAL.
  const uint64_t min_log = dbfull()->MinLogNumberToKeep();
  ASSERT_GT(min_log, 0u);
  ASSERT_LE(min_log, dbfull()->TEST_LogfileNumber());
  ASSERT_LE(cfd->GetLogNumber(), dbfull()->TEST_LogfileNumber());

  ASSERT_OK(Put("post", "q"));
  ASSERT_EQ(Get(Key(42)), Val(42));
  // Both WAL-only writes survive recovery; the block's L0 file is intact.
  Reopen(options);
  ASSERT_EQ(Get("pre"), "p");
  ASSERT_EQ(Get("post"), "q");
  ASSERT_EQ(Get(Key(0)), Val(0));
  ASSERT_EQ(Get(Key(99)), Val(99));
}

// [relink memtable 2026-09-27] Why the log-number fallback is the OLDEST alive
// WAL and not logfile_number_: a reopen with avoid_flush_during_recovery keeps
// the recovered writes in the active memtable, leaves the CF log number at 0
// and keeps their WAL alive next to the newly created one. Stamping the block
// with logfile_number_ (the new WAL) would let the block-only flush declare the
// recovered WAL obsolete and lose its write on the next reopen.
TEST_F(ExternalMemTableTest, BlockOnlyFlushKeepsRecoveredWal) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.avoid_flush_during_recovery = true;
  DestroyAndReopen(options);
  ASSERT_OK(Put("pre", "p"));
  Reopen(options);  // replays the WAL into the memtable without a flush
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())->cfd();
  ASSERT_EQ(cfd->GetLogNumber(), 0u);  // precondition: never flushed
  uint64_t recovered_wal = 0;
  {
    // GetSortedWalFiles lists non-empty WALs only (WalManager skips a WAL
    // with no record): the recovered one, not the empty WAL opened after it.
    VectorLogPtr wals;
    ASSERT_OK(db_->GetSortedWalFiles(wals));
    ASSERT_EQ(wals.size(), 1u);
    ASSERT_EQ(wals[0]->Type(), kAliveLogFile);
    recovered_wal = wals[0]->LogNumber();
  }
  ASSERT_LT(recovered_wal, dbfull()->TEST_LogfileNumber());

  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(SequentialEntries(10, g, /*width_one=*/true), &released);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0), 1);
  // The recovered WAL is still the floor and still alive on disk.
  ASSERT_EQ(dbfull()->MinLogNumberToKeep(), recovered_wal);
  {
    VectorLogPtr wals;
    ASSERT_OK(db_->GetSortedWalFiles(wals));
    ASSERT_EQ(wals.size(), 1u);
    ASSERT_EQ(wals[0]->LogNumber(), recovered_wal);
    ASSERT_EQ(wals[0]->Type(), kAliveLogFile);
  }
  Reopen(options);
  ASSERT_EQ(Get("pre"), "p");
  ASSERT_EQ(Get(Key(0)), Val(0));
  ASSERT_EQ(Get(Key(9)), Val(9));
}

// (f) L0 seqno rule: width-1 window at the DB's latest sequence coexists with
// older unflushed writes (the relink shape); a multi-seqno window is refused
// while anything older is unflushed or while it does not lie above the DB's
// sequence.
TEST_F(ExternalMemTableTest, L0SeqnoRuleWidthOneWindow) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  ASSERT_TRUE(options.force_consistency_checks);
  DestroyAndReopen(options);

  ASSERT_OK(Put("a", "1"));  // unflushed, seq 1
  const SequenceNumber g = db_->GetLatestSequenceNumber();  // gsn_base
  ASSERT_EQ(g, 1u);
  std::atomic<int> released{0};
  ExternalMemTableBlock blk = MakeBlock(
      {{"b", g, kExternalMemTableTypeValue, "2"},
       {"c", g, kExternalMemTableTypeValue, "3"}},
      &released);
  ASSERT_EQ(blk.smallest_seqno, blk.largest_seqno);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  ASSERT_EQ(db_->GetLatestSequenceNumber(), g);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get("c"), "3");
  // The block is flushed alone (auto request) into an external-shape file.
  { FlushOptions fo; fo.wait = true; ASSERT_OK(db_->Flush(fo)); }  // NDEBUG-safe wait (TEST_WaitForFlushMemTable is debug-only)
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0), 1);
  // The older unflushed write plus a newer one form a second, NEWER L0 file
  // whose smallest seqno is below the block's: allowed by the external-file
  // exemption, and the Version must pass force_consistency_checks.
  ASSERT_OK(Put("d", "4"));
  ASSERT_OK(Flush());
  ASSERT_EQ(NumTableFilesAtLevel(0), 2);
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  Reopen(options);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get("c"), "3");
  ASSERT_EQ(Get("d"), "4");

  // Multi-seqno window with an unflushed active memtable -> refused, block
  // not consumed.
  ASSERT_OK(Put("e", "5"));
  const SequenceNumber s = db_->GetLatestSequenceNumber();
  std::atomic<int> released2{0};
  ExternalMemTableBlock multi = MakeBlock(
      {{"f", s + 2, kExternalMemTableTypeValue, "6"},
       {"g", s + 1, kExternalMemTableTypeValue, "7"}},
      &released2);
  Status st = db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                           std::move(multi));
  ASSERT_TRUE(st.IsInvalidArgument()) << st.ToString();
  ASSERT_EQ(released2.load(), 0);
  ASSERT_EQ(Get("f"), "NOT_FOUND");
  // Flush everything, then a multi-seqno window that is NOT above the DB's
  // sequence is refused too.
  ASSERT_OK(Flush());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  multi.smallest_seqno = s;  // pretend: overlaps the DB's sequence
  st = db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                    std::move(multi));
  ASSERT_TRUE(st.IsInvalidArgument()) << st.ToString();
  ASSERT_EQ(released2.load(), 0);
  multi.smallest_seqno = s + 1;
  // Now the CF has nothing unflushed and the window lies above s -> OK.
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(multi)));
  ASSERT_EQ(Get("f"), "6");
  ASSERT_EQ(Get("g"), "7");
  { FlushOptions fo; fo.wait = true; ASSERT_OK(db_->Flush(fo)); }  // NDEBUG-safe wait (TEST_WaitForFlushMemTable is debug-only)
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released2, 1);
  ASSERT_EQ(released2.load(), 1);
  Reopen(options);
  ASSERT_EQ(Get("e"), "5");
  ASSERT_EQ(Get("f"), "6");
}

// (g) argument validation never consumes the block.
TEST_F(ExternalMemTableTest, ValidationFailuresDoNotConsumeBlock) {
  Options options = CurrentOptions();
  DestroyAndReopen(options);
  const SequenceNumber s0 = db_->GetLatestSequenceNumber();
  std::atomic<int> released{0};
  ExternalMemTableBlock good = MakeBlock(SequentialEntries(10, s0 + 1, true),
                                         &released);
  ColumnFamilyHandle* cf = db_->DefaultColumnFamily();

  {
    ExternalMemTableBlock b = good;  // copies pointers + release
    Status st = db_->InstallExternalMemTable(nullptr, std::move(b));
    ASSERT_TRUE(st.IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    b.count = 0;
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    b.smallest_seqno = 0;
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    b.smallest_seqno = b.largest_seqno + 1;
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    b.data_size = static_cast<size_t>(1) << 32;  // rejected before any read
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    std::vector<uint32_t> bad(good.offsets, good.offsets + good.count);
    bad[0] = 1;  // offsets[0] must be 0
    b.offsets = bad.data();
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
    bad[0] = 0;
    bad[2] = bad[1];  // not strictly increasing
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
    bad[2] = bad[1] + 1;
    bad[good.count - 1] = static_cast<uint32_t>(good.data_size);  // == size
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  {
    ExternalMemTableBlock b = good;
    b.data = nullptr;
    ASSERT_TRUE(db_->InstallExternalMemTable(cf, std::move(b))
                    .IsInvalidArgument());
  }
  ASSERT_EQ(released.load(), 0);
  ASSERT_EQ(Get(Key(0)), "NOT_FOUND");

  // [migagent 2026-10-20] Hand-encoded entry with type kTypeMerge (0x2): the
  // builder refuses it, and the installer's release-mode scan must too
  // (Corruption), without consuming the block.
  {
    std::string data;
    PutVarint32(&data, 1 + 8);  // |"m"| + 8
    data.append("m", 1);
    PutFixed64(&data, (static_cast<uint64_t>(s0 + 1) << 8) | 0x2);
    PutVarint32(&data, 1);
    data.append("v", 1);
    std::vector<uint32_t> offsets{0};
    std::atomic<int> merge_released{0};
    ExternalMemTableBlock b;
    b.data = data.data();
    b.data_size = data.size();
    b.offsets = offsets.data();
    b.count = 1;
    b.smallest_seqno = b.largest_seqno = s0 + 1;
    b.release = [&merge_released]() { merge_released.fetch_add(1); };
    Status st = db_->InstallExternalMemTable(cf, std::move(b));
    ASSERT_TRUE(st.IsCorruption()) << st.ToString();
    ASSERT_EQ(merge_released.load(), 0);
    ASSERT_EQ(Get("m"), "NOT_FOUND");
    // A second entry whose header runs past its offset slot (the first
    // entry's slot ends where the second begins) -> InvalidArgument.
    std::string two = data;
    PutVarint32(&two, 1 + 8);
    two.append("n", 1);
    PutFixed64(&two, (static_cast<uint64_t>(s0 + 1) << 8) | 0x1);
    PutVarint32(&two, 200);  // claims 200 value bytes, none follow
    std::vector<uint32_t> offsets2{0, static_cast<uint32_t>(data.size())};
    ExternalMemTableBlock b2;
    b2.data = two.data();
    b2.data_size = two.size();
    b2.offsets = offsets2.data();
    b2.count = 2;
    b2.smallest_seqno = b2.largest_seqno = s0 + 1;
    b2.release = [&merge_released]() { merge_released.fetch_add(1); };
    st = db_->InstallExternalMemTable(cf, std::move(b2));
    ASSERT_TRUE(st.IsInvalidArgument() || st.IsCorruption()) << st.ToString();
    ASSERT_EQ(merge_released.load(), 0);
  }

  // Per-entry memtable checksums are not carried by a block -> NotSupported.
  options.memtable_protection_bytes_per_key = 8;
  Reopen(options);
  {
    ExternalMemTableBlock b = good;
    Status st = db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                             std::move(b));
    ASSERT_TRUE(st.IsNotSupported()) << st.ToString();
  }
  ASSERT_EQ(released.load(), 0);
  // The block was never consumed: the test frees it.
  good.release();
  ASSERT_EQ(released.load(), 1);
}

// StackableDB forwards; a DB that does not override returns NotSupported.
TEST_F(ExternalMemTableTest, StackableDBPassthrough) {
  Options options = CurrentOptions();
  DestroyAndReopen(options);
  const SequenceNumber s0 = db_->GetLatestSequenceNumber();
  std::atomic<int> released{0};
  ExternalMemTableBlock blk = MakeBlock(SequentialEntries(5, s0 + 1, true),
                                        &released, /*request_flush=*/false);
  {
    std::shared_ptr<DB> no_own(db_, [](DB*) {});
    StackableDB sdb(no_own);
    ASSERT_OK(sdb.InstallExternalMemTable(sdb.DefaultColumnFamily(),
                                          std::move(blk)));
    ASSERT_EQ(Get(Key(3)), Val(3));
  }
  // [relink memtable 2026-09-27] allow_write_stall: request_flush=false
  // leaves no queued flush; see InstallReadsIteratorsAndManualFlush.
  {
    FlushOptions fo;
    fo.allow_write_stall = true;
    ASSERT_OK(db_->Flush(fo));
  }
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(Get(Key(3)), Val(3));
}

// [migagent 2026-10-20] A block's tombstones are counted at install (the
// block carries no delete count) and reported like a stock immutable
// memtable's; after the flush the tombstones apply to the older L0 versions.
TEST_F(ExternalMemTableTest, TombstoneBlockCountsDeletes) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  ASSERT_OK(Put("d0", "x"));
  ASSERT_OK(Put("s0", "y"));
  ASSERT_OK(Flush());  // older versions of two of the deleted keys in L0

  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::vector<Entry> entries;  // bytewise order: "d*" < "k*" < "s*"
  for (int i = 0; i < 5; ++i) {
    entries.push_back(
        {"d" + std::to_string(i), g, kExternalMemTableTypeDeletion, ""});
  }
  for (int i = 0; i < 10; ++i) {
    entries.push_back({Key(i), g, kExternalMemTableTypeValue, Val(i)});
  }
  for (int i = 0; i < 3; ++i) {
    entries.push_back(
        {"s" + std::to_string(i), g, kExternalMemTableTypeSingleDeletion, ""});
  }
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(entries, &released, /*request_flush=*/false);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  uint64_t v = 0;
  ASSERT_TRUE(
      db_->GetIntProperty(DB::Properties::kNumDeletesImmMemTables, &v));
  ASSERT_EQ(v, 8u);  // 5 Deletion + 3 SingleDeletion
  ASSERT_TRUE(
      db_->GetIntProperty(DB::Properties::kNumEntriesImmMemTables, &v));
  ASSERT_EQ(v, 18u);
  ASSERT_EQ(Get("d0"), "NOT_FOUND");
  ASSERT_EQ(Get("s0"), "NOT_FOUND");
  ASSERT_EQ(Get(Key(1)), Val(1));
  {
    FlushOptions fo;
    fo.allow_write_stall = true;  // request_flush=false: no queued flush
    ASSERT_OK(db_->Flush(fo));
  }
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_TRUE(
      db_->GetIntProperty(DB::Properties::kNumDeletesImmMemTables, &v));
  ASSERT_EQ(v, 0u);
  ASSERT_EQ(Get("d0"), "NOT_FOUND");
  ASSERT_EQ(Get("s0"), "NOT_FOUND");
  ASSERT_EQ(Get(Key(9)), Val(9));
  Reopen(options);
  ASSERT_EQ(Get("d0"), "NOT_FOUND");
  ASSERT_EQ(Get("s0"), "NOT_FOUND");
  ASSERT_EQ(Get(Key(0)), Val(0));
}

// [migagent 2026-10-20] Multi-CF rules: a DB with more than one column family
// (MyRocks: {default, __system__}) accepts a block for the DEFAULT column
// family only, and only with request_flush = true. Both refusals leave the
// block unconsumed.
TEST_F(ExternalMemTableTest, MultiCFDefaultOnly) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  CreateAndReopenWithCF({"__system__"}, options);
  ASSERT_EQ(handles_.size(), 2u);
  // handles_[0] is a distinct handle object for the default CF (the check in
  // the engine compares the ColumnFamilyData, not the handle).
  ASSERT_EQ(handles_[0]->GetID(), db_->DefaultColumnFamily()->GetID());

  const SequenceNumber s0 = db_->GetLatestSequenceNumber();
  std::atomic<int> released{0};
  ExternalMemTableBlock good = MakeBlock(SequentialEntries(10, s0 + 1, true),
                                         &released);
  // Non-default column family -> NotSupported, block not consumed.
  {
    ExternalMemTableBlock b = good;
    Status st = db_->InstallExternalMemTable(handles_[1], std::move(b));
    ASSERT_TRUE(st.IsNotSupported()) << st.ToString();
    ASSERT_NE(st.ToString().find("only the default column family"),
              std::string::npos)
        << st.ToString();
  }
  ASSERT_EQ(released.load(), 0);
  ASSERT_EQ(Get(1, Key(0)), "NOT_FOUND");
  ASSERT_EQ(Get(0, Key(0)), "NOT_FOUND");
  // request_flush = false with >1 CF -> InvalidArgument, block not consumed.
  {
    ExternalMemTableBlock b = good;
    b.request_flush = false;
    Status st = db_->InstallExternalMemTable(handles_[0], std::move(b));
    ASSERT_TRUE(st.IsInvalidArgument()) << st.ToString();
  }
  ASSERT_EQ(released.load(), 0);
  ASSERT_EQ(Get(0, Key(0)), "NOT_FOUND");
  ASSERT_EQ(db_->GetLatestSequenceNumber(), s0);

  // Default column family with request_flush = true -> installed, visible,
  // flushed by its own request, released exactly once.
  ASSERT_OK(db_->InstallExternalMemTable(handles_[0], std::move(good)));
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(Get(0, Key(i)), Val(i));
    ASSERT_EQ(Get(1, Key(i)), "NOT_FOUND");
  }
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[0]));
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0, 0), 1);
  ASSERT_EQ(NumTableFilesAtLevel(0, 1), 0);
  ASSERT_OK(Put(1, "sys", "1"));
  ReopenWithColumnFamilies({"default", "__system__"}, options);
  ASSERT_EQ(Get(0, Key(0)), Val(0));
  ASSERT_EQ(Get(0, Key(9)), Val(9));
  ASSERT_EQ(Get(1, "sys"), "1");
  ASSERT_EQ(Get(1, Key(0)), "NOT_FOUND");
}

// [migagent 2026-10-20] Multi-CF flush id fence. With >1 CF and a WAL,
// FlushMemTableToOutputFile captures max_memtable_id = the newest immutable's
// id (the block's) before SyncClosedLogs releases the mutex. A memtable
// switched in that window keeps the active's id; if the block had been
// stamped with the active's id too, the switched memtable would pass the
// fence and be flushed on unsynced WAL data. The block therefore carries
// active - 1 in multi-CF DBs; the single-CF path (no fence, max id =
// UINT64_MAX) keeps the active's id and still co-flushes everything.
TEST_F(ExternalMemTableTest, MultiCFFenceExcludesSwitchedActive) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.max_write_buffer_number = 4;  // two immutables must not stall
  CreateAndReopenWithCF({"__system__"}, options);
  ASSERT_GT(dbfull()->TEST_LogfileNumber(), 0u);  // WAL on
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(handles_[0])->cfd();
  ASSERT_OK(Put(0, "pre", "p"));  // WAL-backed, stays in the active memtable
  const uint64_t active_id = cfd->mem()->GetID();
  ASSERT_GE(active_id, 1u);

  std::atomic<int> switches{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncClosedLogs:Start", [&](void*) {
        // The flush job has captured max_memtable_id and dropped the mutex.
        if (switches.fetch_add(1) == 0) {
          ASSERT_OK(dbfull()->TEST_SwitchMemtable(cfd));
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(SequentialEntries(10, g, /*width_one=*/true), &released);
  ASSERT_OK(db_->InstallExternalMemTable(handles_[0], std::move(blk)));
  // The block's own flush commits (release runs); the switched memtable has
  // no flush scheduled, so it must still be there afterwards.
  WaitForRelease(&released, 1);
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(switches.load(), 1);
  ASSERT_EQ(cfd->mem()->GetID(), active_id + 1);  // the switch happened
  uint64_t num_imm = 0;
  ASSERT_TRUE(db_->GetIntProperty(
      handles_[0], DB::Properties::kNumImmutableMemTable, &num_imm));
  ASSERT_EQ(num_imm, 1u);
  ASSERT_EQ(NumTableFilesAtLevel(0, 0), 1);
  {
    // The L0 file holds the block alone: 10 entries at seq g, not "pre".
    ColumnFamilyMetaData meta;
    db_->GetColumnFamilyMetaData(handles_[0], &meta);
    ASSERT_EQ(meta.levels[0].files.size(), 1u);
    ASSERT_EQ(meta.levels[0].files[0].num_entries, 10u);
    ASSERT_EQ(meta.levels[0].files[0].smallest_seqno, g);
    ASSERT_EQ(meta.levels[0].files[0].largest_seqno, g);
  }
  ASSERT_EQ(Get(0, "pre"), "p");
  ASSERT_EQ(Get(0, Key(3)), Val(3));
  // Flush the switched memtable; everything survives a reopen.
  ASSERT_OK(Flush(0));
  ASSERT_EQ(NumTableFilesAtLevel(0, 0), 2);
  ReopenWithColumnFamilies({"default", "__system__"}, options);
  ASSERT_EQ(Get(0, "pre"), "p");
  ASSERT_EQ(Get(0, Key(0)), Val(0));
  ASSERT_EQ(Get(0, Key(9)), Val(9));

  // Single-CF equivalent: no fence, so a memtable switched after the flush
  // request was queued (DBImpl::BGWorkFlush runs before the pick, without
  // the mutex) is picked together with the block (id = active's id).
  options.max_write_buffer_number = 2;
  DestroyAndReopen(options);
  cfd = static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())
            ->cfd();
  ASSERT_OK(Put("pre", "p"));
  const uint64_t active_id1 = cfd->mem()->GetID();
  switches.store(0);
  SyncPoint::GetInstance()->SetCallBack("DBImpl::BGWorkFlush", [&](void*) {
    if (switches.fetch_add(1) == 0) {
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(cfd));
    }
  });
  SyncPoint::GetInstance()->EnableProcessing();
  const SequenceNumber g1 = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released1{0};
  ExternalMemTableBlock blk1 =
      MakeBlock(SequentialEntries(10, g1, /*width_one=*/true), &released1);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk1)));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  WaitForRelease(&released1, 1);
  ASSERT_EQ(released1.load(), 1);
  ASSERT_EQ(switches.load(), 1);
  ASSERT_EQ(cfd->mem()->GetID(), active_id1 + 1);
  ASSERT_TRUE(db_->GetIntProperty(DB::Properties::kNumImmutableMemTable,
                                  &num_imm));
  ASSERT_EQ(num_imm, 0u);  // block AND the switched memtable were picked
  ASSERT_EQ(NumTableFilesAtLevel(0), 1);
  {
    ColumnFamilyMetaData meta;
    db_->GetColumnFamilyMetaData(&meta);
    ASSERT_EQ(meta.levels[0].files.size(), 1u);
    ASSERT_EQ(meta.levels[0].files[0].num_entries, 11u);  // 10 + "pre"
  }
  ASSERT_EQ(Get("pre"), "p");
  ASSERT_EQ(Get(Key(3)), Val(3));
}

// [migagent 2026-10-20] Block-only flush log number, sequential case. The
// block is stamped at install with the CF log number of that moment. An older
// plain immutable, already being flushed (parked inside WriteLevel0Table with
// the mutex released), carries the newer WAL opened at its switch; its commit
// raises the CF log number, and the block's own flush -- queued behind it on
// the single flush thread -- then picks the block alone. Without the clamp in
// FlushJob::PickMemTable its edit carries the lower stamp: the debug assert
// in VersionSet::LogAndApplyHelper fires; a release build writes a
// non-monotonic log number that recovery reports as "NOT monotonically
// increasing".
TEST_F(ExternalMemTableTest, BlockOnlyFlushAfterOlderImmCommitWithWal) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.max_write_buffer_number = 4;
  options.max_background_flushes = 1;  // the block's flush waits for the older
  DestroyAndReopen(options);
  ASSERT_GT(dbfull()->TEST_LogfileNumber(), 0u);  // WAL on
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())->cfd();

  // A committed flush gives the CF a real log number: the block's stamp.
  ASSERT_OK(Put("a", "1"));
  ASSERT_OK(Flush());
  const uint64_t log_at_install = cfd->GetLogNumber();
  ASSERT_GT(log_at_install, 0u);

  // Older plain immutable whose flush parks with the mutex released.
  ASSERT_OK(Put("b", "2"));
  std::atomic<bool> parked{false};
  std::atomic<bool> release_older{false};
  std::atomic<int> arrivals{0};
  SyncPoint::GetInstance()->SetCallBack(
      "FlushJob::WriteLevel0Table:output_compression", [&](void*) {
        if (arrivals.fetch_add(1) == 0) {
          parked.store(true);
          while (!release_older.load()) {
            Env::Default()->SleepForMicroseconds(1000);
          }
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  {
    FlushOptions fo;
    fo.wait = false;
    ASSERT_OK(db_->Flush(fo));  // switches "b" out and schedules its flush
  }
  for (int i = 0; i < 20000 && !parked.load(); ++i) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_TRUE(parked.load());
  const uint64_t imm_next_log = dbfull()->TEST_LogfileNumber();
  ASSERT_GT(imm_next_log, log_at_install);  // the WAL opened at the switch

  // Install while the older flush is in flight: stamp = log_at_install.
  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(SequentialEntries(10, g, /*width_one=*/true), &released);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  ASSERT_EQ(cfd->GetLogNumber(), log_at_install);
  // Let the older flush commit (CF log number -> imm_next_log); the block's
  // flush then runs alone.
  release_older.store(true);
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(arrivals.load(), 2);  // older flush, then the block-only flush
  ASSERT_EQ(NumTableFilesAtLevel(0), 3);
  ASSERT_GE(cfd->GetLogNumber(), imm_next_log);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get(Key(5)), Val(5));

  Reopen(options);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get(Key(0)), Val(0));
  ASSERT_EQ(Get(Key(9)), Val(9));
  std::string log_contents;
  ASSERT_OK(ReadFileToString(env_, dbname_ + "/LOG", &log_contents));
  ASSERT_EQ(log_contents.find("NOT monotonically increasing"),
            std::string::npos);
}

// [migagent 2026-10-20] Block-only flush log number, concurrent case: two
// flush threads. The older immutable's flush is parked, the block's flush
// picks the block alone (the older one is in progress) BEFORE the older
// commit raises the CF log number, then parks itself; the older flush commits
// in between. max(stamp, CF log number) at pick time is too early here; the
// clamp also folds in the NextLogNumber of older memtables still in flight.
TEST_F(ExternalMemTableTest, BlockOnlyFlushConcurrentWithOlderImmCommitWithWal) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.max_write_buffer_number = 4;
  options.max_background_flushes = 2;  // two flush jobs of one CF in flight
  DestroyAndReopen(options);
  ASSERT_GT(dbfull()->TEST_LogfileNumber(), 0u);  // WAL on
  ColumnFamilyData* cfd =
      static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())->cfd();

  ASSERT_OK(Put("a", "1"));
  ASSERT_OK(Flush());
  const uint64_t log_at_install = cfd->GetLogNumber();
  ASSERT_GT(log_at_install, 0u);

  ASSERT_OK(Put("b", "2"));
  std::atomic<bool> older_parked{false}, block_parked{false};
  std::atomic<bool> release_older{false}, release_block{false};
  std::atomic<bool> older_committed{false};
  std::atomic<int> arrivals{0};
  std::atomic<int> finishes{0};
  SyncPoint::GetInstance()->SetCallBack(
      "FlushJob::WriteLevel0Table:output_compression", [&](void*) {
        const int n = arrivals.fetch_add(1);
        if (n == 0) {  // the older immutable's flush
          older_parked.store(true);
          while (!release_older.load()) {
            Env::Default()->SleepForMicroseconds(1000);
          }
        } else if (n == 1) {  // the block-only flush, already picked
          block_parked.store(true);
          while (!release_block.load()) {
            Env::Default()->SleepForMicroseconds(1000);
          }
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::FlushMemTableToOutputFile:Finish", [&](void*) {
        if (finishes.fetch_add(1) == 0) {
          older_committed.store(true);
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  {
    FlushOptions fo;
    fo.wait = false;
    ASSERT_OK(db_->Flush(fo));
  }
  for (int i = 0; i < 20000 && !older_parked.load(); ++i) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_TRUE(older_parked.load());
  const uint64_t imm_next_log = dbfull()->TEST_LogfileNumber();
  ASSERT_GT(imm_next_log, log_at_install);

  const SequenceNumber g = db_->GetLatestSequenceNumber() + 1;
  std::atomic<int> released{0};
  ExternalMemTableBlock blk =
      MakeBlock(SequentialEntries(10, g, /*width_one=*/true), &released);
  ASSERT_OK(db_->InstallExternalMemTable(db_->DefaultColumnFamily(),
                                         std::move(blk)));
  // The second flush slot picks the block alone and parks.
  for (int i = 0; i < 20000 && !block_parked.load(); ++i) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_TRUE(block_parked.load());
  ASSERT_EQ(cfd->GetLogNumber(), log_at_install);  // nothing committed yet
  // Older flush commits first ...
  release_older.store(true);
  for (int i = 0; i < 20000 && !older_committed.load(); ++i) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_TRUE(older_committed.load());
  // ... then the block-only flush installs its result.
  release_block.store(true);
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(arrivals.load(), 2);
  ASSERT_EQ(finishes.load(), 2);
  ASSERT_EQ(NumTableFilesAtLevel(0), 3);
  ASSERT_GE(cfd->GetLogNumber(), imm_next_log);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get(Key(5)), Val(5));

  Reopen(options);
  ASSERT_EQ(Get("a"), "1");
  ASSERT_EQ(Get("b"), "2");
  ASSERT_EQ(Get(Key(0)), Val(0));
  ASSERT_EQ(Get(Key(9)), Val(9));
  std::string log_contents;
  ASSERT_OK(ReadFileToString(env_, dbname_ + "/LOG", &log_contents));
  ASSERT_EQ(log_contents.find("NOT monotonically increasing"),
            std::string::npos);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#else
int main(int, char**) {
  fprintf(stderr, "SKIPPED: external_memtable_test needs a debug build (TEST_* hooks)\n");
  return 0;
}
#endif  // NDEBUG
