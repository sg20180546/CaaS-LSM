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
  ASSERT_OK(Flush());
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
  ASSERT_OK(Flush());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WaitForRelease(&released, 1);
  ASSERT_EQ(released.load(), 1);
  ASSERT_EQ(Get(Key(3)), Val(3));
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
