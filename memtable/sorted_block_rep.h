//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] Read-only MemTableRep over a caller-owned,
// already-sorted block of memtable entries (ExternalMemTableBlock). Lookups
// are binary searches over the offset array with the memtable's
// KeyComparator; iteration walks the offset array. Nothing is copied: the rep
// points at block.data / block.offsets and calls block.release exactly once
// from its destructor (i.e. when the owning MemTable is destroyed).
//
// The rep is only ever installed as an IMMUTABLE memtable, so every insert
// entry point is unreachable; they assert in debug builds and are a logged
// no-op (returning false where a bool is expected) in release builds.

#pragma once

#include <atomic>
#include <cstdint>

#include "rocksdb/external_memtable.h"
#include "rocksdb/memtablerep.h"

namespace ROCKSDB_NAMESPACE {

class Arena;

class SortedBlockMemTableRep : public MemTableRep {
 public:
  // `compare` must outlive the rep (it is MemTable::comparator_). The block
  // is taken by value-move; its pointers are used in place.
  SortedBlockMemTableRep(const MemTableRep::KeyComparator& compare,
                         ExternalMemTableBlock&& block);
  // Runs block.release() exactly once (if set).
  ~SortedBlockMemTableRep() override;

  // ---- write side: unreachable (installed immutable) -------------------
  KeyHandle Allocate(const size_t len, char** buf) override;
  void Insert(KeyHandle handle) override;
  bool InsertKey(KeyHandle handle) override;
  void InsertWithHint(KeyHandle handle, void** hint) override;
  bool InsertKeyWithHint(KeyHandle handle, void** hint) override;
  void InsertWithHintConcurrently(KeyHandle handle, void** hint) override;
  bool InsertKeyWithHintConcurrently(KeyHandle handle, void** hint) override;
  void InsertConcurrently(KeyHandle handle) override;
  bool InsertKeyConcurrently(KeyHandle handle) override;
  void MarkReadOnly() override {}
  void MarkFlushed() override {}

  // ---- read side -------------------------------------------------------
  bool Contains(const char* key) const override;
  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;
  uint64_t ApproximateNumEntries(const Slice& start_ikey,
                                 const Slice& end_ikey) override;
  void UniqueRandomSample(const uint64_t num_entries,
                          const uint64_t target_sample_size,
                          std::unordered_set<const char*>* entries) override;
  // data_size + 4 * count (the block bytes plus the offset array); the rep
  // owns no allocator memory.
  size_t ApproximateMemoryUsage() override;

  MemTableRep::Iterator* GetIterator(Arena* arena = nullptr) override;

  // Accessors for the owning MemTable / diagnostics.
  const ExternalMemTableBlock& block() const { return block_; }
  uint64_t count() const { return block_.count; }
  const char* entry(uint64_t i) const { return block_.data + block_.offsets[i]; }

  class Iterator : public MemTableRep::Iterator {
   public:
    explicit Iterator(const SortedBlockMemTableRep* rep)
        : rep_(rep), pos_(0), valid_(false) {}
    ~Iterator() override {}

    bool Valid() const override { return valid_; }
    const char* key() const override { return rep_->entry(pos_); }
    void Next() override;
    void Prev() override;
    // Advance to the first entry with a key >= target.
    void Seek(const Slice& internal_key, const char* memtable_key) override;
    // Retreat to the last entry with a key <= target.
    void SeekForPrev(const Slice& internal_key,
                     const char* memtable_key) override;
    void RandomSeek() override;
    void SeekToFirst() override;
    void SeekToLast() override;

   private:
    const SortedBlockMemTableRep* rep_;
    uint64_t pos_;
    bool valid_;
  };

 private:
  // First index whose entry compares >= key (== count if none).
  uint64_t LowerBound(const char* memtable_key) const;
  uint64_t LowerBound(const Slice& internal_key) const;
  // First index whose entry compares > key (== count if none).
  uint64_t UpperBound(const char* memtable_key) const;
  uint64_t UpperBound(const Slice& internal_key) const;

  void RejectWrite(const char* what) const;

  const MemTableRep::KeyComparator& cmp_;
  ExternalMemTableBlock block_;
  bool released_;
};

}  // namespace ROCKSDB_NAMESPACE
