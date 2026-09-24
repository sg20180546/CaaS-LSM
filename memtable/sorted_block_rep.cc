//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] see sorted_block_rep.h.

#include "memtable/sorted_block_rep.h"

#include <cassert>
#include <cstdio>
#include <random>
#include <utility>

#include "db/dbformat.h"
#include "db/lookup_key.h"
#include "memory/arena.h"

namespace ROCKSDB_NAMESPACE {

SortedBlockMemTableRep::SortedBlockMemTableRep(
    const MemTableRep::KeyComparator& compare, ExternalMemTableBlock&& block)
    : MemTableRep(nullptr /* allocator: never used */),
      cmp_(compare),
      block_(std::move(block)),
      released_(false) {}

SortedBlockMemTableRep::~SortedBlockMemTableRep() {
  // Exactly once: the owning MemTable's unique_ptr<MemTableRep> destroys us
  // inside ~MemTable, which runs after the flush result is in the MANIFEST
  // and the last MemTableListVersion / SuperVersion reference is dropped.
  if (!released_ && block_.release) {
    released_ = true;
    std::function<void()> f = std::move(block_.release);
    block_.release = nullptr;
    f();
  }
}

// ---- write side: unreachable ------------------------------------------

void SortedBlockMemTableRep::RejectWrite(const char* what) const {
  assert(false && "SortedBlockMemTableRep is read-only");
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    // No logger is reachable from a MemTableRep; say it once on stderr.
    fprintf(stderr,
            "[rocksdb] SortedBlockMemTableRep::%s called on a read-only "
            "external memtable block; ignored\n",
            what);
  }
}

KeyHandle SortedBlockMemTableRep::Allocate(const size_t /*len*/, char** buf) {
  RejectWrite("Allocate");
  *buf = nullptr;
  return nullptr;
}
void SortedBlockMemTableRep::Insert(KeyHandle /*handle*/) {
  RejectWrite("Insert");
}
bool SortedBlockMemTableRep::InsertKey(KeyHandle /*handle*/) {
  RejectWrite("InsertKey");
  return false;
}
void SortedBlockMemTableRep::InsertWithHint(KeyHandle /*handle*/,
                                            void** /*hint*/) {
  RejectWrite("InsertWithHint");
}
bool SortedBlockMemTableRep::InsertKeyWithHint(KeyHandle /*handle*/,
                                               void** /*hint*/) {
  RejectWrite("InsertKeyWithHint");
  return false;
}
void SortedBlockMemTableRep::InsertWithHintConcurrently(KeyHandle /*handle*/,
                                                        void** /*hint*/) {
  RejectWrite("InsertWithHintConcurrently");
}
bool SortedBlockMemTableRep::InsertKeyWithHintConcurrently(
    KeyHandle /*handle*/, void** /*hint*/) {
  RejectWrite("InsertKeyWithHintConcurrently");
  return false;
}
void SortedBlockMemTableRep::InsertConcurrently(KeyHandle /*handle*/) {
  RejectWrite("InsertConcurrently");
}
bool SortedBlockMemTableRep::InsertKeyConcurrently(KeyHandle /*handle*/) {
  RejectWrite("InsertKeyConcurrently");
  return false;
}

// ---- binary search helpers ---------------------------------------------
// Both KeyComparator overloads compare a memtable-encoded entry (length
// prefixed internal key) against either another encoded entry or a bare
// internal key, in InternalKeyComparator order (user key asc, seq desc).

uint64_t SortedBlockMemTableRep::LowerBound(const char* memtable_key) const {
  uint64_t lo = 0, hi = block_.count;
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    if (cmp_(entry(mid), memtable_key) < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

uint64_t SortedBlockMemTableRep::LowerBound(const Slice& internal_key) const {
  uint64_t lo = 0, hi = block_.count;
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    if (cmp_(entry(mid), internal_key) < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

uint64_t SortedBlockMemTableRep::UpperBound(const char* memtable_key) const {
  uint64_t lo = 0, hi = block_.count;
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    if (cmp_(entry(mid), memtable_key) <= 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

uint64_t SortedBlockMemTableRep::UpperBound(const Slice& internal_key) const {
  uint64_t lo = 0, hi = block_.count;
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    if (cmp_(entry(mid), internal_key) <= 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---- read side -----------------------------------------------------------

bool SortedBlockMemTableRep::Contains(const char* key) const {
  const uint64_t i = LowerBound(key);
  return i < block_.count && cmp_(entry(i), key) == 0;
}

void SortedBlockMemTableRep::Get(const LookupKey& k, void* callback_args,
                                 bool (*callback_func)(void* arg,
                                                       const char* entry)) {
  // Same protocol as SkipListRep::Get: position at the first entry >= the
  // lookup key (user key, snapshot seq) and hand entries to the callback in
  // order until it returns false (MemTable::SaveValue stops by itself as
  // soon as the user key differs).
  for (uint64_t i = LowerBound(k.memtable_key().data());
       i < block_.count && callback_func(callback_args, entry(i)); ++i) {
  }
}

uint64_t SortedBlockMemTableRep::ApproximateNumEntries(
    const Slice& start_ikey, const Slice& end_ikey) {
  const uint64_t s = LowerBound(start_ikey);
  const uint64_t e = LowerBound(end_ikey);
  return e > s ? e - s : 0;
}

void SortedBlockMemTableRep::UniqueRandomSample(
    const uint64_t /*num_entries*/, const uint64_t target_sample_size,
    std::unordered_set<const char*>* entries) {
  // Cheap uniform sample over the offset array (only mempurge's sampler ever
  // asks; keeps the base-class assert from firing if it is enabled).
  if (block_.count == 0 || target_sample_size == 0) return;
  if (target_sample_size >= block_.count) {
    for (uint64_t i = 0; i < block_.count; ++i) entries->insert(entry(i));
    return;
  }
  std::mt19937_64 rng(static_cast<uint64_t>(block_.count) * 2654435761ull);
  std::uniform_int_distribution<uint64_t> dist(0, block_.count - 1);
  // Bounded number of draws so a hostile size never loops for long.
  const uint64_t max_draws = target_sample_size * 4;
  for (uint64_t d = 0;
       d < max_draws && entries->size() < static_cast<size_t>(target_sample_size);
       ++d) {
    entries->insert(entry(dist(rng)));
  }
}

size_t SortedBlockMemTableRep::ApproximateMemoryUsage() {
  return block_.data_size + static_cast<size_t>(block_.count) * sizeof(uint32_t);
}

MemTableRep::Iterator* SortedBlockMemTableRep::GetIterator(Arena* arena) {
  void* mem = arena ? arena->AllocateAligned(sizeof(Iterator))
                    : operator new(sizeof(Iterator));
  return new (mem) Iterator(this);
}

// ---- Iterator --------------------------------------------------------------

void SortedBlockMemTableRep::Iterator::Next() {
  assert(valid_);
  ++pos_;
  valid_ = pos_ < rep_->count();
}

void SortedBlockMemTableRep::Iterator::Prev() {
  assert(valid_);
  if (pos_ == 0) {
    valid_ = false;
  } else {
    --pos_;
  }
}

void SortedBlockMemTableRep::Iterator::Seek(const Slice& internal_key,
                                            const char* memtable_key) {
  pos_ = memtable_key != nullptr ? rep_->LowerBound(memtable_key)
                                 : rep_->LowerBound(internal_key);
  valid_ = pos_ < rep_->count();
}

void SortedBlockMemTableRep::Iterator::SeekForPrev(const Slice& internal_key,
                                                   const char* memtable_key) {
  const uint64_t ub = memtable_key != nullptr ? rep_->UpperBound(memtable_key)
                                              : rep_->UpperBound(internal_key);
  if (ub == 0) {
    valid_ = false;
  } else {
    pos_ = ub - 1;
    valid_ = true;
  }
}

void SortedBlockMemTableRep::Iterator::RandomSeek() {
  const uint64_t n = rep_->count();
  if (n == 0) {
    valid_ = false;
    return;
  }
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  pos_ = std::uniform_int_distribution<uint64_t>(0, n - 1)(rng);
  valid_ = true;
}

void SortedBlockMemTableRep::Iterator::SeekToFirst() {
  pos_ = 0;
  valid_ = rep_->count() > 0;
}

void SortedBlockMemTableRep::Iterator::SeekToLast() {
  const uint64_t n = rep_->count();
  if (n == 0) {
    valid_ = false;
  } else {
    pos_ = n - 1;
    valid_ = true;
  }
}

}  // namespace ROCKSDB_NAMESPACE
