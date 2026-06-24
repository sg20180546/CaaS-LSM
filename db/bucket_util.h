//  [BucketLSM / relink] Shared L0-bucket helper.
//
//  When ColumnFamilyOptions::l0_bucket_count > 1 (set on RELINK / G5 shards ONLY), L0 is partitioned
//  into `n` non-overlapping buckets by the user key. BucketOf() is the single source of truth used by
//  BucketFlush (db/builder.cc), per-bucket write-stall / compaction trigger (db/column_family.cc,
//  db/version_set.cc) and bucket-aware scan (db/version_set.cc). It mirrors
//  KeyGroupAlignedPartitioner::GroupOf (db/compaction/sst_partitioner.cc) so L0 buckets and the L1+
//  key-group-aligned SSTs use the SAME key->bucket mapping.
//
//  Off-path (l0_bucket_count <= 1) callers must NOT call this; gate every BucketLSM site behind the
//  option so baselines stay bit-identical (isolation invariant).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Big-endian fixed-width (first 8 bytes) unsigned integer value of a user key.
inline uint64_t KeyValue8(const Slice& user_key) {
  uint64_t v = 0;
  size_t n = user_key.size() < 8 ? user_key.size() : 8;
  for (size_t i = 0; i < n; i++) {
    v = (v << 8) | static_cast<uint8_t>(user_key.data()[i]);
  }
  return v;
}

// Bucket id of a user key, given the key space and bucket count. Keys are big-endian fixed-width
// unsigned integers (first 8 bytes). Returns 0 when bucketing is off/degenerate.
inline uint64_t BucketOf(const Slice& user_key, uint64_t key_space,
                         uint64_t bucket_count) {
  if (key_space == 0 || bucket_count <= 1) return 0;
  uint64_t v = KeyValue8(user_key);
  uint64_t b = static_cast<uint64_t>((unsigned __int128)v * bucket_count / key_space);
  return b >= bucket_count ? bucket_count - 1 : b;
}

// [BucketLSM Phase 7 — dynamic buckets] Explicit (possibly non-uniform) bucket
// boundaries. Entry i = the big-endian-8 value of the FIRST key of bucket i+1
// (exclusive upper bound between bucket i and i+1). Sorted strictly increasing;
// size == (live_bucket_count - 1). Empty => single degenerate bucket 0.
using BucketBoundaries = std::vector<uint64_t>;

// Bucket id of a user key against an explicit boundary list. id = #boundaries <=
// key = upper_bound index. Seeded with UniformBucketBoundaries() this is
// byte-identical to the uniform 3-arg BucketOf above (equivalence invariant).
inline uint64_t BucketOf(const Slice& user_key, const BucketBoundaries& bnd) {
  if (bnd.empty()) return 0;
  uint64_t v = KeyValue8(user_key);
  return static_cast<uint64_t>(
      std::upper_bound(bnd.begin(), bnd.end(), v) - bnd.begin());
}

// Uniform seed boundaries reproducing BucketOf(key, key_space, count) exactly:
// the transition into bucket i happens at the smallest v with
// floor(v*count/key_space) >= i, i.e. v >= ceil(i*key_space/count).
inline BucketBoundaries UniformBucketBoundaries(uint64_t key_space,
                                                uint64_t count) {
  BucketBoundaries b;
  if (key_space == 0 || count <= 1) return b;
  b.reserve(count - 1);
  for (uint64_t i = 1; i < count; i++) {
    b.push_back(static_cast<uint64_t>(
        ((unsigned __int128)i * key_space + (count - 1)) / count));
  }
  return b;
}

// RCU-style holder for the live boundary snapshot. Writers Set() an immutable
// vector; readers Get() a snapshot. Lives in ImmutableCFOptions (shared across
// value-copies), so a publish via any handle is visible at every BucketOf site.
// C++17: std::atomic_*_explicit FREE functions on a plain shared_ptr (the
// codebase already does this in db/memtable.cc); std::atomic<shared_ptr> is C++20.
// Allocated only when l0_bucket_count>1 (nullptr off-path => fully inert).
class BucketBoundaryPublisher {
 public:
  std::shared_ptr<const BucketBoundaries> Get() const {
    return std::atomic_load_explicit(&b_, std::memory_order_acquire);
  }
  void Set(std::shared_ptr<const BucketBoundaries> v) {
    std::atomic_store_explicit(&b_, std::move(v), std::memory_order_release);
  }

 private:
  std::shared_ptr<const BucketBoundaries> b_;
};

}  // namespace ROCKSDB_NAMESPACE
