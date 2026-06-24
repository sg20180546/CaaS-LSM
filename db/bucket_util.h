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
#include <cstdint>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Bucket id of a user key, given the key space and bucket count. Keys are big-endian fixed-width
// unsigned integers (first 8 bytes). Returns 0 when bucketing is off/degenerate.
inline uint64_t BucketOf(const Slice& user_key, uint64_t key_space,
                         uint64_t bucket_count) {
  if (key_space == 0 || bucket_count <= 1) return 0;
  uint64_t v = 0;
  size_t n = user_key.size() < 8 ? user_key.size() : 8;
  for (size_t i = 0; i < n; i++) {
    v = (v << 8) | static_cast<uint8_t>(user_key.data()[i]);
  }
  uint64_t b = static_cast<uint64_t>((unsigned __int128)v * bucket_count / key_space);
  return b >= bucket_count ? bucket_count - 1 : b;
}

}  // namespace ROCKSDB_NAMESPACE
