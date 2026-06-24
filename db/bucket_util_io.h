//  [BucketLSM / relink] Dispatch BucketOf that knows about ImmutableCFOptions.
//
//  Kept in a SEPARATE header from db/bucket_util.h to break the include cycle:
//  options/cf_options.h includes db/bucket_util.h (for the BucketBoundaryPublisher
//  member type), so the dispatch helper — which needs the full ImmutableCFOptions
//  — cannot live in bucket_util.h. Call sites that have an ImmutableCFOptions /
//  ImmutableOptions in scope include THIS header instead.
//
//  Default (no SetBucketBoundaries ever published) => l0_bucket_boundaries is null
//  (or holds an empty/seed vector) => uniform key_space/count mapping => behavior
//  is BIT-IDENTICAL to Phase 0-4. The dynamic boundary path only diverges after the
//  Phase 7 BucketManager publishes an explicit (possibly non-uniform) list.
#pragma once

#include "db/bucket_util.h"
#include "options/cf_options.h"

namespace ROCKSDB_NAMESPACE {

inline uint64_t BucketOfCF(const Slice& user_key,
                           const ImmutableCFOptions& iopt) {
  if (iopt.l0_bucket_boundaries) {
    std::shared_ptr<const BucketBoundaries> bnd = iopt.l0_bucket_boundaries->Get();
    if (bnd && !bnd->empty()) {
      return BucketOf(user_key, *bnd);
    }
  }
  return BucketOf(user_key, iopt.l0_bucket_key_space, iopt.l0_bucket_count);
}

}  // namespace ROCKSDB_NAMESPACE
