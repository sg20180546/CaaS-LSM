//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#include "rocksdb/sst_partitioner.h"

#include <algorithm>

#include "rocksdb/utilities/customizable_util.h"
#include "rocksdb/utilities/object_registry.h"
#include "rocksdb/utilities/options_type.h"

namespace ROCKSDB_NAMESPACE {
static std::unordered_map<std::string, OptionTypeInfo>
    sst_fixed_prefix_type_info = {
#ifndef ROCKSDB_LITE
        {"length",
         {0, OptionType::kSizeT, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
#endif  // ROCKSDB_LITE
};

SstPartitionerFixedPrefixFactory::SstPartitionerFixedPrefixFactory(size_t len)
    : len_(len) {
  RegisterOptions("Length", &len_, &sst_fixed_prefix_type_info);
}

PartitionerResult SstPartitionerFixedPrefix::ShouldPartition(
    const PartitionerRequest& request) {
  Slice last_key_fixed(*request.prev_user_key);
  if (last_key_fixed.size() > len_) {
    last_key_fixed.size_ = len_;
  }
  Slice current_key_fixed(*request.current_user_key);
  if (current_key_fixed.size() > len_) {
    current_key_fixed.size_ = len_;
  }
  return last_key_fixed.compare(current_key_fixed) != 0 ? kRequired
                                                        : kNotRequired;
}

bool SstPartitionerFixedPrefix::CanDoTrivialMove(
    const Slice& smallest_user_key, const Slice& largest_user_key) {
  return ShouldPartition(PartitionerRequest(smallest_user_key, largest_user_key,
                                            0)) == kNotRequired;
}

std::unique_ptr<SstPartitioner>
SstPartitionerFixedPrefixFactory::CreatePartitioner(
    const SstPartitioner::Context& /* context */) const {
  return std::unique_ptr<SstPartitioner>(new SstPartitionerFixedPrefix(len_));
}

std::shared_ptr<SstPartitionerFactory> NewSstPartitionerFixedPrefixFactory(
    size_t prefix_len) {
  return std::make_shared<SstPartitionerFixedPrefixFactory>(prefix_len);
}

// ---- [relink] Key-Group Aligned partitioner ----
static std::unordered_map<std::string, OptionTypeInfo> kg_aligned_type_info = {
#ifndef ROCKSDB_LITE
    {"key_space",
     {offsetof(struct KeyGroupAlignedPartitionerFactory, key_space_),
      OptionType::kUInt64T, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"num_groups",
     {offsetof(struct KeyGroupAlignedPartitionerFactory, num_groups_),
      OptionType::kUInt64T, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
#endif  // ROCKSDB_LITE
};

KeyGroupAlignedPartitionerFactory::KeyGroupAlignedPartitionerFactory(
    uint64_t key_space, uint64_t num_groups)
    : key_space_(key_space), num_groups_(num_groups) {
  RegisterOptions("Options", this, &kg_aligned_type_info);
}

uint64_t KeyGroupAlignedPartitioner::GroupOf(const Slice& k) const {
  uint64_t v = 0;
  size_t n = k.size() < 8 ? k.size() : 8;  // big-endian fixed-width unsigned key
  for (size_t i = 0; i < n; i++) v = (v << 8) | (uint8_t)k.data()[i];
  return (uint64_t)((unsigned __int128)v * groups_ / ks_);
}

PartitionerResult KeyGroupAlignedPartitioner::ShouldPartition(
    const PartitionerRequest& request) {
  return GroupOf(*request.prev_user_key) == GroupOf(*request.current_user_key)
             ? kNotRequired
             : kRequired;  // cut: keys are in different key-groups
}

bool KeyGroupAlignedPartitioner::CanDoTrivialMove(
    const Slice& smallest_user_key, const Slice& largest_user_key) {
  return GroupOf(smallest_user_key) == GroupOf(largest_user_key);
}

std::unique_ptr<SstPartitioner>
KeyGroupAlignedPartitionerFactory::CreatePartitioner(
    const SstPartitioner::Context& context) const {
  if (context.output_level <= 0) return nullptr;  // L0 excluded (flush/intra-L0 not aligned)
  return std::unique_ptr<SstPartitioner>(
      new KeyGroupAlignedPartitioner(key_space_, num_groups_));
}

std::shared_ptr<SstPartitionerFactory> NewKeyGroupAlignedPartitionerFactory(
    uint64_t key_space, uint64_t num_groups) {
  return std::make_shared<KeyGroupAlignedPartitionerFactory>(key_space,
                                                             num_groups);
}

#ifndef ROCKSDB_LITE
namespace {
static int RegisterSstPartitionerFactories(ObjectLibrary& library,
                                           const std::string& /*arg*/) {
  library.AddFactory<SstPartitionerFactory>(
      SstPartitionerFixedPrefixFactory::kClassName(),
      [](const std::string& /*uri*/,
         std::unique_ptr<SstPartitionerFactory>* guard,
         std::string* /* errmsg */) {
        guard->reset(new SstPartitionerFixedPrefixFactory(0));
        return guard->get();
      });
  library.AddFactory<SstPartitionerFactory>(  // [relink] for remote-compaction CSA reconstruction
      KeyGroupAlignedPartitionerFactory::kClassName(),
      [](const std::string& /*uri*/,
         std::unique_ptr<SstPartitionerFactory>* guard,
         std::string* /* errmsg */) {
        guard->reset(new KeyGroupAlignedPartitionerFactory(0, 1));
        return guard->get();
      });
  return 2;
}
}  // namespace
#endif  // ROCKSDB_LITE

Status SstPartitionerFactory::CreateFromString(
    const ConfigOptions& options, const std::string& value,
    std::shared_ptr<SstPartitionerFactory>* result) {
#ifndef ROCKSDB_LITE
  static std::once_flag once;
  std::call_once(once, [&]() {
    RegisterSstPartitionerFactories(*(ObjectLibrary::Default().get()), "");
  });
#endif  // ROCKSDB_LITE
  return LoadSharedObject<SstPartitionerFactory>(options, value, nullptr,
                                                 result);
}
}  // namespace ROCKSDB_NAMESPACE
