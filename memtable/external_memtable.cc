//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] Source-side builder for an
// ExternalMemTableBlock (see include/rocksdb/external_memtable.h).

#include "rocksdb/external_memtable.h"

#include <cstring>
#include <limits>

#include "db/dbformat.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

// The public constants must be the engine's own type codes: the block is read
// back by the stock memtable / flush code paths without any translation.
static_assert(kExternalMemTableTypeDeletion == kTypeDeletion,
              "kExternalMemTableTypeDeletion must equal kTypeDeletion");
static_assert(kExternalMemTableTypeValue == kTypeValue,
              "kExternalMemTableTypeValue must equal kTypeValue");
static_assert(kExternalMemTableTypeSingleDeletion == kTypeSingleDeletion,
              "kExternalMemTableTypeSingleDeletion must equal "
              "kTypeSingleDeletion");
static_assert(((0x1ull << 56) - 1) == kMaxSequenceNumber,
              "ExternalMemTableBlockBuilder::smallest_ default must be "
              "kMaxSequenceNumber");

namespace {
// Entries must stay addressable by a uint32_t offset.
constexpr uint64_t kMaxBlockBytes = (0x1ull << 32);
}  // namespace

ExternalMemTableBlockBuilder::ExternalMemTableBlockBuilder(
    size_t reserve_bytes) {
  if (reserve_bytes > 0) {
    data_.reserve(reserve_bytes);
  }
}

void ExternalMemTableBlockBuilder::Add(const Slice& user_key,
                                       SequenceNumber seq, ValueType type,
                                       const Slice& value) {
  if (!status_.ok()) {
    return;  // already poisoned; Finish() reports the first violation
  }
  if (type != kTypeValue && type != kTypeDeletion &&
      type != kTypeSingleDeletion) {
    status_ = Status::InvalidArgument(
        "ExternalMemTableBlockBuilder: unsupported value type " +
        std::to_string(static_cast<unsigned>(type)));
    return;
  }
  if (seq > kMaxSequenceNumber) {
    status_ = Status::InvalidArgument(
        "ExternalMemTableBlockBuilder: sequence number out of range");
    return;
  }
  // Ordering: InternalKeyComparator order = user key ascending (bytewise
  // here), sequence number DESCENDING for equal user keys.
  if (have_last_) {
    const int c = user_key.compare(Slice(last_user_key_));
    if (c < 0 || (c == 0 && seq >= last_seq_)) {
      status_ = Status::InvalidArgument(
          "ExternalMemTableBlockBuilder: entries not in InternalKeyComparator "
          "order (user key asc, seq desc) at entry " +
          std::to_string(offsets_.size()));
      return;
    }
  }
  const uint32_t key_size = static_cast<uint32_t>(user_key.size());
  const uint32_t val_size = static_cast<uint32_t>(value.size());
  const uint32_t internal_key_size = key_size + 8;
  const uint64_t encoded_len =
      static_cast<uint64_t>(VarintLength(internal_key_size)) +
      internal_key_size + VarintLength(val_size) + val_size;
  if (static_cast<uint64_t>(data_.size()) + encoded_len >= kMaxBlockBytes) {
    status_ = Status::InvalidArgument(
        "ExternalMemTableBlockBuilder: block would reach 4 GiB");
    return;
  }
  offsets_.push_back(static_cast<uint32_t>(data_.size()));
  // Exactly MemTable::Add's layout with protection_bytes_per_key == 0.
  PutVarint32(&data_, internal_key_size);
  data_.append(user_key.data(), key_size);
  PutFixed64(&data_, PackSequenceAndType(seq, type));
  PutVarint32(&data_, val_size);
  data_.append(value.data(), val_size);

  last_user_key_.assign(user_key.data(), user_key.size());
  last_seq_ = seq;
  have_last_ = true;
  if (seq < smallest_) smallest_ = seq;
  if (seq > largest_) largest_ = seq;
}

Status ExternalMemTableBlockBuilder::Finish(std::string* data,
                                            std::vector<uint32_t>* offsets,
                                            SequenceNumber* smallest,
                                            SequenceNumber* largest,
                                            uint64_t* count) {
  if (!status_.ok()) {
    return status_;
  }
  if (offsets_.empty()) {
    return Status::InvalidArgument(
        "ExternalMemTableBlockBuilder: no entries added");
  }
  if (data == nullptr || offsets == nullptr || smallest == nullptr ||
      largest == nullptr || count == nullptr) {
    return Status::InvalidArgument(
        "ExternalMemTableBlockBuilder::Finish: null output argument");
  }
  *count = offsets_.size();
  *smallest = smallest_;
  *largest = largest_;
  *data = std::move(data_);
  *offsets = std::move(offsets_);
  // Reset for hygiene (single use).
  data_.clear();
  offsets_.clear();
  last_user_key_.clear();
  last_seq_ = 0;
  have_last_ = false;
  smallest_ = kMaxSequenceNumber;
  largest_ = 0;
  return Status::OK();
}

}  // namespace ROCKSDB_NAMESPACE
