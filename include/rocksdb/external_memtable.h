//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// [external memtable 2026-09-23] Public types for handing a pointer-free,
// sorted KV-block (the unflushed memtable slice of a migrated key range) to a
// destination DB, which installs it AS AN IMMUTABLE MEMTABLE without
// re-inserting a single entry (DB::InstallExternalMemTable).
//
// Block layout: `data` is a run of RocksDB memtable entries in the engine's
// own memtable entry encoding (db/memtable.cc MemTable::Add, with
// memtable_protection_bytes_per_key == 0):
//     varint32(user_key.size() + 8) | user_key | fixed64(seq << 8 | type) |
//     varint32(value.size())        | value
// `offsets[i]` is the byte offset of entry i inside `data`; entries are in
// InternalKeyComparator order (user key ascending, sequence number
// descending) and `count` of them exist. `data_size` must be < 4 GiB so the
// offsets fit in 32 bits. The destination NEVER copies `data`/`offsets`: the
// installed memtable points straight at them and calls `release` exactly once
// when that memtable is destroyed (after its flush completed and the last
// reader let go of it).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

// `ValueType` is defined in the INTERNAL header db/dbformat.h
// (`enum ValueType : unsigned char { kTypeDeletion = 0x0, kTypeValue = 0x1,
// ..., kTypeSingleDeletion = 0x7, ... }`), which callers that only see
// include/ cannot include. This opaque redeclaration (same fixed underlying
// type, hence a complete type per [dcl.enum]) lets the builder keep the
// signature `Add(user_key, seq, ValueType, value)`. The three constants below
// are the only types a block may carry; memtable/external_memtable.cc
// static_asserts them against dbformat.h.
enum ValueType : unsigned char;
constexpr ValueType kExternalMemTableTypeDeletion = static_cast<ValueType>(0x0);
constexpr ValueType kExternalMemTableTypeValue = static_cast<ValueType>(0x1);
constexpr ValueType kExternalMemTableTypeSingleDeletion =
    static_cast<ValueType>(0x7);

struct ExternalMemTableBlock {
  // Entries, RocksDB memtable entry encoding, sorted by InternalKeyComparator
  // (user key asc, seq desc). Not owned by the engine; must stay valid and
  // unmodified until `release` has been called.
  const char* data = nullptr;
  size_t data_size = 0;
  // offsets[i] = byte offset of entry i in `data`; `count` entries.
  // (data_size < 4 GiB.)
  const uint32_t* offsets = nullptr;
  uint64_t count = 0;
  // Inclusive bounds of the sequence numbers inside the block. smallest must
  // be >= 1. NOTE (L0 seqno consistency, see DB::InstallExternalMemTable): a
  // multi-seqno window (smallest != largest) is only accepted when the column
  // family has no unflushed data and smallest > the DB's latest sequence;
  // otherwise use a width-1 window (smallest == largest) -- the "external
  // file" shape RocksDB's L0 ordering check exempts.
  SequenceNumber smallest_seqno = 0, largest_seqno = 0;
  // Called exactly once when the installed memtable is destroyed (after its
  // flush has been committed to the MANIFEST and every reader released it).
  // Never called if InstallExternalMemTable returns a non-OK status. Frees or
  // unpins `data` / `offsets`.
  std::function<void()> release;
  // Whether InstallExternalMemTable should queue a flush of the block right
  // away (default) or leave it in DRAM until the column family's next
  // memtable switch flushes it together with the active memtable. Not part of
  // the fixed E1 field set; added so the driver can A/B the two policies
  // without an engine rebuild.
  bool request_flush = true;
};

// Used by the SOURCE side to build a block from a memtable walk. Entries must
// be added in InternalKeyComparator order: user keys strictly ascending under
// BYTEWISE comparison (the builder cannot see the column family's comparator;
// only use it with the default bytewise comparator), and for a repeated user
// key strictly descending sequence numbers. A violation is remembered and
// reported by Finish() as InvalidArgument (the offending entries are dropped).
// Single use: Finish() moves the buffers out and resets the builder.
class ExternalMemTableBlockBuilder {
 public:
  explicit ExternalMemTableBlockBuilder(size_t reserve_bytes = 0);

  // `type` must be one of kExternalMemTableType{Value,Deletion,
  // SingleDeletion}; `seq` <= 2^56 - 1.
  void Add(const Slice& user_key, SequenceNumber seq, ValueType type,
           const Slice& value);

  // Hands the encoded entries and the offset array to the caller. Returns
  // InvalidArgument if an Add() violated the ordering or type rules, if
  // nothing was added, or if the encoded size reached 4 GiB.
  Status Finish(std::string* data, std::vector<uint32_t>* offsets,
                SequenceNumber* smallest, SequenceNumber* largest,
                uint64_t* count);

  uint64_t count() const { return offsets_.size(); }
  size_t data_size() const { return data_.size(); }
  // [migagent 2026-10-20] User key of the last Add() (empty if none): the
  // resume point of ExportMemTableRange after Status::Incomplete.
  const std::string& last_user_key() const { return last_user_key_; }
  // First ordering / type violation seen by Add() (OK while none); Add() drops
  // entries once this is set, so callers must check it after every Add().
  const Status& status() const { return status_; }

 private:
  std::string data_;
  std::vector<uint32_t> offsets_;
  std::string last_user_key_;
  SequenceNumber last_seq_ = 0;
  bool have_last_ = false;
  // kMaxSequenceNumber lives in db/dbformat.h; ((1 << 56) - 1) is its value.
  SequenceNumber smallest_ = ((0x1ull << 56) - 1);
  SequenceNumber largest_ = 0;
  Status status_;  // first violation, returned by Finish()
};

class DB;
class ColumnFamilyHandle;
class Snapshot;

// [migagent 2026-10-20] Typed export of the unflushed state of a key range.
// Walks the column family's ACTIVE and IMMUTABLE memtables (merged, under a
// referenced SuperVersion; SST files are not read) for user keys in
// [begin, end) and appends one entry per user key to `out` at sequence
// number `block_seq` (a width-1 block, the shape DB::InstallExternalMemTable
// accepts next to older unflushed data). `db` may be a wrapper (StackableDB /
// TransactionDB): the root DB is reached through GetRootDB() and must be the
// engine's DBImpl (InvalidArgument otherwise). The column family must use the
// default bytewise key order (the builder's ordering rule).
//
// Versions with a sequence number above `snapshot` are skipped (nullptr =
// everything); of the surviving versions of a user key only the newest is
// exported, with this type mapping:
//   Value          -> Value, unless an older surviving version of the key is
//                     a SingleDeletion (NotSupported: the destination cannot
//                     reproduce the single-delete pairing);
//   Deletion       -> Deletion;
//   SingleDeletion -> SingleDeletion when it is the key's only version in the
//                     memtables, Deletion when older versions exist beneath;
//   Merge (any version)               -> NotSupported;
//   a range tombstone overlapping [begin, end] (inclusive end, the engine's
//                     overlap check) -> NotSupported;
//   any other type                    -> Corruption.
// `*num_deletes` (may be nullptr) receives the number of Deletion +
// SingleDeletion entries added by this call.
//
// Byte budget: this call adds at most `max_data_bytes` of encoded entries
// (0 = unlimited; the first entry is always added so progress is guaranteed)
// and returns Status::Incomplete() when the next entry would exceed it. The
// builder then holds everything up to and including out->last_user_key();
// continue with begin = out->last_user_key() + std::string(1, '\0') -- with
// a fresh builder if each call is to become one block. A range with
// begin >= end, or one without surviving memtable entries, returns OK with
// nothing added. The caller provides the consistency cut (snapshot) and the
// cutover discipline described at DB::InstallExternalMemTable; the walk
// takes no DB mutex beyond the SuperVersion reference.
Status ExportMemTableRange(DB* db, ColumnFamilyHandle* cf, const Slice& begin,
                           const Slice& end, const Snapshot* snapshot,
                           SequenceNumber block_seq, size_t max_data_bytes,
                           ExternalMemTableBlockBuilder* out,
                           uint64_t* num_deletes);

}  // namespace ROCKSDB_NAMESPACE
