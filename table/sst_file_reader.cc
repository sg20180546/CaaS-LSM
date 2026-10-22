//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#ifndef ROCKSDB_LITE

#include "rocksdb/sst_file_reader.h"

#include "db/arena_wrapped_db_iter.h"
#include "db/db_iter.h"
#include "db/dbformat.h"
#include "file/random_access_file_reader.h"
#include "options/cf_options.h"
#include "rocksdb/env.h"
#include "rocksdb/file_system.h"
#include "table/get_context.h"
#include "table/table_builder.h"
#include "table/table_reader.h"

namespace ROCKSDB_NAMESPACE {

struct SstFileReader::Rep {
  Options options;
  EnvOptions soptions;
  ImmutableOptions ioptions;
  MutableCFOptions moptions;

  std::unique_ptr<TableReader> table_reader;

  Rep(const Options& opts)
      : options(opts),
        soptions(options),
        ioptions(options),
        moptions(ColumnFamilyOptions(options)) {}
};

SstFileReader::SstFileReader(const Options& options) : rep_(new Rep(options)) {}

SstFileReader::~SstFileReader() {}

Status SstFileReader::Open(const std::string& file_path) {
  auto r = rep_.get();
  Status s;
  uint64_t file_size = 0;
  std::unique_ptr<FSRandomAccessFile> file;
  std::unique_ptr<RandomAccessFileReader> file_reader;
  FileOptions fopts(r->soptions);
  const auto& fs = r->options.env->GetFileSystem();

  s = fs->GetFileSize(file_path, fopts.io_options, &file_size, nullptr);
  if (s.ok()) {
    s = fs->NewRandomAccessFile(file_path, fopts, &file, nullptr);
  }
  if (s.ok()) {
    file_reader.reset(new RandomAccessFileReader(std::move(file), file_path));
  }
  if (s.ok()) {
    TableReaderOptions t_opt(r->ioptions, r->moptions.prefix_extractor,
                             r->soptions, r->ioptions.internal_comparator);
    // Allow open file with global sequence number for backward compatibility.
    t_opt.largest_seqno = kMaxSequenceNumber;
    s = r->options.table_factory->NewTableReader(t_opt, std::move(file_reader),
                                                 file_size, &r->table_reader);
  }
  return s;
}

Iterator* SstFileReader::NewIterator(const ReadOptions& roptions) {
  auto r = rep_.get();
  auto sequence = roptions.snapshot != nullptr
                      ? roptions.snapshot->GetSequenceNumber()
                      : kMaxSequenceNumber;
  ArenaWrappedDBIter* res = new ArenaWrappedDBIter();
  res->Init(r->options.env, roptions, r->ioptions, r->moptions,
            nullptr /* version */, sequence,
            r->moptions.max_sequential_skip_in_iterations,
            0 /* version_number */, nullptr /* read_callback */,
            nullptr /* db_impl */, nullptr /* cfd */,
            true /* expose_blob_index */, false /* allow_refresh */);
  auto internal_iter = r->table_reader->NewIterator(
      res->GetReadOptions(), r->moptions.prefix_extractor.get(),
      res->GetArena(), false /* skip_filters */,
      TableReaderCaller::kSSTFileReader);
  res->SetIterUnderDBIter(internal_iter);
  return res;
}

namespace {
// [migagent 2026-10-21] Public Iterator over a TableReader's InternalIterator (heap-allocated, owned): raw entries,
// internal keys out, user keys in (Seek / SeekForPrev build the internal seek key).
class SstRawIterator : public Iterator {
 public:
  explicit SstRawIterator(InternalIterator* iter) : iter_(iter) {}
  ~SstRawIterator() override { delete iter_; }
  bool Valid() const override { return iter_->Valid(); }
  void SeekToFirst() override { iter_->SeekToFirst(); }
  void SeekToLast() override { iter_->SeekToLast(); }
  void Seek(const Slice& user_key) override {
    InternalKey ik(user_key, kMaxSequenceNumber, kValueTypeForSeek);
    iter_->Seek(ik.Encode());
  }
  void SeekForPrev(const Slice& user_key) override {
    InternalKey ik(user_key, 0, static_cast<ValueType>(0));
    iter_->SeekForPrev(ik.Encode());
  }
  void Next() override { iter_->Next(); }
  void Prev() override { iter_->Prev(); }
  Slice key() const override { return iter_->key(); }
  Slice value() const override { return iter_->value(); }
  Status status() const override { return iter_->status(); }

 private:
  InternalIterator* iter_;
};
}  // namespace

std::unique_ptr<Iterator> SstFileReader::NewRawIterator(const ReadOptions& roptions) {
  auto r = rep_.get();
  InternalIterator* internal_iter = r->table_reader->NewIterator(
      roptions, r->moptions.prefix_extractor.get(), /*arena=*/nullptr,
      false /* skip_filters */, TableReaderCaller::kSSTFileReader);
  if (internal_iter == nullptr) return nullptr;
  return std::unique_ptr<Iterator>(new SstRawIterator(internal_iter));
}

std::shared_ptr<const TableProperties> SstFileReader::GetTableProperties()
    const {
  return rep_->table_reader->GetTableProperties();
}

Status SstFileReader::VerifyChecksum(const ReadOptions& read_options) {
  return rep_->table_reader->VerifyChecksum(read_options,
                                            TableReaderCaller::kSSTFileReader);
}

}  // namespace ROCKSDB_NAMESPACE

#endif  // !ROCKSDB_LITE
