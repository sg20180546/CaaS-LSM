//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once
#ifndef ROCKSDB_LITE

#include <unordered_map>
#include <vector>

#include "cache/cache_key.h"
#include "file/random_access_file_reader.h"
#include "file/writable_file_writer.h"
#include "rocksdb/utilities/cache_dump_load.h"
#include "table/block_based/block.h"
#include "table/block_based/block_like_traits.h"
#include "table/block_based/block_type.h"
#include "table/block_based/cachable_entry.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "table/block_based/reader_common.h"

namespace ROCKSDB_NAMESPACE {

// the read buffer size of for the default CacheDumpReader
const unsigned int kDumpReaderBufferSize = 1024;  // 1KB
static const unsigned int kSizePrefixLen = 4;

enum CacheDumpUnitType : unsigned char {
  kHeader = 1,
  kFooter = 2,
  kData = 3,
  kFilter = 4,
  kProperties = 5,
  kCompressionDictionary = 6,
  kRangeDeletion = 7,
  kHashIndexPrefixes = 8,
  kHashIndexMetadata = 9,
  kMetaIndex = 10,
  kIndex = 11,
  kDeprecatedFilterBlock = 12,  // OBSOLETE / DEPRECATED
  kFilterMetaBlock = 13,
  kBlockTypeMax,
};

// The metadata of a dump unit. After it is serilized, its size is fixed 16
// bytes.
struct DumpUnitMeta {
  // sequence number is a monotonically increasing number to indicate the order
  // of the blocks being written. Header is 0.
  uint32_t sequence_num;
  // The Crc32c checksum of its dump unit.
  uint32_t dump_unit_checksum;
  // The dump unit size after the dump unit is serilized to a string.
  uint64_t dump_unit_size;

  void reset() {
    sequence_num = 0;
    dump_unit_checksum = 0;
    dump_unit_size = 0;
  }
};

// The data structure to hold a block and its information.
struct DumpUnit {
  // The timestamp when the block is identified, copied, and dumped from block
  // cache
  uint64_t timestamp;
  // The type of the block
  CacheDumpUnitType type;
  // The key of this block when the block is referenced by this Cache
  Slice key;
  // The block size
  size_t value_len;
  // The Crc32c checksum of the block
  uint32_t value_checksum;
  // Pointer to the block. Note that, in the dump process, it points to a memory
  // buffer copied from cache block. The buffer is freed when we process the
  // next block. In the load process, we use an std::string to store the
  // serialized dump_unit read from the reader. So it points to the memory
  // address of the begin of the block in this string.
  void* value;

  DumpUnit() { reset(); }

  void reset() {
    timestamp = 0;
    type = CacheDumpUnitType::kBlockTypeMax;
    key.clear();
    value_len = 0;
    value_checksum = 0;
    value = nullptr;
  }
};

// The warmup stream intentionally has a wire format distinct from the stock
// cache dump. A receiver that expects this format rejects an old stock stream,
// and every unit carries the version so a truncated/misaligned stream cannot
// silently turn a priority byte into some other field. The fixed DumpUnitMeta
// checksum covers the whole encoded warmup record, including its payload.
static constexpr uint32_t kCacheWarmupDumpMagic = 0x43575550;  // "CWUP"
static constexpr uint32_t kCacheWarmupDumpFormatVersion = 1;

enum class CacheWarmupDumpUnitType : unsigned char {
  kHeader = 1,
  kData = 2,
  kFooter = 3,
};

struct CacheWarmupDumpUnit {
  CacheWarmupDumpUnitType type = CacheWarmupDumpUnitType::kHeader;
  Cache::Priority priority = Cache::Priority::LOW;
  Slice key;
  size_t value_len = 0;
  void* value = nullptr;
};

// The default implementation of the Cache Dumper
class CacheDumperImpl : public CacheDumper {
 public:
  CacheDumperImpl(const CacheDumpOptions& dump_options,
                  const std::shared_ptr<Cache>& cache,
                  std::unique_ptr<CacheDumpWriter>&& writer)
      : options_(dump_options), cache_(cache), writer_(std::move(writer)) {}
  ~CacheDumperImpl() { writer_.reset(); }
  Status SetDumpFilter(std::vector<DB*> db_list) override;
  Status SetDumpFilterFiles(DB* db,
                            const std::vector<std::string>& sst_paths) override;
  Status SetDumpFilterFiles(const TablePropertiesCollection& ptc,
                            const std::vector<std::string>& sst_paths) override;
  Status SetDumpFilterPrefixes(
      const std::vector<std::string>& prefixes) override;
  IOStatus DumpCacheEntriesToWriter() override;
  IOStatus DumpWarmupCacheEntriesToWriter(
      const CacheWarmupOptions& warmup_options,
      CacheWarmupTransferStats* warmup_stats) override;
  const CacheWarmupTransferStats& GetCacheWarmupTransferStats() const override {
    return warmup_stats_;
  }

 private:
  IOStatus WriteBlock(CacheDumpUnitType type, const Slice& key,
                      const Slice& value);
  IOStatus WriteHeader();
  IOStatus WriteFooter();
  IOStatus WriteWarmupUnit(CacheWarmupDumpUnitType type,
                           Cache::Priority priority, const Slice& key,
                           const Slice& value);
  IOStatus WriteWarmupEncodedUnit();
  IOStatus WriteWarmupHeader();
  IOStatus WriteWarmupFooter();
  bool ShouldFilterOut(const Slice& key);
  std::function<void(const Slice&, void*, size_t, Cache::DeleterFn)>
  DumpOneBlockCallBack();

  CacheDumpOptions options_;
  std::shared_ptr<Cache> cache_;
  std::unique_ptr<CacheDumpWriter> writer_;
  UnorderedMap<Cache::DeleterFn, CacheEntryRole> role_map_;
  SystemClock* clock_;
  uint32_t sequence_num_;
  CacheWarmupTransferStats warmup_stats_;
  // The warmup stream's reusable owned staging slab. A data record is encoded
  // here while its source handle is pinned, then the handle is released before
  // CRC/framing/writer I/O. CacheDumpWriter consumes its Slice synchronously.
  std::string warmup_encoded_data_;
  // The cache key prefix filter. Currently, we use db_session_id as the prefix,
  // so using std::set to store the prefixes as filter is enough. Further
  // improvement can be applied like BloomFilter or others to speedup the
  // filtering.
  std::set<std::string> prefix_filter_;
};

// The default implementation of CacheDumpedLoader
class CacheDumpedLoaderImpl : public CacheDumpedLoader {
 public:
  CacheDumpedLoaderImpl(const CacheDumpOptions& dump_options,
                        const BlockBasedTableOptions& toptions,
                        const std::shared_ptr<SecondaryCache>& secondary_cache,
                        std::unique_ptr<CacheDumpReader>&& reader)
      : options_(dump_options),
        toptions_(toptions),
        secondary_cache_(secondary_cache),
        reader_(std::move(reader)) {}
  // [relink cache handoff] primary-cache variant (secondary_cache_ left null)
  CacheDumpedLoaderImpl(const CacheDumpOptions& dump_options,
                        const BlockBasedTableOptions& toptions,
                        const std::shared_ptr<Cache>& primary_cache,
                        std::unique_ptr<CacheDumpReader>&& reader)
      : options_(dump_options),
        toptions_(toptions),
        primary_cache_(primary_cache),
        reader_(std::move(reader)) {}
  ~CacheDumpedLoaderImpl() {}
  IOStatus RestoreCacheEntriesToSecondaryCache() override;
  IOStatus RestoreCacheEntriesToPrimaryCache() override;
  IOStatus RestoreWarmupCacheEntriesToPrimaryCache(
      const CacheWarmupOptions& warmup_options,
      CacheWarmupTransferStats* warmup_stats) override;
  const CacheWarmupTransferStats& GetCacheWarmupTransferStats() const override {
    return warmup_stats_;
  }

 private:
  IOStatus ReadDumpUnitMeta(std::string* data, DumpUnitMeta* unit_meta);
  IOStatus ReadDumpUnit(size_t len, std::string* data, DumpUnit* unit);
  IOStatus ReadHeader(std::string* data, DumpUnit* dump_unit);
  IOStatus ReadCacheBlock(std::string* data, DumpUnit* dump_unit);
  IOStatus ReadWarmupUnit(size_t len, std::string* data,
                          CacheWarmupDumpUnit* unit);
  IOStatus ReadWarmupHeader(std::string* data, CacheWarmupDumpUnit* unit);
  IOStatus ReadWarmupCacheBlock(std::string* data, CacheWarmupDumpUnit* unit);

  CacheDumpOptions options_;
  const BlockBasedTableOptions& toptions_;
  std::shared_ptr<SecondaryCache> secondary_cache_;
  std::shared_ptr<Cache> primary_cache_;  // [relink cache handoff]
  std::unique_ptr<CacheDumpReader> reader_;
  UnorderedMap<Cache::DeleterFn, CacheEntryRole> role_map_;
  CacheWarmupTransferStats warmup_stats_;
};

// The default implementation of CacheDumpWriter. We write the blocks to a file
// sequentially.
class ToFileCacheDumpWriter : public CacheDumpWriter {
 public:
  explicit ToFileCacheDumpWriter(
      std::unique_ptr<WritableFileWriter>&& file_writer)
      : file_writer_(std::move(file_writer)) {}

  ~ToFileCacheDumpWriter() { Close().PermitUncheckedError(); }

  // Write the serialized metadata to the file
  virtual IOStatus WriteMetadata(const Slice& metadata) override {
    assert(file_writer_ != nullptr);
    std::string prefix;
    PutFixed32(&prefix, static_cast<uint32_t>(metadata.size()));
    IOStatus io_s = file_writer_->Append(Slice(prefix));
    if (!io_s.ok()) {
      return io_s;
    }
    io_s = file_writer_->Append(metadata);
    return io_s;
  }

  // Write the serialized data to the file
  virtual IOStatus WritePacket(const Slice& data) override {
    assert(file_writer_ != nullptr);
    std::string prefix;
    PutFixed32(&prefix, static_cast<uint32_t>(data.size()));
    IOStatus io_s = file_writer_->Append(Slice(prefix));
    if (!io_s.ok()) {
      return io_s;
    }
    io_s = file_writer_->Append(data);
    return io_s;
  }

  // Reset the writer
  virtual IOStatus Close() override {
    file_writer_.reset();
    return IOStatus::OK();
  }

 private:
  std::unique_ptr<WritableFileWriter> file_writer_;
};

// The default implementation of CacheDumpReader. It is implemented based on
// RandomAccessFileReader. Note that, we keep an internal variable to remember
// the current offset.
class FromFileCacheDumpReader : public CacheDumpReader {
 public:
  explicit FromFileCacheDumpReader(
      std::unique_ptr<RandomAccessFileReader>&& reader)
      : file_reader_(std::move(reader)),
        offset_(0),
        buffer_(new char[kDumpReaderBufferSize]) {}

  ~FromFileCacheDumpReader() { delete[] buffer_; }

  virtual IOStatus ReadMetadata(std::string* metadata) override {
    uint32_t metadata_len = 0;
    IOStatus io_s = ReadSizePrefix(&metadata_len);
    if (!io_s.ok()) {
      return io_s;
    }
    return Read(metadata_len, metadata);
  }

  virtual IOStatus ReadPacket(std::string* data) override {
    uint32_t data_len = 0;
    IOStatus io_s = ReadSizePrefix(&data_len);
    if (!io_s.ok()) {
      return io_s;
    }
    return Read(data_len, data);
  }

 private:
  IOStatus ReadSizePrefix(uint32_t* len) {
    std::string prefix;
    IOStatus io_s = Read(kSizePrefixLen, &prefix);
    if (!io_s.ok()) {
      return io_s;
    }
    Slice encoded_slice(prefix);
    if (!GetFixed32(&encoded_slice, len)) {
      return IOStatus::Corruption("Decode size prefix string failed");
    }
    return IOStatus::OK();
  }

  IOStatus Read(size_t len, std::string* data) {
    assert(file_reader_ != nullptr);
    IOStatus io_s;

    unsigned int bytes_to_read = static_cast<unsigned int>(len);
    unsigned int to_read = bytes_to_read > kDumpReaderBufferSize
                               ? kDumpReaderBufferSize
                               : bytes_to_read;

    while (to_read > 0) {
      io_s = file_reader_->Read(IOOptions(), offset_, to_read, &result_,
                                buffer_, nullptr,
                                Env::IO_TOTAL /* rate_limiter_priority */);
      if (!io_s.ok()) {
        return io_s;
      }
      if (result_.size() < to_read) {
        return IOStatus::Corruption("Corrupted cache dump file.");
      }
      data->append(result_.data(), result_.size());

      offset_ += to_read;
      bytes_to_read -= to_read;
      to_read = bytes_to_read > kDumpReaderBufferSize ? kDumpReaderBufferSize
                                                      : bytes_to_read;
    }
    return io_s;
  }
  std::unique_ptr<RandomAccessFileReader> file_reader_;
  Slice result_;
  size_t offset_;
  char* buffer_;
};

// The cache dump and load helper class
class CacheDumperHelper {
 public:
  // serialize the dump_unit_meta to a string, it is fixed 16 bytes size.
  static void EncodeDumpUnitMeta(const DumpUnitMeta& meta, std::string* data) {
    assert(data);
    PutFixed32(data, static_cast<uint32_t>(meta.sequence_num));
    PutFixed32(data, static_cast<uint32_t>(meta.dump_unit_checksum));
    PutFixed64(data, meta.dump_unit_size);
  }

  // Serialize the dump_unit to a string.
  static void EncodeDumpUnit(const DumpUnit& dump_unit, std::string* data) {
    assert(data);
    PutFixed64(data, dump_unit.timestamp);
    data->push_back(dump_unit.type);
    PutLengthPrefixedSlice(data, dump_unit.key);
    PutFixed32(data, static_cast<uint32_t>(dump_unit.value_len));
    PutFixed32(data, dump_unit.value_checksum);
    PutLengthPrefixedSlice(data,
                           Slice((char*)dump_unit.value, dump_unit.value_len));
  }

  // Deserialize the dump_unit_meta from a string
  static Status DecodeDumpUnitMeta(const std::string& encoded_data,
                                   DumpUnitMeta* unit_meta) {
    assert(unit_meta != nullptr);
    Slice encoded_slice = Slice(encoded_data);
    if (!GetFixed32(&encoded_slice, &(unit_meta->sequence_num))) {
      return Status::Incomplete("Decode dumped unit meta sequence_num failed");
    }
    if (!GetFixed32(&encoded_slice, &(unit_meta->dump_unit_checksum))) {
      return Status::Incomplete(
          "Decode dumped unit meta dump_unit_checksum failed");
    }
    if (!GetFixed64(&encoded_slice, &(unit_meta->dump_unit_size))) {
      return Status::Incomplete(
          "Decode dumped unit meta dump_unit_size failed");
    }
    return Status::OK();
  }

  // Deserialize the dump_unit from a string.
  static Status DecodeDumpUnit(const std::string& encoded_data,
                               DumpUnit* dump_unit) {
    assert(dump_unit != nullptr);
    Slice encoded_slice = Slice(encoded_data);

    // Decode timestamp
    if (!GetFixed64(&encoded_slice, &dump_unit->timestamp)) {
      return Status::Incomplete("Decode dumped unit string failed");
    }
    // Decode the block type
    dump_unit->type = static_cast<CacheDumpUnitType>(encoded_slice[0]);
    encoded_slice.remove_prefix(1);
    // Decode the key
    if (!GetLengthPrefixedSlice(&encoded_slice, &(dump_unit->key))) {
      return Status::Incomplete("Decode dumped unit string failed");
    }
    // Decode the value size
    uint32_t value_len;
    if (!GetFixed32(&encoded_slice, &value_len)) {
      return Status::Incomplete("Decode dumped unit string failed");
    }
    dump_unit->value_len = static_cast<size_t>(value_len);
    // Decode the value checksum
    if (!GetFixed32(&encoded_slice, &(dump_unit->value_checksum))) {
      return Status::Incomplete("Decode dumped unit string failed");
    }
    // Decode the block content and copy to the memory space whose pointer
    // will be managed by the cache finally.
    Slice block;
    if (!GetLengthPrefixedSlice(&encoded_slice, &block)) {
      return Status::Incomplete("Decode dumped unit string failed");
    }
    dump_unit->value = (void*)block.data();
    assert(block.size() == dump_unit->value_len);
    return Status::OK();
  }

  // The explicit on-wire encoding is deliberately independent of the enum's
  // implementation values. That keeps HIGH/LOW/BOTTOM stable if Cache::Priority
  // is extended or reordered later.
  static Status EncodeWarmupPriority(Cache::Priority priority,
                                     unsigned char* encoded_priority) {
    assert(encoded_priority != nullptr);
    switch (priority) {
      case Cache::Priority::HIGH:
        *encoded_priority = 1;
        return Status::OK();
      case Cache::Priority::LOW:
        *encoded_priority = 2;
        return Status::OK();
      case Cache::Priority::BOTTOM:
        *encoded_priority = 3;
        return Status::OK();
    }
    return Status::InvalidArgument("unknown cache warmup priority");
  }

  static Status DecodeWarmupPriority(unsigned char encoded_priority,
                                     Cache::Priority* priority) {
    assert(priority != nullptr);
    switch (encoded_priority) {
      case 1:
        *priority = Cache::Priority::HIGH;
        return Status::OK();
      case 2:
        *priority = Cache::Priority::LOW;
        return Status::OK();
      case 3:
        *priority = Cache::Priority::BOTTOM;
        return Status::OK();
      default:
        return Status::Corruption("unknown cache warmup priority");
    }
  }

  static Status EncodeWarmupDumpUnit(const CacheWarmupDumpUnit& unit,
                                     std::string* data) {
    assert(data != nullptr);
    if (unit.value_len > UINT32_MAX) {
      return Status::InvalidArgument("cache warmup entry exceeds wire limit");
    }
    unsigned char encoded_priority = 0;
    Status s = EncodeWarmupPriority(unit.priority, &encoded_priority);
    if (!s.ok()) {
      return s;
    }
    PutFixed32(data, kCacheWarmupDumpMagic);
    PutFixed32(data, kCacheWarmupDumpFormatVersion);
    data->push_back(static_cast<char>(unit.type));
    data->push_back(static_cast<char>(encoded_priority));
    PutLengthPrefixedSlice(data, unit.key);
    PutFixed32(data, static_cast<uint32_t>(unit.value_len));
    PutLengthPrefixedSlice(
        data, Slice(static_cast<const char*>(unit.value), unit.value_len));
    return Status::OK();
  }

  static Status DecodeWarmupDumpUnit(const std::string& encoded_data,
                                     CacheWarmupDumpUnit* unit) {
    assert(unit != nullptr);
    Slice encoded_slice(encoded_data);
    uint32_t magic = 0;
    uint32_t version = 0;
    if (!GetFixed32(&encoded_slice, &magic) ||
        !GetFixed32(&encoded_slice, &version)) {
      return Status::Incomplete("cache warmup unit missing format header");
    }
    if (magic != kCacheWarmupDumpMagic) {
      return Status::Corruption("not a cache warmup stream");
    }
    if (version != kCacheWarmupDumpFormatVersion) {
      return Status::NotSupported("unsupported cache warmup format version");
    }
    if (encoded_slice.size() < 2) {
      return Status::Incomplete("cache warmup unit missing type or priority");
    }
    const unsigned char encoded_type =
        static_cast<unsigned char>(encoded_slice[0]);
    const unsigned char encoded_priority =
        static_cast<unsigned char>(encoded_slice[1]);
    encoded_slice.remove_prefix(2);
    switch (encoded_type) {
      case static_cast<unsigned char>(CacheWarmupDumpUnitType::kHeader):
        unit->type = CacheWarmupDumpUnitType::kHeader;
        break;
      case static_cast<unsigned char>(CacheWarmupDumpUnitType::kData):
        unit->type = CacheWarmupDumpUnitType::kData;
        break;
      case static_cast<unsigned char>(CacheWarmupDumpUnitType::kFooter):
        unit->type = CacheWarmupDumpUnitType::kFooter;
        break;
      default:
        return Status::Corruption("unknown cache warmup unit type");
    }
    Status s = DecodeWarmupPriority(encoded_priority, &unit->priority);
    if (!s.ok()) {
      return s;
    }
    if (!GetLengthPrefixedSlice(&encoded_slice, &unit->key)) {
      return Status::Incomplete("cache warmup unit missing key");
    }
    if (unit->type == CacheWarmupDumpUnitType::kData &&
        unit->key.size() != kCacheKeySize) {
      return Status::Corruption(
          "cache warmup data unit has non-standard cache key size");
    }
    uint32_t value_len = 0;
    if (!GetFixed32(&encoded_slice, &value_len)) {
      return Status::Incomplete("cache warmup unit missing value metadata");
    }
    Slice value;
    if (!GetLengthPrefixedSlice(&encoded_slice, &value)) {
      return Status::Incomplete("cache warmup unit missing value");
    }
    if (value.size() != value_len || !encoded_slice.empty()) {
      return Status::Corruption("cache warmup unit has invalid value length");
    }
    unit->value_len = value_len;
    unit->value = const_cast<char*>(value.data());
    return Status::OK();
  }
};

}  // namespace ROCKSDB_NAMESPACE
#endif  // ROCKSDB_LITE
