//  Copyright (c) 2021-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "hdfs.h"
#include "rocksdb/env.h"
#include "rocksdb/file_system.h"
#include "rocksdb/status.h"

// [relink/Storage-CP] Forward-declare the generated gRPC client stub so this
// header does not need to pull in the generated proto headers. The concrete
// type is only required in env_hdfs_impl.cc.
namespace compactionservice {
class StorageService;
}  // namespace compactionservice

namespace ROCKSDB_NAMESPACE {
class ObjectLibrary;

class HdfsFileSystem : public FileSystemWrapper {
 public:
  static const char* kClassName() { return "HdfsFileSystem"; }
  const char* Name() const override { return kClassName(); }
  static const char* kNickName() { return "hdfs"; }
  static constexpr const char* kProto = "hdfs://";

  const char* NickName() const override { return kNickName(); }
  static Status Create(const std::shared_ptr<FileSystem>& base,
                       const std::string& fsname,
                       std::unique_ptr<FileSystem>* fs);

  explicit HdfsFileSystem(const std::shared_ptr<FileSystem>& base,
                          const std::string& fsname, hdfsFS fileSys);
  ~HdfsFileSystem() override;

  std::string GetId() const override;

  Status ValidateOptions(const DBOptions& db_opts,
                         const ColumnFamilyOptions& cf_opts) const override;

  IOStatus NewSequentialFile(const std::string& /*fname*/,
                             const FileOptions& /*options*/,
                             std::unique_ptr<FSSequentialFile>* /*result*/,
                             IODebugContext* /*dbg*/) override;
  IOStatus NewRandomAccessFile(const std::string& /*fname*/,
                               const FileOptions& /*options*/,
                               std::unique_ptr<FSRandomAccessFile>* /*result*/,
                               IODebugContext* /*dbg*/) override;
  IOStatus NewWritableFile(const std::string& /*fname*/,
                           const FileOptions& /*options*/,
                           std::unique_ptr<FSWritableFile>* /*result*/,
                           IODebugContext* /*dbg*/) override;
  IOStatus NewDirectory(const std::string& /*name*/,
                        const IOOptions& /*options*/,
                        std::unique_ptr<FSDirectory>* /*result*/,
                        IODebugContext* /*dbg*/) override;
  IOStatus FileExists(const std::string& /*fname*/,
                      const IOOptions& /*options*/,
                      IODebugContext* /*dbg*/) override;
  IOStatus GetChildren(const std::string& /*path*/,
                       const IOOptions& /*options*/,
                       std::vector<std::string>* /*result*/,
                       IODebugContext* /*dbg*/) override;
  IOStatus DeleteFile(const std::string& /*fname*/,
                      const IOOptions& /*options*/,
                      IODebugContext* /*dbg*/) override;
  IOStatus CreateDir(const std::string& /*name*/, const IOOptions& /*options*/,
                     IODebugContext* /*dbg*/) override;
  IOStatus CreateDirIfMissing(const std::string& /*name*/,
                              const IOOptions& /*options*/,
                              IODebugContext* /*dbg*/) override;
  IOStatus DeleteDir(const std::string& /*name*/, const IOOptions& /*options*/,
                     IODebugContext* /*dbg*/) override;
  IOStatus GetFileSize(const std::string& /*fname*/,
                       const IOOptions& /*options*/, uint64_t* /*size*/,
                       IODebugContext* /*dbg*/) override;
  IOStatus GetFileModificationTime(const std::string& /*fname*/,
                                   const IOOptions& /*options*/,
                                   uint64_t* /*time*/,
                                   IODebugContext* /*dbg*/) override;
  IOStatus RenameFile(const std::string& /*src*/, const std::string& /*target*/,
                      const IOOptions& /*options*/,
                      IODebugContext* /*dbg*/) override;
  IOStatus LockFile(const std::string& /*fname*/, const IOOptions& /*options*/,
                    FileLock** /*lock*/, IODebugContext* /*dbg*/) override;
  IOStatus UnlockFile(FileLock* /*lock*/, const IOOptions& /*options*/,
                      IODebugContext* /*dbg*/) override;
  IOStatus NewLogger(const std::string& /*fname*/, const IOOptions& /*options*/,
                     std::shared_ptr<Logger>* /*result*/,
                     IODebugContext* /*dbg*/) override;
  IOStatus IsDirectory(const std::string& /*path*/,
                       const IOOptions& /*options*/, bool* /*is_dir*/,
                       IODebugContext* /*dbg*/) override;

 private:
  std::string fsname_;  // string of the form "hdfs://hostname:port/dira"
  hdfsFS fileSys_;      // a single hdfsFS object for all files

  // [relink/Storage-CP] Gated gRPC StorageService client for per-file refcount
  // + GC over shared SSTs on HDFS. DISABLED unless getenv("STORAGE_CP_ADDR") is
  // set & non-empty. When disabled, every hook below is a no-op and HDFS
  // behavior is BIT-IDENTICAL to baseline.
  //
  // The client (gRPC channel + generated Stub) is held behind an opaque,
  // lazily-constructed struct defined in env_hdfs_impl.cc so this header does
  // not need to include the generated proto/grpc headers.
  struct StorageCpClient;
  // Initialized once (lazily) on first hook invocation.
  mutable std::once_flag storage_cp_init_flag_;
  // Owning pointer to the opaque client; nullptr => DISABLED (default).
  mutable std::unique_ptr<StorageCpClient> storage_cp_client_;
  // Shard id from getenv("STORAGE_CP_SHARD") (default 0); cached at init.
  mutable uint32_t storage_cp_shard_ = 0;

  // Returns the opaque client iff Storage-CP is enabled, else nullptr.
  // Performs the one-time lazy init (reads STORAGE_CP_ADDR / STORAGE_CP_SHARD,
  // builds the channel + stub). Thread-safe.
  StorageCpClient* GetStorageCpClient() const;
  // Helper: true iff fname names a real SST (ends in ".sst").
  static bool IsSstFile(const std::string& fname);
};

// Returns a `FileSystem` that hashes file contents when naming files, thus
// deduping them. RocksDB however expects files to be identified based on a
// monotonically increasing counter, so a mapping of RocksDB's name to content
// hash is needed. This mapping is stored in a separate RocksDB instance.
Status NewHdfsEnv(const std::string& fsname, std::unique_ptr<Env>* env);
Status NewHdfsFileSystem(const std::string& fsname,
                         std::shared_ptr<FileSystem>* fs);
extern "C" {
int register_HdfsObjects(ROCKSDB_NAMESPACE::ObjectLibrary& library,
                         const std::string&);
void hdfs_reg();
}  // extern "C"
}  // namespace ROCKSDB_NAMESPACE
