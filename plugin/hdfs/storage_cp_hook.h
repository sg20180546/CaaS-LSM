//  [relink/Storage-CP] Engine-facing hook for the shared-SST refcount.
//
//  Kept deliberately free of hdfs.h / grpc headers so engine code (db_impl.cc)
//  can call it without pulling in libhdfs or the generated protos. The
//  implementation lives in env_hdfs_impl.cc and is compiled only when the hdfs
//  plugin is built (-DHDFS), which is also the only configuration where an
//  external_path relink can exist.
#pragma once

#include <string>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {
class FileSystem;

// Tell the Storage-CP that a SECOND shard now references `path` in place
// (relink / FileDescriptor::external_path), so the file must survive until
// every referencing shard has released it.
//
// Must be called by the shard that ADOPTS the reference, at the moment the
// external_path is installed in its MANIFEST. Before 2026-08-08 this was the
// migration driver's job, but that binary is built without gRPC on the driver
// host, so the call silently compiled away and every relinked file stayed at
// refcount 1 -- the source then deleted files the destination was still
// reading. Doing it here keeps the increment in the same process (and the same
// gRPC client) that installs the reference, so it cannot be configured away.
//
// No-op when the Storage-CP client is disabled (STORAGE_CP_ADDR unset) or when
// `fs` is not backed by the HDFS FileSystem => baseline bit-identical.
void StorageCpNotifyLink(FileSystem* fs, const std::string& path);

}  // namespace ROCKSDB_NAMESPACE
