//  [relink/Storage-CP] Engine-facing hook for the shared-SST refcount.
//
//  Kept deliberately free of hdfs.h / grpc headers so engine code (db_impl.cc)
//  can call it without pulling in libhdfs or the generated protos. The
//  implementation lives in env_hdfs_impl.cc and is compiled only when the hdfs
//  plugin is built (-DHDFS), which is also the only configuration where an
//  external_path relink can exist.
#pragma once

#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
class FileSystem;

// Tell the Storage-CP that a SECOND shard now references `path` in place
// (relink / FileDescriptor::external_path), so the file must survive until
// every referencing shard has released it.
//
// Must be called by the shard that ADOPTS the reference, before the
// external_path is installed in its MANIFEST (ACK required). Before 2026-08-08 this was the
// migration driver's job, but that binary is built without gRPC on the driver
// host, so the call silently compiled away and every relinked file stayed at
// refcount 1 -- the source then deleted files the destination was still
// reading. Doing it here keeps the increment in the same process (and the same
// gRPC client) that installs the reference, so it cannot be configured away.
//
// No-op when the Storage-CP client is disabled (STORAGE_CP_ADDR unset) or when
// `fs` is not backed by the HDFS FileSystem => baseline bit-identical.
Status StorageCpNotifyLink(FileSystem* fs, const std::string& path);

// [batch 2026-09-09] Same claim for every path of ONE register
// (RegisterExternalFilesInPlace): one NotifyLinkBatch RPC instead of N serial
// NotifyLink round trips — the last O(#files) term in relink's stop window.
// Same no-op conditions as StorageCpNotifyLink; empty `paths` is a no-op.
Status StorageCpNotifyLinkBatch(FileSystem* fs,
                              const std::vector<std::string>& paths);

// Only complete, verified FINAL paths are registered. The caller must wait
// for success before LogAndApply, outside the DB mutex. An error may mean the
// server applied the request: AbortOutputs fences retries for a proven abort.
#ifdef HDFS
Status StorageCpPrepareOutputs(FileSystem*, const std::vector<std::string>&);
Status StorageCpAbortOutputs(FileSystem*, const std::vector<std::string>&);
Status StorageCpRecoverReferences(FileSystem*, const std::vector<std::string>&);
// Local claim by the CN that a completed remote job returned a staging output.
Status StorageCpTrackUnpublishedOutput(FileSystem*, const std::string&);
// These two hooks are local only and may run with the DB mutex held.
void StorageCpMarkOutputsPublished(FileSystem*, const std::vector<std::string>&);
void StorageCpPreserveOutputsOnUncertainCommit(
    FileSystem*, const std::vector<std::string>&);
#else
inline Status StorageCpPrepareOutputs(FileSystem*, const std::vector<std::string>&) {
  return Status::OK();
}
inline Status StorageCpAbortOutputs(FileSystem*, const std::vector<std::string>&) {
  return Status::OK();
}
inline Status StorageCpRecoverReferences(FileSystem*, const std::vector<std::string>&) {
  return Status::OK();
}
inline Status StorageCpTrackUnpublishedOutput(FileSystem*, const std::string&) {
  return Status::OK();
}
inline void StorageCpMarkOutputsPublished(FileSystem*, const std::vector<std::string>&) {}
inline void StorageCpPreserveOutputsOnUncertainCommit(
    FileSystem*, const std::vector<std::string>&) {}
#endif

}  // namespace ROCKSDB_NAMESPACE
