// [ft 2026-11-04] PrefixRoutingFileSystem -- every path under `prefix` (env ROCKSDB_LOCAL_PREFIX, an absolute local
// directory such as /nvme/ftwal) is served by `local` (the Posix FileSystem); every other path by the wrapped target
// (HdfsFileSystem). Installed by PosixEnv's constructor under -DHDFS when the variable is set, so a DB whose data lives
// on HDFS can keep DBOptions::wal_dir on the compute node's NVMe (the failure-tolerance cells' write-ahead log).
// Unset variable = the HdfsFileSystem is installed bare, byte-identical to before.
//
// Routing rules: one path -> by prefix; two paths (rename/link/same) -> by the first (both must sit on one side);
// FileLock objects remember their side so UnlockFile goes back to the FS that made them. The wrapper is transparent to
// Customizable::CheckedCast (FileSystemWrapper::Inner), so the Storage-CP hooks still find the HdfsFileSystem.
#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rocksdb/file_system.h"

namespace ROCKSDB_NAMESPACE {

class PrefixRoutingFileSystem : public FileSystemWrapper {
 public:
  PrefixRoutingFileSystem(const std::shared_ptr<FileSystem>& target,
                          const std::shared_ptr<FileSystem>& local, std::string prefix)
      : FileSystemWrapper(target), local_(local), prefix_(std::move(prefix)) {}
  static const char* kClassName() { return "PrefixRoutingFileSystem"; }
  const char* Name() const override { return kClassName(); }

  bool IsLocal(const std::string& p) const {
    return !prefix_.empty() && p.size() >= prefix_.size() && p.compare(0, prefix_.size(), prefix_) == 0 &&
           (p.size() == prefix_.size() || p[prefix_.size()] == '/' || prefix_.back() == '/');
  }
  const std::string& prefix() const { return prefix_; }
  FileSystem* local() const { return local_.get(); }

  IOStatus NewSequentialFile(const std::string& f, const FileOptions& o, std::unique_ptr<FSSequentialFile>* r,
                             IODebugContext* d) override {
    return IsLocal(f) ? local_->NewSequentialFile(f, o, r, d) : target_->NewSequentialFile(f, o, r, d);
  }
  IOStatus NewRandomAccessFile(const std::string& f, const FileOptions& o, std::unique_ptr<FSRandomAccessFile>* r,
                               IODebugContext* d) override {
    return IsLocal(f) ? local_->NewRandomAccessFile(f, o, r, d) : target_->NewRandomAccessFile(f, o, r, d);
  }
  IOStatus NewWritableFile(const std::string& f, const FileOptions& o, std::unique_ptr<FSWritableFile>* r,
                           IODebugContext* d) override {
    return IsLocal(f) ? local_->NewWritableFile(f, o, r, d) : target_->NewWritableFile(f, o, r, d);
  }
  IOStatus ReopenWritableFile(const std::string& f, const FileOptions& o, std::unique_ptr<FSWritableFile>* r,
                              IODebugContext* d) override {
    return IsLocal(f) ? local_->ReopenWritableFile(f, o, r, d) : target_->ReopenWritableFile(f, o, r, d);
  }
  IOStatus ReuseWritableFile(const std::string& f, const std::string& old, const FileOptions& o,
                             std::unique_ptr<FSWritableFile>* r, IODebugContext* d) override {
    return IsLocal(f) ? local_->ReuseWritableFile(f, old, o, r, d) : target_->ReuseWritableFile(f, old, o, r, d);
  }
  IOStatus NewRandomRWFile(const std::string& f, const FileOptions& o, std::unique_ptr<FSRandomRWFile>* r,
                           IODebugContext* d) override {
    return IsLocal(f) ? local_->NewRandomRWFile(f, o, r, d) : target_->NewRandomRWFile(f, o, r, d);
  }
  IOStatus NewMemoryMappedFileBuffer(const std::string& f, std::unique_ptr<MemoryMappedFileBuffer>* r) override {
    return IsLocal(f) ? local_->NewMemoryMappedFileBuffer(f, r) : target_->NewMemoryMappedFileBuffer(f, r);
  }
  IOStatus NewDirectory(const std::string& n, const IOOptions& o, std::unique_ptr<FSDirectory>* r,
                        IODebugContext* d) override {
    return IsLocal(n) ? local_->NewDirectory(n, o, r, d) : target_->NewDirectory(n, o, r, d);
  }
  IOStatus FileExists(const std::string& f, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(f) ? local_->FileExists(f, o, d) : target_->FileExists(f, o, d);
  }
  IOStatus GetChildren(const std::string& dir, const IOOptions& o, std::vector<std::string>* r,
                       IODebugContext* d) override {
    return IsLocal(dir) ? local_->GetChildren(dir, o, r, d) : target_->GetChildren(dir, o, r, d);
  }
  IOStatus GetChildrenFileAttributes(const std::string& dir, const IOOptions& o, std::vector<FileAttributes>* r,
                                     IODebugContext* d) override {
    return IsLocal(dir) ? local_->GetChildrenFileAttributes(dir, o, r, d)
                        : target_->GetChildrenFileAttributes(dir, o, r, d);
  }
  IOStatus DeleteFile(const std::string& f, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(f) ? local_->DeleteFile(f, o, d) : target_->DeleteFile(f, o, d);
  }
  IOStatus Truncate(const std::string& f, size_t size, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(f) ? local_->Truncate(f, size, o, d) : target_->Truncate(f, size, o, d);
  }
  IOStatus CreateDir(const std::string& n, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(n) ? local_->CreateDir(n, o, d) : target_->CreateDir(n, o, d);
  }
  IOStatus CreateDirIfMissing(const std::string& n, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(n) ? local_->CreateDirIfMissing(n, o, d) : target_->CreateDirIfMissing(n, o, d);
  }
  IOStatus DeleteDir(const std::string& n, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(n) ? local_->DeleteDir(n, o, d) : target_->DeleteDir(n, o, d);
  }
  IOStatus GetFileSize(const std::string& f, const IOOptions& o, uint64_t* s, IODebugContext* d) override {
    return IsLocal(f) ? local_->GetFileSize(f, o, s, d) : target_->GetFileSize(f, o, s, d);
  }
  IOStatus GetFileModificationTime(const std::string& f, const IOOptions& o, uint64_t* t,
                                   IODebugContext* d) override {
    return IsLocal(f) ? local_->GetFileModificationTime(f, o, t, d) : target_->GetFileModificationTime(f, o, t, d);
  }
  IOStatus RenameFile(const std::string& s, const std::string& t, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(s) ? local_->RenameFile(s, t, o, d) : target_->RenameFile(s, t, o, d);
  }
  IOStatus LinkFile(const std::string& s, const std::string& t, const IOOptions& o, IODebugContext* d) override {
    return IsLocal(s) ? local_->LinkFile(s, t, o, d) : target_->LinkFile(s, t, o, d);
  }
  IOStatus NumFileLinks(const std::string& f, const IOOptions& o, uint64_t* c, IODebugContext* d) override {
    return IsLocal(f) ? local_->NumFileLinks(f, o, c, d) : target_->NumFileLinks(f, o, c, d);
  }
  IOStatus AreFilesSame(const std::string& a, const std::string& b, const IOOptions& o, bool* r,
                        IODebugContext* d) override {
    return IsLocal(a) ? local_->AreFilesSame(a, b, o, r, d) : target_->AreFilesSame(a, b, o, r, d);
  }
  IOStatus LockFile(const std::string& f, const IOOptions& o, FileLock** l, IODebugContext* d) override {
    if (!IsLocal(f)) return target_->LockFile(f, o, l, d);
    IOStatus s = local_->LockFile(f, o, l, d);
    if (s.ok() && l && *l) {
      std::lock_guard<std::mutex> g(mu_);
      local_locks_.insert(*l);
    }
    return s;
  }
  IOStatus UnlockFile(FileLock* l, const IOOptions& o, IODebugContext* d) override {
    bool is_local = false;
    {
      std::lock_guard<std::mutex> g(mu_);
      is_local = local_locks_.erase(l) > 0;
    }
    return is_local ? local_->UnlockFile(l, o, d) : target_->UnlockFile(l, o, d);
  }
  IOStatus NewLogger(const std::string& f, const IOOptions& o, std::shared_ptr<Logger>* r,
                     IODebugContext* d) override {
    return IsLocal(f) ? local_->NewLogger(f, o, r, d) : target_->NewLogger(f, o, r, d);
  }
  IOStatus GetAbsolutePath(const std::string& p, const IOOptions& o, std::string* out, IODebugContext* d) override {
    return IsLocal(p) ? local_->GetAbsolutePath(p, o, out, d) : target_->GetAbsolutePath(p, o, out, d);
  }
  IOStatus GetFreeSpace(const std::string& p, const IOOptions& o, uint64_t* f, IODebugContext* d) override {
    return IsLocal(p) ? local_->GetFreeSpace(p, o, f, d) : target_->GetFreeSpace(p, o, f, d);
  }
  IOStatus IsDirectory(const std::string& p, const IOOptions& o, bool* r, IODebugContext* d) override {
    return IsLocal(p) ? local_->IsDirectory(p, o, r, d) : target_->IsDirectory(p, o, r, d);
  }

 private:
  std::shared_ptr<FileSystem> local_;
  std::string prefix_;
  std::mutex mu_;
  std::set<FileLock*> local_locks_;
};

}  // namespace ROCKSDB_NAMESPACE
