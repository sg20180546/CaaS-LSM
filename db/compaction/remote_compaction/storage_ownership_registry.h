// Durable CP ownership state. Immutable SST paths are never resurrected after
// release/abort. All mutations are journaled and fsynced before their ACK.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ownership_cp {

enum class ErrorCode { InvalidArgument, FailedPrecondition, IOError, Corruption, Busy };
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
  ErrorCode code() const { return code_; }
 private:
  ErrorCode code_;
};

enum class State : uint8_t { Active = 0, Pending = 1, Deleting = 2, Deleted = 3 };
struct EntryView {
  bool known = false;
  bool born = false;
  bool birth_live = false;
  bool aborted = false;
  uint32_t creator = 0;
  uint64_t count = 0;
  uint64_t claim_token = 0;
  State state = State::Active;
  std::map<uint32_t, uint64_t> linked_refs;
  std::set<uint32_t> released_by;
  std::map<uint32_t, uint64_t> last_release_sequence;
};
struct ReleaseResult {
  uint64_t count = 0;
  bool unknown = false;
  bool duplicate = false;
};
struct DeleteClaim { std::string path; uint64_t token = 0; };
struct Stats {
  size_t tracked = 0, shared = 0, pending = 0, deleting = 0, deleted = 0;
  uint64_t unknown_releases = 0, journal_sequence = 0;
  bool poisoned = false;
};

class Registry {
 public:
  explicit Registry(const std::string& directory) {
    try {
      if (directory.empty() || directory.front() != '/') {
        throw Error(ErrorCode::InvalidArgument, "ownership state directory must be explicit and absolute");
      }
      EnsureDirectory(directory);
      int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECTORY
      flags |= O_DIRECTORY;
#endif
#ifdef O_NOFOLLOW
      flags |= O_NOFOLLOW;
#endif
      dir_fd_ = ::open(directory.c_str(), flags);
      if (dir_fd_ < 0) IoError("open ownership state directory");
      flags = O_RDWR | O_CREAT | O_CLOEXEC;
#ifdef O_NOFOLLOW
      flags |= O_NOFOLLOW;
#endif
      lock_fd_ = ::openat(dir_fd_, "ownership.lock", flags, 0600);
      if (lock_fd_ < 0) IoError("open ownership lock");
      if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        throw Error(ErrorCode::Busy, "ownership state directory already has a writer: " + directory);
      }
      flags = O_RDWR | O_APPEND | O_CLOEXEC;
#ifdef O_NOFOLLOW
      flags |= O_NOFOLLOW;
#endif
      journal_fd_ = ::openat(dir_fd_, "ownership.journal", flags);
      if (journal_fd_ < 0 && errno == ENOENT) {
        journal_fd_ = ::openat(dir_fd_, "ownership.journal", flags | O_CREAT | O_EXCL, 0600);
        if (journal_fd_ < 0) IoError("create ownership journal");
        WriteAll(journal_fd_, Magic(), "initialize ownership journal");
        if (::fsync(journal_fd_) != 0 || ::fsync(dir_fd_) != 0) IoError("sync ownership journal creation");
      } else if (journal_fd_ < 0) {
        IoError("open ownership journal");
      }
      struct stat statbuf;
      if (::fstat(journal_fd_, &statbuf) != 0) IoError("stat ownership journal");
      if (!S_ISREG(statbuf.st_mode)) {
        throw Error(ErrorCode::Corruption, "ownership journal is not a regular file");
      }
      Replay(statbuf.st_size);
      // Claims interrupted by a CP crash remain fenced and are retryable once.
      for (const auto& item : entries_) {
        if (item.second.state == State::Deleting) recovered_claims_.insert(item.first);
      }
    } catch (...) {
      Close();
      throw;
    }
  }
  ~Registry() { Close(); }
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  void Create(std::vector<std::string> paths, uint32_t owner, const std::string& operation_id = "") {
    std::lock_guard<std::mutex> lock(mu_);
    Record record = MakeRecord(Kind::Create, std::move(paths), owner, operation_id);
    Commit(record);
  }
  void Link(std::vector<std::string> paths, uint32_t owner, const std::string& operation_id = "") {
    std::lock_guard<std::mutex> lock(mu_);
    Record record = MakeRecord(Kind::Link, std::move(paths), owner, operation_id);
    Commit(record);
  }
  ReleaseResult Release(const std::string& path, uint32_t owner) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureHealthy();
    ValidatePath(path);
    auto found = entries_.find(path);
    if (found == entries_.end()) {
      Record record = MakeRecord(Kind::Release, {path}, owner, "");
      Commit(record);  // Persist the fail-safe diagnostic; never queue unknown releases.
      return {0, true, false};
    }
    const EntryView& entry = found->second;
    if (entry.state != State::Active || entry.released_by.count(owner)) {
      return {entry.count, false, true};
    }
    Record record = MakeRecord(Kind::Release, {path}, owner, "");
    Commit(record);
    return {entries_.at(path).count, false, false};
  }
  void AbortUnpublished(std::vector<std::string> paths, uint32_t owner,
                        const std::string& operation_id = "") {
    std::lock_guard<std::mutex> lock(mu_);
    Record record = MakeRecord(Kind::Abort, std::move(paths), owner, operation_id);
    Commit(record);
  }
  void RecoverReferences(std::vector<std::string> paths, uint32_t owner) const {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureHealthy();
    Normalize(paths, true);
    for (const auto& path : paths) {
      const auto found = entries_.find(path);
      if (found == entries_.end() || found->second.state != State::Active || found->second.aborted) {
        throw Error(ErrorCode::FailedPrecondition, "live MANIFEST path has no active durable ownership: " + path);
      }
      const auto& entry = found->second;
      const auto linked = entry.linked_refs.find(owner);
      if (entry.released_by.count(owner) ||
          (!(entry.birth_live && entry.creator == owner) &&
           !(linked != entry.linked_refs.end() && linked->second > 0))) {
        throw Error(ErrorCode::FailedPrecondition, "live MANIFEST owner is missing from durable ownership: " + path);
      }
    }
  }
  std::vector<DeleteClaim> ClaimDeletes(size_t maximum = 4096) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureHealthy();
    std::vector<std::string> paths;
    for (const auto& path : pending_paths_) {
      if (paths.size() == maximum) break;
      paths.push_back(path);
    }
    for (const auto& path : recovered_claims_) {
      if (paths.size() == maximum) break;
      paths.push_back(path);
    }
    if (paths.empty()) return {};
    Record record = MakeRecord(Kind::Claim, std::move(paths), 0, "");
    record.token = sequence_ + 1;
    Commit(record);  // The immutable tombstone/claim is durable BEFORE HDFS deletion.
    std::vector<DeleteClaim> result;
    for (const auto& path : record.paths) result.push_back({path, record.token});
    return result;
  }
  void CompleteDelete(const DeleteClaim& claim, bool success) {
    std::lock_guard<std::mutex> lock(mu_);
    Record record = MakeRecord(Kind::Complete, {claim.path}, 0, "");
    record.token = claim.token;
    record.success = success;
    Commit(record);
  }
  EntryView Inspect(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = entries_.find(path);
    return found == entries_.end() ? EntryView{} : found->second;
  }
  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats stats;
    stats.unknown_releases = unknown_releases_;
    stats.journal_sequence = sequence_;
    stats.poisoned = poisoned_;
    for (const auto& item : entries_) {
      const auto& entry = item.second;
      if (entry.state == State::Active) {
        stats.tracked++;
        stats.shared += entry.count > 1;
      } else if (entry.state == State::Pending) stats.pending++;
      else if (entry.state == State::Deleting) stats.deleting++;
      else stats.deleted++;
    }
    return stats;
  }
#ifdef OWNERSHIP_REGISTRY_TESTING
  void FailNextSyncForTest() { std::lock_guard<std::mutex> lock(mu_); fail_next_sync_ = true; }
#endif

 private:
  enum class Kind : uint8_t { Create = 1, Link = 2, Release = 3, Abort = 4, Claim = 5, Complete = 6 };
  struct Record {
    Kind kind = Kind::Create;
    uint64_t sequence = 0, token = 0;
    uint32_t owner = 0;
    bool success = false;
    std::string operation_id;
    std::vector<std::string> paths;
  };
  struct AppliedOperation { std::string fingerprint; uint64_t sequence = 0; };
  static std::string Magic() { return "CP-OWNERSHIP-JOURNAL-v1\n"; }
  static constexpr uint32_t kMaximumRecord = 64 * 1024 * 1024;
  static constexpr uint32_t kMaximumPaths = 1000000;
  static void IoError(const std::string& action) {
    const int saved = errno;
    throw Error(ErrorCode::IOError, action + ": " + std::strerror(saved));
  }
  static void EnsureDirectory(const std::string& path) {
    struct stat statbuf;
    if (::lstat(path.c_str(), &statbuf) == 0) {
      if (!S_ISDIR(statbuf.st_mode)) throw Error(ErrorCode::InvalidArgument, "ownership state path is not a directory: " + path);
      return;
    }
    if (errno != ENOENT) IoError("stat ownership state directory");
    const auto slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) EnsureDirectory(path.substr(0, slash));
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) IoError("create ownership state directory");
    if (::lstat(path.c_str(), &statbuf) != 0 || !S_ISDIR(statbuf.st_mode)) {
      throw Error(ErrorCode::InvalidArgument, "ownership state directory changed unexpectedly: " + path);
    }
    // Persist the newly created directory entry in its parent.
    const std::string parent = slash == 0 ? "/" : path.substr(0, slash);
    int fd = ::open(parent.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) IoError("open ownership directory parent");
    const int result = ::fsync(fd);
    const int saved = errno;
    ::close(fd);
    errno = saved;
    if (result != 0) IoError("sync ownership directory parent");
  }
  static void ValidatePath(const std::string& path) {
    if (path.empty() || path.front() != '/' || path.find("//") != std::string::npos ||
        path.find('\\') != std::string::npos || path.size() > 1024 * 1024 ||
        path.size() < 4 || path.compare(path.size() - 4, 4, ".sst") != 0 ||
        path.find("/../") != std::string::npos || path.find("/./") != std::string::npos) {
      throw Error(ErrorCode::InvalidArgument, "ownership requires one canonical bare absolute HDFS SST path");
    }
    for (unsigned char character : path) {
      if (character < 32 || character == 127) {
        throw Error(ErrorCode::InvalidArgument, "ownership path contains a control character");
      }
    }
  }
  static void Normalize(std::vector<std::string>& paths, bool deduplicate) {
    if (paths.size() > kMaximumPaths) throw Error(ErrorCode::InvalidArgument, "ownership batch is too large");
    for (const auto& path : paths) ValidatePath(path);
    std::sort(paths.begin(), paths.end());
    if (deduplicate) paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  }
  Record MakeRecord(Kind kind, std::vector<std::string> paths, uint32_t owner,
                    const std::string& operation_id) const {
    EnsureHealthy();
    if (operation_id.size() > 1024 * 1024) throw Error(ErrorCode::InvalidArgument, "operation_id is too large");
    Normalize(paths, kind != Kind::Link);
    Record record;
    record.kind = kind;
    record.owner = owner;
    record.paths = std::move(paths);
    record.operation_id = operation_id;
    return record;
  }
  void EnsureHealthy() const {
    if (poisoned_) throw Error(ErrorCode::IOError, "ownership journal failed; registry is fenced until verified replay");
  }
  static void Put32(std::string& output, uint32_t value) {
    for (int i = 0; i < 4; ++i) output.push_back(static_cast<char>(value >> (8 * i)));
  }
  static void Put64(std::string& output, uint64_t value) {
    for (int i = 0; i < 8; ++i) output.push_back(static_cast<char>(value >> (8 * i)));
  }
  static void PutString(std::string& output, const std::string& value) {
    Put32(output, static_cast<uint32_t>(value.size())); output += value;
  }
  static uint64_t Take(const std::string& input, size_t& position, size_t bytes) {
    if (bytes > input.size() - position) throw Error(ErrorCode::Corruption, "short ownership journal field");
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) value |= static_cast<uint64_t>(static_cast<unsigned char>(input[position++])) << (8 * i);
    return value;
  }
  static std::string TakeString(const std::string& input, size_t& position) {
    const uint64_t length = Take(input, position, 4);
    if (length > 1024 * 1024 || length > input.size() - position) {
      throw Error(ErrorCode::Corruption, "invalid ownership journal string length");
    }
    std::string value = input.substr(position, length); position += length; return value;
  }
  static std::string Encode(const Record& record) {
    std::string output;
    output.push_back(static_cast<char>(record.kind));
    Put64(output, record.sequence); Put32(output, record.owner); Put64(output, record.token);
    output.push_back(record.success ? 1 : 0);
    PutString(output, record.operation_id); Put32(output, static_cast<uint32_t>(record.paths.size()));
    for (const auto& path : record.paths) PutString(output, path);
    return output;
  }
  static Record Decode(const std::string& input) {
    size_t position = 0;
    Record record;
    const auto kind = Take(input, position, 1);
    if (kind < 1 || kind > 6) throw Error(ErrorCode::Corruption, "unknown ownership journal operation");
    record.kind = static_cast<Kind>(kind);
    record.sequence = Take(input, position, 8); record.owner = static_cast<uint32_t>(Take(input, position, 4));
    record.token = Take(input, position, 8);
    const auto success = Take(input, position, 1);
    if (success > 1) throw Error(ErrorCode::Corruption, "invalid ownership journal outcome");
    record.success = success != 0;
    record.operation_id = TakeString(input, position);
    const auto count = Take(input, position, 4);
    if (count > kMaximumPaths) throw Error(ErrorCode::Corruption, "invalid ownership journal path count");
    for (uint64_t i = 0; i < count; ++i) record.paths.push_back(TakeString(input, position));
    if (position != input.size()) throw Error(ErrorCode::Corruption, "trailing ownership journal payload bytes");
    return record;
  }
  static uint64_t Checksum(const std::string& bytes) {
    uint64_t value = 14695981039346656037ULL;
    for (unsigned char byte : bytes) { value ^= byte; value *= 1099511628211ULL; }
    return value;
  }
  static void WriteAll(int fd, const std::string& bytes, const std::string& action) {
    size_t written = 0;
    while (written < bytes.size()) {
      const ssize_t result = ::write(fd, bytes.data() + written, bytes.size() - written);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) IoError(action);
      written += static_cast<size_t>(result);
    }
  }
  static std::string ReadAt(int fd, off_t offset, size_t bytes) {
    std::string output(bytes, '\0');
    size_t read = 0;
    while (read < bytes) {
      const ssize_t result = ::pread(fd, &output[read], bytes - read, offset + read);
      if (result < 0 && errno == EINTR) continue;
      if (result < 0) IoError("read ownership journal");
      if (result == 0) throw Error(ErrorCode::Corruption, "truncated ownership journal");
      read += static_cast<size_t>(result);
    }
    return output;
  }
  static std::string Fingerprint(Record record) {
    record.sequence = 0; record.operation_id.clear(); return Encode(record);
  }
  bool AlreadyApplied(const Record& record) const {
    if (record.operation_id.empty()) return false;
    const auto found = operations_.find(record.operation_id);
    if (found == operations_.end()) return false;
    if (found->second.fingerprint != Fingerprint(record)) {
      throw Error(ErrorCode::FailedPrecondition, "operation_id reused with different ownership arguments");
    }
    if (record.kind == Kind::Link) {
      for (const auto& path : record.paths) {
        const auto entry = entries_.find(path);
        if (entry == entries_.end()) throw Error(ErrorCode::FailedPrecondition, "linked SST is no longer known");
        const auto release = entry->second.last_release_sequence.find(record.owner);
        if (release != entry->second.last_release_sequence.end() && release->second >= found->second.sequence) {
          throw Error(ErrorCode::FailedPrecondition, "link operation was released; refusing stale reacquisition ACK: " + path);
        }
      }
    }
    return true;
  }
  void Validate(const Record& record) const {
    if ((record.kind == Kind::Release || record.kind == Kind::Complete) && record.paths.size() != 1) {
      throw Error(ErrorCode::InvalidArgument, "single-path ownership operation has invalid count");
    }
    std::map<std::string, uint64_t> link_additions;
    for (const auto& path : record.paths) {
      ValidatePath(path);
      const auto found = entries_.find(path);
      const EntryView* entry = found == entries_.end() ? nullptr : &found->second;
      if (record.kind == Kind::Create) {
        if (entry && (entry->state != State::Active || entry->aborted || !entry->born ||
                      !entry->birth_live || entry->creator != record.owner)) {
          throw Error(ErrorCode::FailedPrecondition, "SST path already retired or belongs to a different birth: " + path);
        }
      } else if (record.kind == Kind::Link) {
        if (!entry || entry->state != State::Active || entry->aborted || !entry->born ||
            entry->released_by.count(record.owner)) {
          throw Error(ErrorCode::FailedPrecondition, "link requires a known live immutable SST: " + path);
        }
        const uint64_t additions = ++link_additions[path];
        if (entry->count > std::numeric_limits<uint64_t>::max() - additions) {
          throw Error(ErrorCode::FailedPrecondition, "ownership reference count overflow");
        }
      } else if (record.kind == Kind::Release) {
        if (entry && entry->state == State::Active && !entry->released_by.count(record.owner)) {
          const auto linked = entry->linked_refs.find(record.owner);
          if (!(entry->birth_live && entry->creator == record.owner) &&
              (linked == entry->linked_refs.end() || linked->second == 0)) {
            throw Error(ErrorCode::FailedPrecondition, "release has no remaining reference for this owner: " + path);
          }
        }
      } else if (record.kind == Kind::Abort) {
        if (entry && entry->state == State::Active &&
            (!entry->birth_live || entry->count != 1 || !entry->linked_refs.empty() || entry->creator != record.owner)) {
          throw Error(ErrorCode::FailedPrecondition, "cannot abort an SST with linked or foreign ownership: " + path);
        }
      } else if (record.kind == Kind::Claim) {
        if (!entry || entry->count != 0 ||
            (entry->state != State::Pending && entry->state != State::Deleting)) {
          throw Error(ErrorCode::FailedPrecondition, "deletion claim requires a retired SST: " + path);
        }
      } else if (!entry || entry->state != State::Deleting || entry->claim_token != record.token) {
        throw Error(ErrorCode::FailedPrecondition, "stale or absent physical deletion claim: " + path);
      }
    }
  }
  void Apply(const Record& record) {
    if (AlreadyApplied(record)) return;
    for (const auto& path : record.paths) {
      if (record.kind == Kind::Release && entries_.find(path) == entries_.end()) {
        unknown_releases_++;
        continue;
      }
      EntryView& entry = entries_[path];
      if (record.kind == Kind::Create) {
        if (!entry.known) {
          entry.known = entry.born = entry.birth_live = true;
          entry.creator = record.owner; entry.count = 1;
        }
      } else if (record.kind == Kind::Link) {
        entry.linked_refs[record.owner]++; entry.count++;
        // No reference-generation token exists in legacy RequestDelete. Never
        // re-arm a released owner: a delayed old release could eat the new ref.
      } else if (record.kind == Kind::Release) {
        if (entry.state != State::Active || entry.released_by.count(record.owner)) continue;
        auto linked = entry.linked_refs.find(record.owner);
        if (entry.birth_live && entry.creator == record.owner) {
          entry.birth_live = false;
        } else if (linked != entry.linked_refs.end() && linked->second > 0) {
          if (--linked->second == 0) entry.linked_refs.erase(linked);
        }
        entry.count--; entry.released_by.insert(record.owner);
        entry.last_release_sequence[record.owner] = record.sequence;
        if (entry.count == 0) { entry.state = State::Pending; pending_paths_.insert(path); }
      } else if (record.kind == Kind::Abort) {
        entry.known = true; entry.aborted = true;
        if (entry.state == State::Active) {
          entry.birth_live = false; entry.count = 0; entry.state = State::Pending;
          pending_paths_.insert(path);
        }
      } else if (record.kind == Kind::Claim) {
        entry.state = State::Deleting; entry.claim_token = record.token;
        pending_paths_.erase(path); recovered_claims_.erase(path);
      } else {
        entry.state = record.success ? State::Deleted : State::Pending;
        if (!record.success) pending_paths_.insert(path);
      }
    }
    if (!record.operation_id.empty()) operations_.emplace(record.operation_id,
                                                         AppliedOperation{Fingerprint(record), record.sequence});
  }
  void Commit(Record& record) {
    EnsureHealthy();
    Validate(record);  // Whole batch validated BEFORE any state/journal mutation.
    if (AlreadyApplied(record)) return;
    if (record.paths.empty()) return;
    if (sequence_ == std::numeric_limits<uint64_t>::max()) {
      throw Error(ErrorCode::IOError, "ownership journal sequence exhausted");
    }
    record.sequence = sequence_ + 1;
    const std::string payload = Encode(record);
    if (payload.size() > kMaximumRecord) throw Error(ErrorCode::InvalidArgument, "ownership journal record is too large");
    std::string frame;
    Put32(frame, static_cast<uint32_t>(payload.size())); frame += payload; Put64(frame, Checksum(payload));
    try {
      WriteAll(journal_fd_, frame, "append ownership journal");
#ifdef OWNERSHIP_REGISTRY_TESTING
      if (fail_next_sync_) { fail_next_sync_ = false; errno = EIO; IoError("injected ownership journal sync failure"); }
#endif
      if (::fsync(journal_fd_) != 0) IoError("sync ownership journal before ACK");
      Apply(record);
      sequence_ = record.sequence;
    } catch (...) {
      poisoned_ = true;  // Never ACK another mutation/GC claim after ambiguous I/O.
      throw;
    }
  }
  void Replay(off_t size) {
    const std::string magic = Magic();
    if (size < static_cast<off_t>(magic.size()) || ReadAt(journal_fd_, 0, magic.size()) != magic) {
      throw Error(ErrorCode::Corruption, "ownership journal header is absent or corrupt; refusing fresh state");
    }
    off_t offset = magic.size();
    while (offset < size) {
      if (size - offset < 12) throw Error(ErrorCode::Corruption, "torn ownership journal frame");
      const auto header = ReadAt(journal_fd_, offset, 4);
      size_t position = 0;
      const uint64_t length = Take(header, position, 4);
      if (length == 0 || length > kMaximumRecord || length > static_cast<uint64_t>(size - offset - 12)) {
        throw Error(ErrorCode::Corruption, "invalid or torn ownership journal frame length");
      }
      const auto payload = ReadAt(journal_fd_, offset + 4, length);
      const auto trailer = ReadAt(journal_fd_, offset + 4 + length, 8);
      position = 0;
      if (Checksum(payload) != Take(trailer, position, 8)) {
        throw Error(ErrorCode::Corruption, "ownership journal checksum mismatch");
      }
      Record record = Decode(payload);
      if (record.sequence != sequence_ + 1) throw Error(ErrorCode::Corruption, "ownership journal sequence discontinuity");
      try { Validate(record); Apply(record); }
      catch (const Error& error) { throw Error(ErrorCode::Corruption, std::string("invalid ownership journal transition: ") + error.what()); }
      sequence_ = record.sequence;
      offset += 12 + length;
    }
  }
  void Close() {
    if (journal_fd_ >= 0) ::close(journal_fd_);
    if (lock_fd_ >= 0) ::close(lock_fd_);
    if (dir_fd_ >= 0) ::close(dir_fd_);
    journal_fd_ = lock_fd_ = dir_fd_ = -1;
  }

  mutable std::mutex mu_;
  int dir_fd_ = -1, lock_fd_ = -1, journal_fd_ = -1;
  bool poisoned_ = false;
  uint64_t sequence_ = 0, unknown_releases_ = 0;
  std::unordered_map<std::string, EntryView> entries_;
  std::unordered_map<std::string, AppliedOperation> operations_;
  std::set<std::string> pending_paths_, recovered_claims_;
#ifdef OWNERSHIP_REGISTRY_TESTING
  bool fail_next_sync_ = false;
#endif
};

}  // namespace ownership_cp
