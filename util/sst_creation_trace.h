// Opt-in, per-physical-SST experiment tracing. No clocks, path copies, or
// output are used unless SST_CREATION_TRACE names a local JSONL output file.
// Open-to-close is wall time, including interleaved merge iteration, encoding,
// synchronous ownership registration, writes, Sync and Close. Work before
// opening the file is deliberately not attributed to that file.
#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {
namespace sst_creation_trace {

class PreserveErrno {
 public:
  PreserveErrno() : saved_(errno) {}
  ~PreserveErrno() { errno = saved_; }
 private:
  int saved_;
};

inline uint64_t NowNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline uint64_t WallUs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

inline std::string Quote(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      char escaped[7];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
      out += escaped;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + '"';
}

class Sink {
 public:
  Sink() {
    PreserveErrno preserve_errno;
    const char* path = std::getenv("SST_CREATION_TRACE");
    if (path != nullptr && *path != '\0') {
      file_ = std::fopen(path, "a");
      if (file_ == nullptr) {
        std::fprintf(stderr, "[sst-creation-trace] cannot open %s\n", path);
      }
    }
  }
  bool enabled() const { return file_ != nullptr; }
  void Emit(const std::string& fields) {
    PreserveErrno preserve_errno;
    std::ostringstream row;
    row << "{\"schema\":1,\"pid\":" << getpid()
        << ",\"tid\":" << std::hash<std::thread::id>{}(std::this_thread::get_id())
        << ",\"wall_us\":" << WallUs() << ',' << fields << "}\n";
    const std::string line = row.str();
    std::lock_guard<std::mutex> lock(mu_);
    // Emission is outside the measured file interval. Flushing each complete
    // record permits collection while long-lived DB/CSA processes are running.
    const bool wrote = std::fwrite(line.data(), 1, line.size(), file_) == line.size();
    const bool flushed = std::fflush(file_) == 0;
    if ((!wrote || !flushed) && !warned_) {
      warned_ = true;
      std::fprintf(stderr, "[sst-creation-trace] write failed\n");
    }
  }

 private:
  FILE* file_ = nullptr;
  std::mutex mu_;
  bool warned_ = false;
};

inline Sink& GetSink() {
  // Intentionally live through process exit: writer destruction may happen
  // during teardown, after unrelated function-local static destructors.
  static Sink* sink = new Sink();
  return *sink;
}
inline bool Enabled() { return GetSink().enabled(); }

inline void OwnershipRpc(const char* op, const std::string& path,
                         uint64_t duration_ns, bool success,
                         uint64_t files = 1) {
  if (!Enabled()) return;
  std::ostringstream row;
  row << "\"event\":\"ownership_rpc\",\"op\":" << Quote(op)
      << ",\"path\":" << Quote(path) << ",\"duration_ns\":" << duration_ns
      << ",\"success\":" << (success ? "true" : "false")
      << ",\"files\":" << files;
  GetSink().Emit(row.str());
}

class RpcTimer {
 public:
  RpcTimer() : enabled_(Enabled()), start_ns_(enabled_ ? NowNs() : 0) {}
  void Finish(const char* op, const std::string& path, bool success,
              uint64_t files = 1) const {
    if (!enabled_) return;
    const uint64_t duration_ns = NowNs() - start_ns_;
    OwnershipRpc(op, path, duration_ns, success, files);
  }
 private:
  bool enabled_;
  uint64_t start_ns_;
};

class Scope;
inline Scope*& CurrentScope() {
  static thread_local Scope* scope = nullptr;
  return scope;
}
inline uint64_t NextScopeId() {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

class Scope {
 public:
  Scope(const char* origin, int job_id) {
    if (!Enabled()) return;
    enabled_ = true;
    origin_ = origin;
    job_id_ = job_id;
    id_ = NextScopeId();
    start_ns_ = NowNs();
    previous_ = CurrentScope();
    CurrentScope() = this;
  }
  ~Scope() {
    if (!enabled_) return;
    const uint64_t end_ns = NowNs();
    CurrentScope() = previous_;
    std::ostringstream row;
    row << "\"event\":\"scope_end\",\"origin\":" << Quote(origin_)
        << ",\"job_id\":" << job_id_ << ",\"scope_id\":" << id_
        << ",\"duration_ns\":" << end_ns - start_ns_
        << ",\"files_opened\":" << files_opened_
        << ",\"success_known\":" << (success_known_ ? "true" : "false")
        << ",\"success\":" << (success_ ? "true" : "false");
    GetSink().Emit(row.str());
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  void SetResult(bool success) { success_known_ = true; success_ = success; }
  const char* origin() const { return origin_; }
  int job_id() const { return job_id_; }
  uint64_t id() const { return id_; }
  uint64_t start_ns() const { return start_ns_; }
  bool FileOpened() { return files_opened_++ == 0; }

 private:
  bool enabled_ = false;
  bool success_known_ = false;
  bool success_ = false;
  const char* origin_ = "unknown";
  int job_id_ = -1;
  uint64_t id_ = 0;
  uint64_t start_ns_ = 0;
  uint64_t files_opened_ = 0;
  Scope* previous_ = nullptr;
};

// Independent state per physical writer; neither append nor merge loops lock.
class File {
 public:
  void Begin(const std::string& path) {
    if (!Enabled()) return;
    if (path.size() < 4 || path.compare(path.size() - 4, 4, ".sst") != 0) return;
    enabled_ = true;
    path_ = path;
    Scope* scope = CurrentScope();
    if (scope != nullptr) {
      origin_ = scope->origin();
      job_id_ = scope->job_id();
      scope_id_ = scope->id();
      first_ = scope->FileOpened();
      scope_start_ns_ = scope->start_ns();
    }
    std::ostringstream row;
    row << "\"event\":\"file_open\",\"path\":" << Quote(path_)
        << ",\"origin\":" << Quote(origin_) << ",\"job_id\":" << job_id_
        << ",\"scope_id\":" << scope_id_;
    // Record intent before starting the measured interval. An unmatched intent
    // identifies a failed/unfinished file instead of silently biasing its CDF.
    GetSink().Emit(row.str());
    start_wall_us_ = WallUs();
    start_ns_ = NowNs();
  }
  bool enabled() const { return enabled_; }
  void Opened(bool success) {
    if (!enabled_) return;
    PreserveErrno preserve_errno;
    open_ns_ = NowNs() - start_ns_;
    open_ok_ = success;
    if (!success) End(false, "open_failed");
  }
  void AddBytes(uint64_t bytes) { if (enabled_) bytes_ += bytes; }
  void IoFailed() { if (enabled_) io_ok_ = false; }
  void OwnRpc(uint64_t duration_ns, bool success) {
    if (!enabled_) return;
    own_attempted_ = true;
    own_ok_ = success;
    own_ns_ = duration_ns;
  }
  void End(bool close_ok, const char* close_kind) {
    if (!enabled_ || ended_) return;
    PreserveErrno preserve_errno;
    const uint64_t end_ns = NowNs();
    ended_ = true;
    std::ostringstream row;
    row << "\"event\":\"file_close\",\"path\":" << Quote(path_)
        << ",\"origin\":" << Quote(origin_) << ",\"job_id\":" << job_id_
        << ",\"scope_id\":" << scope_id_ << ",\"start_wall_us\":" << start_wall_us_
        << ",\"open_start_ns\":" << start_ns_ << ",\"close_end_ns\":" << end_ns
        << ",\"open_to_close_ns\":" << end_ns - start_ns_
        << ",\"hdfs_open_ns\":" << open_ns_ << ",\"bytes\":" << bytes_
        << ",\"own_attempted\":" << (own_attempted_ ? "true" : "false")
        << ",\"own_ok\":" << (own_ok_ ? "true" : "false")
        << ",\"own_rpc_ns\":" << own_ns_
        << ",\"open_ok\":" << (open_ok_ ? "true" : "false")
        << ",\"io_ok\":" << (io_ok_ ? "true" : "false")
        << ",\"close_ok\":" << (close_ok ? "true" : "false")
        << ",\"close_kind\":" << Quote(close_kind)
        << ",\"first_output_preopen_ns\":";
    if (first_) row << start_ns_ - scope_start_ns_;
    else row << "null";
    GetSink().Emit(row.str());
    // Initial-own RPC emission is deferred until the measured file interval
    // ends so JSONL I/O is not included in the SST's open-to-close duration.
    if (own_attempted_) OwnershipRpc("NotifyCreate", path_, own_ns_, own_ok_);
  }

 private:
  bool enabled_ = false;
  bool ended_ = false;
  bool open_ok_ = false;
  bool io_ok_ = true;
  bool own_attempted_ = false;
  bool own_ok_ = false;
  bool first_ = false;
  std::string path_;
  const char* origin_ = "unknown";
  int job_id_ = -1;
  uint64_t scope_id_ = 0;
  uint64_t scope_start_ns_ = 0;
  uint64_t start_wall_us_ = 0;
  uint64_t start_ns_ = 0;
  uint64_t open_ns_ = 0;
  uint64_t bytes_ = 0;
  uint64_t own_ns_ = 0;
};

inline void BuildResult(const std::string& path, uint64_t bytes,
                        bool success, bool empty) {
  if (!Enabled()) return;
  Scope* scope = CurrentScope();
  std::ostringstream row;
  row << "\"event\":\"build_result\",\"path\":" << Quote(path)
      << ",\"origin\":" << Quote(scope == nullptr ? "unknown" : scope->origin())
      << ",\"job_id\":" << (scope == nullptr ? -1 : scope->job_id())
      << ",\"scope_id\":" << (scope == nullptr ? 0 : scope->id())
      << ",\"bytes\":" << bytes
      << ",\"success\":" << (success ? "true" : "false")
      << ",\"empty\":" << (empty ? "true" : "false");
  GetSink().Emit(row.str());
}

}  // namespace sst_creation_trace
}  // namespace ROCKSDB_NAMESPACE
