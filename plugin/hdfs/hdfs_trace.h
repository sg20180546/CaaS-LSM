// HDFS API call tracer — characterization only, env-gated, zero-overhead when off.
// Enable:  export CAAS_HDFS_TRACE=/nvme/sungjin/htrace/<run>   (directory, must exist)
// Output:  per-thread files  <dir>/hdfs_trace.<pid>.<tid>.csv   (no lock → no contention on the hot pread path)
// Schema:  ts_us,tid,api,size,offset,lat_us
//   size   = bytes actually moved (0 for metadata/NameNode RPC)
//   offset = file offset for pread (-1 when N/A)
//   lat_us = wall time of the single libhdfs call (µs)
// Merge at analysis time: cat <dir>/hdfs_trace.*.csv (dedupe the header rows).
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <unistd.h>

namespace ROCKSDB_NAMESPACE {

class HdfsTracer {
 public:
  static HdfsTracer& Get() {
    static HdfsTracer t;
    return t;
  }
  bool on() const { return dir_ != nullptr; }

  static inline long long NowUs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
  }

  // Fully buffered, per-thread FILE → no mutex on the hot path.
  inline void Log(const char* api, long long size, long long offset,
                  long long lat_us) {
    if (!dir_) return;
    FILE* f = TLFile();
    if (!f) return;
    fprintf(f, "%lld,%u,%s,%lld,%lld,%lld\n", NowUs(), TLId(), api, size,
            offset, lat_us);
  }

 private:
  HdfsTracer() { dir_ = getenv("CAAS_HDFS_TRACE"); if (dir_ && !*dir_) dir_ = nullptr; }

  static unsigned TLId() {
    static std::atomic<unsigned> next{0};
    static thread_local unsigned id = next.fetch_add(1);
    return id;
  }
  FILE* TLFile() {
    static thread_local FILE* f = OpenTL();
    return f;
  }
  FILE* OpenTL() {
    if (!dir_) return nullptr;
    char path[512];
    snprintf(path, sizeof(path), "%s/hdfs_trace.%d.%u.csv", dir_, (int)getpid(),
             TLId());
    FILE* f = fopen(path, "w");
    if (f) {
      setvbuf(f, nullptr, _IOFBF, 1 << 20);  // 1MB buffer
      fprintf(f, "ts_us,tid,api,size,offset,lat_us\n");
    }
    return f;
  }

  const char* dir_ = nullptr;
};

// Time a call that returns a value, log it, forward the value. (GCC/Clang stmt-expr)
#define HDFS_TRACE_CALL(api, size, off, call)                          \
  ({                                                                   \
    long long _t0 = ::ROCKSDB_NAMESPACE::HdfsTracer::NowUs();          \
    auto _ret = (call);                                                \
    ::ROCKSDB_NAMESPACE::HdfsTracer::Get().Log(                        \
        (api), (long long)(size), (long long)(off),                    \
        ::ROCKSDB_NAMESPACE::HdfsTracer::NowUs() - _t0);               \
    _ret;                                                              \
  })

}  // namespace ROCKSDB_NAMESPACE
