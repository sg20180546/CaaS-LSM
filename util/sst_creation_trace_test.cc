// Standalone instrumentation test, built by tools/test_sst_creation_trace.py.
// The same source is built into two translation units to check that inline
// sink/TLS state is shared across engine and filesystem call sites.
#include "util/sst_creation_trace.h"

#include <cassert>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace trace = ROCKSDB_NAMESPACE::sst_creation_trace;
void WriteTraceTestSst(const std::string& path, const std::string& mode);

#ifdef SST_TRACE_TEST_WRITER
void WriteTraceTestSst(const std::string& path, const std::string& mode) {
  trace::File file;
  file.Begin(path);
  errno = ENOSPC;
  file.Opened(mode != "open_failed");
  assert(errno == ENOSPC);
  if (mode == "open_failed") {
    file.End(false, "duplicate_must_not_appear");
    trace::BuildResult(path, 0, false, true);
    return;
  }
  if (mode == "timed") {
    const uint64_t start = trace::NowNs();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    file.OwnRpc(trace::NowNs() - start, true);
  } else {
    file.OwnRpc(0, mode != "own_failed");
  }
  const bool empty = mode == "empty";
  if (!empty) {
    file.AddBytes(64);
    file.AddBytes(128);
  }
  if (mode == "io_failed") file.IoFailed();
  const bool explicit_close = mode != "destructor";
  errno = EIO;
  file.End(mode != "close_failed", explicit_close ? "explicit" : "destructor");
  assert(errno == EIO);
  file.End(false, "duplicate_must_not_appear");
  trace::BuildResult(path, empty ? 0 : 192,
                     mode != "io_failed" && mode != "close_failed", empty);
}
#else
int main() {
  {
    trace::Scope outer("flush", 7);
    WriteTraceTestSst("/quoted\"\\\n/first.sst", "timed");
    {
      trace::Scope inner("compaction", 8);
      WriteTraceTestSst("/nested.sst", "normal");
      inner.SetResult(true);
    }
    WriteTraceTestSst("/second.sst", "normal");
    outer.SetResult(true);
  }
  {
    trace::Scope failures("flush", 9);
    for (const char* mode : {"open_failed", "io_failed", "own_failed",
                             "close_failed", "empty", "destructor"}) {
      WriteTraceTestSst(std::string("/") + mode + ".sst", mode);
    }
    failures.SetResult(false);
  }
  {
    trace::File manifest;
    manifest.Begin("/MANIFEST-1");
    assert(!manifest.enabled());
    manifest.Opened(true);
    manifest.End(true, "explicit");
  }
  std::vector<std::thread> writers;
  for (int i = 0; i < 8; ++i) {
    writers.emplace_back([i] {
      for (int j = 0; j < 20; ++j) {
        trace::Scope scope(i % 2 == 0 ? "flush" : "compaction", i * 100 + j);
        WriteTraceTestSst("/thread" + std::to_string(i) + "/" +
                              std::to_string(j) + ".sst", "normal");
        scope.SetResult(true);
      }
    });
  }
  for (auto& writer : writers) writer.join();
  trace::RpcTimer rpc;
  rpc.Finish("RequestDelete(rename)", "/renamed.sst", false);
}
#endif
