#define OWNERSHIP_REGISTRY_TESTING
#include "storage_ownership_registry.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/wait.h>
#include <thread>

using ownership_cp::Error;
using ownership_cp::ErrorCode;
using ownership_cp::Registry;
using ownership_cp::State;

namespace {
std::string scratch;
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void Fails(ErrorCode code, Function action) {
  try { action(); }
  catch (const Error& error) { Require(error.code() == code, "unexpected registry error: " + std::string(error.what())); return; }
  throw std::runtime_error("operation unexpectedly succeeded");
}
std::string Directory(const std::string& label) {
  std::string pattern = scratch + "/" + label + "-XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back(0);
  char* created = ::mkdtemp(writable.data());
  Require(created != nullptr, "mkdtemp failed");
  return created;
}
std::string Read(const std::string& path) {
  std::ifstream source(path, std::ios::binary);
  Require(source.good(), "cannot read test journal");
  return std::string(std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>());
}
void Write(const std::string& path, const std::string& bytes) {
  std::ofstream target(path, std::ios::binary | std::ios::trunc);
  target.write(bytes.data(), bytes.size());
  Require(target.good(), "cannot mutate test journal");
}

void BirthAndBatchAtomicity() {
  Registry registry(Directory("birth"));
  registry.Create({"/db/a.sst", "/db/b.sst"}, 0, "birth-a-b");
  registry.Create({"/db/b.sst", "/db/a.sst"}, 0, "birth-a-b");
  registry.Create({"/db/a.sst"}, 0);  // Old NotifyCreate and new CreateBatch share one birth.
  Require(registry.Inspect("/db/a.sst").count == 1, "birth retry inflated refcount");
  registry.RecoverReferences({"/db/a.sst", "/db/b.sst"}, 0);
  Fails(ErrorCode::FailedPrecondition, [&] { registry.RecoverReferences({"/db/a.sst"}, 9); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Release("/db/a.sst", 9); });
  Require(registry.Inspect("/db/a.sst").count == 1, "foreign release consumed birth");
  registry.Release("/db/b.sst", 0);
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/fresh.sst", "/db/b.sst"}, 0); });
  Require(!registry.Inspect("/db/fresh.sst").known, "failed create batch partly applied");
  Fails(ErrorCode::InvalidArgument, [&] { registry.Create({"/db/../unsafe.sst"}, 0); });
  for (const auto& alias : {"/db//alias.sst", "hdfs://namenode/db/alias.sst", "/db/./alias.sst",
                            "/db/back\\slash.sst", "/db/control\n.sst"}) {
    Fails(ErrorCode::InvalidArgument, [&] { registry.Create({alias}, 0); });
  }
  registry.Create({"/db/alias.sst"}, 0);
  Require(registry.Inspect("/db/alias.sst").count == 1, "canonical path was rejected");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Link({"/db/missing.sst"}, 1); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.RecoverReferences({"/db/missing.sst"}, 1); });
  registry.Create({"/db/legacy.sst"}, 0);
  registry.Link({"/db/legacy.sst"}, 1);
  registry.Link({"/db/legacy.sst"}, 1);
  Require(registry.Inspect("/db/legacy.sst").count == 3, "legacy counted Link semantics changed");
}

void ReleaseAndStaleOperationFences() {
  Registry registry(Directory("stale"));
  registry.Create({"/db/p.sst"}, 0, "birth-p");
  registry.Link({"/db/p.sst"}, 1, "link-old");
  registry.Link({"/db/p.sst"}, 1, "link-old");
  Require(registry.Inspect("/db/p.sst").count == 2, "link operation retry inflated refcount");
  registry.Release("/db/p.sst", 1);
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Link({"/db/p.sst"}, 1, "link-old"); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Link({"/db/p.sst"}, 1, "link-new"); });
  const auto late_duplicate = registry.Release("/db/p.sst", 1);
  Require(late_duplicate.duplicate && late_duplicate.count == 1, "late release burned a reference after reacquire attempt");
  registry.Create({"/db/q.sst"}, 0, "birth-q");
  registry.Link({"/db/q.sst"}, 1, "link-q");
  registry.Release("/db/q.sst", 0);
  Require(registry.Inspect("/db/q.sst").count == 1, "creator release consumed linked reference");
  registry.RecoverReferences({"/db/q.sst"}, 1);
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/q.sst"}, 0, "birth-q"); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.RecoverReferences({"/db/q.sst"}, 0); });
  const auto duplicate = registry.Release("/db/q.sst", 0);
  Require(duplicate.duplicate && duplicate.count == 1, "release dedup burned linked owner");
  registry.Release("/db/q.sst", 1);
  registry.Release("/db/p.sst", 0);
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/p.sst"}, 0, "birth-p"); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Link({"/db/p.sst"}, 1, "link-new"); });
  const auto claims = registry.ClaimDeletes();
  Require(claims.size() == 2, "retired paths not claimed exactly once");
  Require(registry.ClaimDeletes().empty(), "live claim duplicated by next GC poll");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/p.sst"}, 0); });
  for (const auto& claim : claims) registry.CompleteDelete(claim, true);
  Require(registry.Inspect("/db/p.sst").state == State::Deleted, "completion lost tombstone");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/p.sst"}, 0, "birth-p"); });
}

void AbortAndUnknownRelease() {
  Registry registry(Directory("abort"));
  registry.Create({"/db/shared.sst", "/db/unpublished.sst", "/db/foreign.sst"}, 0);
  registry.Link({"/db/shared.sst"}, 1, "shared-link");
  Fails(ErrorCode::FailedPrecondition, [&] {
    registry.AbortUnpublished({"/db/staging.sst", "/db/unpublished.sst", "/db/shared.sst"}, 0);
  });
  Require(!registry.Inspect("/db/staging.sst").known && registry.Inspect("/db/unpublished.sst").count == 1,
          "failed abort batch partly applied");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.AbortUnpublished({"/db/foreign.sst"}, 1); });
  registry.Create({"/db/unpublished.sst"}, 0, "birth-unpublished");
  registry.AbortUnpublished({"/db/staging.sst", "/db/unpublished.sst"}, 0, "abort-two");
  registry.AbortUnpublished({"/db/unpublished.sst", "/db/staging.sst"}, 0, "abort-two");
  Require(registry.Inspect("/db/shared.sst").count == 2, "abort consumed shared reference");
  Require(registry.Inspect("/db/staging.sst").aborted && registry.GetStats().pending == 2,
          "unknown staging abort was not durably queued");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/staging.sst"}, 0); });
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Create({"/db/unpublished.sst"}, 0, "birth-unpublished"); });
  const auto unknown = registry.Release("/db/unknown.sst", 0);
  Require(unknown.unknown && unknown.count == 0 && !registry.Inspect("/db/unknown.sst").known,
          "ordinary unknown release became deletion authority");
  Require(registry.GetStats().pending == 2, "unknown release entered GC queue");
  registry.Create({"/db/unknown.sst"}, 0);
  Require(registry.Inspect("/db/unknown.sst").count == 1, "unknown release fenced future legitimate birth");
}

void ConcurrentRetriesAndClaimRace() {
  Registry registry(Directory("concurrent"));
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 12; ++i) threads.emplace_back([&] {
    try {
      for (int j = 0; j < 5; ++j) {
        registry.Create({"/db/retry.sst"}, 0, "birth-retry");
        registry.Link({"/db/retry.sst"}, 1, "link-retry");
      }
    } catch (...) { ++failures; }
  });
  for (auto& thread : threads) thread.join();
  Require(failures == 0 && registry.Inspect("/db/retry.sst").count == 2, "concurrent retries inflated/lost references");
  std::thread a([&] { registry.Release("/db/retry.sst", 0); });
  std::thread b([&] { registry.Release("/db/retry.sst", 1); });
  a.join(); b.join();
  Require(registry.GetStats().pending == 1, "concurrent releases queued twice");
  const auto claims = registry.ClaimDeletes();
  Fails(ErrorCode::FailedPrecondition, [&] { registry.Link({"/db/retry.sst"}, 1, "new-link"); });
  registry.CompleteDelete(claims[0], false);
  const auto retried = registry.ClaimDeletes();
  Require(retried[0].token > claims[0].token, "GC retry reused stale claim token");
  Fails(ErrorCode::FailedPrecondition, [&] { registry.CompleteDelete(claims[0], true); });
  registry.CompleteDelete(retried[0], true);
}

void CrashReplayAndProcessLock() {
  const std::string directory = Directory("replay");
  const pid_t child = ::fork();
  Require(child >= 0, "fork failed");
  if (child == 0) {
    try {
      Registry registry(directory);
      registry.Create({"/db/live.sst"}, 0, "live-birth");
      registry.Link({"/db/live.sst"}, 1, "live-link");
      registry.Release("/db/live.sst", 0);
      registry.Create({"/db/done.sst"}, 0);
      registry.Release("/db/done.sst", 0);
      registry.CompleteDelete(registry.ClaimDeletes()[0], true);
      registry.Create({"/db/deleting.sst"}, 0);
      registry.Release("/db/deleting.sst", 0);
      registry.ClaimDeletes();
      registry.Create({"/db/pending.sst"}, 0);
      registry.Release("/db/pending.sst", 0);
      registry.AbortUnpublished({"/db/unknown-temp.sst"}, 0, "abort-temp");
      ::_exit(0);  // Simulate CP death: no Registry destructor/teardown.
    } catch (...) { ::_exit(2); }
  }
  int status;
  Require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "crash producer failed");
  {
    Registry registry(directory);
    Require(registry.Inspect("/db/live.sst").count == 1 && !registry.Inspect("/db/live.sst").birth_live,
            "replay lost linked reference or creator release");
    registry.RecoverReferences({"/db/live.sst"}, 1);
    registry.Link({"/db/live.sst"}, 1, "live-link");
    Require(registry.Inspect("/db/live.sst").count == 1, "crash replay forgot link idempotency");
    Require(registry.Inspect("/db/done.sst").state == State::Deleted &&
            registry.Inspect("/db/deleting.sst").state == State::Deleting &&
            registry.Inspect("/db/pending.sst").state == State::Pending &&
            registry.Inspect("/db/unknown-temp.sst").aborted,
            "abrupt crash replay lost lifecycle state");
  }
}

void ReplayAllStatesAndExclusiveLock() {
  const std::string directory = Directory("states");
  {
    Registry registry(directory);
    registry.Create({"/db/live.sst"}, 0);
    registry.Link({"/db/live.sst"}, 1, "live-link");
    registry.Release("/db/live.sst", 0);
    registry.Create({"/db/deleted.sst"}, 0);
    registry.Release("/db/deleted.sst", 0);
    registry.CompleteDelete(registry.ClaimDeletes()[0], true);
    registry.AbortUnpublished({"/db/deleting.sst"}, 0);
    registry.ClaimDeletes();
    registry.AbortUnpublished({"/db/pending.sst"}, 0);
    registry.Release("/db/unknown.sst", 0);
    const pid_t child = ::fork();
    Require(child >= 0, "lock test fork failed");
    if (child == 0) {
      try { Registry other(directory); ::_exit(2); }
      catch (const Error& error) { ::_exit(error.code() == ErrorCode::Busy ? 0 : 3); }
    }
    int status;
    Require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "second CP writer acquired the state directory");
  }
  {
    Registry registry(directory);
    const auto stats = registry.GetStats();
    Require(stats.tracked == 1 && stats.pending == 1 && stats.deleting == 1 && stats.deleted == 1 && stats.unknown_releases == 1,
            "journal replay did not preserve all lifecycle states");
    registry.RecoverReferences({"/db/live.sst"}, 1);
    registry.Link({"/db/live.sst"}, 1, "live-link");
    Require(registry.Inspect("/db/live.sst").count == 1, "replay forgot link operation dedup");
    const uint64_t old_token = registry.Inspect("/db/deleting.sst").claim_token;
    const auto claims = registry.ClaimDeletes();
    Require(claims.size() == 2, "crash deletion claim was not retried");
    for (const auto& claim : claims) {
      Require(claim.token > old_token, "replayed claim not fenced by new token");
      registry.CompleteDelete(claim, true);
    }
    Require(registry.GetStats().deleted == 3, "replayed GC lost permanent tombstones");
  }
}

void CorruptionFailsClosed() {
  const std::string source = Directory("journal-source");
  { Registry registry(source); registry.Create({"/db/a.sst"}, 0); }
  const auto clean = Read(source + "/ownership.journal");
  std::vector<std::string> corrupted;
  auto bytes = clean; bytes[0] ^= 1; corrupted.push_back(bytes);
  bytes = clean; bytes[bytes.size() - 9] ^= 1; corrupted.push_back(bytes);
  corrupted.push_back(clean.substr(0, clean.size() - 1));
  corrupted.push_back(clean + "xx");
  corrupted.push_back("");
  for (const auto& invalid : corrupted) {
    const std::string directory = Directory("corrupt");
    { Registry registry(directory); }
    Write(directory + "/ownership.journal", invalid);
    Fails(ErrorCode::Corruption, [&] { Registry registry(directory); });
    Require(Read(directory + "/ownership.journal") == invalid, "corrupt state was silently reset/truncated");
  }
}

void SyncFailureHasNoAckAndPoisons() {
  const std::string directory = Directory("sync-failure");
  {
    Registry registry(directory);
    registry.Create({"/db/old.sst"}, 0);
    registry.FailNextSyncForTest();
    Fails(ErrorCode::IOError, [&] { registry.Create({"/db/ambiguous.sst"}, 0, "ambiguous-birth"); });
    Require(registry.GetStats().poisoned && !registry.Inspect("/db/ambiguous.sst").known,
            "failed sync produced an in-memory success");
    Fails(ErrorCode::IOError, [&] { registry.Release("/db/old.sst", 0); });
    Fails(ErrorCode::IOError, [&] { registry.ClaimDeletes(); });
    Fails(ErrorCode::IOError, [&] { registry.RecoverReferences({"/db/old.sst"}, 0); });
  }
  // The unacked record may have reached storage. Verified replay plus a stable
  // retry must recover it exactly once, never by resetting the state directory.
  Registry replay(directory);
  replay.Create({"/db/ambiguous.sst"}, 0, "ambiguous-birth");
  Require(replay.Inspect("/db/ambiguous.sst").count == 1, "ambiguous retry inflated durable birth");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || argv[1][0] != '/') return 2;
  scratch = argv[1];
  const std::vector<std::pair<std::string, void (*)()>> tests = {
      {"birth-batch-atomicity", BirthAndBatchAtomicity},
      {"release-stale-operation-fences", ReleaseAndStaleOperationFences},
      {"abort-unknown-release", AbortAndUnknownRelease},
      {"concurrent-retries-gc-claim", ConcurrentRetriesAndClaimRace},
      {"abrupt-process-crash-replay", CrashReplayAndProcessLock},
      {"replay-lifecycle-process-lock", ReplayAllStatesAndExclusiveLock},
      {"journal-corruption-fail-closed", CorruptionFailsClosed},
      {"sync-failure-no-ack-poison", SyncFailureHasNoAckAndPoisons},
  };
  try {
    for (const auto& test : tests) { test.second(); std::cout << "PASS " << test.first << '\n'; }
    std::cout << "PASS all " << tests.size() << " durable ownership cases\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n'; return 1;
  }
}
