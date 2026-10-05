#include "plugin/hdfs/storage_cp_recovery_plan.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>

namespace recovery = rocksdb_storage_recovery;
#define REQUIRE(condition) do { if (!(condition)) { \
  std::cerr << "failed at line " << __LINE__ << ": " #condition << '\n'; \
  std::abort(); } } while (0)

int main() {
  const std::string live = "/db/s0/000101.sst";
  const std::string prepared = "/db/s0/000102.sst";
  const std::string staging = "/db/s0/job123/000001.sst";
  recovery::Plan plan;
  REQUIRE(recovery::BuildPlan({live, live}, {live, prepared, staging, prepared},
                             {"/db/s0"}, &plan).empty());
  REQUIRE(plan.live.size() == 1 && plan.keep.size() == 1 && plan.discard.size() == 2);

  // Crash after own but before MANIFEST: the live file stays owned, while both
  // the prepared final file and the never-owned CSA staging file are discarded.
  // All deletions in this model are CP decisions; the client cannot erase bytes.
  std::map<std::string, std::set<int>> owners{{live, {0}}, {prepared, {0}}};
  std::set<std::string> tombstones, cp_delete_queue;
  int recover_calls = 0, discard_calls = 0;
  auto recover = [&](const std::vector<std::string>& paths) -> std::string {
    ++recover_calls;
    for (const auto& path : paths) {
      if (!owners.count(path) || !owners[path].count(0)) return "unknown live owner";
    }
    return {};
  };
  auto discard = [&](const std::vector<std::string>& paths) -> std::string {
    ++discard_calls;
    for (const auto& path : paths) {
      if (!owners[path].empty() && owners[path] != std::set<int>{0}) {
        return "linked or foreign ownership; abort refused";
      }
    }
    for (const auto& path : paths) {
      owners[path].erase(0);
      if (owners[path].empty()) {
        tombstones.insert(path);
        cp_delete_queue.insert(path);
      }
    }
    return {};
  };
  auto unchanged = []() -> std::string { return {}; };

  // An active CN or orphan CSA must stop the entire operation, even though its
  // in-flight output is absent from the MANIFEST.
  REQUIRE(!recovery::ApplyPlan(plan, false, unchanged, recover, discard).empty());
  REQUIRE(recover_calls == 0 && discard_calls == 0);
  REQUIRE(cp_delete_queue.empty());

  // If MANIFEST sync was uncertain and later proves committed, that output is
  // discovered as live and cannot be sent to the discard RPC.
  recovery::Plan committed;
  REQUIRE(recovery::BuildPlan({live, prepared}, {prepared, staging}, {"/db/s0"},
                             &committed).empty());
  REQUIRE(committed.keep == std::vector<std::string>{prepared});
  REQUIRE(committed.discard == std::vector<std::string>{staging});

  // Detect another writer changing MANIFEST during registry validation. No
  // candidate may be discarded from an obsolete snapshot.
  int snapshots = 0;
  auto changed = [&]() -> std::string {
    return ++snapshots == 2 ? "MANIFEST changed" : "";
  };
  REQUIRE(!recovery::ApplyPlan(plan, true, changed, recover, discard).empty());
  REQUIRE(discard_calls == 0 && cp_delete_queue.empty());

  REQUIRE(recovery::ApplyPlan(plan, true, unchanged, recover, discard).empty());
  REQUIRE(owners[live].count(0) == 1);
  REQUIRE(cp_delete_queue == (std::set<std::string>{prepared, staging}));
  REQUIRE(tombstones.count(prepared) && tombstones.count(staging));
  // A crash after CP accepted the abort but before its ACK is safe to retry.
  REQUIRE(recovery::ApplyPlan(plan, true, unchanged, recover, discard).empty());
  REQUIRE(cp_delete_queue.size() == 2);

  // Linked ownership makes the whole abort batch fail closed.
  owners[prepared] = {0, 1};
  cp_delete_queue.clear();
  REQUIRE(!recovery::ApplyPlan(plan, true, unchanged, recover, discard).empty());
  REQUIRE(owners[prepared] == (std::set<int>{0, 1}));
  REQUIRE(cp_delete_queue.count(prepared) == 0);

  // Lost CP state is not reconstructed from a single DB's partial view.
  owners.erase(live);
  const auto old_calls = discard_calls;
  REQUIRE(!recovery::ApplyPlan(plan, true, unchanged, recover, discard).empty());
  REQUIRE(discard_calls == old_calls);

  for (const auto& invalid : {"/db/s01/000001.sst", "/db/s0/../s1/000001.sst",
                              "/db/s0//000001.sst", "/db/s0/MANIFEST-000001"}) {
    REQUIRE(!recovery::BuildPlan({live}, {invalid}, {"/db/s0"}, &plan).empty());
  }
  std::cout << "ownership recovery planner: crash, fencing, uncertain commit, "
               "partial CP loss, shared owner, retry, and scope tests passed\n";
}
