// Copyright (c) 2026-present. All rights reserved.
// Offline ownership recovery deliberately has no age-based orphan heuristic.
#pragma once

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace rocksdb_storage_recovery {

struct Plan {
  std::vector<std::string> live;
  std::vector<std::string> keep;
  std::vector<std::string> discard;
};

// Keep registry keys byte-for-byte. Silently normalizing an alias could protect
// one key while deleting the same physical file through another key.
inline bool IsCanonicalAbsolute(const std::string& path) {
  size_t begin = 0;
  if (path.empty() || path[begin] != '/' ||
      path.find_first_of("\n\r\t\\") != std::string::npos ||
      path.find('\0') != std::string::npos) return false;
  std::string suffix = path.substr(begin);
  if (suffix.find("//") != std::string::npos) return false;
  for (size_t p = 1; p <= suffix.size();) {
    size_t next = suffix.find('/', p);
    if (next == std::string::npos) next = suffix.size();
    const auto part = suffix.substr(p, next - p);
    if (part == "." || part == "..") return false;
    if (next == suffix.size()) break;
    p = next + 1;
  }
  return true;
}

inline std::string TrimDirectory(std::string path) {
  while (path.size() > 1 && path.back() == '/') path.pop_back();
  return path;
}

inline bool InDirectory(const std::string& path, const std::string& root) {
  const auto directory = TrimDirectory(root);
  return directory == "/" ? path.front() == '/'
                          : path.size() > directory.size() &&
                                path.compare(0, directory.size(), directory) == 0 &&
                                path[directory.size()] == '/';
}

inline std::string BuildPlan(const std::vector<std::string>& live,
                             const std::vector<std::string>& candidates,
                             const std::vector<std::string>& directories,
                             Plan* plan) {
  *plan = Plan();
  for (const auto& dir : directories) {
    if (!IsCanonicalAbsolute(dir)) return "non-canonical DB directory: " + dir;
  }
  std::set<std::string> live_set(live.begin(), live.end());
  for (const auto& path : live_set) {
    if (!IsCanonicalAbsolute(path)) return "non-canonical live path: " + path;
  }
  plan->live.assign(live_set.begin(), live_set.end());
  const std::set<std::string> candidate_set(candidates.begin(), candidates.end());
  for (const auto& path : candidate_set) {
    if (!IsCanonicalAbsolute(path) || path.size() < 4 ||
        path.compare(path.size() - 4, 4, ".sst") != 0) {
      return "candidate is not a canonical absolute SST path: " + path;
    }
    bool inside = false;
    for (const auto& dir : directories) inside |= InDirectory(path, dir);
    if (!inside) return "candidate outside this DB's configured directories: " + path;
    (live_set.count(path) ? plan->keep : plan->discard).push_back(path);
  }
  return {};
}

// The caller must keep every CN/CSA producer fenced for the entire operation.
// HDFS LockFile is currently a no-op; the boolean is an explicit maintenance
// precondition, NOT an implementation of a distributed lock. Checking MANIFEST
// twice detects an accidentally changing snapshot but cannot fence a writer.
inline std::string ApplyPlan(
    const Plan& plan, bool producers_stopped,
    const std::function<std::string()>& check_snapshot,
    const std::function<std::string(const std::vector<std::string>&)>& recover,
    const std::function<std::string(const std::vector<std::string>&)>& discard) {
  if (!producers_stopped) return "all CN/CSA producers must be stopped and fenced";
  auto error = check_snapshot();
  if (!error.empty()) return error;
  error = recover(plan.live);
  if (!error.empty()) return error;
  error = check_snapshot();
  if (!error.empty()) return error;
  return discard(plan.discard);
}

}  // namespace rocksdb_storage_recovery
