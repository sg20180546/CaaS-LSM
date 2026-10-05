// Local publication state. CP journal remains authoritative across restart.
#pragma once
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace storage_cp {
class PublicationTracker {
 public:
  enum class State { kUnknown, kPending, kPrepared, kPublished, kUncertain,
                     kReleased, kRenamedAway };
  State Get(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = states_.find(path);
    return it == states_.end() ? State::kUnknown : it->second;
  }
  bool CanAbort(const std::string& path) const {
    const auto state = Get(path);
    return state == State::kUnknown || state == State::kPending ||
           state == State::kPrepared || state == State::kReleased ||
           state == State::kRenamedAway;
  }
  void Set(const std::vector<std::string>& paths, State state) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& path : paths) states_[path] = state;
  }
  void Renamed(const std::string& src, const std::string& dst) {
    std::lock_guard<std::mutex> lock(mu_);
    states_[src] = State::kRenamedAway;
    states_[dst] = State::kPending;
  }
 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, State> states_;
};
}  // namespace storage_cp
