#include <grpc/grpc.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <set>

#include "compaction_service.grpc.pb.h"
#include "queue"
#include "rocksdb/options.h"
#include "thread"
#include "utils.h"
#include <cstdlib>
#include <chrono>  // [heartbeat] steady_clock for per-task liveness timestamps
#include <vector>
#include "hdfs.h"  // [relink/Storage-CP] libhdfs C API: the CP now owns the physical SST delete
#include "storage_ownership_registry.h"

ROCKSDB_NAMESPACE::OpenAndCompactOptions compaction_service_options;
std::unordered_map<uint64_t, compactionservice::AddTaskArgs> task_args_map_;
struct TaskCmp {
  bool operator()(const uint64_t& task1, const uint64_t& task2) const {
    if (task_args_map_[task1].compaction_addition_info().start_level() !=
        task_args_map_[task2].compaction_addition_info().start_level()) {
      return task_args_map_[task1].compaction_addition_info().start_level() >
             task_args_map_[task2].compaction_addition_info().start_level();
    }
    if (task_args_map_[task1].compaction_addition_info().score() < 0 &&
        task_args_map_[task2].compaction_addition_info().score() < 0) {
      return task1 > task2;
    }
    if (task_args_map_[task1].compaction_addition_info().score() < 0 ||
        task_args_map_[task2].compaction_addition_info().score() < 0) {
      return task_args_map_[task1].compaction_addition_info().score() >
             task_args_map_[task2].compaction_addition_info().score();
    }
    if (task_args_map_[task1].compaction_addition_info().score() !=
        task_args_map_[task2].compaction_addition_info().score()) {
      return task_args_map_[task1].compaction_addition_info().score() <
             task_args_map_[task2].compaction_addition_info().score();
    }
    return task1 > task2;
  }
};

std::unordered_map<std::string, compactionservice::CSAStatus> csa_status_map_;
std::unordered_map<std::string, std::vector<uint64_t>> csa_task_list_;
std::unordered_map<std::string,
                   std::unique_ptr<compactionservice::CSAService::Stub>>
    csa_client_map_;
std::unordered_map<uint64_t, compactionservice::CompactionReply>
    task_reply_map_;
std::atomic<uint64_t> next_task_id_ = 0;
std::unordered_map<uint64_t, uint64_t> reschedule_num;
std::unordered_map<uint64_t, uint64_t> submit_fail_num_;  // [F2b] CSA-reported failures per task
// [heartbeat 2026-06-29] Last time the CSA pinged that a task is still PROGRESSING. CheckTask tells the
// CN to give up (fall back to local) ONLY when a task goes silent > kHeartbeatStaleSec, so a slow-but-
// running remote compaction is no longer killed+retried (the retry-churn / src-freeze bug). All accesses
// are under scheduler_latch_ (SubmitTask / CheckTask / ConsumeTask-dispatch), so no extra lock needed.
std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> last_heartbeat_;
static constexpr int kHeartbeatCode = 424242;   // SubmitTask sentinel: a ping, not a result (!=0,!=99,!=10)
static constexpr int kHeartbeatStaleSec = 150;  // 2.5x the CSA's 60s ping interval -> declared stuck
std::priority_queue<uint64_t, std::vector<uint64_t>, TaskCmp>
    task_priority_queue_;
std::mutex monitor_latch_;
std::mutex scheduler_latch_;

// Initialized and replayed before the server starts accepting Storage RPCs.
// Remote-compaction-only baselines may leave this disabled; Storage handlers
// then fail closed instead of silently creating a volatile reference table.
std::unique_ptr<ownership_cp::Registry> storage_registry_;

class ProCPImpl final : public compactionservice::ProCPService::Service {
 public:
  bool JudgeFallback(const compactionservice::AddTaskArgs* request) {
    return false;
  }

  grpc::Status AddTask(grpc::ServerContext* context,
                       const compactionservice::AddTaskArgs* request,
                       compactionservice::TaskId* reply) override {
    std::lock_guard<std::mutex> lock(scheduler_latch_);
    if (JudgeFallback(request)) {
      return grpc::Status::CANCELLED;
    }
    task_args_map_[next_task_id_].mutable_compaction_args()->CopyFrom(
        request->compaction_args());
    task_args_map_[next_task_id_].mutable_compaction_addition_info()->CopyFrom(
        request->compaction_addition_info());
    task_priority_queue_.push(next_task_id_);
    ++next_task_id_;
    reply->set_task_id(next_task_id_ - 1);
    std::cout << "Add compaction task: (" << reply->task_id() << ")"
              << std::endl;
    return grpc::Status::OK;
  }

  grpc::Status SubmitTask(grpc::ServerContext* context,
                          const compactionservice::SubmitTaskArgs* request,
                          google::protobuf::Empty* response) override {
    std::cout << GetTime() << "Submit compaction task: (" << request->task_id()
              << ")" << std::endl;
    std::lock_guard<std::mutex> lock(scheduler_latch_);
    if (request->compaction_reply().code() == kHeartbeatCode) {
      // [heartbeat] Not a result -> just record liveness so CheckTask keeps the CN waiting
      // instead of timing the (still-progressing) remote compaction out and retrying it.
      last_heartbeat_[request->task_id()] = std::chrono::steady_clock::now();
      return grpc::Status::OK;
    }
    if (task_args_map_.count(request->task_id()) == 0) {
      return grpc::Status::OK;
    }
    if (request->compaction_reply().code() != 0) {
      // [F2b, 2026-06-26] Bound failure retries. A deterministically-failing remote compaction (e.g.
      // a real Corruption) was re-queued here FOREVER, so CheckTask kept returning 99 and the CN's
      // poll parked -> src freeze. After max_reschedule failures make it TERMINAL: store the failed
      // reply so CheckTask returns its real code -> the CN gets a failure (kUseLocal) instead of hanging.
      if (++submit_fail_num_[request->task_id()] >
          compaction_service_options.max_reschedule) {
        std::cout << GetTime() << "Compaction task (" << request->task_id()
                  << ") failed " << submit_fail_num_[request->task_id()]
                  << "x -> TERMINAL (returning failure to CN)" << std::endl;
        task_reply_map_[request->task_id()] = request->compaction_reply();
        task_args_map_.erase(request->task_id());
        submit_fail_num_.erase(request->task_id());
        return grpc::Status::OK;
      }
      std::cout << GetTime() << "Compaction task (" << request->task_id()
                << ") failed (retry " << submit_fail_num_[request->task_id()]
                << ")" << std::endl;
      task_priority_queue_.push(request->task_id());
      return grpc::Status::OK;
    }
    std::cout << GetTime() << "Compaction task (" << request->task_id()
              << ") success" << std::endl;
    task_reply_map_[request->task_id()] = request->compaction_reply();
    task_args_map_.erase(request->task_id());
    submit_fail_num_.erase(request->task_id());
    return grpc::Status::OK;
  }

  grpc::Status CheckTask(grpc::ServerContext* context,
                         const compactionservice::TaskId* request,
                         compactionservice::CompactionReply* reply) override {
    std::lock_guard<std::mutex> lock(scheduler_latch_);
    if (task_reply_map_.count(request->task_id()) == 0) {
      // [heartbeat] Abort (tell the CN to fall back to local) ONLY if the task went silent for
      // > kHeartbeatStaleSec. A slow-but-progressing remote compaction keeps pinging -> stays 99 ->
      // CN keeps waiting instead of killing+retrying it. (Routine "Not finished" logging removed:
      // it was ~193k lines/run of pure noise + CPU on node53.)
      auto hb = last_heartbeat_.find(request->task_id());
      if (hb != last_heartbeat_.end() &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - hb->second)
                  .count() > kHeartbeatStaleSec) {
        std::cout << GetTime() << "Check compaction task (" << request->task_id()
                  << "): STALE (no heartbeat > " << kHeartbeatStaleSec
                  << "s) -> abort to CN" << std::endl;
        last_heartbeat_.erase(hb);
        reply->set_code(10);  // kAborted: CN sees code!=0 -> kUseLocal -> runs it locally
        return grpc::Status::OK;
      }
      reply->set_code(99);
      return grpc::Status::OK;
    }
    auto res = task_reply_map_[request->task_id()];
    task_reply_map_.erase(request->task_id());
    std::cout << GetTime() << "Check compaction task (" << request->task_id()
              << "): Finished " << std::endl;
    reply->set_code(res.code());
    reply->mutable_result()->assign(res.result());
    reply->set_process_latency(res.process_latency());
    reply->set_open_db_latency(res.open_db_latency());
    return grpc::Status::OK;
  }

  grpc::Status RegisterCSA(grpc::ServerContext* context,
                           const compactionservice::CSAStatus* request,
                           google::protobuf::Empty* response) override {
    csa_status_map_[request->address()] = *request;
    csa_client_map_[request->address()] =
        compactionservice::CSAService::NewStub(grpc::CreateChannel(
            request->address(), grpc::InsecureChannelCredentials()));
    std::vector<uint64_t> temp;
    csa_task_list_[request->address()] = temp;
    std::cout << GetTime() << "Register CSA (" << request->address() << ")"
              << std::endl;
    return grpc::Status::OK;
  }
};

// [relink/Storage-CP] Distributed refcount registry for shared SSTs on HDFS.
// Storage publication is acknowledged only after durable ownership journaling.
// Every final path has one immutable birth; logical links add references.
// Physical deletion is exclusively performed by the CP after a durable claim.
template <typename Function>
grpc::Status StorageCall(Function function) {
  if (!storage_registry_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "durable ownership disabled: set STORAGE_CP_STATE_DIR");
  }
  try {
    function(*storage_registry_);
    return grpc::Status::OK;
  } catch (const ownership_cp::Error& error) {
    grpc::StatusCode code = grpc::StatusCode::FAILED_PRECONDITION;
    if (error.code() == ownership_cp::ErrorCode::InvalidArgument) code = grpc::StatusCode::INVALID_ARGUMENT;
    else if (error.code() == ownership_cp::ErrorCode::IOError) code = grpc::StatusCode::UNAVAILABLE;
    else if (error.code() == ownership_cp::ErrorCode::Corruption) code = grpc::StatusCode::DATA_LOSS;
    std::cerr << GetTime() << "[storage] RPC rejected: " << error.what() << std::endl;
    return grpc::Status(code, error.what());
  } catch (const std::exception& error) {
    std::cerr << GetTime() << "[storage] RPC failed: " << error.what() << std::endl;
    return grpc::Status(grpc::StatusCode::INTERNAL, error.what());
  }
}

std::vector<std::string> StoragePaths(const compactionservice::FileRefBatch& request) {
  return std::vector<std::string>(request.path().begin(), request.path().end());
}

class StorageImpl final : public compactionservice::StorageService::Service {
 public:
  grpc::Status NotifyCreate(grpc::ServerContext*, const compactionservice::FileRef* request,
                            google::protobuf::Empty*) override {
    return StorageCall([&](ownership_cp::Registry& registry) {
      registry.Create({request->path()}, request->shard_id());
      std::cout << GetTime() << "[storage] NotifyCreate path=" << request->path()
                << " shard=" << request->shard_id()
                << " refcount=" << registry.Inspect(request->path()).count << std::endl;
    });
  }
  grpc::Status NotifyCreateBatch(grpc::ServerContext*, const compactionservice::FileRefBatch* request,
                                 google::protobuf::Empty*) override {
    return StorageCall([&](ownership_cp::Registry& registry) {
      registry.Create(StoragePaths(*request), request->shard_id(), request->operation_id());
      for (const auto& path : request->path()) {
        std::cout << GetTime() << "[storage] NotifyCreate path=" << path
                  << " shard=" << request->shard_id() << " refcount=" << registry.Inspect(path).count
                  << " (durable batch)\n";
      }
      std::cout << GetTime() << "[storage] create-batch n=" << request->path_size() << std::endl;
    });
  }
  grpc::Status NotifyLink(grpc::ServerContext*, const compactionservice::FileRef* request,
                          google::protobuf::Empty*) override {
    return StorageCall([&](ownership_cp::Registry& registry) {
      registry.Link({request->path()}, request->shard_id());
      std::cout << GetTime() << "[storage] NotifyLink path=" << request->path()
                << " shard=" << request->shard_id()
                << " refcount=" << registry.Inspect(request->path()).count << std::endl;
    });
  }
  grpc::Status NotifyLinkBatch(grpc::ServerContext*, const compactionservice::FileRefBatch* request,
                               google::protobuf::Empty*) override {
    return LinkBatch(request, false);
  }
  grpc::Status PrepareReferences(grpc::ServerContext*, const compactionservice::FileRefBatch* request,
                                 google::protobuf::Empty*) override {
    if (request->operation_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "PrepareReferences requires a stable operation_id");
    }
    return LinkBatch(request, true);
  }
  grpc::Status AbortUnpublished(grpc::ServerContext*, const compactionservice::FileRefBatch* request,
                                google::protobuf::Empty*) override {
    return StorageCall([&](ownership_cp::Registry& registry) {
      // The engine/offline recovery caller must have proved these paths were
      // never committed. Unknown ordinary releases do not grant this authority.
      registry.AbortUnpublished(StoragePaths(*request), request->shard_id(), request->operation_id());
      for (const auto& path : request->path()) {
        std::cout << GetTime() << "[storage] AbortUnpublished path=" << path
                  << " shard=" << request->shard_id() << '\n';
      }
      std::cout << GetTime() << "[storage] abort-batch n=" << request->path_size() << std::endl;
    });
  }
  grpc::Status RecoverReferences(grpc::ServerContext*, const compactionservice::FileRefBatch* request,
                                 google::protobuf::Empty*) override {
    return StorageCall([&](ownership_cp::Registry& registry) {
      registry.RecoverReferences(StoragePaths(*request), request->shard_id());
      std::cout << GetTime() << "[storage] recover-verified n=" << request->path_size()
                << " shard=" << request->shard_id() << std::endl;
    });
  }
  grpc::Status RequestDelete(grpc::ServerContext*, const compactionservice::FileRef* request,
                             compactionservice::DeleteReply* reply) override {
    reply->set_deleted(false);  // No Storage RPC ever authorizes client-side HDFS deletion.
    return StorageCall([&](ownership_cp::Registry& registry) {
      const auto result = registry.Release(request->path(), request->shard_id());
      reply->set_refcount(static_cast<int>(std::min<uint64_t>(result.count, std::numeric_limits<int>::max())));
      if (result.unknown) {
        std::cout << GetTime() << "[storage] ★UNTRACKED RequestDelete path=" << request->path()
                  << " shard=" << request->shard_id() << " -> KEEP (fail-safe)" << std::endl;
      } else {
        std::cout << GetTime() << "[storage] RequestDelete path=" << request->path()
                  << " shard=" << request->shard_id() << " deleted=false refcount=" << result.count
                  << " duplicate=" << result.duplicate << std::endl;
      }
    });
  }

 private:
  grpc::Status LinkBatch(const compactionservice::FileRefBatch* request, bool prepared) {
    return StorageCall([&](ownership_cp::Registry& registry) {
      registry.Link(StoragePaths(*request), request->shard_id(), request->operation_id());
      for (const auto& path : request->path()) {
        std::cout << GetTime() << "[storage] NotifyLink path=" << path
                  << " shard=" << request->shard_id() << " refcount=" << registry.Inspect(path).count
                  << " (durable batch)\n";
      }
      std::cout << GetTime() << "[storage] " << (prepared ? "prepare" : "link")
                << "-batch n=" << request->path_size() << std::endl;
    });
  }
};

// Fenced deletion claims are fsynced before any HDFS operation. A crash after
// physical deletion but before its completion record safely retries the same
// immutable path. Unknown staging files enter this queue only through an
// explicit AbortUnpublished; neither TTL nor MANIFEST absence grants deletion.
[[noreturn]] void RunStorageGC() {
  int interval = 60;
  if (const char* s = getenv("STORAGE_GC_INTERVAL_S")) { int v = atoi(s); if (v > 0) interval = v; }
  const char* nn_host = getenv("STORAGE_GC_NN_HOST"); if (!nn_host || !*nn_host) nn_host = "192.168.88.87";
  int nn_port = 9000;
  if (const char* p = getenv("STORAGE_GC_NN_PORT")) { int v = atoi(p); if (v > 0) nn_port = v; }
  const char* user = getenv("HADOOP_USER_NAME"); if (!user || !*user) user = "sbyeon12";
  hdfsFS fs = nullptr;
  while (true) {
    sleep(interval);
    try {
      const auto claims = storage_registry_->ClaimDeletes();
      if (!claims.empty() && fs == nullptr) {
        fs = hdfsConnectAsUser(nn_host, nn_port, user);
        if (fs == nullptr) {
          std::cerr << GetTime() << "[storage-gc] hdfsConnect FAILED " << nn_host << ':' << nn_port << std::endl;
        }
      }
      size_t deleted = 0, requeued = 0;
      for (const auto& claim : claims) {
        bool reclaimed = false;
        if (fs != nullptr) {
          errno = 0;
          reclaimed = hdfsDelete(fs, claim.path.c_str(), /*recursive=*/0) == 0;
          if (!reclaimed) {
            // Do not interpret an arbitrary HDFS error as 'already absent'.
            errno = 0;
            hdfsFileInfo* info = hdfsGetPathInfo(fs, claim.path.c_str());
            if (info != nullptr) hdfsFreeFileInfo(info, 1);
            else reclaimed = errno == ENOENT;
          }
        }
        storage_registry_->CompleteDelete(claim, reclaimed);
        if (reclaimed) ++deleted; else ++requeued;
      }
      const auto stats = storage_registry_->GetStats();
      std::cout << GetTime() << "[storage-gc] interval=" << interval << "s tracked=" << stats.tracked
                << " shared=" << stats.shared << " batch=" << claims.size() << " deleted=" << deleted
                << " requeued=" << requeued << " pending=" << stats.pending
                << " deleting=" << stats.deleting << " tombstones=" << stats.deleted
                << " untracked_deletes=" << stats.unknown_releases
                << " journal_sequence=" << stats.journal_sequence << std::endl;
    } catch (const std::exception& error) {
      // Journal failure poisons the registry: all subsequent mutation/claim
      // attempts fail closed, and this loop performs no further physical I/O.
      std::cerr << GetTime() << "[storage-gc] durable registry FAILED: " << error.what() << std::endl;
    }
  }
}

grpc::Status DistributeCompactionJob(
    const compactionservice::CompactionTaskArgs& compact_task_args,
    const std::string& csa_address) {
  grpc::ClientContext context;
  google::protobuf::Empty response;
  grpc::Status status = csa_client_map_[csa_address]->ExecuteCompactionTask(
      &context, compact_task_args, &response);
  return status;
}

void UpdateOneCSAStatus(const std::string& address) {
  auto& stub = csa_client_map_[address];
  grpc::ClientContext context;
  google::protobuf::Empty request;
  compactionservice::CSAStatus csa_status;
  grpc::Status status = stub->CheckCSAStatus(&context, request, &csa_status);
  monitor_latch_.lock();
  if (status.ok()) {
    csa_status_map_[address] = csa_status;
  } else {
    std::cout << GetTime() << "CSA (" << address << ") Offline" << std::endl;
    scheduler_latch_.lock();
    csa_status_map_.erase(address);
    csa_client_map_.erase(address);
    auto task_list = csa_task_list_[address];
    for (uint64_t i : task_list) {
      if (task_args_map_.count(i) > 0) {
        task_priority_queue_.push(i);
      }
    }
    csa_task_list_.erase(address);
    scheduler_latch_.unlock();
  }
  monitor_latch_.unlock();
}

[[noreturn]] void UpdateCSAStatus() {
  while (true) {
    sleep(5);
    for (auto iter = csa_status_map_.begin(); iter != csa_status_map_.end();) {
      UpdateOneCSAStatus(iter++->first);
    }
  }
}

std::string ScheduleCSA(
    const compactionservice::CompactionAdditionInfo& compaction_addition_info) {
  std::lock_guard<std::mutex> lock(monitor_latch_);
  if (csa_status_map_.empty()) {
    return "";
  }
  size_t min_task_nums = csa_status_map_.begin()->second.local_task_nums();
  auto best_worker = csa_status_map_.begin()->first;
  for (const auto& iter : csa_status_map_) {
    if (iter.second.local_task_nums() < min_task_nums &&
        iter.second.local_task_nums() < iter.second.max_task_nums()) {
      best_worker = iter.first;
      min_task_nums = iter.second.local_task_nums();
    }
  }
  if (csa_status_map_[best_worker].local_task_nums() >=
          csa_status_map_[best_worker].max_task_nums() ||
      csa_status_map_[best_worker].memory_usage() < 0.3) {
    return "";
  }
  return best_worker;
}

[[noreturn]] void ConsumeTask() {
  while (true) {
    scheduler_latch_.lock();
    if (task_priority_queue_.empty()) {
      scheduler_latch_.unlock();
      //      std::cout << GetTime() << "No task" << std::endl;
      sleep(1);
      continue;
    }
    auto task_id = task_priority_queue_.top();
    task_priority_queue_.pop();
    std::cout << GetTime() << "Schedule compaction task (" << task_id << ")"
              << std::endl;
    auto compaction_job_info = task_args_map_[task_id];
    if (csa_status_map_.empty() ||
        task_priority_queue_.size() >
            compaction_service_options.max_accumulation_in_procp ||
        reschedule_num[task_id] > compaction_service_options.max_reschedule) {
      // [diag 2026-06-27] Record WHICH of the three conditions forced this local
      // fallback, plus the live values, all read here under scheduler_latch_ so they
      // match the branch decision. 0626_8 showed mass "immediate" fallbacks (no prior
      // "all busy") even though the CSAs were registered ~70s earlier and never went
      // Offline, with fresh unique task_ids and a tiny queue -- none of the three
      // conditions should have been true. This pins it: csa_empty (map empty / not
      // visible to this thread) vs queue_over vs reschedule_over.
      const char* why =
          csa_status_map_.empty()
              ? "csa_empty"
              : (task_priority_queue_.size() >
                         compaction_service_options.max_accumulation_in_procp
                     ? "queue_over"
                     : "reschedule_over");
      uint64_t diag_q = task_priority_queue_.size();
      uint64_t diag_resched = reschedule_num[task_id];
      size_t diag_csa = csa_status_map_.size();
      compactionservice::CompactionReply compactionReply;
      compactionReply.set_code(1);
      task_reply_map_[task_id] = compactionReply;
      task_args_map_.erase(task_id);
      scheduler_latch_.unlock();
      std::cout << GetTime() << "Fallback compaction task (" << task_id
                << ") reason=" << why << " csa_map=" << diag_csa
                << " qsize=" << diag_q << " resched=" << diag_resched << std::endl;
      continue;
    }
    auto worker_address =
        ScheduleCSA(compaction_job_info.compaction_addition_info());
    if (worker_address.empty()) {
      reschedule_num[task_id]++;
      std::cout << GetTime() << "Workers are all busy, reschedule " << task_id
                << std::endl;
      task_priority_queue_.push(task_id);
      scheduler_latch_.unlock();
      sleep(1);
      continue;
    }
    csa_task_list_[worker_address].emplace_back(task_id);
    last_heartbeat_[task_id] = std::chrono::steady_clock::now();  // [heartbeat] grace until 1st CSA ping
    scheduler_latch_.unlock();
    monitor_latch_.lock();
    std::cout << GetTime() << "Memory usage is "
              << csa_status_map_[worker_address].memory_usage() << std::endl;
    csa_status_map_[worker_address].set_local_task_nums(
        csa_status_map_[worker_address].local_task_nums() + 1);
    monitor_latch_.unlock();
    compactionservice::CompactionTaskArgs compaction_task_args;
    compaction_task_args.set_task_id(task_id);
    compaction_task_args.mutable_compaction_args()->CopyFrom(
        compaction_job_info.compaction_args());
    grpc::Status status =
        DistributeCompactionJob(compaction_task_args, worker_address);
    if (!status.ok()) {
      std::cout << GetTime()
                << " Failed to send compaction task to CSA: " << worker_address
                << std::endl;
      scheduler_latch_.lock();
      task_priority_queue_.push(task_id);
      scheduler_latch_.unlock();
    }
  }
}

int main() {
  // ProCP dispatch knobs — env-overridable so we can push the post-migration compaction
  // burst to the (idle) CSAs instead of falling back to local CN. Defaults stay 5 so
  // baselines are unchanged. PROCP_MAX_ACCUMULATION = queue depth before shedding to
  // local; PROCP_MAX_RESCHEDULE = per-task CSA-busy retries before local fallback.
  if (const char* s = getenv("PROCP_MAX_ACCUMULATION")) {
    unsigned long long v = strtoull(s, nullptr, 10);
    if (v > 0) compaction_service_options.max_accumulation_in_procp = v;
  }
  if (const char* s = getenv("PROCP_MAX_RESCHEDULE")) {
    unsigned long long v = strtoull(s, nullptr, 10);
    if (v > 0) compaction_service_options.max_reschedule = v;
  }
  // Listen address — env-overridable (PRO_CP_ADDR="host:port") so procp can run on a
  // node other than the compiled default. Unset/empty keeps the default.
  bool pro_cp_addr_from_env = false;
  if (const char* s = getenv("PRO_CP_ADDR")) {
    if (*s) {
      compaction_service_options.pro_cp_address = s;
      pro_cp_addr_from_env = true;
    }
  }
  std::cout << GetTime() << "ProCP knobs: max_accumulation_in_procp="
            << compaction_service_options.max_accumulation_in_procp
            << " max_reschedule=" << compaction_service_options.max_reschedule
            << " pro_cp_address=" << compaction_service_options.pro_cp_address
            << (pro_cp_addr_from_env ? " (from PRO_CP_ADDR)" : " (compiled default)")
            << std::endl;
  std::string server_address(compaction_service_options.pro_cp_address);
  const char* state_directory = getenv("STORAGE_CP_STATE_DIR");
  if (state_directory != nullptr && *state_directory != '\0') {
    try {
      storage_registry_.reset(new ownership_cp::Registry(state_directory));
      const auto stats = storage_registry_->GetStats();
      std::cout << GetTime() << "[storage] durable registry READY state_dir=" << state_directory
                << " tracked=" << stats.tracked << " pending=" << stats.pending
                << " deleting=" << stats.deleting << " tombstones=" << stats.deleted
                << " journal_sequence=" << stats.journal_sequence << std::endl;
    } catch (const std::exception& error) {
      std::cerr << GetTime() << "FATAL: durable ownership startup failed: " << error.what() << std::endl;
      return 2;
    }
  } else {
    std::cout << GetTime() << "[storage] DISABLED: STORAGE_CP_STATE_DIR unset; Storage RPCs fail closed, GC disabled"
              << std::endl;
  }
  ProCPImpl service;
  StorageImpl storage_service;

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(&service);
  builder.RegisterService(&storage_service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  // A stale procp still holding pro_cp_address makes AddListeningPort fail; gRPC
  // leaves selected_port==0 (and may return a non-listening server). Previously we
  // printed "Server listening" unconditionally and ran on -- the GC thread ticked
  // while 8020 was dead, so every NotifyCreate/Link/RequestDelete got "Connection
  // refused" and relink silently fell back to GC. Fail LOUD + exit so the launcher
  // (start_cp.sh) detects it instead of leaving a non-listening zombie.
  if (!server || selected_port == 0) {
    std::cerr << GetTime() << "FATAL: procp failed to bind " << server_address
              << " (selected_port=" << selected_port
              << "; stale procp still holding the port?). Exiting." << std::endl;
    return 1;
  }
  std::cout << GetTime() << "Server listening on " << server_address
            << " (port=" << selected_port << ")" << std::endl;
  std::thread scheduler(ConsumeTask);
  std::thread monitor(UpdateCSAStatus);
  if (storage_registry_) std::thread(RunStorageGC).detach();
  scheduler.join();
  monitor.join();
  server->Wait();
}
