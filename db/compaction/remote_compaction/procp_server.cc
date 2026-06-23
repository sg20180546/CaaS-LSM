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
#include <vector>
#include "hdfs.h"  // [relink/Storage-CP] libhdfs C API: the CP now owns the physical SST delete

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
std::priority_queue<uint64_t, std::vector<uint64_t>, TaskCmp>
    task_priority_queue_;
std::mutex monitor_latch_;
std::mutex scheduler_latch_;

// [relink/Storage-CP] Per-file refcount over shared SSTs on HDFS.
// path(bare HDFS path, e.g. /kg/s0/000123.sst) -> {count, owning shard ids}.
// Eager: NotifyCreate(=1)/NotifyLink(++)/RequestDelete(--). When refcount reaches 0 no live
// shard references the file anymore, so the CP (which now links libhdfs) OWNS the physical
// delete: RequestDelete queues the path on storage_pending_delete_ and the periodic GC thread
// (RunStorageGC, every STORAGE_GC_INTERVAL_S, default 60s) batch-hdfsDeletes it; the CN never
// hdfsDeletes a tracked file. refcount==0 is definitive (every reference released => no live
// MANIFEST points at it), so the batch delete needs NO MANIFEST scan. A MANIFEST mark-sweep is
// only a *future* backstop for files leaked by a lost RequestDelete (CN crash), out of scope here.
struct RefEntry {
  int count = 0;
  std::set<uint32_t> owners;
};
std::unordered_map<std::string, RefEntry> storage_refcount_map_;
std::vector<std::string> storage_pending_delete_;  // refcount-0 paths awaiting the CP's batch hdfsDelete
std::mutex storage_latch_;                          // guards storage_refcount_map_ + storage_pending_delete_

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
    if (task_args_map_.count(request->task_id()) == 0) {
      return grpc::Status::OK;
    }
    if (request->compaction_reply().code() != 0) {
      std::cout << GetTime() << "Compaction task (" << request->task_id()
                << ") failed" << std::endl;
      task_priority_queue_.push(request->task_id());
      return grpc::Status::OK;
    }
    std::cout << GetTime() << "Compaction task (" << request->task_id()
              << ") success" << std::endl;
    task_reply_map_[request->task_id()] = request->compaction_reply();
    task_args_map_.erase(request->task_id());
    return grpc::Status::OK;
  }

  grpc::Status CheckTask(grpc::ServerContext* context,
                         const compactionservice::TaskId* request,
                         compactionservice::CompactionReply* reply) override {
    std::lock_guard<std::mutex> lock(scheduler_latch_);
    if (task_reply_map_.count(request->task_id()) == 0) {
      std::cout << GetTime() << "Check compaction task (" << request->task_id()
                << "): Not finished" << std::endl;
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
// CN/CSA notify the CP when a shared (aligned/relinked) .sst is physically
// created or logically linked into another shard, and ask the CP before
// deleting one. The CP owns the authoritative refcount keyed by full HDFS URI;
// it never touches HDFS itself this milestone -- the CN performs the actual
// hdfsDelete iff RequestDelete returns deleted=true (refcount reached 0).
// ISOLATION: this service only ever runs for files registered via these RPCs.
// When the relink feature is disabled, FileDescriptor::external_path is empty,
// CN/CSA never call NotifyCreate/NotifyLink/RequestDelete, storage_refcount_map_
// stays empty, and librocksdb/serverclient behavior is bit-identical to today.
class StorageImpl final : public compactionservice::StorageService::Service {
 public:
  // A new physical reference to a shared SST. PLAIN COUNT (++), NOT keyed by
  // owner identity: with remote compaction the CSA (one shard_id) CREATES the
  // aligned SST while the CN (a different shard_id) later DELETEs it, so gating
  // the count on owner-set membership would miss the decrement. owners is kept
  // for debug only. One NewWritableFile => one NotifyCreate per file.
  grpc::Status NotifyCreate(grpc::ServerContext* context,
                            const compactionservice::FileRef* request,
                            google::protobuf::Empty* response) override {
    std::lock_guard<std::mutex> lock(storage_latch_);
    auto& e = storage_refcount_map_[request->path()];
    e.count++;
    e.owners.insert(request->shard_id());  // debug only
    std::cout << GetTime() << "[storage] NotifyCreate path=" << request->path()
              << " shard=" << request->shard_id() << " refcount=" << e.count
              << std::endl;
    return grpc::Status::OK;
  }

  // Logical link of an already-existing shared SST into another shard (relink).
  // PLAIN COUNT (++) like NotifyCreate.
  grpc::Status NotifyLink(grpc::ServerContext* context,
                          const compactionservice::FileRef* request,
                          google::protobuf::Empty* response) override {
    std::lock_guard<std::mutex> lock(storage_latch_);
    auto& e = storage_refcount_map_[request->path()];
    e.count++;
    e.owners.insert(request->shard_id());  // debug only
    std::cout << GetTime() << "[storage] NotifyLink   path=" << request->path()
              << " shard=" << request->shard_id() << " refcount=" << e.count
              << std::endl;
    return grpc::Status::OK;
  }

  // Drop a shard's reference. For a TRACKED file the CP owns deletion: when the
  // refcount hits 0 the path is queued for the CP's batch GC (RunStorageGC) and the
  // reply is deleted=false (the CN must NOT delete it). An UNTRACKED path (never saw a
  // NotifyCreate) is treated as unshared -> deleted=true so the CN's normal delete path
  // stays intact for non-relinked files / pre-STORAGE_CP files.
  grpc::Status RequestDelete(grpc::ServerContext* context,
                             const compactionservice::FileRef* request,
                             compactionservice::DeleteReply* reply) override {
    std::lock_guard<std::mutex> lock(storage_latch_);
    auto it = storage_refcount_map_.find(request->path());
    if (it == storage_refcount_map_.end()) {
      // Untracked => unshared: tell the CN to delete normally.
      reply->set_deleted(true);
      reply->set_refcount(0);
    } else {
      auto& e = it->second;
      e.count--;  // unconditional: the deleter need not be the creator (CSA vs CN)
      e.owners.erase(request->shard_id());  // debug only
      if (e.count <= 0) {
        // No shard references it anymore. The CP owns the physical delete: queue it
        // for the periodic batch GC and tell the CN NOT to delete. refcount==0 is
        // definitive, so no MANIFEST check is needed.
        storage_refcount_map_.erase(it);
        storage_pending_delete_.push_back(request->path());
        reply->set_deleted(false);
        reply->set_refcount(0);
      } else {
        reply->set_deleted(false);
        reply->set_refcount(e.count);
      }
    }
    std::cout << GetTime() << "[storage] RequestDelete path=" << request->path()
              << " shard=" << request->shard_id()
              << " deleted=" << reply->deleted()
              << " refcount=" << reply->refcount() << std::endl;
    return grpc::Status::OK;
  }
};

// [relink/Storage-CP] Lazy/batched physical delete owned by the CP.
// Every STORAGE_GC_INTERVAL_S (default 60s) the CP drains the refcount-0 paths that
// RequestDelete queued and hdfsDeletes them in one pass. This is the "Lazy GC": instead of
// the CN deleting each obsolete SST inline, the CP accumulates them and reclaims in batch.
// The CP connects to the NameNode EXPLICITLY (a bare path on the default fs would hit the
// LOCAL fs -- fs.defaultFS is unset on this cluster). A failed delete is re-queued, never
// leaked. (A MANIFEST mark-sweep for lost-RPC/crash leaks would slot in here as a later
// backstop; refcount-0 reclamation needs no MANIFEST.) HDFS I/O is done outside storage_latch_.
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
    std::vector<std::string> batch;
    size_t tracked = 0, shared = 0;
    {
      std::lock_guard<std::mutex> lock(storage_latch_);
      batch.swap(storage_pending_delete_);
      tracked = storage_refcount_map_.size();
      for (const auto& kv : storage_refcount_map_) if (kv.second.count > 1) shared++;
    }
    if (!batch.empty() && fs == nullptr) {
      fs = hdfsConnectAsUser(nn_host, nn_port, user);
      if (fs == nullptr)
        std::cout << GetTime() << "[storage-gc] hdfsConnect FAILED " << nn_host << ":" << nn_port
                  << " (retry next cycle)" << std::endl;
    }
    int deleted = 0;
    std::vector<std::string> requeue;
    for (const auto& path : batch) {
      if (fs != nullptr && hdfsDelete(fs, path.c_str(), /*recursive=*/0) == 0) deleted++;
      else requeue.push_back(path);  // transient failure -> retry next cycle, never leak
    }
    if (!requeue.empty()) {
      std::lock_guard<std::mutex> lock(storage_latch_);
      for (auto& p : requeue) storage_pending_delete_.push_back(std::move(p));
    }
    std::cout << GetTime() << "[storage-gc] interval=" << interval << "s tracked=" << tracked
              << " shared=" << shared << " batch=" << batch.size() << " deleted=" << deleted
              << " requeued=" << requeue.size() << std::endl;
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
      compactionservice::CompactionReply compactionReply;
      compactionReply.set_code(1);
      task_reply_map_[task_id] = compactionReply;
      task_args_map_.erase(task_id);
      scheduler_latch_.unlock();
      std::cout << GetTime() << "Fallback compaction task (" << task_id << ")"
                << std::endl;
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
  std::string server_address(compaction_service_options.pro_cp_address);
  ProCPImpl service;
  StorageImpl storage_service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.RegisterService(&storage_service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << GetTime() << "Server listening on " << server_address
            << std::endl;
  std::thread scheduler(ConsumeTask);
  std::thread monitor(UpdateCSAStatus);
  std::thread storage_gc(RunStorageGC);
  storage_gc.detach();
  scheduler.join();
  monitor.join();
  server->Wait();
}