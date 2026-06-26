#include "compaction_service.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "compaction_service.grpc.pb.h"
#include "rocksdb/db.h"

class ProCPClient {
 public:
  ProCPClient(const std::shared_ptr<grpc::Channel>& channel)
      : stub_(compactionservice::ProCPService::NewStub(channel)) {}

  ROCKSDB_NAMESPACE::Status OpenAndCompact(
      const ROCKSDB_NAMESPACE::OpenAndCompactOptions& options,
      const std::string& name, const std::string& output_directory,
      const std::string& input, std::string* output,
      const ROCKSDB_NAMESPACE::CompactionServiceOptionsOverride&
          override_options,
      compactionservice::CompactionAdditionInfo* compaction_addition_info,
      uint64_t trigger_ms, uint64_t* compact_process_latency,
      uint64_t* open_db_latency) {
    compactionservice::CompactionArgs compaction_args;
    compactionservice::CompactionReply compaction_reply;
    compaction_args.set_name(name);
    compaction_args.set_output_directory(output_directory);
    compaction_args.set_input(input);
    compaction_args.set_trigger_ms(trigger_ms);
    compactionservice::TaskId taskId;
    grpc::ClientContext context;
    compactionservice::AddTaskArgs add_task_args;
    add_task_args.mutable_compaction_args()->CopyFrom(compaction_args);
    add_task_args.mutable_compaction_addition_info()->CopyFrom(
        *compaction_addition_info);
    grpc::Status status = stub_->AddTask(&context, add_task_args, &taskId);
    if (!status.ok()) {
      return ROCKSDB_NAMESPACE::Status::IOError(status.error_message());
    }
    // [F2a, 2026-06-26] Bound this otherwise-deadline-less poll. code()==99 means "no terminal result
    // yet". If a remote task never produces one (e.g. ProCP infinitely re-queues a persistently-failing
    // compaction), this loop would park the CN compaction thread FOREVER (the migration_mechanism src
    // freeze: num-running-compactions pinned, 0-byte progress). After max_poll intervals give up ->
    // Aborted -> WaitForCompleteV2 returns kUseLocal -> the CN runs the compaction LOCALLY (thread freed).
    int poll_count = 0;
    const int max_poll =
        options.check_time_interval > 0 ? 600 / options.check_time_interval : 600;
    do {
      grpc::ClientContext check_task_context;
      sleep(options.check_time_interval);
      status = stub_->CheckTask(&check_task_context, taskId, &compaction_reply);
      if (!status.ok()) {
        return ROCKSDB_NAMESPACE::Status::IOError(status.error_message());
      }
    } while (compaction_reply.code() == 99 && ++poll_count < max_poll);
    if (compaction_reply.code() == 99) {
      return ROCKSDB_NAMESPACE::Status::Aborted("remote compaction poll timed out");
    }
    if (compaction_reply.code() != 0) {
      return ROCKSDB_NAMESPACE::Status::Aborted();
    }
    output->assign(compaction_reply.result());
    if (status.ok()) {
      *compact_process_latency = compaction_reply.process_latency();
      *open_db_latency = compaction_reply.open_db_latency();
      return ROCKSDB_NAMESPACE::Status::OK();
    } else {
      return ROCKSDB_NAMESPACE::Status::IOError(status.error_message());
    }
  };

 private:
  std::unique_ptr<compactionservice::ProCPService::Stub> stub_;
};

namespace ROCKSDB_NAMESPACE {
CompactionServiceJobStatus MyTestCompactionService::StartV2(
    const CompactionServiceJobInfo& info,
    const std::string& compaction_service_input) {
  InstrumentedMutexLock l(&mutex_);
  start_info_ = info;
  assert(info.db_name == db_path_);
  jobs_.emplace(info.job_id, compaction_service_input);
  CompactionServiceJobStatus s = CompactionServiceJobStatus::kSuccess;
  if (is_override_start_status_) {
    return override_start_status_;
  }
  return s;
}

CompactionServiceJobStatus MyTestCompactionService::WaitForCompleteV2(
    const CompactionServiceJobInfo& info,
    CompactionAdditionInfo* compaction_addition_info,
    std::string* compaction_service_result) {
  std::string compaction_input;
  assert(info.db_name == db_path_);
  {
    InstrumentedMutexLock l(&mutex_);
    wait_info_ = info;
    auto i = jobs_.find(info.job_id);
    if (i == jobs_.end()) {
      return CompactionServiceJobStatus::kFailure;
    }
    compaction_input = std::move(i->second);
    jobs_.erase(i);
  }

  if (is_override_wait_status_) {
    return override_wait_status_;
  }

  CompactionServiceOptionsOverride options_override;
  options_override.env = options_.env;
  options_override.file_checksum_gen_factory =
      options_.file_checksum_gen_factory;
  options_override.comparator = options_.comparator;
  options_override.merge_operator = options_.merge_operator;
  options_override.compaction_filter = options_.compaction_filter;
  options_override.compaction_filter_factory =
      options_.compaction_filter_factory;
  options_override.prefix_extractor = options_.prefix_extractor;
  options_override.table_factory = options_.table_factory;
  options_override.sst_partitioner_factory = options_.sst_partitioner_factory;
  options_override.statistics = statistics_;
  if (!listeners_.empty()) {
    options_override.listeners = listeners_;
  }

  if (!table_properties_collector_factories_.empty()) {
    options_override.table_properties_collector_factories =
        table_properties_collector_factories_;
  }

  OpenAndCompactOptions options;
  options.canceled = &canceled_;

  //  Status s = DB::OpenAndCompact(
  //      options, db_path_, db_path_ + "/" + std::to_string(info.job_id),
  //      compaction_input, compaction_service_result, options_override);
  ProCPClient proCp_client(grpc::CreateChannel(
      options.pro_cp_address, grpc::InsecureChannelCredentials()));
  compactionservice::CompactionAdditionInfo addition_info;

  addition_info.set_score(compaction_addition_info->score);
  addition_info.set_num_entries(compaction_addition_info->num_entries);
  addition_info.set_num_deletions(compaction_addition_info->num_deletions);
  addition_info.set_compensated_file_size(
      compaction_addition_info->compensated_file_size);
  addition_info.set_output_level(compaction_addition_info->output_level);
  addition_info.set_start_level(compaction_addition_info->start_level);
  uint64_t compact_process_latency;
  uint64_t open_db_latency;
  Status s = proCp_client.OpenAndCompact(
      options, db_path_, db_path_ + "/" + std::to_string(info.job_id),
      compaction_input, compaction_service_result, options_override,
      &addition_info, compaction_addition_info->trigger_ms,
      &compact_process_latency, &open_db_latency);
  if (is_override_wait_result_) {
    *compaction_service_result = override_wait_result_;
  }
  compaction_num_.fetch_add(1);
  if (s.ok()) {
    compaction_addition_info->trigger_ms = compact_process_latency;
    compaction_addition_info->num_entries = open_db_latency;
    return CompactionServiceJobStatus::kSuccess;
  } else {
    return CompactionServiceJobStatus::kUseLocal;
  }
}

// [relink] Public factory: lets an external app (no protobuf deps) enable CaaS remote compaction.
std::shared_ptr<CompactionService> NewMyTestCompactionService(
    const std::string& db_path, const Options& options,
    std::shared_ptr<Statistics> statistics) {
  Options opts = options;
  std::shared_ptr<Statistics> stats = std::move(statistics);
  std::vector<std::shared_ptr<EventListener>> listeners;
  std::vector<std::shared_ptr<TablePropertiesCollectorFactory>> tpcf;
  return std::make_shared<MyTestCompactionService>(db_path, opts, stats, listeners, tpcf);
}
};  // namespace ROCKSDB_NAMESPACE
