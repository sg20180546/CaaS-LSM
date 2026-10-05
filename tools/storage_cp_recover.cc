// Offline recovery of SST ownership after an interrupted publication.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "compaction_service.grpc.pb.h"
#include "file/filename.h"
#include "plugin/hdfs/storage_cp_recovery_plan.h"
#include "rocksdb/convenience.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/utilities/options_util.h"

namespace r = ROCKSDB_NAMESPACE;
namespace recovery = rocksdb_storage_recovery;

namespace {
class SilentLogger final : public r::Logger {
 public:
  void Logv(const char*, va_list) override {}
};

struct Snapshot {
  std::string current;
  std::string manifest;
  std::string manifest_name;
};

r::Status ReadSnapshot(r::Env* env, const std::string& db, Snapshot* snap) {
  auto s = r::ReadFileToString(env, r::CurrentFileName(db), &snap->current);
  if (!s.ok()) return s;
  snap->manifest_name = snap->current;
  if (!snap->manifest_name.empty() && snap->manifest_name.back() == '\n') {
    snap->manifest_name.pop_back();
  }
  uint64_t number = 0;
  r::FileType type;
  if (snap->manifest_name.find('/') != std::string::npos ||
      !r::ParseFileName(snap->manifest_name, &number, &type) ||
      type != r::kDescriptorFile) {
    return r::Status::Corruption("invalid CURRENT; refusing ownership recovery");
  }
  return r::ReadFileToString(env, db + "/" + snap->manifest_name, &snap->manifest);
}

int Fail(const std::string& error) {
  std::cerr << "REFUSED: " << error << '\n';
  return 1;
}

void Usage() {
  std::cerr
      << "storage_cp_recover --db /path --fs-uri hdfs://host:port "
         "--cp host:port --shard N --candidates LOCAL_FILE "
         "[--apply --producers-stopped]\n"
         "Default: dry run. Candidates are one absolute .sst path per line.\n"
         "Apply requires all CN/CSA producers and restart controllers to remain "
         "stopped/fenced until completion. This utility does not acquire a lease.\n";
}
}  // namespace

int main(int argc, char** argv) {
  std::string db_path, fs_uri, cp_address, candidates_file;
  uint32_t shard = 0;
  bool have_shard = false, apply = false, producers_stopped = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--apply") { apply = true; continue; }
    if (key == "--producers-stopped") { producers_stopped = true; continue; }
    if (key == "--help") { Usage(); return 0; }
    if (i + 1 == argc) { Usage(); return Fail("missing option value"); }
    const std::string value(argv[++i]);
    if (key == "--db") db_path = value;
    else if (key == "--fs-uri") fs_uri = value;
    else if (key == "--cp") cp_address = value;
    else if (key == "--candidates") candidates_file = value;
    else if (key == "--shard") {
      try {
        size_t end = 0;
        const auto n = std::stoull(value, &end);
        if (end != value.size() || value.empty() || value.front() == '-' ||
            n > std::numeric_limits<uint32_t>::max()) return Fail("invalid shard");
        shard = static_cast<uint32_t>(n);
        have_shard = true;
      } catch (...) { return Fail("invalid shard"); }
    } else { Usage(); return Fail("unknown option: " + key); }
  }
  db_path = recovery::TrimDirectory(db_path);
  if (db_path.empty() || fs_uri.empty() || cp_address.empty() ||
      candidates_file.empty() || !have_shard) { Usage(); return 1; }
  if (fs_uri.compare(0, 7, "hdfs://") != 0) return Fail("--fs-uri must select HDFS");
  if (!recovery::IsCanonicalAbsolute(db_path)) return Fail("DB path must be canonical and absolute");
  if (apply && !producers_stopped) return Fail("--apply requires --producers-stopped");

  std::vector<std::string> candidates;
  std::ifstream input(candidates_file);
  if (!input) return Fail("cannot open candidate file");
  for (std::string line; std::getline(input, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) candidates.push_back(line);
  }
  if (input.bad()) return Fail("cannot read candidate file");

  r::ConfigOptions config;
  r::Env* env = r::Env::Default();
  std::shared_ptr<r::Env> env_guard;
  auto s = r::Env::CreateFromUri(config, "", fs_uri, &env, &env_guard);
  if (!s.ok()) return Fail(s.ToString());
  config.env = env;
  config.ignore_unknown_options = false;
  config.ignore_unsupported_options = false;
  r::DBOptions options;
  std::vector<r::ColumnFamilyDescriptor> columns;
  s = r::LoadLatestOptions(config, db_path, &options, &columns);
  if (!s.ok()) return Fail("cannot load DB options: " + s.ToString());
  options.env = env;
  options.create_if_missing = false;
  options.create_missing_column_families = false;
  options.best_efforts_recovery = false;
  options.paranoid_checks = true;
  options.info_log = std::make_shared<SilentLogger>();

  Snapshot before;
  s = ReadSnapshot(env, db_path, &before);
  if (!s.ok()) return Fail(s.ToString());
  // Read-only Open accepts a subset of column families. An old OPTIONS file
  // must never turn an omitted CF's live SSTs into deletion candidates.
  std::vector<std::string> actual_columns, configured_columns;
  s = r::DB::ListColumnFamilies(options, db_path, &actual_columns);
  if (!s.ok()) return Fail("cannot enumerate all column families: " + s.ToString());
  for (const auto& column : columns) configured_columns.push_back(column.name);
  std::sort(actual_columns.begin(), actual_columns.end());
  std::sort(configured_columns.begin(), configured_columns.end());
  if (actual_columns != configured_columns) {
    return Fail("OPTIONS does not describe every live column family");
  }
  r::DB* raw_db = nullptr;
  std::vector<r::ColumnFamilyHandle*> handles;
  s = r::DB::OpenForReadOnly(options, db_path, columns, &handles, &raw_db);
  if (!s.ok()) return Fail("MANIFEST/WAL recovery failed: " + s.ToString());
  std::unique_ptr<r::DB> db(raw_db);
  std::vector<r::LiveFileMetaData> metadata;
  db->GetLiveFilesMetaData(&metadata);
  for (auto* handle : handles) db->DestroyColumnFamilyHandle(handle);
  db.reset();

  std::vector<std::string> live, directories{db_path};
  for (const auto& path : options.db_paths) directories.push_back(path.path);
  for (const auto& column : columns) {
    for (const auto& path : column.options.cf_paths) directories.push_back(path.path);
  }
  for (const auto& file : metadata) {
    live.push_back(file.external_path.empty()
                       ? file.directory + "/" + file.relative_filename
                       : file.external_path);
  }
  recovery::Plan plan;
  auto error = recovery::BuildPlan(live, candidates, directories, &plan);
  if (!error.empty()) return Fail(error);
  const auto check_snapshot = [&]() -> std::string {
    Snapshot now;
    const auto status = ReadSnapshot(env, db_path, &now);
    if (!status.ok()) return status.ToString();
    if (now.current != before.current || now.manifest != before.manifest) {
      return "CURRENT/MANIFEST changed; keep files and fence all producers first";
    }
    return {};
  };
  error = check_snapshot();
  if (!error.empty()) return Fail(error);
  for (const auto& path : plan.keep) std::cout << "KEEP " << path << '\n';
  for (const auto& path : plan.discard) std::cout << "DISCARD_OWNER " << path << '\n';
  std::cout << "live=" << plan.live.size() << " keep=" << plan.keep.size()
            << " discard=" << plan.discard.size() << " mode="
            << (apply ? "apply" : "dry-run") << '\n';
  if (!apply) return 0;

  auto stub = compactionservice::StorageService::NewStub(
      grpc::CreateChannel(cp_address, grpc::InsecureChannelCredentials()));
  const auto send = [&](const std::vector<std::string>& paths,
                        bool recover) -> std::string {
    constexpr size_t kBatchSize = 512;
    for (size_t i = 0; i < paths.size(); i += kBatchSize) {
      compactionservice::FileRefBatch request;
      request.set_shard_id(shard);
      for (size_t j = i; j < std::min(paths.size(), i + kBatchSize); ++j) {
        request.add_path(paths[j]);
      }
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
      google::protobuf::Empty response;
      const auto status = recover
          ? stub->RecoverReferences(&context, request, &response)
          : stub->AbortUnpublished(&context, request, &response);
      if (!status.ok()) return status.error_message();
    }
    return {};
  };
  error = recovery::ApplyPlan(plan, producers_stopped, check_snapshot,
      [&](const std::vector<std::string>& paths) { return send(paths, true); },
      [&](const std::vector<std::string>& paths) { return send(paths, false); });
  if (!error.empty()) return Fail(error + "; safe to rerun after resolving the cause");
  std::cout << "DONE: CP acknowledged discards; physical deletion belongs to CP GC.\n";
  return 0;
}
