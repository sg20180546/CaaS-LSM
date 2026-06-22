// [relink] Public factory for the CaaS remote-compaction service. Lets an external app
// (e.g. migration_mechansim_exp/serverclient, no gRPC/protobuf deps) enable remote compaction
// offload (ProCP@53:8020 -> CSA@87/88:8010) without pulling in the internal protobuf headers.
//   Options opt; opt.compaction_service = NewMyTestCompactionService(db_path, opt, opt.statistics);
#pragma once

#include <memory>
#include <string>

#include "rocksdb/options.h"

namespace ROCKSDB_NAMESPACE {

class CompactionService;
class Statistics;

extern std::shared_ptr<CompactionService> NewMyTestCompactionService(
    const std::string& db_path, const Options& options,
    std::shared_ptr<Statistics> statistics);

}  // namespace ROCKSDB_NAMESPACE
