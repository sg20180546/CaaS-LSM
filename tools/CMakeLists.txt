set(CORE_TOOLS
  sst_dump.cc
  ldb.cc)
foreach(src ${CORE_TOOLS})
  get_filename_component(exename ${src} NAME_WE)
  add_executable(${exename}${ARTIFACT_SUFFIX}
    ${src})
  target_link_libraries(${exename}${ARTIFACT_SUFFIX} ${ROCKSDB_LIB})
  list(APPEND core_tool_deps ${exename})
endforeach()

if(WITH_HDFS)
  add_executable(storage_cp_recover${ARTIFACT_SUFFIX} storage_cp_recover.cc)
  target_link_libraries(storage_cp_recover${ARTIFACT_SUFFIX}
    ${ROCKSDB_LIB} compaction_service_grpc_proto)
  list(APPEND core_tool_deps storage_cp_recover)
endif()

if(WITH_TESTS)
  add_executable(storage_ownership_registry_test
    ../db/compaction/remote_compaction/storage_ownership_registry_test.cc)
  target_link_libraries(storage_ownership_registry_test Threads::Threads)
  set_property(TARGET storage_ownership_registry_test PROPERTY CXX_STANDARD 17)
  add_executable(storage_cp_recovery_plan_test
    ../plugin/hdfs/storage_cp_recovery_plan_test.cc)
  add_test(NAME storage_cp_recovery_plan_test COMMAND storage_cp_recovery_plan_test)
endif()

if(WITH_TOOLS)
  set(TOOLS
    db_sanity_test.cc
    write_stress.cc
    db_repl_stress.cc
    dump/rocksdb_dump.cc
    dump/rocksdb_undump.cc)
  foreach(src ${TOOLS})
    get_filename_component(exename ${src} NAME_WE)
    add_executable(${exename}${ARTIFACT_SUFFIX}
      ${src})
    target_link_libraries(${exename}${ARTIFACT_SUFFIX} ${ROCKSDB_LIB} ${THIRDPARTY_LIBS})
    list(APPEND tool_deps ${exename})
  endforeach()

  add_custom_target(ldb_tests
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/ldb_tests.py
    DEPENDS ldb)
endif()
