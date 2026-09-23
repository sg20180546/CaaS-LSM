//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/lru_cache.h"

#include <array>
#include <atomic>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/cache_key.h"
#include "cache/clock_cache.h"
#include "cache/fast_lru_cache.h"
#include "db/db_test_util.h"
#include "file/sst_file_manager_impl.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/cache.h"
#include "rocksdb/io_status.h"
#include "rocksdb/sst_file_manager.h"
#include "rocksdb/utilities/cache_dump_load.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/random.h"
#include "utilities/cache_dump_load_impl.h"
#include "utilities/fault_injection_fs.h"
#include "utilities/memory_allocators.h"

namespace ROCKSDB_NAMESPACE {

namespace {

void WarmupIncrementDeleter(const Slice& /*key*/, void* value) {
  ++*static_cast<int*>(value);
}

}  // namespace

class LRUCacheTest : public testing::Test {
 public:
  LRUCacheTest() {}
  ~LRUCacheTest() override { DeleteCache(); }

  void DeleteCache() {
    if (cache_ != nullptr) {
      cache_->~LRUCacheShard();
      port::cacheline_aligned_free(cache_);
      cache_ = nullptr;
    }
  }

  void NewCache(size_t capacity, double high_pri_pool_ratio = 0.0,
                double low_pri_pool_ratio = 1.0,
                bool use_adaptive_mutex = kDefaultToAdaptiveMutex) {
    DeleteCache();
    cache_ = reinterpret_cast<LRUCacheShard*>(
        port::cacheline_aligned_alloc(sizeof(LRUCacheShard)));
    new (cache_) LRUCacheShard(capacity, /*strict_capacity_limit=*/false,
                               high_pri_pool_ratio, low_pri_pool_ratio,
                               use_adaptive_mutex, kDontChargeCacheMetadata,
                               /*max_upper_hash_bits=*/24,
                               /*secondary_cache=*/nullptr);
  }

  void Insert(const std::string& key,
              Cache::Priority priority = Cache::Priority::LOW) {
    EXPECT_OK(cache_->Insert(key, 0 /*hash*/, nullptr /*value*/, 1 /*charge*/,
                             nullptr /*deleter*/, nullptr /*handle*/,
                             priority));
  }

  void Insert(char key, Cache::Priority priority = Cache::Priority::LOW) {
    Insert(std::string(1, key), priority);
  }

  bool Lookup(const std::string& key) {
    auto handle = cache_->Lookup(key, 0 /*hash*/);
    if (handle) {
      cache_->Release(handle, true /*useful*/, false /*erase*/);
      return true;
    }
    return false;
  }

  bool Lookup(char key) { return Lookup(std::string(1, key)); }

  void Erase(const std::string& key) { cache_->Erase(key, 0 /*hash*/); }

  void ValidateLRUList(std::vector<std::string> keys,
                       size_t num_high_pri_pool_keys = 0,
                       size_t num_low_pri_pool_keys = 0,
                       size_t num_bottom_pri_pool_keys = 0) {
    LRUHandle* lru;
    LRUHandle* lru_low_pri;
    LRUHandle* lru_bottom_pri;
    cache_->TEST_GetLRUList(&lru, &lru_low_pri, &lru_bottom_pri);

    LRUHandle* iter = lru;

    bool in_low_pri_pool = false;
    bool in_high_pri_pool = false;

    size_t high_pri_pool_keys = 0;
    size_t low_pri_pool_keys = 0;
    size_t bottom_pri_pool_keys = 0;

    if (iter == lru_bottom_pri) {
      in_low_pri_pool = true;
      in_high_pri_pool = false;
    }
    if (iter == lru_low_pri) {
      in_low_pri_pool = false;
      in_high_pri_pool = true;
    }

    for (const auto& key : keys) {
      iter = iter->next;
      ASSERT_NE(lru, iter);
      ASSERT_EQ(key, iter->key().ToString());
      ASSERT_EQ(in_high_pri_pool, iter->InHighPriPool());
      ASSERT_EQ(in_low_pri_pool, iter->InLowPriPool());
      if (in_high_pri_pool) {
        ASSERT_FALSE(iter->InLowPriPool());
        high_pri_pool_keys++;
      } else if (in_low_pri_pool) {
        ASSERT_FALSE(iter->InHighPriPool());
        low_pri_pool_keys++;
      } else {
        bottom_pri_pool_keys++;
      }
      if (iter == lru_bottom_pri) {
        ASSERT_FALSE(in_low_pri_pool);
        ASSERT_FALSE(in_high_pri_pool);
        in_low_pri_pool = true;
        in_high_pri_pool = false;
      }
      if (iter == lru_low_pri) {
        ASSERT_TRUE(in_low_pri_pool);
        ASSERT_FALSE(in_high_pri_pool);
        in_low_pri_pool = false;
        in_high_pri_pool = true;
      }
    }
    ASSERT_EQ(lru, iter->next);
    ASSERT_FALSE(in_low_pri_pool);
    ASSERT_TRUE(in_high_pri_pool);
    ASSERT_EQ(num_high_pri_pool_keys, high_pri_pool_keys);
    ASSERT_EQ(num_low_pri_pool_keys, low_pri_pool_keys);
    ASSERT_EQ(num_bottom_pri_pool_keys, bottom_pri_pool_keys);
  }

 protected:
  LRUCacheShard* cache_ = nullptr;
};

TEST_F(LRUCacheTest, BasicLRU) {
  NewCache(5);
  for (char ch = 'a'; ch <= 'e'; ch++) {
    Insert(ch);
  }
  ValidateLRUList({"a", "b", "c", "d", "e"}, 0, 5);
  for (char ch = 'x'; ch <= 'z'; ch++) {
    Insert(ch);
  }
  ValidateLRUList({"d", "e", "x", "y", "z"}, 0, 5);
  ASSERT_FALSE(Lookup("b"));
  ValidateLRUList({"d", "e", "x", "y", "z"}, 0, 5);
  ASSERT_TRUE(Lookup("e"));
  ValidateLRUList({"d", "x", "y", "z", "e"}, 0, 5);
  ASSERT_TRUE(Lookup("z"));
  ValidateLRUList({"d", "x", "y", "e", "z"}, 0, 5);
  Erase("x");
  ValidateLRUList({"d", "y", "e", "z"}, 0, 4);
  ASSERT_TRUE(Lookup("d"));
  ValidateLRUList({"y", "e", "z", "d"}, 0, 4);
  Insert("u");
  ValidateLRUList({"y", "e", "z", "d", "u"}, 0, 5);
  Insert("v");
  ValidateLRUList({"e", "z", "d", "u", "v"}, 0, 5);
}

TEST_F(LRUCacheTest, LowPriorityMidpointInsertion) {
  // Allocate 2 cache entries to high-pri pool and 3 to low-pri pool.
  NewCache(5, /* high_pri_pool_ratio */ 0.40, /* low_pri_pool_ratio */ 0.60);

  Insert("a", Cache::Priority::LOW);
  Insert("b", Cache::Priority::LOW);
  Insert("c", Cache::Priority::LOW);
  Insert("x", Cache::Priority::HIGH);
  Insert("y", Cache::Priority::HIGH);
  ValidateLRUList({"a", "b", "c", "x", "y"}, 2, 3);

  // Low-pri entries inserted to the tail of low-pri list (the midpoint).
  // After lookup, it will move to the tail of the full list.
  Insert("d", Cache::Priority::LOW);
  ValidateLRUList({"b", "c", "d", "x", "y"}, 2, 3);
  ASSERT_TRUE(Lookup("d"));
  ValidateLRUList({"b", "c", "x", "y", "d"}, 2, 3);

  // High-pri entries will be inserted to the tail of full list.
  Insert("z", Cache::Priority::HIGH);
  ValidateLRUList({"c", "x", "y", "d", "z"}, 2, 3);
}

TEST_F(LRUCacheTest, BottomPriorityMidpointInsertion) {
  // Allocate 2 cache entries to high-pri pool and 2 to low-pri pool.
  NewCache(6, /* high_pri_pool_ratio */ 0.35, /* low_pri_pool_ratio */ 0.35);

  Insert("a", Cache::Priority::BOTTOM);
  Insert("b", Cache::Priority::BOTTOM);
  Insert("i", Cache::Priority::LOW);
  Insert("j", Cache::Priority::LOW);
  Insert("x", Cache::Priority::HIGH);
  Insert("y", Cache::Priority::HIGH);
  ValidateLRUList({"a", "b", "i", "j", "x", "y"}, 2, 2, 2);

  // Low-pri entries will be inserted to the tail of low-pri list (the
  // midpoint). After lookup, 'k' will move to the tail of the full list, and
  // 'x' will spill over to the low-pri pool.
  Insert("k", Cache::Priority::LOW);
  ValidateLRUList({"b", "i", "j", "k", "x", "y"}, 2, 2, 2);
  ASSERT_TRUE(Lookup("k"));
  ValidateLRUList({"b", "i", "j", "x", "y", "k"}, 2, 2, 2);

  // High-pri entries will be inserted to the tail of full list. Although y was
  // inserted with high priority, it got spilled over to the low-pri pool. As
  // a result, j also got spilled over to the bottom-pri pool.
  Insert("z", Cache::Priority::HIGH);
  ValidateLRUList({"i", "j", "x", "y", "k", "z"}, 2, 2, 2);
  Erase("x");
  ValidateLRUList({"i", "j", "y", "k", "z"}, 2, 1, 2);
  Erase("y");
  ValidateLRUList({"i", "j", "k", "z"}, 2, 0, 2);

  // Bottom-pri entries will be inserted to the tail of bottom-pri list.
  Insert("c", Cache::Priority::BOTTOM);
  ValidateLRUList({"i", "j", "c", "k", "z"}, 2, 0, 3);
  Insert("d", Cache::Priority::BOTTOM);
  ValidateLRUList({"i", "j", "c", "d", "k", "z"}, 2, 0, 4);
  Insert("e", Cache::Priority::BOTTOM);
  ValidateLRUList({"j", "c", "d", "e", "k", "z"}, 2, 0, 4);

  // Low-pri entries will be inserted to the tail of low-pri list (the
  // midpoint).
  Insert("l", Cache::Priority::LOW);
  ValidateLRUList({"c", "d", "e", "l", "k", "z"}, 2, 1, 3);
  Insert("m", Cache::Priority::LOW);
  ValidateLRUList({"d", "e", "l", "m", "k", "z"}, 2, 2, 2);

  Erase("k");
  ValidateLRUList({"d", "e", "l", "m", "z"}, 1, 2, 2);
  Erase("z");
  ValidateLRUList({"d", "e", "l", "m"}, 0, 2, 2);

  // Bottom-pri entries will be inserted to the tail of bottom-pri list.
  Insert("f", Cache::Priority::BOTTOM);
  ValidateLRUList({"d", "e", "f", "l", "m"}, 0, 2, 3);
  Insert("g", Cache::Priority::BOTTOM);
  ValidateLRUList({"d", "e", "f", "g", "l", "m"}, 0, 2, 4);

  // High-pri entries will be inserted to the tail of full list.
  Insert("o", Cache::Priority::HIGH);
  ValidateLRUList({"e", "f", "g", "l", "m", "o"}, 1, 2, 3);
  Insert("p", Cache::Priority::HIGH);
  ValidateLRUList({"f", "g", "l", "m", "o", "p"}, 2, 2, 2);
}

TEST_F(LRUCacheTest, EntriesWithPriority) {
  // Allocate 2 cache entries to high-pri pool and 2 to low-pri pool.
  NewCache(6, /* high_pri_pool_ratio */ 0.35, /* low_pri_pool_ratio */ 0.35);

  Insert("a", Cache::Priority::LOW);
  Insert("b", Cache::Priority::LOW);
  ValidateLRUList({"a", "b"}, 0, 2, 0);
  // Low-pri entries can overflow to bottom-pri pool.
  Insert("c", Cache::Priority::LOW);
  ValidateLRUList({"a", "b", "c"}, 0, 2, 1);

  // Bottom-pri entries can take high-pri pool capacity if available
  Insert("t", Cache::Priority::LOW);
  Insert("u", Cache::Priority::LOW);
  ValidateLRUList({"a", "b", "c", "t", "u"}, 0, 2, 3);
  Insert("v", Cache::Priority::LOW);
  ValidateLRUList({"a", "b", "c", "t", "u", "v"}, 0, 2, 4);
  Insert("w", Cache::Priority::LOW);
  ValidateLRUList({"b", "c", "t", "u", "v", "w"}, 0, 2, 4);

  Insert("X", Cache::Priority::HIGH);
  Insert("Y", Cache::Priority::HIGH);
  ValidateLRUList({"t", "u", "v", "w", "X", "Y"}, 2, 2, 2);

  // After lookup, the high-pri entry 'X' got spilled over to the low-pri pool.
  // The low-pri entry 'v' got spilled over to the bottom-pri pool.
  Insert("Z", Cache::Priority::HIGH);
  ValidateLRUList({"u", "v", "w", "X", "Y", "Z"}, 2, 2, 2);

  // Low-pri entries will be inserted to head of low-pri pool.
  Insert("a", Cache::Priority::LOW);
  ValidateLRUList({"v", "w", "X", "a", "Y", "Z"}, 2, 2, 2);

  // After lookup, the high-pri entry 'Y' got spilled over to the low-pri pool.
  // The low-pri entry 'X' got spilled over to the bottom-pri pool.
  ASSERT_TRUE(Lookup("v"));
  ValidateLRUList({"w", "X", "a", "Y", "Z", "v"}, 2, 2, 2);

  // After lookup, the high-pri entry 'Z' got spilled over to the low-pri pool.
  // The low-pri entry 'a' got spilled over to the bottom-pri pool.
  ASSERT_TRUE(Lookup("X"));
  ValidateLRUList({"w", "a", "Y", "Z", "v", "X"}, 2, 2, 2);

  // After lookup, the low pri entry 'Z' got promoted back to high-pri pool. The
  // high-pri entry 'v' got spilled over to the low-pri pool.
  ASSERT_TRUE(Lookup("Z"));
  ValidateLRUList({"w", "a", "Y", "v", "X", "Z"}, 2, 2, 2);

  Erase("Y");
  ValidateLRUList({"w", "a", "v", "X", "Z"}, 2, 1, 2);
  Erase("X");
  ValidateLRUList({"w", "a", "v", "Z"}, 1, 1, 2);

  Insert("d", Cache::Priority::LOW);
  Insert("e", Cache::Priority::LOW);
  ValidateLRUList({"w", "a", "v", "d", "e", "Z"}, 1, 2, 3);

  Insert("f", Cache::Priority::LOW);
  Insert("g", Cache::Priority::LOW);
  ValidateLRUList({"v", "d", "e", "f", "g", "Z"}, 1, 2, 3);
  ASSERT_TRUE(Lookup("d"));
  ValidateLRUList({"v", "e", "f", "g", "Z", "d"}, 2, 2, 2);

  // Erase some entries.
  Erase("e");
  Erase("f");
  Erase("Z");
  ValidateLRUList({"v", "g", "d"}, 1, 1, 1);

  // Bottom-pri entries can take low- and high-pri pool capacity if available
  Insert("o", Cache::Priority::BOTTOM);
  ValidateLRUList({"v", "o", "g", "d"}, 1, 1, 2);
  Insert("p", Cache::Priority::BOTTOM);
  ValidateLRUList({"v", "o", "p", "g", "d"}, 1, 1, 3);
  Insert("q", Cache::Priority::BOTTOM);
  ValidateLRUList({"v", "o", "p", "q", "g", "d"}, 1, 1, 4);

  // High-pri entries can overflow to low-pri pool, and bottom-pri entries will
  // be evicted.
  Insert("x", Cache::Priority::HIGH);
  ValidateLRUList({"o", "p", "q", "g", "d", "x"}, 2, 1, 3);
  Insert("y", Cache::Priority::HIGH);
  ValidateLRUList({"p", "q", "g", "d", "x", "y"}, 2, 2, 2);
  Insert("z", Cache::Priority::HIGH);
  ValidateLRUList({"q", "g", "d", "x", "y", "z"}, 2, 2, 2);

  // 'g' is bottom-pri before this lookup, it will be inserted to head of
  // high-pri pool after lookup.
  ASSERT_TRUE(Lookup("g"));
  ValidateLRUList({"q", "d", "x", "y", "z", "g"}, 2, 2, 2);

  // High-pri entries will be inserted to head of high-pri pool after lookup.
  ASSERT_TRUE(Lookup("z"));
  ValidateLRUList({"q", "d", "x", "y", "g", "z"}, 2, 2, 2);

  // Bottom-pri entries will be inserted to head of high-pri pool after lookup.
  ASSERT_TRUE(Lookup("d"));
  ValidateLRUList({"q", "x", "y", "g", "z", "d"}, 2, 2, 2);

  // Bottom-pri entries will be inserted to the tail of bottom-pri list.
  Insert("m", Cache::Priority::BOTTOM);
  ValidateLRUList({"x", "m", "y", "g", "z", "d"}, 2, 2, 2);

  // Bottom-pri entries will be inserted to head of high-pri pool after lookup.
  ASSERT_TRUE(Lookup("m"));
  ValidateLRUList({"x", "y", "g", "z", "d", "m"}, 2, 2, 2);
}

TEST_F(LRUCacheTest, CacheWarmupMetadataAndLookupDoNotRecordHit) {
  NewCache(3, /*high_pri_pool_ratio=*/0.34,
           /*low_pri_pool_ratio=*/0.34);
  Insert("bottom", Cache::Priority::BOTTOM);
  Insert("low", Cache::Priority::LOW);
  Insert("high", Cache::Priority::HIGH);

  std::map<std::string, Cache::Priority> priorities;
  size_t state = 0;
  do {
    cache_->ApplyToSomeEntriesForCacheWarmup(
        [&](const Slice& key, size_t charge, Cache::DeleterFn deleter,
            Cache::Priority priority) {
          EXPECT_EQ(1U, charge);
          EXPECT_EQ(nullptr, deleter);
          priorities[key.ToString()] = priority;
        },
        /*average_entries_per_lock=*/1, &state);
  } while (state != SIZE_MAX);
  ASSERT_EQ(3U, priorities.size());
  EXPECT_EQ(Cache::Priority::BOTTOM, priorities["bottom"]);
  EXPECT_EQ(Cache::Priority::LOW, priorities["low"]);
  EXPECT_EQ(Cache::Priority::HIGH, priorities["high"]);

  Cache::Priority priority = Cache::Priority::HIGH;
  LRUHandle* handle =
      cache_->LookupForCacheWarmup("bottom", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, handle);
  EXPECT_EQ(Cache::Priority::BOTTOM, priority);
  EXPECT_FALSE(handle->HasHit());
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(handle, priority));

  // If the warmup lookup had called SetHit(), "bottom" would have been
  // promoted and there would be no BOTTOM victim for this LOW admission.
  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmup("incoming-low", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_FALSE(Lookup("bottom"));
  EXPECT_TRUE(Lookup("low"));
  EXPECT_TRUE(Lookup("high"));
  EXPECT_TRUE(Lookup("incoming-low"));
}

TEST_F(LRUCacheTest, CacheWarmupReleasePreservesDemotedClass) {
  NewCache(3, /*high_pri_pool_ratio=*/0.34,
           /*low_pri_pool_ratio=*/0.34);
  Insert("high-a", Cache::Priority::HIGH);
  Insert("high-b", Cache::Priority::HIGH);
  Insert("high-c", Cache::Priority::HIGH);

  // Pool limits spill the oldest HIGH-tagged entry all the way to BOTTOM.
  // A normal Release() would look at its immutable HIGH tag and promote it.
  Cache::Priority priority = Cache::Priority::HIGH;
  LRUHandle* handle =
      cache_->LookupForCacheWarmup("high-a", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, handle);
  ASSERT_EQ(Cache::Priority::BOTTOM, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(handle, priority));

  std::map<std::string, Cache::Priority> priorities;
  size_t state = 0;
  do {
    cache_->ApplyToSomeEntriesForCacheWarmup(
        [&](const Slice& key, size_t /*charge*/, Cache::DeleterFn /*deleter*/,
            Cache::Priority effective_priority) {
          priorities[key.ToString()] = effective_priority;
        },
        /*average_entries_per_lock=*/1, &state);
  } while (state != SIZE_MAX);
  EXPECT_EQ(Cache::Priority::BOTTOM, priorities["high-a"]);
}

TEST_F(LRUCacheTest, CacheWarmupLeasePreservesExactRecency) {
  NewCache(3, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/1.0);
  Insert("a");
  Insert("b");
  Insert("c");
  ValidateLRUList({"a", "b", "c"}, 0, 3);

  Cache::Priority priority;
  LRUHandle* middle = cache_->LookupForCacheWarmup("b", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, middle);
  EXPECT_EQ(1U, cache_->GetPinnedUsage());
  ValidateLRUList({"a", "b", "c"}, 0, 3);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(middle, priority));
  EXPECT_EQ(0U, cache_->GetPinnedUsage());
  ValidateLRUList({"a", "b", "c"}, 0, 3);

  // A leased oldest entry is pinned, so normal admission evicts the next
  // oldest entry without moving the lease. Once released, it is still oldest.
  LRUHandle* oldest = cache_->LookupForCacheWarmup("a", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, oldest);
  Insert("d");
  ValidateLRUList({"a", "c", "d"}, 0, 3);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(oldest, priority));
  ValidateLRUList({"a", "c", "d"}, 0, 3);
  Insert("e");
  ValidateLRUList({"c", "d", "e"}, 0, 3);
}

TEST_F(LRUCacheTest, CacheWarmupReleaseHonorsConcurrentRealHit) {
  NewCache(3, /*high_pri_pool_ratio=*/0.34,
           /*low_pri_pool_ratio=*/0.34);
  Insert("high-a", Cache::Priority::HIGH);
  Insert("high-b", Cache::Priority::HIGH);
  Insert("high-c", Cache::Priority::HIGH);

  Cache::Priority priority = Cache::Priority::HIGH;
  LRUHandle* warmup =
      cache_->LookupForCacheWarmup("high-a", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, warmup);
  ASSERT_EQ(Cache::Priority::BOTTOM, priority);

  // A foreground lookup during the lease is a real touch and must retain its
  // normal HIGH promotion even when the warmup reference happens to be last.
  LRUHandle* foreground = cache_->Lookup("high-a", 0 /*hash*/);
  ASSERT_NE(nullptr, foreground);
  EXPECT_FALSE(cache_->Release(foreground, true /*useful*/, false /*erase*/));
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));

  Cache::Priority after = Cache::Priority::BOTTOM;
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("high-a", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));
}

TEST_F(LRUCacheTest, CacheWarmupReleaseBeforeConcurrentRealHit) {
  NewCache(3, /*high_pri_pool_ratio=*/0.34,
           /*low_pri_pool_ratio=*/0.34);
  Insert("high-a", Cache::Priority::HIGH);
  Insert("high-b", Cache::Priority::HIGH);
  Insert("high-c", Cache::Priority::HIGH);

  Cache::Priority priority;
  LRUHandle* warmup =
      cache_->LookupForCacheWarmup("high-a", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, warmup);
  ASSERT_EQ(Cache::Priority::BOTTOM, priority);
  LRUHandle* foreground = cache_->Lookup("high-a", 0 /*hash*/);
  ASSERT_NE(nullptr, foreground);

  // The opposite release order must retain the foreground hit as well.
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));
  EXPECT_FALSE(cache_->Release(foreground, true /*useful*/, false /*erase*/));

  Cache::Priority after;
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("high-a", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));
}

TEST_F(LRUCacheTest, CacheWarmupConcurrentHitReleaseOrderStress) {
  NewCache(8, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  Insert("key", Cache::Priority::BOTTOM);

  for (int i = 0; i < 500; ++i) {
    Cache::Priority priority;
    LRUHandle* warmup =
        cache_->LookupForCacheWarmup("key", 0 /*hash*/, &priority);
    ASSERT_NE(nullptr, warmup);

    std::atomic<bool> acquired{false};
    std::atomic<bool> release_foreground{false};
    std::atomic<bool> lookup_failed{false};
    std::thread foreground_thread([&] {
      LRUHandle* foreground = cache_->Lookup("key", 0 /*hash*/);
      if (foreground == nullptr) {
        lookup_failed.store(true, std::memory_order_relaxed);
        acquired.store(true, std::memory_order_release);
        return;
      }
      acquired.store(true, std::memory_order_release);
      while (!release_foreground.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      cache_->Release(foreground, true /*useful*/, false /*erase*/);
    });
    while (!acquired.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (lookup_failed.load(std::memory_order_relaxed)) {
      release_foreground.store(true, std::memory_order_release);
      foreground_thread.join();
      cache_->ReleaseForCacheWarmup(warmup, priority);
      FAIL() << "foreground lookup unexpectedly missed";
    }

    if ((i & 1) == 0) {
      release_foreground.store(true, std::memory_order_release);
      foreground_thread.join();
      EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));
    } else {
      EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));
      release_foreground.store(true, std::memory_order_release);
      foreground_thread.join();
    }
  }

  Cache::Priority priority;
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("key", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, priority));
}

TEST_F(LRUCacheTest, CacheWarmupLeasesNormallyPinnedEntryBothReleaseOrders) {
  NewCache(8, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int first_deleted = 0;
  int second_deleted = 0;

  // Model TableCache's resident table_reader_handle: insertion itself returns
  // a long-lived normal pin without recording a cache hit.
  LRUHandle* first_normal = nullptr;
  ASSERT_OK(cache_->Insert("first", 0 /*hash*/, &first_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, &first_normal,
                           Cache::Priority::LOW));
  ASSERT_NE(nullptr, first_normal);
  ASSERT_EQ(1U, cache_->GetPinnedUsage());

  Cache::Priority first_priority = Cache::Priority::BOTTOM;
  LRUHandle* first_warmup =
      cache_->LookupForCacheWarmup("first", 0 /*hash*/, &first_priority);
  ASSERT_NE(nullptr, first_warmup);
  EXPECT_EQ(Cache::Priority::LOW, first_priority);
  EXPECT_EQ(&first_deleted, first_warmup->value);
  // References, rather than bytes, increased, so pinned usage stays exact.
  EXPECT_EQ(1U, cache_->GetPinnedUsage());
  Cache::Priority ignored;
  EXPECT_EQ(nullptr,
            cache_->LookupForCacheWarmup("first", 0 /*hash*/, &ignored));

  // Normal pin released first; the warmup release becomes the last release and
  // must restore LOW without manufacturing a hit.
  EXPECT_FALSE(cache_->Release(first_normal, true /*useful*/, false /*erase*/));
  EXPECT_EQ(1U, cache_->GetPinnedUsage());
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(first_warmup, first_priority));
  EXPECT_EQ(0U, cache_->GetPinnedUsage());
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("first", 0 /*hash*/, &ignored);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::LOW, ignored);
  EXPECT_EQ(&first_deleted, verify->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, ignored));

  LRUHandle* second_normal = nullptr;
  ASSERT_OK(cache_->Insert("second", 0 /*hash*/, &second_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, &second_normal,
                           Cache::Priority::LOW));
  ASSERT_NE(nullptr, second_normal);
  Cache::Priority second_priority;
  LRUHandle* second_warmup =
      cache_->LookupForCacheWarmup("second", 0 /*hash*/, &second_priority);
  ASSERT_NE(nullptr, second_warmup);
  ASSERT_EQ(Cache::Priority::LOW, second_priority);

  // Warmup released first; the ordinary release remains responsible for the
  // same LOW insertion policy.
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(second_warmup, second_priority));
  EXPECT_EQ(1U, cache_->GetPinnedUsage());
  EXPECT_FALSE(
      cache_->Release(second_normal, true /*useful*/, false /*erase*/));
  EXPECT_EQ(0U, cache_->GetPinnedUsage());
  verify = cache_->LookupForCacheWarmup("second", 0 /*hash*/, &ignored);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::LOW, ignored);
  EXPECT_EQ(&second_deleted, verify->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, ignored));

  cache_->Erase("first", 0 /*hash*/);
  cache_->Erase("second", 0 /*hash*/);
  EXPECT_EQ(1, first_deleted);
  EXPECT_EQ(1, second_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupLeasesLookupPinnedEntryBothReleaseOrders) {
  NewCache(8, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  Insert("first", Cache::Priority::LOW);
  Insert("second", Cache::Priority::LOW);

  // Unlike an insertion handle, Lookup() is a real hit. Acquiring the warmup
  // lease afterward must neither lose nor add to that hit, regardless of which
  // reference is released last.
  LRUHandle* first_normal = cache_->Lookup("first", 0 /*hash*/);
  ASSERT_NE(nullptr, first_normal);
  Cache::Priority first_before;
  LRUHandle* first_warmup =
      cache_->LookupForCacheWarmup("first", 0 /*hash*/, &first_before);
  ASSERT_NE(nullptr, first_warmup);
  ASSERT_EQ(Cache::Priority::LOW, first_before);
  EXPECT_FALSE(cache_->Release(first_normal, true /*useful*/, false /*erase*/));
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(first_warmup, first_before));

  Cache::Priority after;
  LRUHandle* verify = cache_->LookupForCacheWarmup("first", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));

  LRUHandle* second_normal = cache_->Lookup("second", 0 /*hash*/);
  ASSERT_NE(nullptr, second_normal);
  Cache::Priority second_before;
  LRUHandle* second_warmup =
      cache_->LookupForCacheWarmup("second", 0 /*hash*/, &second_before);
  ASSERT_NE(nullptr, second_warmup);
  ASSERT_EQ(Cache::Priority::LOW, second_before);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(second_warmup, second_before));
  EXPECT_FALSE(
      cache_->Release(second_normal, true /*useful*/, false /*erase*/));
  verify = cache_->LookupForCacheWarmup("second", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));
}

TEST_F(LRUCacheTest, CacheWarmupNormallyPinnedLifetimeAcrossErase) {
  NewCache(8, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int first_deleted = 0;
  int second_deleted = 0;

  LRUHandle* first_normal = nullptr;
  ASSERT_OK(cache_->Insert("first", 0 /*hash*/, &first_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, &first_normal,
                           Cache::Priority::LOW));
  Cache::Priority first_priority;
  LRUHandle* first_warmup =
      cache_->LookupForCacheWarmup("first", 0 /*hash*/, &first_priority);
  ASSERT_NE(nullptr, first_warmup);
  cache_->Erase("first", 0 /*hash*/);
  EXPECT_EQ(0, first_deleted);
  EXPECT_FALSE(cache_->Release(first_normal, true /*useful*/, false /*erase*/));
  EXPECT_EQ(0, first_deleted);
  EXPECT_TRUE(cache_->ReleaseForCacheWarmup(first_warmup, first_priority));
  EXPECT_EQ(1, first_deleted);

  LRUHandle* second_normal = nullptr;
  ASSERT_OK(cache_->Insert("second", 0 /*hash*/, &second_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, &second_normal,
                           Cache::Priority::LOW));
  Cache::Priority second_priority;
  LRUHandle* second_warmup =
      cache_->LookupForCacheWarmup("second", 0 /*hash*/, &second_priority);
  ASSERT_NE(nullptr, second_warmup);
  cache_->Erase("second", 0 /*hash*/);
  EXPECT_EQ(0, second_deleted);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(second_warmup, second_priority));
  EXPECT_EQ(0, second_deleted);
  EXPECT_TRUE(cache_->Release(second_normal, true /*useful*/, false /*erase*/));
  EXPECT_EQ(1, second_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupLookupTreatsSecondaryDummyAsMiss) {
  NewCache(1);
  ASSERT_OK(cache_->Insert("dummy", 0 /*hash*/, lru_cache::kDummyValueMarker,
                           0 /*charge*/, nullptr /*deleter*/,
                           nullptr /*handle*/, Cache::Priority::LOW));

  Cache::Priority priority = Cache::Priority::HIGH;
  LRUHandle* handle =
      cache_->LookupForCacheWarmup("dummy", 0 /*hash*/, &priority);
  EXPECT_EQ(nullptr, handle);
  cache_->Erase("dummy", 0 /*hash*/);
}

TEST_F(LRUCacheTest, CacheWarmupDuplicateKeepsDestinationOwnership) {
  NewCache(4, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int destination_deleted = 0;
  int incoming_deleted = 0;
  ASSERT_OK(cache_->Insert("duplicate", 0 /*hash*/, &destination_deleted,
                           1 /*charge*/, WarmupIncrementDeleter,
                           nullptr /*handle*/, Cache::Priority::LOW));

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmup(
      "duplicate", 0 /*hash*/, &incoming_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  EXPECT_EQ(0, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);

  Cache::Priority priority;
  LRUHandle* handle =
      cache_->LookupForCacheWarmup("duplicate", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, handle);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_EQ(&destination_deleted, handle->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(handle, priority));
  cache_->Erase("duplicate", 0 /*hash*/);
  EXPECT_EQ(1, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupDuplicatePromotionWaitsForPinnedResident) {
  NewCache(4, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int destination_deleted = 0;
  int incoming_deleted = 0;
  ASSERT_OK(cache_->Insert("duplicate", 0 /*hash*/, &destination_deleted,
                           1 /*charge*/, WarmupIncrementDeleter,
                           nullptr /*handle*/, Cache::Priority::BOTTOM));

  Cache::Priority before;
  LRUHandle* lease =
      cache_->LookupForCacheWarmup("duplicate", 0 /*hash*/, &before);
  ASSERT_NE(nullptr, lease);
  ASSERT_EQ(Cache::Priority::BOTTOM, before);

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmup(
      "duplicate", 0 /*hash*/, &incoming_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  EXPECT_EQ(0, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(lease, before));

  Cache::Priority after;
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("duplicate", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_EQ(&destination_deleted, verify->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));
  cache_->Erase("duplicate", 0 /*hash*/);
  EXPECT_EQ(1, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);

  // The same deferred promotion is applied by ordinary Release() for an entry
  // that was already pinned before the duplicate arrived.
  destination_deleted = 0;
  incoming_deleted = 0;
  LRUHandle* pinned = nullptr;
  ASSERT_OK(cache_->Insert("normal-pinned", 0 /*hash*/, &destination_deleted,
                           1 /*charge*/, WarmupIncrementDeleter, &pinned,
                           Cache::Priority::BOTTOM));
  ASSERT_NE(nullptr, pinned);
  ASSERT_OK(cache_->InsertForCacheWarmup(
      "normal-pinned", 0 /*hash*/, &incoming_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  EXPECT_FALSE(cache_->Release(pinned, true /*useful*/, false /*erase*/));
  verify = cache_->LookupForCacheWarmup("normal-pinned", 0 /*hash*/, &after);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::HIGH, after);
  EXPECT_EQ(&destination_deleted, verify->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, after));
  cache_->Erase("normal-pinned", 0 /*hash*/);
  EXPECT_EQ(1, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupLookupReportsDeferredPromotion) {
  NewCache(4, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  LRUHandle* pinned = nullptr;
  ASSERT_OK(cache_->Insert("pinned", 0 /*hash*/, nullptr /*value*/,
                           1 /*charge*/, nullptr /*deleter*/, &pinned,
                           Cache::Priority::BOTTOM));
  ASSERT_NE(nullptr, pinned);

  ASSERT_OK(cache_->PromoteForCacheWarmup("pinned", 0 /*hash*/,
                                          Cache::Priority::HIGH));

  // The ordinary pin prevents an immediate pool move, but warmup capture must
  // still observe and forward the deferred promotion.
  Cache::Priority priority = Cache::Priority::BOTTOM;
  LRUHandle* warmup =
      cache_->LookupForCacheWarmup("pinned", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, warmup);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));

  EXPECT_FALSE(cache_->Release(pinned, true /*useful*/, false /*erase*/));
  warmup = cache_->LookupForCacheWarmup("pinned", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, warmup);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, priority));
  cache_->Erase("pinned", 0 /*hash*/);
}

TEST_F(LRUCacheTest, CacheWarmupNormalizesDisabledDestinationPools) {
  Cache::CacheWarmupInsertResult result;
  Cache::Priority priority;

  // HIGH maps to LOW when the destination has no HIGH pool, so it cannot
  // displace an equal-priority LOW resident.
  NewCache(1, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/1.0);
  Insert("low", Cache::Priority::LOW);
  ASSERT_OK(cache_->InsertForCacheWarmup("incoming", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("low"));
  EXPECT_FALSE(Lookup("incoming"));

  // With free capacity, that same HIGH source entry is admitted as LOW.
  NewCache(2, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/1.0);
  ASSERT_OK(cache_->InsertForCacheWarmup("incoming", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result));
  ASSERT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  LRUHandle* verify =
      cache_->LookupForCacheWarmup("incoming", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::LOW, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, priority));

  // LOW maps to BOTTOM when LOW is disabled and therefore cannot evict even a
  // BOTTOM entry. HIGH remains usable because its destination pool exists.
  NewCache(1, /*high_pri_pool_ratio=*/1.0,
           /*low_pri_pool_ratio=*/0.0);
  Insert("bottom", Cache::Priority::BOTTOM);
  ASSERT_OK(cache_->InsertForCacheWarmup("incoming-low", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("bottom"));

  // Duplicate promotion is normalized by the same rule (HIGH -> LOW).
  NewCache(4, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/1.0);
  Insert("duplicate", Cache::Priority::BOTTOM);
  ASSERT_OK(cache_->InsertForCacheWarmup("duplicate", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  verify = cache_->LookupForCacheWarmup("duplicate", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, verify);
  EXPECT_EQ(Cache::Priority::LOW, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(verify, priority));

  // With neither optional pool configured, every incoming class normalizes to
  // BOTTOM and cannot displace an existing BOTTOM entry.
  NewCache(1, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/0.0);
  Insert("bottom", Cache::Priority::BOTTOM);
  ASSERT_OK(cache_->InsertForCacheWarmup("incoming-high", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("bottom"));
}

TEST_F(LRUCacheTest, CacheWarmupAtomicExplicitVictimReplacement) {
  NewCache(4, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int victim_deleted = 0;
  int incoming_deleted = 0;
  ASSERT_OK(cache_->Insert("victim", 0 /*hash*/, &victim_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, nullptr /*handle*/,
                           Cache::Priority::BOTTOM));
  Insert("keeper", Cache::Priority::HIGH);

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "victim", 0 /*victim_hash*/, "incoming", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_EQ(1, victim_deleted);
  EXPECT_EQ(0, incoming_deleted);
  EXPECT_FALSE(Lookup("victim"));
  EXPECT_TRUE(Lookup("keeper"));

  Cache::Priority priority;
  LRUHandle* incoming =
      cache_->LookupForCacheWarmup("incoming", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, incoming);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_EQ(&incoming_deleted, incoming->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(incoming, priority));
  cache_->Erase("incoming", 0 /*hash*/);
  EXPECT_EQ(1, incoming_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupAtomicReplacementAllowsEqualNormalizedClass) {
  // Both source priorities normalize to BOTTOM when both destination pools
  // are disabled. The explicitly selected victim is still authoritative.
  NewCache(2, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/0.0);
  int victim_deleted = 0;
  int incoming_deleted = 0;
  ASSERT_OK(cache_->Insert("victim", 0 /*hash*/, &victim_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, nullptr /*handle*/,
                           Cache::Priority::LOW));

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "victim", 0 /*victim_hash*/, "incoming", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_EQ(1, victim_deleted);

  Cache::Priority priority;
  LRUHandle* incoming =
      cache_->LookupForCacheWarmup("incoming", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, incoming);
  EXPECT_EQ(Cache::Priority::BOTTOM, priority);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(incoming, priority));
  cache_->Erase("incoming", 0 /*hash*/);
  EXPECT_EQ(1, incoming_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupNoEvictUsesOnlyFreeShardCapacity) {
  NewCache(2, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int keeper_deleted = 0;
  int destination_deleted = 0;
  int rejected_deleted = 0;
  ASSERT_OK(cache_->Insert("keeper", 0 /*hash*/, &keeper_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, nullptr /*handle*/,
                           Cache::Priority::BOTTOM));

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmupNoEvict(
      "destination", 0 /*hash*/, &destination_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);

  ASSERT_OK(cache_->InsertForCacheWarmupNoEvict(
      "rejected", 0 /*hash*/, &rejected_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("keeper"));
  EXPECT_TRUE(Lookup("destination"));
  EXPECT_FALSE(Lookup("rejected"));
  EXPECT_EQ(0, rejected_deleted);

  // Duplicate admission keeps the destination value/deleter and can still
  // promote its policy class without needing any free capacity.
  ASSERT_OK(cache_->InsertForCacheWarmupNoEvict(
      "destination", 0 /*hash*/, &rejected_deleted, 1 /*charge*/,
      WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  Cache::Priority priority;
  LRUHandle* destination =
      cache_->LookupForCacheWarmup("destination", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, destination);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_EQ(&destination_deleted, destination->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(destination, priority));

  cache_->Erase("keeper", 0 /*hash*/);
  cache_->Erase("destination", 0 /*hash*/);
  EXPECT_EQ(1, keeper_deleted);
  EXPECT_EQ(1, destination_deleted);
  EXPECT_EQ(0, rejected_deleted);
}

TEST_F(LRUCacheTest, CacheWarmupAtomicReplacementRejectsWithoutEviction) {
  NewCache(4, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  int victim_deleted = 0;
  int destination_deleted = 0;
  int incoming_deleted = 0;
  ASSERT_OK(cache_->Insert("victim", 0 /*hash*/, &victim_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, nullptr /*handle*/,
                           Cache::Priority::BOTTOM));
  ASSERT_OK(cache_->Insert("duplicate", 0 /*hash*/, &destination_deleted,
                           1 /*charge*/, WarmupIncrementDeleter,
                           nullptr /*handle*/, Cache::Priority::LOW));

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "victim", 0 /*victim_hash*/, "duplicate", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kDuplicate, result);
  EXPECT_EQ(0, victim_deleted);
  EXPECT_EQ(0, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);
  EXPECT_TRUE(Lookup("victim"));
  Cache::Priority priority;
  LRUHandle* duplicate =
      cache_->LookupForCacheWarmup("duplicate", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, duplicate);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_EQ(&destination_deleted, duplicate->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(duplicate, priority));

  // A pinned victim, including a recency-preserving warmup lease, rejects the
  // replacement and leaves both the victim and unrelated entries untouched.
  LRUHandle* pinned = cache_->Lookup("victim", 0 /*hash*/);
  ASSERT_NE(nullptr, pinned);
  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "victim", 0 /*victim_hash*/, "new", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_FALSE(Lookup("new"));
  EXPECT_EQ(0, victim_deleted);
  EXPECT_FALSE(cache_->Release(pinned, true /*useful*/, false /*erase*/));

  Cache::Priority victim_priority;
  LRUHandle* warmup =
      cache_->LookupForCacheWarmup("victim", 0 /*hash*/, &victim_priority);
  ASSERT_NE(nullptr, warmup);
  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "victim", 0 /*victim_hash*/, "new", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_FALSE(Lookup("new"));
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(warmup, victim_priority));

  ASSERT_OK(cache_->ReplaceForCacheWarmup(
      "missing", 0 /*victim_hash*/, "new", 0 /*hash*/, &incoming_deleted,
      1 /*charge*/, WarmupIncrementDeleter, Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("victim"));
  EXPECT_TRUE(Lookup("duplicate"));

  cache_->Erase("victim", 0 /*hash*/);
  cache_->Erase("duplicate", 0 /*hash*/);
  EXPECT_EQ(1, victim_deleted);
  EXPECT_EQ(1, destination_deleted);
  EXPECT_EQ(0, incoming_deleted);
}

TEST(LRUCacheWarmupTest, AtomicReplacementRequiresSameShard) {
  LRUCache cache(8 /*capacity*/, 1 /*num_shard_bits*/,
                 false /*strict_capacity_limit*/, 0.5 /*high_pri_pool_ratio*/,
                 0.5 /*low_pri_pool_ratio*/);
  std::string victim_key = "victim-0";
  std::string incoming_key;
  const uint32_t victim_shard =
      LRUCacheShard::ComputeHash(victim_key) & uint32_t{1};
  for (int i = 0; i < 1000; ++i) {
    std::string candidate = "incoming-" + std::to_string(i);
    if ((LRUCacheShard::ComputeHash(candidate) & uint32_t{1}) != victim_shard) {
      incoming_key = candidate;
      break;
    }
  }
  ASSERT_FALSE(incoming_key.empty());

  Cache::CacheWarmupInsertResult result;
  Status s = cache.ReplaceForCacheWarmup(victim_key, incoming_key, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result);
  EXPECT_TRUE(s.IsInvalidArgument());
}

TEST(LRUCacheWarmupTest, PromoteExistingEntryImmediateAndDeferred) {
  LRUCache cache(4 /*capacity*/, 0 /*num_shard_bits*/,
                 false /*strict_capacity_limit*/, 0.5 /*high_pri_pool_ratio*/,
                 0.5 /*low_pri_pool_ratio*/);
  ASSERT_OK(cache.Insert("immediate", nullptr /*value*/, 1 /*charge*/,
                         nullptr /*deleter*/, nullptr /*handle*/,
                         Cache::Priority::BOTTOM));
  ASSERT_OK(cache.PromoteForCacheWarmup("immediate", Cache::Priority::HIGH));

  Cache::Handle* warmup = nullptr;
  Cache::Priority priority;
  ASSERT_OK(cache.LookupForCacheWarmup("immediate", &warmup, &priority));
  ASSERT_NE(nullptr, warmup);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_FALSE(cache.ReleaseForCacheWarmup(warmup, priority));

  Cache::Handle* pinned = nullptr;
  ASSERT_OK(cache.Insert("pinned", nullptr /*value*/, 1 /*charge*/,
                         nullptr /*deleter*/, &pinned,
                         Cache::Priority::BOTTOM));
  ASSERT_NE(nullptr, pinned);
  ASSERT_OK(cache.PromoteForCacheWarmup("pinned", Cache::Priority::HIGH));
  EXPECT_FALSE(cache.Release(pinned));
  ASSERT_OK(cache.LookupForCacheWarmup("pinned", &warmup, &priority));
  ASSERT_NE(nullptr, warmup);
  EXPECT_EQ(Cache::Priority::HIGH, priority);
  EXPECT_FALSE(cache.ReleaseForCacheWarmup(warmup, priority));

  EXPECT_TRUE(cache.PromoteForCacheWarmup("missing", Cache::Priority::HIGH)
                  .IsNotFound());
  cache.Erase("immediate");
  cache.Erase("pinned");
}

TEST(LRUCacheWarmupTest, PhysicalShardIntrospectionUsesChargeUnits) {
  LRUCache cache(8 /*capacity*/, 2 /*num_shard_bits*/,
                 false /*strict_capacity_limit*/, 0.0 /*high_pri_pool_ratio*/,
                 1.0 /*low_pri_pool_ratio*/);
  ASSERT_EQ(4U, cache.GetCacheWarmupShardCount());

  std::array<std::string, 4> keys;
  for (int i = 0; i < 1000; ++i) {
    std::string candidate = "shard-key-" + std::to_string(i);
    const size_t shard = cache.GetCacheWarmupShardIndex(candidate);
    ASSERT_LT(shard, keys.size());
    if (keys[shard].empty()) {
      keys[shard] = std::move(candidate);
    }
  }
  for (size_t shard = 0; shard < keys.size(); ++shard) {
    ASSERT_FALSE(keys[shard].empty());
    EXPECT_EQ(2U, cache.GetCacheWarmupShardCapacity(shard));
    EXPECT_EQ(0U, cache.GetCacheWarmupShardUsage(shard));
    ASSERT_OK(cache.Insert(keys[shard], nullptr /*value*/, 1 /*charge*/,
                           nullptr /*deleter*/, nullptr /*handle*/,
                           Cache::Priority::LOW));
    EXPECT_EQ(1U, cache.GetCacheWarmupShardUsage(shard));
  }
  EXPECT_EQ(0U, cache.GetCacheWarmupShardCapacity(keys.size()));
  EXPECT_EQ(0U, cache.GetCacheWarmupShardUsage(keys.size()));
  for (const std::string& key : keys) {
    cache.Erase(key);
  }
}

TEST_F(LRUCacheTest, CacheWarmupPriorityAdmissionRules) {
  Cache::CacheWarmupInsertResult result;

  // HIGH can replace enough LOW and BOTTOM bytes, while retaining HIGH.
  NewCache(3, /*high_pri_pool_ratio=*/0.34,
           /*low_pri_pool_ratio=*/0.34);
  Insert("bottom", Cache::Priority::BOTTOM);
  Insert("low", Cache::Priority::LOW);
  Insert("high", Cache::Priority::HIGH);
  ASSERT_OK(cache_->InsertForCacheWarmup("new-high", 0 /*hash*/, nullptr,
                                         2 /*charge*/, nullptr,
                                         Cache::Priority::HIGH, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_FALSE(Lookup("bottom"));
  EXPECT_FALSE(Lookup("low"));
  EXPECT_TRUE(Lookup("high"));
  EXPECT_TRUE(Lookup("new-high"));

  // LOW can replace BOTTOM but not another LOW.
  NewCache(2, /*high_pri_pool_ratio=*/0.5,
           /*low_pri_pool_ratio=*/0.5);
  Insert("bottom", Cache::Priority::BOTTOM);
  Insert("low", Cache::Priority::LOW);
  ASSERT_OK(cache_->InsertForCacheWarmup("new-low", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_FALSE(Lookup("bottom"));
  EXPECT_TRUE(Lookup("low"));
  EXPECT_TRUE(Lookup("new-low"));

  // With no BOTTOM space left, LOW rejects without replacing LOW.
  NewCache(1, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/1.0);
  Insert("low", Cache::Priority::LOW);
  ASSERT_OK(cache_->InsertForCacheWarmup("other-low", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("low"));
  EXPECT_FALSE(Lookup("other-low"));

  // BOTTOM never evicts, including another BOTTOM entry.
  NewCache(1, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/0.0);
  Insert("bottom", Cache::Priority::BOTTOM);
  ASSERT_OK(cache_->InsertForCacheWarmup("other-bottom", 0 /*hash*/, nullptr,
                                         1 /*charge*/, nullptr,
                                         Cache::Priority::BOTTOM, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);
  EXPECT_TRUE(Lookup("bottom"));
  EXPECT_FALSE(Lookup("other-bottom"));
}

TEST_F(LRUCacheTest, CacheWarmupAtomicRejectionWithPinnedVictim) {
  NewCache(2, /*high_pri_pool_ratio=*/0.0,
           /*low_pri_pool_ratio=*/0.0);
  Insert("bottom-a", Cache::Priority::BOTTOM);
  Insert("bottom-b", Cache::Priority::BOTTOM);

  Cache::Priority priority;
  LRUHandle* pinned =
      cache_->LookupForCacheWarmup("bottom-a", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, pinned);
  ASSERT_EQ(Cache::Priority::BOTTOM, priority);

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmup("new-low", 0 /*hash*/, nullptr,
                                         2 /*charge*/, nullptr,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kRejectedNoSpace, result);

  // The one eligible, unpinned entry was not partially evicted.
  EXPECT_TRUE(Lookup("bottom-b"));
  EXPECT_FALSE(Lookup("new-low"));
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(pinned, priority));
}

TEST_F(LRUCacheTest, CacheWarmupLookupPinsLifetimeAcrossReplacement) {
  NewCache(2);
  int old_deleted = 0;
  int new_deleted = 0;
  ASSERT_OK(cache_->Insert("key", 0 /*hash*/, &old_deleted, 1 /*charge*/,
                           WarmupIncrementDeleter, nullptr /*handle*/,
                           Cache::Priority::LOW));

  Cache::Priority priority;
  LRUHandle* old_handle =
      cache_->LookupForCacheWarmup("key", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, old_handle);
  cache_->Erase("key", 0 /*hash*/);
  EXPECT_EQ(0, old_deleted);

  Cache::CacheWarmupInsertResult result;
  ASSERT_OK(cache_->InsertForCacheWarmup("key", 0 /*hash*/, &new_deleted,
                                         1 /*charge*/, WarmupIncrementDeleter,
                                         Cache::Priority::LOW, &result));
  EXPECT_EQ(Cache::CacheWarmupInsertResult::kInserted, result);
  EXPECT_EQ(&old_deleted, old_handle->value);
  EXPECT_TRUE(cache_->ReleaseForCacheWarmup(old_handle, priority));
  EXPECT_EQ(1, old_deleted);

  LRUHandle* new_handle =
      cache_->LookupForCacheWarmup("key", 0 /*hash*/, &priority);
  ASSERT_NE(nullptr, new_handle);
  EXPECT_EQ(&new_deleted, new_handle->value);
  EXPECT_FALSE(cache_->ReleaseForCacheWarmup(new_handle, priority));
  cache_->Erase("key", 0 /*hash*/);
  EXPECT_EQ(1, new_deleted);
}

// TODO: FastLRUCache and ClockCache use the same tests. We can probably remove
// them from FastLRUCache after ClockCache becomes productive, and we don't plan
// to use or maintain FastLRUCache any more.
namespace fast_lru_cache {

// TODO(guido) Replicate LRU policy tests from LRUCache here.
class FastLRUCacheTest : public testing::Test {
 public:
  FastLRUCacheTest() {}
  ~FastLRUCacheTest() override { DeleteCache(); }

  void DeleteCache() {
    if (cache_ != nullptr) {
      cache_->~LRUCacheShard();
      port::cacheline_aligned_free(cache_);
      cache_ = nullptr;
    }
  }

  void NewCache(size_t capacity) {
    DeleteCache();
    cache_ = reinterpret_cast<LRUCacheShard*>(
        port::cacheline_aligned_alloc(sizeof(LRUCacheShard)));
    new (cache_) LRUCacheShard(capacity, 1 /*estimated_value_size*/,
                               false /*strict_capacity_limit*/,
                               kDontChargeCacheMetadata);
  }

  Status Insert(const std::string& key) {
    return cache_->Insert(key, 0 /*hash*/, nullptr /*value*/, 1 /*charge*/,
                          nullptr /*deleter*/, nullptr /*handle*/,
                          Cache::Priority::LOW);
  }

  Status Insert(char key, size_t len) { return Insert(std::string(len, key)); }

  size_t CalcEstimatedHandleChargeWrapper(
      size_t estimated_value_size,
      CacheMetadataChargePolicy metadata_charge_policy) {
    return LRUCacheShard::CalcEstimatedHandleCharge(estimated_value_size,
                                                    metadata_charge_policy);
  }

  int CalcHashBitsWrapper(size_t capacity, size_t estimated_value_size,
                          CacheMetadataChargePolicy metadata_charge_policy) {
    return LRUCacheShard::CalcHashBits(capacity, estimated_value_size,
                                       metadata_charge_policy);
  }

  // Maximum number of items that a shard can hold.
  double CalcMaxOccupancy(size_t capacity, size_t estimated_value_size,
                          CacheMetadataChargePolicy metadata_charge_policy) {
    size_t handle_charge = LRUCacheShard::CalcEstimatedHandleCharge(
        estimated_value_size, metadata_charge_policy);
    return capacity / (kLoadFactor * handle_charge);
  }
  bool TableSizeIsAppropriate(int hash_bits, double max_occupancy) {
    if (hash_bits == 0) {
      return max_occupancy <= 1;
    } else {
      return (1 << hash_bits >= max_occupancy) &&
             (1 << (hash_bits - 1) <= max_occupancy);
    }
  }

 private:
  LRUCacheShard* cache_ = nullptr;
};

TEST_F(FastLRUCacheTest, ValidateKeySize) {
  NewCache(3);
  EXPECT_OK(Insert('a', 16));
  EXPECT_NOK(Insert('b', 15));
  EXPECT_OK(Insert('b', 16));
  EXPECT_NOK(Insert('c', 17));
  EXPECT_NOK(Insert('d', 1000));
  EXPECT_NOK(Insert('e', 11));
  EXPECT_NOK(Insert('f', 0));
}

TEST_F(FastLRUCacheTest, CalcHashBitsTest) {
  size_t capacity;
  size_t estimated_value_size;
  double max_occupancy;
  int hash_bits;
  CacheMetadataChargePolicy metadata_charge_policy;
  // Vary the cache capacity, fix the element charge.
  for (int i = 0; i < 2048; i++) {
    capacity = i;
    estimated_value_size = 0;
    metadata_charge_policy = kFullChargeCacheMetadata;
    max_occupancy = CalcMaxOccupancy(capacity, estimated_value_size,
                                     metadata_charge_policy);
    hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                    metadata_charge_policy);
    EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, max_occupancy));
  }
  // Fix the cache capacity, vary the element charge.
  for (int i = 0; i < 1024; i++) {
    capacity = 1024;
    estimated_value_size = i;
    metadata_charge_policy = kFullChargeCacheMetadata;
    max_occupancy = CalcMaxOccupancy(capacity, estimated_value_size,
                                     metadata_charge_policy);
    hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                    metadata_charge_policy);
    EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, max_occupancy));
  }
  // Zero-capacity cache, and only values have charge.
  capacity = 0;
  estimated_value_size = 1;
  metadata_charge_policy = kDontChargeCacheMetadata;
  hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                  metadata_charge_policy);
  EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, 0 /* max_occupancy */));
  // Zero-capacity cache, and only metadata has charge.
  capacity = 0;
  estimated_value_size = 0;
  metadata_charge_policy = kFullChargeCacheMetadata;
  hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                  metadata_charge_policy);
  EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, 0 /* max_occupancy */));
  // Small cache, large elements.
  capacity = 1024;
  estimated_value_size = 8192;
  metadata_charge_policy = kFullChargeCacheMetadata;
  hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                  metadata_charge_policy);
  EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, 0 /* max_occupancy */));
  // Large capacity.
  capacity = 31924172;
  estimated_value_size = 8192;
  metadata_charge_policy = kFullChargeCacheMetadata;
  max_occupancy =
      CalcMaxOccupancy(capacity, estimated_value_size, metadata_charge_policy);
  hash_bits = CalcHashBitsWrapper(capacity, estimated_value_size,
                                  metadata_charge_policy);
  EXPECT_TRUE(TableSizeIsAppropriate(hash_bits, max_occupancy));
}

}  // namespace fast_lru_cache

namespace clock_cache {

class ClockCacheTest : public testing::Test {
 public:
  using Shard = HyperClockCache::Shard;
  using Table = HyperClockTable;
  using HandleImpl = Shard::HandleImpl;

  ClockCacheTest() {}
  ~ClockCacheTest() override { DeleteShard(); }

  void DeleteShard() {
    if (shard_ != nullptr) {
      shard_->~ClockCacheShard();
      port::cacheline_aligned_free(shard_);
      shard_ = nullptr;
    }
  }

  void NewShard(size_t capacity, bool strict_capacity_limit = true) {
    DeleteShard();
    shard_ =
        reinterpret_cast<Shard*>(port::cacheline_aligned_alloc(sizeof(Shard)));

    Table::Opts opts;
    opts.estimated_value_size = 1;
    new (shard_)
        Shard(capacity, strict_capacity_limit, kDontChargeCacheMetadata, opts);
  }

  Status Insert(const UniqueId64x2& hashed_key,
                Cache::Priority priority = Cache::Priority::LOW) {
    return shard_->Insert(TestKey(hashed_key), hashed_key, nullptr /*value*/,
                          1 /*charge*/, nullptr /*deleter*/, nullptr /*handle*/,
                          priority);
  }

  Status Insert(char key, Cache::Priority priority = Cache::Priority::LOW) {
    return Insert(TestHashedKey(key), priority);
  }

  Status InsertWithLen(char key, size_t len) {
    std::string skey(len, key);
    return shard_->Insert(skey, TestHashedKey(key), nullptr /*value*/,
                          1 /*charge*/, nullptr /*deleter*/, nullptr /*handle*/,
                          Cache::Priority::LOW);
  }

  bool Lookup(const Slice& key, const UniqueId64x2& hashed_key,
              bool useful = true) {
    auto handle = shard_->Lookup(key, hashed_key);
    if (handle) {
      shard_->Release(handle, useful, /*erase_if_last_ref=*/false);
      return true;
    }
    return false;
  }

  bool Lookup(const UniqueId64x2& hashed_key, bool useful = true) {
    return Lookup(TestKey(hashed_key), hashed_key, useful);
  }

  bool Lookup(char key, bool useful = true) {
    return Lookup(TestHashedKey(key), useful);
  }

  void Erase(char key) {
    UniqueId64x2 hashed_key = TestHashedKey(key);
    shard_->Erase(TestKey(hashed_key), hashed_key);
  }

  static inline Slice TestKey(const UniqueId64x2& hashed_key) {
    return Slice(reinterpret_cast<const char*>(&hashed_key), 16U);
  }

  static inline UniqueId64x2 TestHashedKey(char key) {
    // For testing hash near-collision behavior, put the variance in
    // hashed_key in bits that are unlikely to be used as hash bits.
    return {(static_cast<uint64_t>(key) << 56) + 1234U, 5678U};
  }

  Shard* shard_ = nullptr;
};

TEST_F(ClockCacheTest, Misc) {
  NewShard(3);

  // Key size stuff
  EXPECT_OK(InsertWithLen('a', 16));
  EXPECT_NOK(InsertWithLen('b', 15));
  EXPECT_OK(InsertWithLen('b', 16));
  EXPECT_NOK(InsertWithLen('c', 17));
  EXPECT_NOK(InsertWithLen('d', 1000));
  EXPECT_NOK(InsertWithLen('e', 11));
  EXPECT_NOK(InsertWithLen('f', 0));

  // Some of this is motivated by code coverage
  std::string wrong_size_key(15, 'x');
  EXPECT_FALSE(Lookup(wrong_size_key, TestHashedKey('x')));
  EXPECT_FALSE(shard_->Ref(nullptr));
  EXPECT_FALSE(shard_->Release(nullptr));
  shard_->Erase(wrong_size_key, TestHashedKey('x'));  // no-op
}

TEST_F(ClockCacheTest, Limits) {
  constexpr size_t kCapacity = 3;
  NewShard(kCapacity, false /*strict_capacity_limit*/);
  for (bool strict_capacity_limit : {false, true, false}) {
    SCOPED_TRACE("strict_capacity_limit = " +
                 std::to_string(strict_capacity_limit));

    // Also tests switching between strict limit and not
    shard_->SetStrictCapacityLimit(strict_capacity_limit);

    UniqueId64x2 hkey = TestHashedKey('x');

    // Single entry charge beyond capacity
    {
      Status s = shard_->Insert(TestKey(hkey), hkey, nullptr /*value*/,
                                5 /*charge*/, nullptr /*deleter*/,
                                nullptr /*handle*/, Cache::Priority::LOW);
      if (strict_capacity_limit) {
        EXPECT_TRUE(s.IsMemoryLimit());
      } else {
        EXPECT_OK(s);
      }
    }

    // Single entry fills capacity
    {
      HandleImpl* h;
      ASSERT_OK(shard_->Insert(TestKey(hkey), hkey, nullptr /*value*/,
                               3 /*charge*/, nullptr /*deleter*/, &h,
                               Cache::Priority::LOW));
      // Try to insert more
      Status s = Insert('a');
      if (strict_capacity_limit) {
        EXPECT_TRUE(s.IsMemoryLimit());
      } else {
        EXPECT_OK(s);
      }
      // Release entry filling capacity.
      // Cover useful = false case.
      shard_->Release(h, false /*useful*/, false /*erase_if_last_ref*/);
    }

    // Insert more than table size can handle to exceed occupancy limit.
    // (Cleverly using mostly zero-charge entries, but some non-zero to
    // verify usage tracking on detached entries.)
    {
      size_t n = shard_->GetTableAddressCount() + 1;
      std::unique_ptr<HandleImpl*[]> ha{new HandleImpl*[n]{}};
      Status s;
      for (size_t i = 0; i < n && s.ok(); ++i) {
        hkey[1] = i;
        s = shard_->Insert(TestKey(hkey), hkey, nullptr /*value*/,
                           (i + kCapacity < n) ? 0 : 1 /*charge*/,
                           nullptr /*deleter*/, &ha[i], Cache::Priority::LOW);
        if (i == 0) {
          EXPECT_OK(s);
        }
      }
      if (strict_capacity_limit) {
        EXPECT_TRUE(s.IsMemoryLimit());
      } else {
        EXPECT_OK(s);
      }
      // Same result if not keeping a reference
      s = Insert('a');
      if (strict_capacity_limit) {
        EXPECT_TRUE(s.IsMemoryLimit());
      } else {
        EXPECT_OK(s);
      }

      // Regardless, we didn't allow table to actually get full
      EXPECT_LT(shard_->GetOccupancyCount(), shard_->GetTableAddressCount());

      // Release handles
      for (size_t i = 0; i < n; ++i) {
        if (ha[i]) {
          shard_->Release(ha[i]);
        }
      }
    }
  }
}

TEST_F(ClockCacheTest, ClockEvictionTest) {
  for (bool strict_capacity_limit : {false, true}) {
    SCOPED_TRACE("strict_capacity_limit = " +
                 std::to_string(strict_capacity_limit));

    NewShard(6, strict_capacity_limit);
    EXPECT_OK(Insert('a', Cache::Priority::BOTTOM));
    EXPECT_OK(Insert('b', Cache::Priority::LOW));
    EXPECT_OK(Insert('c', Cache::Priority::HIGH));
    EXPECT_OK(Insert('d', Cache::Priority::BOTTOM));
    EXPECT_OK(Insert('e', Cache::Priority::LOW));
    EXPECT_OK(Insert('f', Cache::Priority::HIGH));

    EXPECT_TRUE(Lookup('a', /*use*/ false));
    EXPECT_TRUE(Lookup('b', /*use*/ false));
    EXPECT_TRUE(Lookup('c', /*use*/ false));
    EXPECT_TRUE(Lookup('d', /*use*/ false));
    EXPECT_TRUE(Lookup('e', /*use*/ false));
    EXPECT_TRUE(Lookup('f', /*use*/ false));

    // Ensure bottom are evicted first, even if new entries are low
    EXPECT_OK(Insert('g', Cache::Priority::LOW));
    EXPECT_OK(Insert('h', Cache::Priority::LOW));

    EXPECT_FALSE(Lookup('a', /*use*/ false));
    EXPECT_TRUE(Lookup('b', /*use*/ false));
    EXPECT_TRUE(Lookup('c', /*use*/ false));
    EXPECT_FALSE(Lookup('d', /*use*/ false));
    EXPECT_TRUE(Lookup('e', /*use*/ false));
    EXPECT_TRUE(Lookup('f', /*use*/ false));
    // Mark g & h useful
    EXPECT_TRUE(Lookup('g', /*use*/ true));
    EXPECT_TRUE(Lookup('h', /*use*/ true));

    // Then old LOW entries
    EXPECT_OK(Insert('i', Cache::Priority::LOW));
    EXPECT_OK(Insert('j', Cache::Priority::LOW));

    EXPECT_FALSE(Lookup('b', /*use*/ false));
    EXPECT_TRUE(Lookup('c', /*use*/ false));
    EXPECT_FALSE(Lookup('e', /*use*/ false));
    EXPECT_TRUE(Lookup('f', /*use*/ false));
    // Mark g & h useful once again
    EXPECT_TRUE(Lookup('g', /*use*/ true));
    EXPECT_TRUE(Lookup('h', /*use*/ true));
    EXPECT_TRUE(Lookup('i', /*use*/ false));
    EXPECT_TRUE(Lookup('j', /*use*/ false));

    // Then old HIGH entries
    EXPECT_OK(Insert('k', Cache::Priority::LOW));
    EXPECT_OK(Insert('l', Cache::Priority::LOW));

    EXPECT_FALSE(Lookup('c', /*use*/ false));
    EXPECT_FALSE(Lookup('f', /*use*/ false));
    EXPECT_TRUE(Lookup('g', /*use*/ false));
    EXPECT_TRUE(Lookup('h', /*use*/ false));
    EXPECT_TRUE(Lookup('i', /*use*/ false));
    EXPECT_TRUE(Lookup('j', /*use*/ false));
    EXPECT_TRUE(Lookup('k', /*use*/ false));
    EXPECT_TRUE(Lookup('l', /*use*/ false));

    // Then the (roughly) least recently useful
    EXPECT_OK(Insert('m', Cache::Priority::HIGH));
    EXPECT_OK(Insert('n', Cache::Priority::HIGH));

    EXPECT_TRUE(Lookup('g', /*use*/ false));
    EXPECT_TRUE(Lookup('h', /*use*/ false));
    EXPECT_FALSE(Lookup('i', /*use*/ false));
    EXPECT_FALSE(Lookup('j', /*use*/ false));
    EXPECT_TRUE(Lookup('k', /*use*/ false));
    EXPECT_TRUE(Lookup('l', /*use*/ false));

    // Now try changing capacity down
    shard_->SetCapacity(4);
    // Insert to ensure evictions happen
    EXPECT_OK(Insert('o', Cache::Priority::LOW));
    EXPECT_OK(Insert('p', Cache::Priority::LOW));

    EXPECT_FALSE(Lookup('g', /*use*/ false));
    EXPECT_FALSE(Lookup('h', /*use*/ false));
    EXPECT_FALSE(Lookup('k', /*use*/ false));
    EXPECT_FALSE(Lookup('l', /*use*/ false));
    EXPECT_TRUE(Lookup('m', /*use*/ false));
    EXPECT_TRUE(Lookup('n', /*use*/ false));
    EXPECT_TRUE(Lookup('o', /*use*/ false));
    EXPECT_TRUE(Lookup('p', /*use*/ false));

    // Now try changing capacity up
    EXPECT_TRUE(Lookup('m', /*use*/ true));
    EXPECT_TRUE(Lookup('n', /*use*/ true));
    shard_->SetCapacity(6);
    EXPECT_OK(Insert('q', Cache::Priority::HIGH));
    EXPECT_OK(Insert('r', Cache::Priority::HIGH));
    EXPECT_OK(Insert('s', Cache::Priority::HIGH));
    EXPECT_OK(Insert('t', Cache::Priority::HIGH));

    EXPECT_FALSE(Lookup('o', /*use*/ false));
    EXPECT_FALSE(Lookup('p', /*use*/ false));
    EXPECT_TRUE(Lookup('m', /*use*/ false));
    EXPECT_TRUE(Lookup('n', /*use*/ false));
    EXPECT_TRUE(Lookup('q', /*use*/ false));
    EXPECT_TRUE(Lookup('r', /*use*/ false));
    EXPECT_TRUE(Lookup('s', /*use*/ false));
    EXPECT_TRUE(Lookup('t', /*use*/ false));
  }
}

void IncrementIntDeleter(const Slice& /*key*/, void* value) {
  *reinterpret_cast<int*>(value) += 1;
}

// Testing calls to CorrectNearOverflow in Release
TEST_F(ClockCacheTest, ClockCounterOverflowTest) {
  NewShard(6, /*strict_capacity_limit*/ false);
  HandleImpl* h;
  int deleted = 0;
  UniqueId64x2 hkey = TestHashedKey('x');
  ASSERT_OK(shard_->Insert(TestKey(hkey), hkey, &deleted, 1,
                           IncrementIntDeleter, &h, Cache::Priority::HIGH));

  // Some large number outstanding
  shard_->TEST_RefN(h, 123456789);
  // Simulate many lookup/ref + release, plenty to overflow counters
  for (int i = 0; i < 10000; ++i) {
    shard_->TEST_RefN(h, 1234567);
    shard_->TEST_ReleaseN(h, 1234567);
  }
  // Mark it invisible (to reach a different CorrectNearOverflow() in Release)
  shard_->Erase(TestKey(hkey), hkey);
  // Simulate many more lookup/ref + release (one-by-one would be too
  // expensive for unit test)
  for (int i = 0; i < 10000; ++i) {
    shard_->TEST_RefN(h, 1234567);
    shard_->TEST_ReleaseN(h, 1234567);
  }
  // Free all but last 1
  shard_->TEST_ReleaseN(h, 123456789);
  // Still alive
  ASSERT_EQ(deleted, 0);
  // Free last ref, which will finalize erasure
  shard_->Release(h);
  // Deleted
  ASSERT_EQ(deleted, 1);
}

// This test is mostly to exercise some corner case logic, by forcing two
// keys to have the same hash, and more
TEST_F(ClockCacheTest, CollidingInsertEraseTest) {
  NewShard(6, /*strict_capacity_limit*/ false);
  int deleted = 0;
  UniqueId64x2 hkey1 = TestHashedKey('x');
  Slice key1 = TestKey(hkey1);
  UniqueId64x2 hkey2 = TestHashedKey('y');
  Slice key2 = TestKey(hkey2);
  UniqueId64x2 hkey3 = TestHashedKey('z');
  Slice key3 = TestKey(hkey3);
  HandleImpl* h1;
  ASSERT_OK(shard_->Insert(key1, hkey1, &deleted, 1, IncrementIntDeleter, &h1,
                           Cache::Priority::HIGH));
  HandleImpl* h2;
  ASSERT_OK(shard_->Insert(key2, hkey2, &deleted, 1, IncrementIntDeleter, &h2,
                           Cache::Priority::HIGH));
  HandleImpl* h3;
  ASSERT_OK(shard_->Insert(key3, hkey3, &deleted, 1, IncrementIntDeleter, &h3,
                           Cache::Priority::HIGH));

  // Can repeatedly lookup+release despite the hash collision
  HandleImpl* tmp_h;
  for (bool erase_if_last_ref : {true, false}) {  // but not last ref
    tmp_h = shard_->Lookup(key1, hkey1);
    ASSERT_EQ(h1, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));

    tmp_h = shard_->Lookup(key2, hkey2);
    ASSERT_EQ(h2, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));

    tmp_h = shard_->Lookup(key3, hkey3);
    ASSERT_EQ(h3, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));
  }

  // Make h1 invisible
  shard_->Erase(key1, hkey1);
  // Redundant erase
  shard_->Erase(key1, hkey1);

  // All still alive
  ASSERT_EQ(deleted, 0);

  // Invisible to Lookup
  tmp_h = shard_->Lookup(key1, hkey1);
  ASSERT_EQ(nullptr, tmp_h);

  // Can still find h2, h3
  for (bool erase_if_last_ref : {true, false}) {  // but not last ref
    tmp_h = shard_->Lookup(key2, hkey2);
    ASSERT_EQ(h2, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));

    tmp_h = shard_->Lookup(key3, hkey3);
    ASSERT_EQ(h3, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));
  }

  // Also Insert with invisible entry there
  ASSERT_OK(shard_->Insert(key1, hkey1, &deleted, 1, IncrementIntDeleter,
                           nullptr, Cache::Priority::HIGH));
  tmp_h = shard_->Lookup(key1, hkey1);
  // Found but distinct handle
  ASSERT_NE(nullptr, tmp_h);
  ASSERT_NE(h1, tmp_h);
  ASSERT_TRUE(shard_->Release(tmp_h, /*erase_if_last_ref*/ true));

  // tmp_h deleted
  ASSERT_EQ(deleted--, 1);

  // Release last ref on h1 (already invisible)
  ASSERT_TRUE(shard_->Release(h1, /*erase_if_last_ref*/ false));

  // h1 deleted
  ASSERT_EQ(deleted--, 1);
  h1 = nullptr;

  // Can still find h2, h3
  for (bool erase_if_last_ref : {true, false}) {  // but not last ref
    tmp_h = shard_->Lookup(key2, hkey2);
    ASSERT_EQ(h2, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));

    tmp_h = shard_->Lookup(key3, hkey3);
    ASSERT_EQ(h3, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));
  }

  // Release last ref on h2
  ASSERT_FALSE(shard_->Release(h2, /*erase_if_last_ref*/ false));

  // h2 still not deleted (unreferenced in cache)
  ASSERT_EQ(deleted, 0);

  // Can still find it
  tmp_h = shard_->Lookup(key2, hkey2);
  ASSERT_EQ(h2, tmp_h);

  // Release last ref on h2, with erase
  ASSERT_TRUE(shard_->Release(h2, /*erase_if_last_ref*/ true));

  // h2 deleted
  ASSERT_EQ(deleted--, 1);
  tmp_h = shard_->Lookup(key2, hkey2);
  ASSERT_EQ(nullptr, tmp_h);

  // Can still find h3
  for (bool erase_if_last_ref : {true, false}) {  // but not last ref
    tmp_h = shard_->Lookup(key3, hkey3);
    ASSERT_EQ(h3, tmp_h);
    ASSERT_FALSE(shard_->Release(tmp_h, erase_if_last_ref));
  }

  // Release last ref on h3, without erase
  ASSERT_FALSE(shard_->Release(h3, /*erase_if_last_ref*/ false));

  // h3 still not deleted (unreferenced in cache)
  ASSERT_EQ(deleted, 0);

  // Explicit erase
  shard_->Erase(key3, hkey3);

  // h3 deleted
  ASSERT_EQ(deleted--, 1);
  tmp_h = shard_->Lookup(key3, hkey3);
  ASSERT_EQ(nullptr, tmp_h);
}

// This uses the public API to effectively test CalcHashBits etc.
TEST_F(ClockCacheTest, TableSizesTest) {
  for (size_t est_val_size : {1U, 5U, 123U, 2345U, 345678U}) {
    SCOPED_TRACE("est_val_size = " + std::to_string(est_val_size));
    for (double est_count : {1.1, 2.2, 511.9, 512.1, 2345.0}) {
      SCOPED_TRACE("est_count = " + std::to_string(est_count));
      size_t capacity = static_cast<size_t>(est_val_size * est_count);
      // kDontChargeCacheMetadata
      auto cache = HyperClockCacheOptions(
                       capacity, est_val_size, /*num shard_bits*/ -1,
                       /*strict_capacity_limit*/ false,
                       /*memory_allocator*/ nullptr, kDontChargeCacheMetadata)
                       .MakeSharedCache();
      // Table sizes are currently only powers of two
      EXPECT_GE(cache->GetTableAddressCount(), est_count / kLoadFactor);
      EXPECT_LE(cache->GetTableAddressCount(), est_count / kLoadFactor * 2.0);
      EXPECT_EQ(cache->GetUsage(), 0);

      // kFullChargeMetaData
      // Because table sizes are currently only powers of two, sizes get
      // really weird when metadata is a huge portion of capacity. For example,
      // doubling the table size could cut by 90% the space available to
      // values. Therefore, we omit those weird cases for now.
      if (est_val_size >= 512) {
        cache = HyperClockCacheOptions(
                    capacity, est_val_size, /*num shard_bits*/ -1,
                    /*strict_capacity_limit*/ false,
                    /*memory_allocator*/ nullptr, kFullChargeCacheMetadata)
                    .MakeSharedCache();
        double est_count_after_meta =
            (capacity - cache->GetUsage()) * 1.0 / est_val_size;
        EXPECT_GE(cache->GetTableAddressCount(),
                  est_count_after_meta / kLoadFactor);
        EXPECT_LE(cache->GetTableAddressCount(),
                  est_count_after_meta / kLoadFactor * 2.0);
      }
    }
  }
}

}  // namespace clock_cache

class TestSecondaryCache : public SecondaryCache {
 public:
  // Specifies what action to take on a lookup for a particular key
  enum ResultType {
    SUCCESS,
    // Fail lookup immediately
    FAIL,
    // Defer the result. It will returned after Wait/WaitAll is called
    DEFER,
    // Defer the result and eventually return failure
    DEFER_AND_FAIL
  };

  using ResultMap = std::unordered_map<std::string, ResultType>;

  explicit TestSecondaryCache(size_t capacity)
      : num_inserts_(0), num_lookups_(0), inject_failure_(false) {
    cache_ =
        NewLRUCache(capacity, 0, false, 0.5 /* high_pri_pool_ratio */, nullptr,
                    kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  }
  ~TestSecondaryCache() override { cache_.reset(); }

  const char* Name() const override { return "TestSecondaryCache"; }

  void InjectFailure() { inject_failure_ = true; }

  void ResetInjectFailure() { inject_failure_ = false; }

  Status Insert(const Slice& key, void* value,
                const Cache::CacheItemHelper* helper) override {
    if (inject_failure_) {
      return Status::Corruption("Insertion Data Corrupted");
    }
    CheckCacheKeyCommonPrefix(key);
    size_t size;
    char* buf;
    Status s;

    num_inserts_++;
    size = (*helper->size_cb)(value);
    buf = new char[size + sizeof(uint64_t)];
    EncodeFixed64(buf, size);
    s = (*helper->saveto_cb)(value, 0, size, buf + sizeof(uint64_t));
    if (!s.ok()) {
      delete[] buf;
      return s;
    }
    return cache_->Insert(key, buf, size,
                          [](const Slice& /*key*/, void* val) -> void {
                            delete[] static_cast<char*>(val);
                          });
  }

  std::unique_ptr<SecondaryCacheResultHandle> Lookup(
      const Slice& key, const Cache::CreateCallback& create_cb, bool /*wait*/,
      bool /*advise_erase*/, bool& is_in_sec_cache) override {
    std::string key_str = key.ToString();
    TEST_SYNC_POINT_CALLBACK("TestSecondaryCache::Lookup", &key_str);

    std::unique_ptr<SecondaryCacheResultHandle> secondary_handle;
    is_in_sec_cache = false;
    ResultType type = ResultType::SUCCESS;
    auto iter = result_map_.find(key.ToString());
    if (iter != result_map_.end()) {
      type = iter->second;
    }
    if (type == ResultType::FAIL) {
      return secondary_handle;
    }

    Cache::Handle* handle = cache_->Lookup(key);
    num_lookups_++;
    if (handle) {
      void* value = nullptr;
      size_t charge = 0;
      Status s;
      if (type != ResultType::DEFER_AND_FAIL) {
        char* ptr = (char*)cache_->Value(handle);
        size_t size = DecodeFixed64(ptr);
        ptr += sizeof(uint64_t);
        s = create_cb(ptr, size, &value, &charge);
      }
      if (s.ok()) {
        secondary_handle.reset(new TestSecondaryCacheResultHandle(
            cache_.get(), handle, value, charge, type));
        is_in_sec_cache = true;
      } else {
        cache_->Release(handle);
      }
    }
    return secondary_handle;
  }

  bool SupportForceErase() const override { return false; }

  void Erase(const Slice& /*key*/) override {}

  void WaitAll(std::vector<SecondaryCacheResultHandle*> handles) override {
    for (SecondaryCacheResultHandle* handle : handles) {
      TestSecondaryCacheResultHandle* sec_handle =
          static_cast<TestSecondaryCacheResultHandle*>(handle);
      sec_handle->SetReady();
    }
  }

  std::string GetPrintableOptions() const override { return ""; }

  void SetResultMap(ResultMap&& map) { result_map_ = std::move(map); }

  uint32_t num_inserts() { return num_inserts_; }

  uint32_t num_lookups() { return num_lookups_; }

  void CheckCacheKeyCommonPrefix(const Slice& key) {
    Slice current_prefix(key.data(), OffsetableCacheKey::kCommonPrefixSize);
    if (ckey_prefix_.empty()) {
      ckey_prefix_ = current_prefix.ToString();
    } else {
      EXPECT_EQ(ckey_prefix_, current_prefix.ToString());
    }
  }

 private:
  class TestSecondaryCacheResultHandle : public SecondaryCacheResultHandle {
   public:
    TestSecondaryCacheResultHandle(Cache* cache, Cache::Handle* handle,
                                   void* value, size_t size, ResultType type)
        : cache_(cache),
          handle_(handle),
          value_(value),
          size_(size),
          is_ready_(true) {
      if (type != ResultType::SUCCESS) {
        is_ready_ = false;
      }
    }

    ~TestSecondaryCacheResultHandle() override { cache_->Release(handle_); }

    bool IsReady() override { return is_ready_; }

    void Wait() override {}

    void* Value() override {
      assert(is_ready_);
      return value_;
    }

    size_t Size() override { return Value() ? size_ : 0; }

    void SetReady() { is_ready_ = true; }

   private:
    Cache* cache_;
    Cache::Handle* handle_;
    void* value_;
    size_t size_;
    bool is_ready_;
  };

  std::shared_ptr<Cache> cache_;
  uint32_t num_inserts_;
  uint32_t num_lookups_;
  bool inject_failure_;
  std::string ckey_prefix_;
  ResultMap result_map_;
};

class DBSecondaryCacheTest : public DBTestBase {
 public:
  DBSecondaryCacheTest()
      : DBTestBase("db_secondary_cache_test", /*env_do_fsync=*/true) {
    fault_fs_.reset(new FaultInjectionTestFS(env_->GetFileSystem()));
    fault_env_.reset(new CompositeEnvWrapper(env_, fault_fs_));
  }

  std::shared_ptr<FaultInjectionTestFS> fault_fs_;
  std::unique_ptr<Env> fault_env_;
};

class LRUCacheSecondaryCacheTest : public LRUCacheTest {
 public:
  LRUCacheSecondaryCacheTest() : fail_create_(false) {}
  ~LRUCacheSecondaryCacheTest() {}

 protected:
  class TestItem {
   public:
    TestItem(const char* buf, size_t size) : buf_(new char[size]), size_(size) {
      memcpy(buf_.get(), buf, size);
    }
    ~TestItem() {}

    char* Buf() { return buf_.get(); }
    size_t Size() { return size_; }
    std::string ToString() { return std::string(Buf(), Size()); }

   private:
    std::unique_ptr<char[]> buf_;
    size_t size_;
  };

  static size_t SizeCallback(void* obj) {
    return reinterpret_cast<TestItem*>(obj)->Size();
  }

  static Status SaveToCallback(void* from_obj, size_t from_offset,
                               size_t length, void* out) {
    TestItem* item = reinterpret_cast<TestItem*>(from_obj);
    char* buf = item->Buf();
    EXPECT_EQ(length, item->Size());
    EXPECT_EQ(from_offset, 0);
    memcpy(out, buf, length);
    return Status::OK();
  }

  static void DeletionCallback(const Slice& /*key*/, void* obj) {
    delete reinterpret_cast<TestItem*>(obj);
  }

  static Cache::CacheItemHelper helper_;

  static Status SaveToCallbackFail(void* /*obj*/, size_t /*offset*/,
                                   size_t /*size*/, void* /*out*/) {
    return Status::NotSupported();
  }

  static Cache::CacheItemHelper helper_fail_;

  Cache::CreateCallback test_item_creator = [&](const void* buf, size_t size,
                                                void** out_obj,
                                                size_t* charge) -> Status {
    if (fail_create_) {
      return Status::NotSupported();
    }
    *out_obj = reinterpret_cast<void*>(new TestItem((char*)buf, size));
    *charge = size;
    return Status::OK();
  };

  void SetFailCreate(bool fail) { fail_create_ = fail; }

 private:
  bool fail_create_;
};

Cache::CacheItemHelper LRUCacheSecondaryCacheTest::helper_(
    LRUCacheSecondaryCacheTest::SizeCallback,
    LRUCacheSecondaryCacheTest::SaveToCallback,
    LRUCacheSecondaryCacheTest::DeletionCallback);

Cache::CacheItemHelper LRUCacheSecondaryCacheTest::helper_fail_(
    LRUCacheSecondaryCacheTest::SizeCallback,
    LRUCacheSecondaryCacheTest::SaveToCallbackFail,
    LRUCacheSecondaryCacheTest::DeletionCallback);

TEST_F(LRUCacheSecondaryCacheTest, BasicTest) {
  LRUCacheOptions opts(1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  std::shared_ptr<Statistics> stats = CreateDBStatistics();
  CacheKey k1 = CacheKey::CreateUniqueForCacheLifetime(cache.get());
  CacheKey k2 = CacheKey::CreateUniqueForCacheLifetime(cache.get());

  Random rnd(301);
  std::string str1 = rnd.RandomString(1020);
  TestItem* item1 = new TestItem(str1.data(), str1.length());
  ASSERT_OK(cache->Insert(k1.AsSlice(), item1,
                          &LRUCacheSecondaryCacheTest::helper_, str1.length()));
  std::string str2 = rnd.RandomString(1021);
  TestItem* item2 = new TestItem(str2.data(), str2.length());
  // k1 should be demoted to NVM
  ASSERT_OK(cache->Insert(k2.AsSlice(), item2,
                          &LRUCacheSecondaryCacheTest::helper_, str2.length()));

  get_perf_context()->Reset();
  Cache::Handle* handle;
  handle =
      cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                    test_item_creator, Cache::Priority::LOW, true, stats.get());
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  // This lookup should promote k1 and demote k2
  handle =
      cache->Lookup(k1.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                    test_item_creator, Cache::Priority::LOW, true, stats.get());
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 1u);
  ASSERT_EQ(stats->getTickerCount(SECONDARY_CACHE_HITS),
            secondary_cache->num_lookups());
  PerfContext perf_ctx = *get_perf_context();
  ASSERT_EQ(perf_ctx.secondary_cache_hit_count, secondary_cache->num_lookups());

  cache.reset();
  secondary_cache.reset();
}

TEST_F(LRUCacheSecondaryCacheTest, BasicFailTest) {
  LRUCacheOptions opts(1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  CacheKey k1 = CacheKey::CreateUniqueForCacheLifetime(cache.get());
  CacheKey k2 = CacheKey::CreateUniqueForCacheLifetime(cache.get());

  Random rnd(301);
  std::string str1 = rnd.RandomString(1020);
  auto item1 = std::make_unique<TestItem>(str1.data(), str1.length());
  ASSERT_TRUE(cache->Insert(k1.AsSlice(), item1.get(), nullptr, str1.length())
                  .IsInvalidArgument());
  ASSERT_OK(cache->Insert(k1.AsSlice(), item1.get(),
                          &LRUCacheSecondaryCacheTest::helper_, str1.length()));
  item1.release();  // Appease clang-analyze "potential memory leak"

  Cache::Handle* handle;
  handle = cache->Lookup(k2.AsSlice(), nullptr, test_item_creator,
                         Cache::Priority::LOW, true);
  ASSERT_EQ(handle, nullptr);
  handle = cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, false);
  ASSERT_EQ(handle, nullptr);

  cache.reset();
  secondary_cache.reset();
}

TEST_F(LRUCacheSecondaryCacheTest, SaveFailTest) {
  LRUCacheOptions opts(1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  CacheKey k1 = CacheKey::CreateUniqueForCacheLifetime(cache.get());
  CacheKey k2 = CacheKey::CreateUniqueForCacheLifetime(cache.get());

  Random rnd(301);
  std::string str1 = rnd.RandomString(1020);
  TestItem* item1 = new TestItem(str1.data(), str1.length());
  ASSERT_OK(cache->Insert(k1.AsSlice(), item1,
                          &LRUCacheSecondaryCacheTest::helper_fail_,
                          str1.length()));
  std::string str2 = rnd.RandomString(1020);
  TestItem* item2 = new TestItem(str2.data(), str2.length());
  // k1 should be demoted to NVM
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_OK(cache->Insert(k2.AsSlice(), item2,
                          &LRUCacheSecondaryCacheTest::helper_fail_,
                          str2.length()));
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);

  Cache::Handle* handle;
  handle =
      cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_fail_,
                    test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  // This lookup should fail, since k1 demotion would have failed
  handle =
      cache->Lookup(k1.AsSlice(), &LRUCacheSecondaryCacheTest::helper_fail_,
                    test_item_creator, Cache::Priority::LOW, true);
  ASSERT_EQ(handle, nullptr);
  // Since k1 didn't get promoted, k2 should still be in cache
  handle =
      cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_fail_,
                    test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 1u);

  cache.reset();
  secondary_cache.reset();
}

TEST_F(LRUCacheSecondaryCacheTest, CreateFailTest) {
  LRUCacheOptions opts(1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  CacheKey k1 = CacheKey::CreateUniqueForCacheLifetime(cache.get());
  CacheKey k2 = CacheKey::CreateUniqueForCacheLifetime(cache.get());

  Random rnd(301);
  std::string str1 = rnd.RandomString(1020);
  TestItem* item1 = new TestItem(str1.data(), str1.length());
  ASSERT_OK(cache->Insert(k1.AsSlice(), item1,
                          &LRUCacheSecondaryCacheTest::helper_, str1.length()));
  std::string str2 = rnd.RandomString(1020);
  TestItem* item2 = new TestItem(str2.data(), str2.length());
  // k1 should be demoted to NVM
  ASSERT_OK(cache->Insert(k2.AsSlice(), item2,
                          &LRUCacheSecondaryCacheTest::helper_, str2.length()));

  Cache::Handle* handle;
  SetFailCreate(true);
  handle = cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  // This lookup should fail, since k1 creation would have failed
  handle = cache->Lookup(k1.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, true);
  ASSERT_EQ(handle, nullptr);
  // Since k1 didn't get promoted, k2 should still be in cache
  handle = cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 1u);

  cache.reset();
  secondary_cache.reset();
}

TEST_F(LRUCacheSecondaryCacheTest, FullCapacityTest) {
  LRUCacheOptions opts(1024 /* capacity */, 0 /* num_shard_bits */,
                       true /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  CacheKey k1 = CacheKey::CreateUniqueForCacheLifetime(cache.get());
  CacheKey k2 = CacheKey::CreateUniqueForCacheLifetime(cache.get());

  Random rnd(301);
  std::string str1 = rnd.RandomString(1020);
  TestItem* item1 = new TestItem(str1.data(), str1.length());
  ASSERT_OK(cache->Insert(k1.AsSlice(), item1,
                          &LRUCacheSecondaryCacheTest::helper_, str1.length()));
  std::string str2 = rnd.RandomString(1020);
  TestItem* item2 = new TestItem(str2.data(), str2.length());
  // k1 should be demoted to NVM
  ASSERT_OK(cache->Insert(k2.AsSlice(), item2,
                          &LRUCacheSecondaryCacheTest::helper_, str2.length()));

  Cache::Handle* handle;
  handle = cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  // k1 promotion should fail due to the block cache being at capacity,
  // but the lookup should still succeed
  Cache::Handle* handle2;
  handle2 = cache->Lookup(k1.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                          test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle2, nullptr);
  // Since k1 didn't get inserted, k2 should still be in cache
  cache->Release(handle);
  cache->Release(handle2);
  handle = cache->Lookup(k2.AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
                         test_item_creator, Cache::Priority::LOW, true);
  ASSERT_NE(handle, nullptr);
  cache->Release(handle);
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 1u);

  cache.reset();
  secondary_cache.reset();
}

// In this test, the block cache size is set to 4096, after insert 6 KV-pairs
// and flush, there are 5 blocks in this SST file, 2 data blocks and 3 meta
// blocks. block_1 size is 4096 and block_2 size is 2056. The total size
// of the meta blocks are about 900 to 1000. Therefore, in any situation,
// if we try to insert block_1 to the block cache, it will always fails. Only
// block_2 will be successfully inserted into the block cache.
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheCorrectness1) {
  LRUCacheOptions opts(4 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);

  // Set the file paranoid check, so after flush, the file will be read
  // all the blocks will be accessed.
  options.paranoid_file_checks = true;
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());
  // After Flush is successful, RocksDB will do the paranoid check for the new
  // SST file. Meta blocks are always cached in the block cache and they
  // will not be evicted. When block_2 is cache miss and read out, it is
  // inserted to the block cache. Note that, block_1 is never successfully
  // inserted to the block cache. Here are 2 lookups in the secondary cache
  // for block_1 and block_2
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  Compact("a", "z");
  // Compaction will create the iterator to scan the whole file. So all the
  // blocks are needed. Meta blocks are always cached. When block_1 is read
  // out, block_2 is evicted from block cache and inserted to secondary
  // cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 3u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // The first data block is not in the cache, similarly, trigger the block
  // cache Lookup and secondary cache lookup for block_1. But block_1 will not
  // be inserted successfully due to the size. Currently, cache only has
  // the meta blocks.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // The second data block is not in the cache, similarly, trigger the block
  // cache Lookup and secondary cache lookup for block_2 and block_2 is found
  // in the secondary cache. Now block cache has block_2
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // block_2 is in the block cache. There is a block cache hit. No need to
  // lookup or insert the secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // Lookup the first data block, not in the block cache, so lookup the
  // secondary cache. Also not in the secondary cache. After Get, still
  // block_1 is will not be cached.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 6u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // Lookup the first data block, not in the block cache, so lookup the
  // secondary cache. Also not in the secondary cache. After Get, still
  // block_1 is will not be cached.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 7u);

  Destroy(options);
}

// In this test, the block cache size is set to 6100, after insert 6 KV-pairs
// and flush, there are 5 blocks in this SST file, 2 data blocks and 3 meta
// blocks. block_1 size is 4096 and block_2 size is 2056. The total size
// of the meta blocks are about 900 to 1000. Therefore, we can successfully
// insert and cache block_1 in the block cache (this is the different place
// from TestSecondaryCacheCorrectness1)
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheCorrectness2) {
  LRUCacheOptions opts(6100 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.paranoid_file_checks = true;
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());
  // After Flush is successful, RocksDB will do the paranoid check for the new
  // SST file. Meta blocks are always cached in the block cache and they
  // will not be evicted. When block_2 is cache miss and read out, it is
  // inserted to the block cache. Thefore, block_1 is evicted from block
  // cache and successfully inserted to the secondary cache. Here are 2
  // lookups in the secondary cache for block_1 and block_2.
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  Compact("a", "z");
  // Compaction will create the iterator to scan the whole file. So all the
  // blocks are needed. After Flush, only block_2 is cached in block cache
  // and block_1 is in the secondary cache. So when read block_1, it is
  // read out from secondary cache and inserted to block cache. At the same
  // time, block_2 is inserted to secondary cache. Now, secondary cache has
  // both block_1 and block_2. After compaction, block_1 is in the cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 3u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // This Get needs to access block_1, since block_1 is cached in block cache
  // there is no secondary cache lookup.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 3u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // This Get needs to access block_2 which is not in the block cache. So
  // it will lookup the secondary cache for block_2 and cache it in the
  // block_cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // This Get needs to access block_2 which is already in the block cache.
  // No need to lookup secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // This Get needs to access block_1, since block_1 is not in block cache
  // there is one econdary cache lookup. Then, block_1 is cached in the
  // block cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // This Get needs to access block_1, since block_1 is cached in block cache
  // there is no secondary cache lookup.
  ASSERT_EQ(secondary_cache->num_inserts(), 2u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  Destroy(options);
}

// The block cache size is set to 1024*1024, after insert 6 KV-pairs
// and flush, there are 5 blocks in this SST file, 2 data blocks and 3 meta
// blocks. block_1 size is 4096 and block_2 size is 2056. The total size
// of the meta blocks are about 900 to 1000. Therefore, we can successfully
// cache all the blocks in the block cache and there is not secondary cache
// insertion. 2 lookup is needed for the blocks.
TEST_F(DBSecondaryCacheTest, NoSecondaryCacheInsertion) {
  LRUCacheOptions opts(1024 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.paranoid_file_checks = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);

  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1000);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());
  // After Flush is successful, RocksDB will do the paranoid check for the new
  // SST file. Meta blocks are always cached in the block cache and they
  // will not be evicted. Now, block cache is large enough, it cache
  // both block_1 and block_2. When first time read block_1 and block_2
  // there are cache misses. So 2 secondary cache lookups are needed for
  // the 2 blocks
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  Compact("a", "z");
  // Compaction will iterate the whole SST file. Since all the data blocks
  // are in the block cache. No need to lookup the secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1000, v.size());
  // Since the block cache is large enough, all the blocks are cached. we
  // do not need to lookup the seondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  Destroy(options);
}

TEST_F(DBSecondaryCacheTest, SecondaryCacheIntensiveTesting) {
  LRUCacheOptions opts(8 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 256;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1000);
    ASSERT_OK(Put(Key(i), p_v));
  }
  ASSERT_OK(Flush());
  Compact("a", "z");

  Random r_index(47);
  std::string v;
  for (int i = 0; i < 1000; i++) {
    uint32_t key_i = r_index.Next() % N;
    v = Get(Key(key_i));
  }

  // We have over 200 data blocks there will be multiple insertion
  // and lookups.
  ASSERT_GE(secondary_cache->num_inserts(), 1u);
  ASSERT_GE(secondary_cache->num_lookups(), 1u);

  Destroy(options);
}

// In this test, the block cache size is set to 4096, after insert 6 KV-pairs
// and flush, there are 5 blocks in this SST file, 2 data blocks and 3 meta
// blocks. block_1 size is 4096 and block_2 size is 2056. The total size
// of the meta blocks are about 900 to 1000. Therefore, in any situation,
// if we try to insert block_1 to the block cache, it will always fails. Only
// block_2 will be successfully inserted into the block cache.
TEST_F(DBSecondaryCacheTest, SecondaryCacheFailureTest) {
  LRUCacheOptions opts(4 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.paranoid_file_checks = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());
  // After Flush is successful, RocksDB will do the paranoid check for the new
  // SST file. Meta blocks are always cached in the block cache and they
  // will not be evicted. When block_2 is cache miss and read out, it is
  // inserted to the block cache. Note that, block_1 is never successfully
  // inserted to the block cache. Here are 2 lookups in the secondary cache
  // for block_1 and block_2
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  // Fail the insertion, in LRU cache, the secondary insertion returned status
  // is not checked, therefore, the DB will not be influenced.
  secondary_cache->InjectFailure();
  Compact("a", "z");
  // Compaction will create the iterator to scan the whole file. So all the
  // blocks are needed. Meta blocks are always cached. When block_1 is read
  // out, block_2 is evicted from block cache and inserted to secondary
  // cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 3u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // The first data block is not in the cache, similarly, trigger the block
  // cache Lookup and secondary cache lookup for block_1. But block_1 will not
  // be inserted successfully due to the size. Currently, cache only has
  // the meta blocks.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // The second data block is not in the cache, similarly, trigger the block
  // cache Lookup and secondary cache lookup for block_2 and block_2 is found
  // in the secondary cache. Now block cache has block_2
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());
  // block_2 is in the block cache. There is a block cache hit. No need to
  // lookup or insert the secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 5u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // Lookup the first data block, not in the block cache, so lookup the
  // secondary cache. Also not in the secondary cache. After Get, still
  // block_1 is will not be cached.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 6u);

  v = Get(Key(0));
  ASSERT_EQ(1007, v.size());
  // Lookup the first data block, not in the block cache, so lookup the
  // secondary cache. Also not in the secondary cache. After Get, still
  // block_1 is will not be cached.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 7u);
  secondary_cache->ResetInjectFailure();

  Destroy(options);
}

TEST_F(LRUCacheSecondaryCacheTest, BasicWaitAllTest) {
  LRUCacheOptions opts(1024 /* capacity */, 2 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(32 * 1024);
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  const int num_keys = 32;
  OffsetableCacheKey ock{"foo", "bar", 1};

  Random rnd(301);
  std::vector<std::string> values;
  for (int i = 0; i < num_keys; ++i) {
    std::string str = rnd.RandomString(1020);
    values.emplace_back(str);
    TestItem* item = new TestItem(str.data(), str.length());
    ASSERT_OK(cache->Insert(ock.WithOffset(i).AsSlice(), item,
                            &LRUCacheSecondaryCacheTest::helper_,
                            str.length()));
  }
  // Force all entries to be evicted to the secondary cache
  cache->SetCapacity(0);
  ASSERT_EQ(secondary_cache->num_inserts(), 32u);
  cache->SetCapacity(32 * 1024);

  secondary_cache->SetResultMap(
      {{ock.WithOffset(3).AsSlice().ToString(),
        TestSecondaryCache::ResultType::DEFER},
       {ock.WithOffset(4).AsSlice().ToString(),
        TestSecondaryCache::ResultType::DEFER_AND_FAIL},
       {ock.WithOffset(5).AsSlice().ToString(),
        TestSecondaryCache::ResultType::FAIL}});
  std::vector<Cache::Handle*> results;
  for (int i = 0; i < 6; ++i) {
    results.emplace_back(cache->Lookup(
        ock.WithOffset(i).AsSlice(), &LRUCacheSecondaryCacheTest::helper_,
        test_item_creator, Cache::Priority::LOW, false));
  }
  cache->WaitAll(results);
  for (int i = 0; i < 6; ++i) {
    if (i == 4) {
      ASSERT_EQ(cache->Value(results[i]), nullptr);
    } else if (i == 5) {
      ASSERT_EQ(results[i], nullptr);
      continue;
    } else {
      TestItem* item = static_cast<TestItem*>(cache->Value(results[i]));
      ASSERT_EQ(item->ToString(), values[i]);
    }
    cache->Release(results[i]);
  }

  cache.reset();
  secondary_cache.reset();
}

// In this test, we have one KV pair per data block. We indirectly determine
// the cache key associated with each data block (and thus each KV) by using
// a sync point callback in TestSecondaryCache::Lookup. We then control the
// lookup result by setting the ResultMap.
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheMultiGet) {
  LRUCacheOptions opts(1 << 20 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  table_options.cache_index_and_filter_blocks = false;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.paranoid_file_checks = true;
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 8;
  std::vector<std::string> keys;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(4000);
    keys.emplace_back(p_v);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());
  // After Flush is successful, RocksDB does the paranoid check for the new
  // SST file. This will try to lookup all data blocks in the secondary
  // cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 8u);

  cache->SetCapacity(0);
  ASSERT_EQ(secondary_cache->num_inserts(), 8u);
  cache->SetCapacity(1 << 20);

  std::vector<std::string> cache_keys;
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "TestSecondaryCache::Lookup", [&cache_keys](void* key) -> void {
        cache_keys.emplace_back(*(static_cast<std::string*>(key)));
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  for (int i = 0; i < N; ++i) {
    std::string v = Get(Key(i));
    ASSERT_EQ(4000, v.size());
    ASSERT_EQ(v, keys[i]);
  }
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ASSERT_EQ(secondary_cache->num_lookups(), 16u);
  cache->SetCapacity(0);
  cache->SetCapacity(1 << 20);

  ASSERT_EQ(Get(Key(2)), keys[2]);
  ASSERT_EQ(Get(Key(7)), keys[7]);
  secondary_cache->SetResultMap(
      {{cache_keys[3], TestSecondaryCache::ResultType::DEFER},
       {cache_keys[4], TestSecondaryCache::ResultType::DEFER_AND_FAIL},
       {cache_keys[5], TestSecondaryCache::ResultType::FAIL}});

  std::vector<std::string> mget_keys(
      {Key(0), Key(1), Key(2), Key(3), Key(4), Key(5), Key(6), Key(7)});
  std::vector<PinnableSlice> values(mget_keys.size());
  std::vector<Status> s(keys.size());
  std::vector<Slice> key_slices;
  for (const std::string& key : mget_keys) {
    key_slices.emplace_back(key);
  }
  uint32_t num_lookups = secondary_cache->num_lookups();
  dbfull()->MultiGet(ReadOptions(), dbfull()->DefaultColumnFamily(),
                     key_slices.size(), key_slices.data(), values.data(),
                     s.data(), false);
  ASSERT_EQ(secondary_cache->num_lookups(), num_lookups + 5);
  for (int i = 0; i < N; ++i) {
    ASSERT_OK(s[i]);
    ASSERT_EQ(values[i].ToString(), keys[i]);
    values[i].Reset();
  }
  Destroy(options);
}

class LRUCacheWithStat : public LRUCache {
 public:
  LRUCacheWithStat(
      size_t _capacity, int _num_shard_bits, bool _strict_capacity_limit,
      double _high_pri_pool_ratio, double _low_pri_pool_ratio,
      std::shared_ptr<MemoryAllocator> _memory_allocator = nullptr,
      bool _use_adaptive_mutex = kDefaultToAdaptiveMutex,
      CacheMetadataChargePolicy _metadata_charge_policy =
          kDontChargeCacheMetadata,
      const std::shared_ptr<SecondaryCache>& _secondary_cache = nullptr)
      : LRUCache(_capacity, _num_shard_bits, _strict_capacity_limit,
                 _high_pri_pool_ratio, _low_pri_pool_ratio, _memory_allocator,
                 _use_adaptive_mutex, _metadata_charge_policy,
                 _secondary_cache) {
    insert_count_ = 0;
    lookup_count_ = 0;
  }
  ~LRUCacheWithStat() {}

  Status Insert(const Slice& key, void* value, size_t charge, DeleterFn deleter,
                Handle** handle, Priority priority) override {
    insert_count_++;
    return LRUCache::Insert(key, value, charge, deleter, handle, priority);
  }
  Status Insert(const Slice& key, void* value, const CacheItemHelper* helper,
                size_t charge, Handle** handle = nullptr,
                Priority priority = Priority::LOW) override {
    insert_count_++;
    return LRUCache::Insert(key, value, helper, charge, handle, priority);
  }
  Handle* Lookup(const Slice& key, Statistics* stats) override {
    lookup_count_++;
    return LRUCache::Lookup(key, stats);
  }
  Handle* Lookup(const Slice& key, const CacheItemHelper* helper,
                 const CreateCallback& create_cb, Priority priority, bool wait,
                 Statistics* stats = nullptr) override {
    lookup_count_++;
    return LRUCache::Lookup(key, helper, create_cb, priority, wait, stats);
  }

  uint32_t GetInsertCount() { return insert_count_; }
  uint32_t GetLookupcount() { return lookup_count_; }
  void ResetCount() {
    insert_count_ = 0;
    lookup_count_ = 0;
  }

 private:
  uint32_t insert_count_;
  uint32_t lookup_count_;
};

#ifndef ROCKSDB_LITE

TEST(CacheWarmupDumpFormatTest, RoundTripsPriorityAndRejectsCorruption) {
  std::array<char, kCacheKeySize> key{};
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = static_cast<char>(i + 1);
  }
  const std::string value = "warm-block";
  for (Cache::Priority priority :
       {Cache::Priority::HIGH, Cache::Priority::LOW, Cache::Priority::BOTTOM}) {
    CacheWarmupDumpUnit input;
    input.type = CacheWarmupDumpUnitType::kData;
    input.priority = priority;
    input.key = Slice(key.data(), key.size());
    input.value_len = value.size();
    input.value = const_cast<char*>(value.data());
    std::string encoded;
    ASSERT_OK(CacheDumperHelper::EncodeWarmupDumpUnit(input, &encoded));

    CacheWarmupDumpUnit decoded;
    ASSERT_OK(CacheDumperHelper::DecodeWarmupDumpUnit(encoded, &decoded));
    EXPECT_EQ(CacheWarmupDumpUnitType::kData, decoded.type);
    EXPECT_EQ(priority, decoded.priority);
    EXPECT_EQ(input.key, decoded.key);
    EXPECT_EQ(
        value,
        Slice(static_cast<char*>(decoded.value), decoded.value_len).ToString());

    std::string bad_magic = encoded;
    bad_magic[0] ^= 1;
    EXPECT_TRUE(CacheDumperHelper::DecodeWarmupDumpUnit(bad_magic, &decoded)
                    .IsCorruption());

    std::string bad_priority = encoded;
    bad_priority[9] = static_cast<char>(0xff);
    EXPECT_TRUE(CacheDumperHelper::DecodeWarmupDumpUnit(bad_priority, &decoded)
                    .IsCorruption());
  }

  CacheWarmupDumpUnit bad_key;
  bad_key.type = CacheWarmupDumpUnitType::kData;
  bad_key.priority = Cache::Priority::LOW;
  bad_key.key = Slice("short");
  bad_key.value_len = value.size();
  bad_key.value = const_cast<char*>(value.data());
  std::string encoded_bad_key;
  ASSERT_OK(CacheDumperHelper::EncodeWarmupDumpUnit(bad_key, &encoded_bad_key));
  CacheWarmupDumpUnit decoded;
  EXPECT_TRUE(CacheDumperHelper::DecodeWarmupDumpUnit(encoded_bad_key, &decoded)
                  .IsCorruption());
}

TEST_F(DBSecondaryCacheTest, LRUCacheDumpLoadBasic) {
  LRUCacheOptions cache_opts(1024 * 1024 /* capacity */, 0 /* num_shard_bits */,
                             false /* strict_capacity_limit */,
                             0.5 /* high_pri_pool_ratio */,
                             nullptr /* memory_allocator */,
                             kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  LRUCacheWithStat* tmp_cache = new LRUCacheWithStat(
      cache_opts.capacity, cache_opts.num_shard_bits,
      cache_opts.strict_capacity_limit, cache_opts.high_pri_pool_ratio,
      cache_opts.low_pri_pool_ratio, cache_opts.memory_allocator,
      cache_opts.use_adaptive_mutex, cache_opts.metadata_charge_policy,
      cache_opts.secondary_cache);
  std::shared_ptr<Cache> cache(tmp_cache);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  DestroyAndReopen(options);
  fault_fs_->SetFailGetUniqueId(true);

  Random rnd(301);
  const int N = 256;
  std::vector<std::string> value;
  char buf[1000];
  memset(buf, 'a', 1000);
  value.resize(N);
  for (int i = 0; i < N; i++) {
    // std::string p_v = rnd.RandomString(1000);
    std::string p_v(buf, 1000);
    value[i] = p_v;
    ASSERT_OK(Put(Key(i), p_v));
  }
  ASSERT_OK(Flush());
  Compact("a", "z");

  // do th eread for all the key value pairs, so all the blocks should be in
  // cache
  uint32_t start_insert = tmp_cache->GetInsertCount();
  uint32_t start_lookup = tmp_cache->GetLookupcount();
  std::string v;
  for (int i = 0; i < N; i++) {
    v = Get(Key(i));
    ASSERT_EQ(v, value[i]);
  }
  uint32_t dump_insert = tmp_cache->GetInsertCount() - start_insert;
  uint32_t dump_lookup = tmp_cache->GetLookupcount() - start_lookup;
  ASSERT_EQ(63,
            static_cast<int>(dump_insert));  // the insert in the block cache
  ASSERT_EQ(256,
            static_cast<int>(dump_lookup));  // the lookup in the block cache
  // We have enough blocks in the block cache

  CacheDumpOptions cd_options;
  cd_options.clock = fault_env_->GetSystemClock().get();
  std::string dump_path = db_->GetName() + "/cache_dump";
  std::unique_ptr<CacheDumpWriter> dump_writer;
  Status s = NewToFileCacheDumpWriter(fault_fs_, FileOptions(), dump_path,
                                      &dump_writer);
  ASSERT_OK(s);
  std::unique_ptr<CacheDumper> cache_dumper;
  s = NewDefaultCacheDumper(cd_options, cache, std::move(dump_writer),
                            &cache_dumper);
  ASSERT_OK(s);
  std::vector<DB*> db_list;
  db_list.push_back(db_);
  s = cache_dumper->SetDumpFilter(db_list);
  ASSERT_OK(s);
  s = cache_dumper->DumpCacheEntriesToWriter();
  ASSERT_OK(s);
  cache_dumper.reset();

  // we have a new cache it is empty, then, before we do the Get, we do the
  // dumpload
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048 * 1024);
  cache_opts.secondary_cache = secondary_cache;
  tmp_cache = new LRUCacheWithStat(
      cache_opts.capacity, cache_opts.num_shard_bits,
      cache_opts.strict_capacity_limit, cache_opts.high_pri_pool_ratio,
      cache_opts.low_pri_pool_ratio, cache_opts.memory_allocator,
      cache_opts.use_adaptive_mutex, cache_opts.metadata_charge_policy,
      cache_opts.secondary_cache);
  std::shared_ptr<Cache> cache_new(tmp_cache);
  table_options.block_cache = cache_new;
  table_options.block_size = 4 * 1024;
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();

  // start to load the data to new block cache
  start_insert = secondary_cache->num_inserts();
  start_lookup = secondary_cache->num_lookups();
  std::unique_ptr<CacheDumpReader> dump_reader;
  s = NewFromFileCacheDumpReader(fault_fs_, FileOptions(), dump_path,
                                 &dump_reader);
  ASSERT_OK(s);
  std::unique_ptr<CacheDumpedLoader> cache_loader;
  s = NewDefaultCacheDumpedLoader(cd_options, table_options, secondary_cache,
                                  std::move(dump_reader), &cache_loader);
  ASSERT_OK(s);
  s = cache_loader->RestoreCacheEntriesToSecondaryCache();
  ASSERT_OK(s);
  uint32_t load_insert = secondary_cache->num_inserts() - start_insert;
  uint32_t load_lookup = secondary_cache->num_lookups() - start_lookup;
  // check the number we inserted
  ASSERT_EQ(64, static_cast<int>(load_insert));
  ASSERT_EQ(0, static_cast<int>(load_lookup));
  ASSERT_OK(s);

  Reopen(options);

  // After load, we do the Get again
  start_insert = secondary_cache->num_inserts();
  start_lookup = secondary_cache->num_lookups();
  uint32_t cache_insert = tmp_cache->GetInsertCount();
  uint32_t cache_lookup = tmp_cache->GetLookupcount();
  for (int i = 0; i < N; i++) {
    v = Get(Key(i));
    ASSERT_EQ(v, value[i]);
  }
  uint32_t final_insert = secondary_cache->num_inserts() - start_insert;
  uint32_t final_lookup = secondary_cache->num_lookups() - start_lookup;
  // no insert to secondary cache
  ASSERT_EQ(0, static_cast<int>(final_insert));
  // lookup the secondary to get all blocks
  ASSERT_EQ(64, static_cast<int>(final_lookup));
  uint32_t block_insert = tmp_cache->GetInsertCount() - cache_insert;
  uint32_t block_lookup = tmp_cache->GetLookupcount() - cache_lookup;
  // Check the new block cache insert and lookup, should be no insert since all
  // blocks are from the secondary cache.
  ASSERT_EQ(0, static_cast<int>(block_insert));
  ASSERT_EQ(256, static_cast<int>(block_lookup));

  fault_fs_->SetFailGetUniqueId(false);
  Destroy(options);
}

TEST_F(DBSecondaryCacheTest, LRUCacheDumpLoadWithFilter) {
  LRUCacheOptions cache_opts(1024 * 1024 /* capacity */, 0 /* num_shard_bits */,
                             false /* strict_capacity_limit */,
                             0.5 /* high_pri_pool_ratio */,
                             nullptr /* memory_allocator */,
                             kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  LRUCacheWithStat* tmp_cache = new LRUCacheWithStat(
      cache_opts.capacity, cache_opts.num_shard_bits,
      cache_opts.strict_capacity_limit, cache_opts.high_pri_pool_ratio,
      cache_opts.low_pri_pool_ratio, cache_opts.memory_allocator,
      cache_opts.use_adaptive_mutex, cache_opts.metadata_charge_policy,
      cache_opts.secondary_cache);
  std::shared_ptr<Cache> cache(tmp_cache);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  std::string dbname1 = test::PerThreadDBPath("db_1");
  ASSERT_OK(DestroyDB(dbname1, options));
  DB* db1 = nullptr;
  ASSERT_OK(DB::Open(options, dbname1, &db1));
  std::string dbname2 = test::PerThreadDBPath("db_2");
  ASSERT_OK(DestroyDB(dbname2, options));
  DB* db2 = nullptr;
  ASSERT_OK(DB::Open(options, dbname2, &db2));
  fault_fs_->SetFailGetUniqueId(true);

  // write the KVs to db1
  Random rnd(301);
  const int N = 256;
  std::vector<std::string> value1;
  WriteOptions wo;
  char buf[1000];
  memset(buf, 'a', 1000);
  value1.resize(N);
  for (int i = 0; i < N; i++) {
    std::string p_v(buf, 1000);
    value1[i] = p_v;
    ASSERT_OK(db1->Put(wo, Key(i), p_v));
  }
  ASSERT_OK(db1->Flush(FlushOptions()));
  Slice bg("a");
  Slice ed("b");
  ASSERT_OK(db1->CompactRange(CompactRangeOptions(), &bg, &ed));

  // Write the KVs to DB2
  std::vector<std::string> value2;
  memset(buf, 'b', 1000);
  value2.resize(N);
  for (int i = 0; i < N; i++) {
    std::string p_v(buf, 1000);
    value2[i] = p_v;
    ASSERT_OK(db2->Put(wo, Key(i), p_v));
  }
  ASSERT_OK(db2->Flush(FlushOptions()));
  ASSERT_OK(db2->CompactRange(CompactRangeOptions(), &bg, &ed));

  // do th eread for all the key value pairs, so all the blocks should be in
  // cache
  uint32_t start_insert = tmp_cache->GetInsertCount();
  uint32_t start_lookup = tmp_cache->GetLookupcount();
  ReadOptions ro;
  std::string v;
  for (int i = 0; i < N; i++) {
    ASSERT_OK(db1->Get(ro, Key(i), &v));
    ASSERT_EQ(v, value1[i]);
  }
  for (int i = 0; i < N; i++) {
    ASSERT_OK(db2->Get(ro, Key(i), &v));
    ASSERT_EQ(v, value2[i]);
  }
  uint32_t dump_insert = tmp_cache->GetInsertCount() - start_insert;
  uint32_t dump_lookup = tmp_cache->GetLookupcount() - start_lookup;
  ASSERT_EQ(128,
            static_cast<int>(dump_insert));  // the insert in the block cache
  ASSERT_EQ(512,
            static_cast<int>(dump_lookup));  // the lookup in the block cache
  // We have enough blocks in the block cache

  CacheDumpOptions cd_options;
  cd_options.clock = fault_env_->GetSystemClock().get();
  std::string dump_path = db1->GetName() + "/cache_dump";
  std::unique_ptr<CacheDumpWriter> dump_writer;
  Status s = NewToFileCacheDumpWriter(fault_fs_, FileOptions(), dump_path,
                                      &dump_writer);
  ASSERT_OK(s);
  std::unique_ptr<CacheDumper> cache_dumper;
  s = NewDefaultCacheDumper(cd_options, cache, std::move(dump_writer),
                            &cache_dumper);
  ASSERT_OK(s);
  std::vector<DB*> db_list;
  db_list.push_back(db1);
  s = cache_dumper->SetDumpFilter(db_list);
  ASSERT_OK(s);
  s = cache_dumper->DumpCacheEntriesToWriter();
  ASSERT_OK(s);
  cache_dumper.reset();

  // we have a new cache it is empty, then, before we do the Get, we do the
  // dumpload
  std::shared_ptr<TestSecondaryCache> secondary_cache =
      std::make_shared<TestSecondaryCache>(2048 * 1024);
  cache_opts.secondary_cache = secondary_cache;
  tmp_cache = new LRUCacheWithStat(
      cache_opts.capacity, cache_opts.num_shard_bits,
      cache_opts.strict_capacity_limit, cache_opts.high_pri_pool_ratio,
      cache_opts.low_pri_pool_ratio, cache_opts.memory_allocator,
      cache_opts.use_adaptive_mutex, cache_opts.metadata_charge_policy,
      cache_opts.secondary_cache);
  std::shared_ptr<Cache> cache_new(tmp_cache);
  table_options.block_cache = cache_new;
  table_options.block_size = 4 * 1024;
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();

  // Start the cache loading process
  start_insert = secondary_cache->num_inserts();
  start_lookup = secondary_cache->num_lookups();
  std::unique_ptr<CacheDumpReader> dump_reader;
  s = NewFromFileCacheDumpReader(fault_fs_, FileOptions(), dump_path,
                                 &dump_reader);
  ASSERT_OK(s);
  std::unique_ptr<CacheDumpedLoader> cache_loader;
  s = NewDefaultCacheDumpedLoader(cd_options, table_options, secondary_cache,
                                  std::move(dump_reader), &cache_loader);
  ASSERT_OK(s);
  s = cache_loader->RestoreCacheEntriesToSecondaryCache();
  ASSERT_OK(s);
  uint32_t load_insert = secondary_cache->num_inserts() - start_insert;
  uint32_t load_lookup = secondary_cache->num_lookups() - start_lookup;
  // check the number we inserted
  ASSERT_EQ(64, static_cast<int>(load_insert));
  ASSERT_EQ(0, static_cast<int>(load_lookup));
  ASSERT_OK(s);

  ASSERT_OK(db1->Close());
  delete db1;
  ASSERT_OK(DB::Open(options, dbname1, &db1));

  // After load, we do the Get again. To validate the cache, we do not allow any
  // I/O, so we set the file system to false.
  IOStatus error_msg = IOStatus::IOError("Retryable IO Error");
  fault_fs_->SetFilesystemActive(false, error_msg);
  start_insert = secondary_cache->num_inserts();
  start_lookup = secondary_cache->num_lookups();
  uint32_t cache_insert = tmp_cache->GetInsertCount();
  uint32_t cache_lookup = tmp_cache->GetLookupcount();
  for (int i = 0; i < N; i++) {
    ASSERT_OK(db1->Get(ro, Key(i), &v));
    ASSERT_EQ(v, value1[i]);
  }
  uint32_t final_insert = secondary_cache->num_inserts() - start_insert;
  uint32_t final_lookup = secondary_cache->num_lookups() - start_lookup;
  // no insert to secondary cache
  ASSERT_EQ(0, static_cast<int>(final_insert));
  // lookup the secondary to get all blocks
  ASSERT_EQ(64, static_cast<int>(final_lookup));
  uint32_t block_insert = tmp_cache->GetInsertCount() - cache_insert;
  uint32_t block_lookup = tmp_cache->GetLookupcount() - cache_lookup;
  // Check the new block cache insert and lookup, should be no insert since all
  // blocks are from the secondary cache.
  ASSERT_EQ(0, static_cast<int>(block_insert));
  ASSERT_EQ(256, static_cast<int>(block_lookup));
  fault_fs_->SetFailGetUniqueId(false);
  fault_fs_->SetFilesystemActive(true);
  delete db1;
  delete db2;
  ASSERT_OK(DestroyDB(dbname1, options));
  ASSERT_OK(DestroyDB(dbname2, options));
}

// [relink cache handoff, RDMA pull] Engine-side contract of the pull
// transport, without any DB, network or cluster: the catalog leases the
// selected data blocks in place (no copy), the leases survive eviction
// pressure, Release() makes the blocks evictable again, and the per-unit
// loader entry point admits a block straight from a catalog pointer into a
// second cache with duplicate detection.
namespace {

// A minimal well-formed kDataBlockBinarySearch block: `size - 8` random
// payload bytes, one restart offset (0) and num_restarts = 1, so Block's
// constructor accepts it and reports size() == size.
std::string MakeWarmupPullTestBlock(Random* rnd, size_t size) {
  std::string block =
      rnd->RandomString(static_cast<int>(size - 2 * sizeof(uint32_t)));
  PutFixed32(&block, 0);  // restart[0]
  PutFixed32(&block, 1);  // num_restarts
  return block;
}

// 16-byte cache key = stable 8-byte prefix + 8-byte offset, the shape the
// warmup prefix filter expects.
std::string MakeWarmupPullTestKey(const std::string& prefix8,
                                  uint64_t offset) {
  std::string key = prefix8;
  PutFixed64(&key, offset);
  EXPECT_EQ(static_cast<size_t>(kCacheKeySize), key.size());
  return key;
}

// Inserts a Block with the real kData/kIndex helper deleter (so the dumper's
// role map classifies it) and returns the live Block::data() pointer the
// cache now owns, i.e. the address an RDMA READ would target.
const char* InsertWarmupPullTestBlock(Cache* cache, const std::string& key,
                                      const std::string& bytes,
                                      Cache::Priority priority,
                                      BlockType block_type) {
  CacheAllocationPtr buf = AllocateBlock(bytes.size(), nullptr);
  memcpy(buf.get(), bytes.data(), bytes.size());
  BlockContents contents(std::move(buf), bytes.size());
  std::unique_ptr<Block> block(new Block(std::move(contents)));
  EXPECT_EQ(bytes.size(), block->size());
  Cache::CacheItemHelper* helper =
      BlocklikeTraits<Block>::GetCacheItemHelper(block_type);
  EXPECT_NE(nullptr, helper);
  const char* data = block->data();
  EXPECT_OK(cache->Insert(key, block.get(), block->ApproximateMemoryUsage(),
                          helper->del_cb, nullptr /*handle*/, priority));
  block.release();  // owned by the cache now
  return data;
}

// No-touch enumeration of resident keys (does not record hits or promote).
std::set<std::string> ResidentWarmupPullTestKeys(Cache* cache) {
  std::set<std::string> keys;
  Cache::ApplyToAllEntriesOptions opts;
  EXPECT_OK(cache->ApplyToAllEntriesForCacheWarmup(
      [&](const Slice& key, size_t /*charge*/, Cache::DeleterFn /*deleter*/,
          Cache::Priority /*priority*/) { keys.insert(key.ToString()); },
      opts));
  return keys;
}

}  // namespace

TEST(CacheWarmupPullTest, CatalogLeasesAndInsertRoundTrip) {
  Random rnd(301);
  const size_t kBlockSize = 4096;
  const std::string kPrefix("\x01\x02\x03\x04\x05\x06\x07\x08", 8);
  const std::string kOtherPrefix("\x11\x12\x13\x14\x15\x16\x17\x18", 8);

  // Every test block has the same size, hence the same cache charge; size the
  // caches in whole blocks so the eviction arithmetic below is exact.
  size_t block_charge = 0;
  {
    std::string probe = MakeWarmupPullTestBlock(&rnd, kBlockSize);
    CacheAllocationPtr buf = AllocateBlock(probe.size(), nullptr);
    memcpy(buf.get(), probe.data(), probe.size());
    Block block(BlockContents(std::move(buf), probe.size()));
    block_charge = block.ApproximateMemoryUsage();
  }
  ASSERT_GT(block_charge, kBlockSize);
  const size_t kCapacityBlocks = 12;
  LRUCacheOptions cache_opts(
      kCapacityBlocks * block_charge, 0 /*num_shard_bits*/,
      false /*strict_capacity_limit*/, 0.5 /*high_pri_pool_ratio*/,
      nullptr /*memory_allocator*/, kDefaultToAdaptiveMutex,
      kDontChargeCacheMetadata, 0.5 /*low_pri_pool_ratio*/);
  std::shared_ptr<Cache> src = NewLRUCache(cache_opts);
  ASSERT_NE(nullptr, src);

  // 8 data blocks under the migrated prefix: 4 HIGH, 4 LOW.
  const size_t kBlocks = 8;
  std::vector<std::string> keys(kBlocks);
  std::vector<std::string> bytes(kBlocks);
  std::vector<const char*> src_data(kBlocks);
  for (size_t i = 0; i < kBlocks; ++i) {
    keys[i] = MakeWarmupPullTestKey(kPrefix, i);
    bytes[i] = MakeWarmupPullTestBlock(&rnd, kBlockSize);
    src_data[i] = InsertWarmupPullTestBlock(
        src.get(), keys[i], bytes[i],
        i < 4 ? Cache::Priority::HIGH : Cache::Priority::LOW,
        BlockType::kData);
  }
  // A data block of another file set (silently filtered by prefix) and an
  // index block of the migrated set (skipped_unsupported: data blocks only).
  InsertWarmupPullTestBlock(src.get(), MakeWarmupPullTestKey(kOtherPrefix, 0),
                            MakeWarmupPullTestBlock(&rnd, kBlockSize),
                            Cache::Priority::LOW, BlockType::kData);
  InsertWarmupPullTestBlock(src.get(), MakeWarmupPullTestKey(kPrefix, 99),
                            MakeWarmupPullTestBlock(&rnd, kBlockSize),
                            Cache::Priority::HIGH, BlockType::kIndex);
  ASSERT_EQ(kBlocks + 2, ResidentWarmupPullTestKeys(src.get()).size());

  // Catalog: the pull path never touches the writer, so none is needed.
  CacheDumpOptions cd_options;
  cd_options.clock = SystemClock::Default().get();
  std::unique_ptr<CacheDumper> dumper;
  ASSERT_OK(NewDefaultCacheDumper(cd_options, src, nullptr /*writer*/,
                                  &dumper));
  ASSERT_OK(dumper->SetDumpFilterPrefixes({kPrefix}));
  CacheWarmupOptions warmup_options;
  warmup_options.max_entry_bytes = kBlockSize;  // exactly fits
  std::unique_ptr<CacheWarmupPullCatalog> catalog;
  CacheWarmupTransferStats stats;
  ASSERT_OK(
      dumper->CatalogWarmupDataBlocksForPull(warmup_options, &catalog, &stats));
  ASSERT_NE(nullptr, catalog);
  EXPECT_EQ(kBlocks, stats.cataloged_entries);
  EXPECT_EQ(kBlocks, stats.data_candidates);
  EXPECT_EQ(kBlocks, stats.entries_staged);
  EXPECT_EQ(0U, stats.entries_written);  // nothing is written on this path
  EXPECT_EQ(kBlocks * kBlockSize, stats.payload_bytes);
  EXPECT_EQ(1U, stats.skipped_unsupported);  // the index block
  EXPECT_EQ(0U, stats.skipped_disappeared);
  EXPECT_EQ(0U, stats.skipped_replaced);
  EXPECT_EQ(0U, stats.skipped_too_large);
  EXPECT_EQ(0U, stats.priority_changed_after_catalog);
  EXPECT_EQ(4U, stats.high.entries_staged);
  EXPECT_EQ(4U, stats.low.entries_staged);
  EXPECT_EQ(0U, stats.bottom.entries_staged);
  EXPECT_EQ(4U * kBlockSize, stats.high.payload_bytes);
  EXPECT_EQ(4U * kBlockSize, stats.low.payload_bytes);
  EXPECT_EQ(kBlocks, dumper->GetCacheWarmupTransferStats().entries_staged);

  const std::vector<CacheWarmupPulledBlock>& blocks = catalog->blocks();
  ASSERT_EQ(kBlocks, blocks.size());
  EXPECT_EQ(kBlocks * kBlockSize, catalog->payload_bytes());
  std::set<std::string> cataloged_keys;
  for (size_t n = 0; n < blocks.size(); ++n) {
    const CacheWarmupPulledBlock& b = blocks[n];
    // HIGH -> LOW order, as the streamed transport emits it.
    EXPECT_EQ(n < 4 ? Cache::Priority::HIGH : Cache::Priority::LOW,
              b.priority);
    ASSERT_EQ(static_cast<size_t>(kCacheKeySize), b.key.size());
    ASSERT_EQ(kBlockSize, b.size);
    ASSERT_NE(nullptr, b.data);
    size_t i = 0;
    for (; i < kBlocks; ++i) {
      if (keys[i] == b.key) {
        break;
      }
    }
    ASSERT_LT(i, kBlocks) << "cataloged key is not one of the inserted keys";
    // No copy: the catalog exposes the cache-owned Block's own bytes.
    EXPECT_EQ(src_data[i], static_cast<const char*>(b.data));
    EXPECT_EQ(0, memcmp(b.data, bytes[i].data(), kBlockSize));
    cataloged_keys.insert(b.key);
  }
  EXPECT_EQ(kBlocks, cataloged_keys.size());

  // Eviction pressure while leased: 20 HIGH fillers of one block charge each
  // into a 12-block cache. A leased entry is skipped by eviction, so all 8
  // stay resident and their bytes stay readable at the cataloged address.
  auto insert_fillers = [&](const char* tag) {
    for (int k = 0; k < 20; ++k) {
      std::string key = std::string(tag) + std::to_string(k);
      ASSERT_OK(src->Insert(key, nullptr /*value*/, block_charge,
                            nullptr /*deleter*/, nullptr /*handle*/,
                            Cache::Priority::HIGH));
    }
  };
  insert_fillers("filler-leased-");
  {
    std::set<std::string> resident = ResidentWarmupPullTestKeys(src.get());
    for (size_t i = 0; i < kBlocks; ++i) {
      EXPECT_EQ(1U, resident.count(keys[i])) << "leased block " << i
                                              << " was evicted";
    }
  }
  for (const CacheWarmupPulledBlock& b : blocks) {
    for (size_t i = 0; i < kBlocks; ++i) {
      if (keys[i] == b.key) {
        EXPECT_EQ(0, memcmp(b.data, bytes[i].data(), kBlockSize));
      }
    }
  }

  // Destination: admit each block from the catalog pointer, as the RDMA
  // receiver does after its READ lands, through the shared per-unit path.
  std::shared_ptr<Cache> dst = NewLRUCache(cache_opts);
  ASSERT_NE(nullptr, dst);
  BlockBasedTableOptions toptions;
  std::unique_ptr<CacheDumpedLoader> loader;
  ASSERT_OK(NewDefaultCacheDumpedLoaderToPrimary(
      cd_options, toptions, dst, nullptr /*reader*/, &loader));
  CacheWarmupTransferStats lstats;
  for (const CacheWarmupPulledBlock& b : blocks) {
    ASSERT_OK(loader->InsertWarmupDataBlock(
        Slice(b.key), b.priority, static_cast<const char*>(b.data), b.size,
        &lstats));
  }
  EXPECT_EQ(kBlocks, lstats.entries_received);
  EXPECT_EQ(kBlocks, lstats.entries_inserted);
  EXPECT_EQ(0U, lstats.entries_duplicate);
  EXPECT_EQ(0U, lstats.entries_rejected_no_space);
  EXPECT_EQ(kBlocks * kBlockSize, lstats.payload_bytes);
  EXPECT_EQ(4U, lstats.high.entries_inserted);
  EXPECT_EQ(4U, lstats.low.entries_inserted);
  // A second pass is all duplicates; the destination keeps its own copies.
  for (const CacheWarmupPulledBlock& b : blocks) {
    ASSERT_OK(loader->InsertWarmupDataBlock(
        Slice(b.key), b.priority, static_cast<const char*>(b.data), b.size,
        &lstats));
  }
  EXPECT_EQ(2 * kBlocks, lstats.entries_received);
  EXPECT_EQ(kBlocks, lstats.entries_inserted);
  EXPECT_EQ(kBlocks, lstats.entries_duplicate);
  EXPECT_EQ(4U, lstats.high.entries_duplicate);
  EXPECT_EQ(4U, lstats.low.entries_duplicate);
  // A malformed unit is refused and counted, never admitted.
  EXPECT_TRUE(loader
                  ->InsertWarmupDataBlock(Slice("short-key"),
                                          Cache::Priority::LOW,
                                          bytes[0].data(), bytes[0].size(),
                                          &lstats)
                  .IsCorruption());
  EXPECT_EQ(1U, lstats.skipped_invalid);
  EXPECT_EQ(2 * kBlocks, lstats.entries_received);
  // The destination holds byte-identical, independently owned copies.
  for (size_t i = 0; i < kBlocks; ++i) {
    Cache::Handle* h = dst->Lookup(keys[i]);
    ASSERT_NE(nullptr, h) << "block " << i << " not admitted";
    const Block* blk = static_cast<const Block*>(dst->Value(h));
    ASSERT_EQ(kBlockSize, blk->size());
    EXPECT_EQ(0, memcmp(blk->data(), bytes[i].data(), kBlockSize));
    EXPECT_NE(src_data[i], blk->data());
    dst->Release(h);
  }

  // Release() ends the leases (idempotently); the same pressure now evicts
  // every previously resident entry, the 8 pulled blocks included.
  catalog->Release();
  catalog->Release();
  for (const CacheWarmupPulledBlock& b : catalog->blocks()) {
    EXPECT_EQ(nullptr, b.data);  // dangling pointers are nulled on release
  }
  insert_fillers("filler-released-");
  {
    std::set<std::string> resident = ResidentWarmupPullTestKeys(src.get());
    for (size_t i = 0; i < kBlocks; ++i) {
      EXPECT_EQ(0U, resident.count(keys[i])) << "released block " << i
                                              << " is still resident";
    }
  }
  catalog.reset();  // destructor after Release() must be a no-op
  dumper.reset();
  loader.reset();
}

// [relink cache handoff, zero-copy landing, 2026-09-23] The owned-buffer
// admission entry point: on kInserted the cache keeps the caller's buffer
// (Block::data() is the very pointer handed in, no copy); on kDuplicate,
// kRejectedNoSpace and on a gate failure the buffer goes back through its
// allocator exactly once before the call returns; concurrent owned inserts
// from several threads, each with its own loader and stats, admit every
// distinct key. A counting allocator plays the role of the pinned arena.
TEST(CacheWarmupPullTest, OwnedInsertOwnershipAndConcurrency) {
  Random rnd(302);
  const size_t kBlockSize = 4096;
  const std::string kPrefix("\x21\x22\x23\x24\x25\x26\x27\x28", 8);
  CountedMemoryAllocator alloc;

  // Charge of one test block whose payload came from the counted allocator
  // (its UsableSize is the allocation size, so every block charges the same).
  size_t block_charge = 0;
  {
    std::string probe = MakeWarmupPullTestBlock(&rnd, kBlockSize);
    CacheAllocationPtr buf = AllocateBlock(probe.size(), &alloc);
    memcpy(buf.get(), probe.data(), probe.size());
    Block block(BlockContents(std::move(buf), probe.size()));
    block_charge = block.ApproximateMemoryUsage();
  }
  ASSERT_GT(block_charge, kBlockSize);
  ASSERT_EQ(1U, alloc.GetNumAllocations());
  ASSERT_EQ(1U, alloc.GetNumDeallocations());

  CacheDumpOptions cd_options;
  cd_options.clock = SystemClock::Default().get();
  BlockBasedTableOptions toptions;
  auto make_cache = [&](size_t capacity_blocks, int num_shard_bits) {
    LRUCacheOptions opts(capacity_blocks * block_charge, num_shard_bits,
                         false /*strict_capacity_limit*/,
                         0.5 /*high_pri_pool_ratio*/,
                         nullptr /*memory_allocator*/, kDefaultToAdaptiveMutex,
                         kDontChargeCacheMetadata, 0.5 /*low_pri_pool_ratio*/);
    return NewLRUCache(opts);
  };
  auto make_loader = [&](const std::shared_ptr<Cache>& cache) {
    std::unique_ptr<CacheDumpedLoader> loader;
    EXPECT_OK(NewDefaultCacheDumpedLoaderToPrimary(cd_options, toptions, cache,
                                                   nullptr /*reader*/,
                                                   &loader));
    return loader;
  };
  auto make_owned = [&](const std::string& bytes) {
    CacheAllocationPtr buf = AllocateBlock(bytes.size(), &alloc);
    memcpy(buf.get(), bytes.data(), bytes.size());
    return buf;
  };

  // kInserted keeps the buffer; kDuplicate and a gate failure release it.
  {
    std::shared_ptr<Cache> cache = make_cache(12, 0 /*num_shard_bits*/);
    ASSERT_NE(nullptr, cache);
    std::unique_ptr<CacheDumpedLoader> loader = make_loader(cache);
    ASSERT_NE(nullptr, loader);
    CacheWarmupTransferStats st;
    const std::string key = MakeWarmupPullTestKey(kPrefix, 0);
    const std::string bytes = MakeWarmupPullTestBlock(&rnd, kBlockSize);

    CacheAllocationPtr buf = make_owned(bytes);
    const char* raw = buf.get();
    const uint64_t allocs_before = alloc.GetNumAllocations();
    const uint64_t frees_before = alloc.GetNumDeallocations();
    ASSERT_OK(loader->InsertWarmupDataBlockOwned(
        Slice(key), Cache::Priority::HIGH, std::move(buf), bytes.size(), &st));
    EXPECT_EQ(nullptr, buf.get());
    EXPECT_EQ(1U, st.entries_received);
    EXPECT_EQ(1U, st.entries_inserted);
    EXPECT_EQ(1U, st.high.entries_inserted);
    EXPECT_EQ(kBlockSize, st.payload_bytes);
    EXPECT_EQ(allocs_before, alloc.GetNumAllocations());
    EXPECT_EQ(frees_before, alloc.GetNumDeallocations());
    {
      Cache::Handle* h = cache->Lookup(key);
      ASSERT_NE(nullptr, h);
      const Block* blk = static_cast<const Block*>(cache->Value(h));
      ASSERT_EQ(kBlockSize, blk->size());
      EXPECT_EQ(raw, blk->data());  // zero copy: the very buffer handed in
      EXPECT_EQ(0, memcmp(blk->data(), bytes.data(), kBlockSize));
      cache->Release(h);
    }

    // Same key again with a fresh buffer: duplicate, freed exactly once.
    buf = make_owned(bytes);
    ASSERT_OK(loader->InsertWarmupDataBlockOwned(
        Slice(key), Cache::Priority::HIGH, std::move(buf), bytes.size(), &st));
    EXPECT_EQ(nullptr, buf.get());
    EXPECT_EQ(2U, st.entries_received);
    EXPECT_EQ(1U, st.entries_inserted);
    EXPECT_EQ(1U, st.entries_duplicate);
    EXPECT_EQ(1U, st.high.entries_duplicate);
    EXPECT_EQ(allocs_before + 1, alloc.GetNumAllocations());
    EXPECT_EQ(frees_before + 1, alloc.GetNumDeallocations());

    // Gate failure (malformed key): refused, counted, freed exactly once.
    buf = make_owned(bytes);
    EXPECT_TRUE(loader
                    ->InsertWarmupDataBlockOwned(Slice("short-key"),
                                                 Cache::Priority::LOW,
                                                 std::move(buf), bytes.size(),
                                                 &st)
                    .IsCorruption());
    EXPECT_EQ(nullptr, buf.get());
    EXPECT_EQ(1U, st.skipped_invalid);
    EXPECT_EQ(2U, st.entries_received);
    EXPECT_EQ(allocs_before + 2, alloc.GetNumAllocations());
    EXPECT_EQ(frees_before + 2, alloc.GetNumDeallocations());

    // The one admitted block goes back through the allocator with the cache.
    loader.reset();
    cache.reset();
    EXPECT_EQ(frees_before + 3, alloc.GetNumDeallocations());
  }

  // kRejectedNoSpace: a block larger than the whole (single) shard can never
  // be admitted; the buffer is freed exactly once and nothing is resident.
  {
    LRUCacheOptions opts(block_charge / 2, 0 /*num_shard_bits*/,
                         false /*strict_capacity_limit*/,
                         0.5 /*high_pri_pool_ratio*/,
                         nullptr /*memory_allocator*/, kDefaultToAdaptiveMutex,
                         kDontChargeCacheMetadata, 0.5 /*low_pri_pool_ratio*/);
    std::shared_ptr<Cache> cache = NewLRUCache(opts);
    ASSERT_NE(nullptr, cache);
    std::unique_ptr<CacheDumpedLoader> loader = make_loader(cache);
    ASSERT_NE(nullptr, loader);
    CacheWarmupTransferStats st;
    const std::string key = MakeWarmupPullTestKey(kPrefix, 1);
    const std::string bytes = MakeWarmupPullTestBlock(&rnd, kBlockSize);
    CacheAllocationPtr buf = make_owned(bytes);
    const uint64_t frees_before = alloc.GetNumDeallocations();
    ASSERT_OK(loader->InsertWarmupDataBlockOwned(
        Slice(key), Cache::Priority::HIGH, std::move(buf), bytes.size(), &st));
    EXPECT_EQ(nullptr, buf.get());
    EXPECT_EQ(1U, st.entries_received);
    EXPECT_EQ(0U, st.entries_inserted);
    EXPECT_EQ(1U, st.entries_rejected_no_space);
    EXPECT_EQ(1U, st.high.entries_rejected_no_space);
    EXPECT_EQ(frees_before + 1, alloc.GetNumDeallocations());
    EXPECT_EQ(nullptr, cache->Lookup(key));
  }

  // Concurrency: 4 threads, each with its own loader and stats, 64 distinct
  // keys each, all owned inserts at once. 16 shards with 8x headroom (HIGH
  // entries cannot evict each other, so no shard may fill).
  {
    const int kThreads = 4;
    const size_t kPerThread = 64;
    const size_t kTotal = kThreads * kPerThread;
    std::shared_ptr<Cache> cache = make_cache(kTotal * 8, 4 /*num_shard_bits*/);
    ASSERT_NE(nullptr, cache);
    std::vector<std::string> keys(kTotal);
    std::vector<std::string> bytes(kTotal);
    std::vector<CacheAllocationPtr> bufs(kTotal);
    const uint64_t allocs_before = alloc.GetNumAllocations();
    const uint64_t frees_before = alloc.GetNumDeallocations();
    for (size_t n = 0; n < kTotal; ++n) {
      keys[n] = MakeWarmupPullTestKey(kPrefix, 1000 + n);
      bytes[n] = MakeWarmupPullTestBlock(&rnd, kBlockSize);
      bufs[n] = make_owned(bytes[n]);  // Random is not thread-safe: fill here
    }
    EXPECT_EQ(allocs_before + kTotal, alloc.GetNumAllocations());
    std::vector<std::unique_ptr<CacheDumpedLoader>> loaders(kThreads);
    std::vector<CacheWarmupTransferStats> stats(kThreads);
    std::vector<int> failures(kThreads, 0);
    for (int t = 0; t < kThreads; ++t) {
      loaders[t] = make_loader(cache);
      ASSERT_NE(nullptr, loaders[t]);
    }
    std::vector<port::Thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t]() {
        for (size_t i = 0; i < kPerThread; ++i) {
          const size_t n = static_cast<size_t>(t) * kPerThread + i;
          IOStatus s = loaders[t]->InsertWarmupDataBlockOwned(
              Slice(keys[n]), Cache::Priority::HIGH, std::move(bufs[n]),
              kBlockSize, &stats[t]);
          if (!s.ok()) {
            ++failures[t];
          }
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }
    uint64_t received = 0, inserted = 0, duplicate = 0, rejected = 0;
    for (int t = 0; t < kThreads; ++t) {
      EXPECT_EQ(0, failures[t]) << "thread " << t;
      received += stats[t].entries_received;
      inserted += stats[t].entries_inserted;
      duplicate += stats[t].entries_duplicate;
      rejected += stats[t].entries_rejected_no_space;
    }
    EXPECT_EQ(kTotal, received);
    EXPECT_EQ(kTotal, inserted);
    EXPECT_EQ(0U, duplicate);
    EXPECT_EQ(0U, rejected);
    EXPECT_EQ(allocs_before + kTotal, alloc.GetNumAllocations());
    EXPECT_EQ(frees_before, alloc.GetNumDeallocations());  // all kept
    for (size_t n = 0; n < kTotal; ++n) {
      Cache::Handle* h = cache->Lookup(keys[n]);
      ASSERT_NE(nullptr, h) << "block " << n << " not admitted";
      const Block* blk = static_cast<const Block*>(cache->Value(h));
      ASSERT_EQ(kBlockSize, blk->size());
      EXPECT_EQ(0, memcmp(blk->data(), bytes[n].data(), kBlockSize));
      cache->Release(h);
    }
    for (auto& l : loaders) {
      l.reset();
    }
    cache.reset();
  }

  // Every buffer ever handed out came back through the allocator exactly once.
  EXPECT_EQ(alloc.GetNumAllocations(), alloc.GetNumDeallocations());
}

// Test the option not to use the secondary cache in a certain DB.
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheOptionBasic) {
  LRUCacheOptions opts(4 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);
  options.lowest_used_cache_tier = CacheTier::kVolatileTier;

  // Set the file paranoid check, so after flush, the file will be read
  // all the blocks will be accessed.
  options.paranoid_file_checks = true;
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());

  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i + 70), p_v));
  }

  ASSERT_OK(Flush());

  // Flush will trigger the paranoid check and read blocks. But only block cache
  // will be read. No operations for secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  Compact("a", "z");

  // Compaction will also insert and evict blocks, no operations to the block
  // cache. No operations for secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1007, v.size());

  // Check the data in first block. Cache miss, direclty read from SST file.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());

  // Check the second block.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());

  // block cache hit
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(70));
  ASSERT_EQ(1007, v.size());

  // Check the first block in the second SST file. Cache miss and trigger SST
  // file read. No operations for secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(75));
  ASSERT_EQ(1007, v.size());

  // Check the second block in the second SST file. Cache miss and trigger SST
  // file read. No operations for secondary cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  Destroy(options);
}

// We disable the secondary cache in DBOptions at first. Close and reopen the DB
// with new options, which set the lowest_used_cache_tier to
// kNonVolatileBlockTier. So secondary cache will be used.
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheOptionChange) {
  LRUCacheOptions opts(4 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  fault_fs_->SetFailGetUniqueId(true);
  options.lowest_used_cache_tier = CacheTier::kVolatileTier;

  // Set the file paranoid check, so after flush, the file will be read
  // all the blocks will be accessed.
  options.paranoid_file_checks = true;
  DestroyAndReopen(options);
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i), p_v));
  }

  ASSERT_OK(Flush());

  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(Put(Key(i + 70), p_v));
  }

  ASSERT_OK(Flush());

  // Flush will trigger the paranoid check and read blocks. But only block cache
  // will be read.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  Compact("a", "z");

  // Compaction will also insert and evict blocks, no operations to the block
  // cache.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  std::string v = Get(Key(0));
  ASSERT_EQ(1007, v.size());

  // Check the data in first block. Cache miss, direclty read from SST file.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());

  // Check the second block.
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  v = Get(Key(5));
  ASSERT_EQ(1007, v.size());

  // block cache hit
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);

  // Change the option to enable secondary cache after we Reopen the DB
  options.lowest_used_cache_tier = CacheTier::kNonVolatileBlockTier;
  Reopen(options);

  v = Get(Key(70));
  ASSERT_EQ(1007, v.size());

  // Enable the secondary cache, trigger lookup of the first block in second SST
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 1u);

  v = Get(Key(75));
  ASSERT_EQ(1007, v.size());

  // trigger lookup of the second block in second SST
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);
  Destroy(options);
}

// Two DB test. We create 2 DBs sharing the same block cache and secondary
// cache. We diable the secondary cache option for DB2.
TEST_F(DBSecondaryCacheTest, TestSecondaryCacheOptionTwoDB) {
  LRUCacheOptions opts(4 * 1024 /* capacity */, 0 /* num_shard_bits */,
                       false /* strict_capacity_limit */,
                       0.5 /* high_pri_pool_ratio */,
                       nullptr /* memory_allocator */, kDefaultToAdaptiveMutex,
                       kDontChargeCacheMetadata);
  std::shared_ptr<TestSecondaryCache> secondary_cache(
      new TestSecondaryCache(2048 * 1024));
  opts.secondary_cache = secondary_cache;
  std::shared_ptr<Cache> cache = NewLRUCache(opts);
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 4 * 1024;
  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options.env = fault_env_.get();
  options.paranoid_file_checks = true;
  std::string dbname1 = test::PerThreadDBPath("db_t_1");
  ASSERT_OK(DestroyDB(dbname1, options));
  DB* db1 = nullptr;
  ASSERT_OK(DB::Open(options, dbname1, &db1));
  std::string dbname2 = test::PerThreadDBPath("db_t_2");
  ASSERT_OK(DestroyDB(dbname2, options));
  DB* db2 = nullptr;
  Options options2 = options;
  options2.lowest_used_cache_tier = CacheTier::kVolatileTier;
  ASSERT_OK(DB::Open(options2, dbname2, &db2));
  fault_fs_->SetFailGetUniqueId(true);

  WriteOptions wo;
  Random rnd(301);
  const int N = 6;
  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(db1->Put(wo, Key(i), p_v));
  }

  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 0u);
  ASSERT_OK(db1->Flush(FlushOptions()));

  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  for (int i = 0; i < N; i++) {
    std::string p_v = rnd.RandomString(1007);
    ASSERT_OK(db2->Put(wo, Key(i), p_v));
  }

  // No change in the secondary cache, since it is disabled in DB2
  ASSERT_EQ(secondary_cache->num_inserts(), 0u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);
  ASSERT_OK(db2->Flush(FlushOptions()));
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  Slice bg("a");
  Slice ed("b");
  ASSERT_OK(db1->CompactRange(CompactRangeOptions(), &bg, &ed));
  ASSERT_OK(db2->CompactRange(CompactRangeOptions(), &bg, &ed));

  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 2u);

  ReadOptions ro;
  std::string v;
  ASSERT_OK(db1->Get(ro, Key(0), &v));
  ASSERT_EQ(1007, v.size());

  // DB 1 has lookup block 1 and it is miss in block cache, trigger secondary
  // cache lookup
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 3u);

  ASSERT_OK(db1->Get(ro, Key(5), &v));
  ASSERT_EQ(1007, v.size());

  // DB 1 lookup the second block and it is miss in block cache, trigger
  // secondary cache lookup
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  ASSERT_OK(db2->Get(ro, Key(0), &v));
  ASSERT_EQ(1007, v.size());

  // For db2, it is not enabled with secondary cache, so no search in the
  // secondary cache
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  ASSERT_OK(db2->Get(ro, Key(5), &v));
  ASSERT_EQ(1007, v.size());

  // For db2, it is not enabled with secondary cache, so no search in the
  // secondary cache
  ASSERT_EQ(secondary_cache->num_inserts(), 1u);
  ASSERT_EQ(secondary_cache->num_lookups(), 4u);

  fault_fs_->SetFailGetUniqueId(false);
  fault_fs_->SetFilesystemActive(true);
  delete db1;
  delete db2;
  ASSERT_OK(DestroyDB(dbname1, options));
  ASSERT_OK(DestroyDB(dbname2, options));
}

#endif  // ROCKSDB_LITE

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
