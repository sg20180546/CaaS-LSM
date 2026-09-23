//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#pragma once

// 2026-09-23 [relink cache handoff]: CustomDeleter, CacheAllocationPtr and
// AllocateBlock now live in the public header so CacheDumpedLoader can take
// an owned buffer. This shim keeps the internal include path working.
#include "rocksdb/memory_allocator.h"
