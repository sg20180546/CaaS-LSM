#pragma once
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

typedef struct MEMPACKED {
  char name1[20];
  unsigned long MemTotal;
  char name2[20];
  unsigned long MemFree;
  char name3[20];
  unsigned long Buffers;
  char name4[20];
  unsigned long Cached;
  char name5[20];
  unsigned long SwapCached;
} MEM_OCCUPY;

double get_memoccupy() {
  FILE* fd;
  char buff[256];

#ifdef HDFS
  fd = fopen("/proc/meminfo", "r");
#else
  fd = fopen("/Users/liin/meminfo", "r");
#endif
  if (!fd) return 1.0;  // can't read -> assume idle, don't wrongly reject the CSA

  // Return the fraction of memory AVAILABLE for new work, not strictly-free.
  // The old code read /proc/meminfo positionally and returned MemFree/MemTotal.
  // MemFree EXCLUDES the page cache, so on an HDFS storage node with a warm cache
  // it sits far below ScheduleCSA's 0.3 gate (e.g. node87: MemFree 23% but
  // MemAvailable 86%) -> the CSA is permanently rejected -> remote compaction never
  // runs (Distributed~=0, everything falls back to local CN). MemAvailable counts
  // the reclaimable page cache, which is the right "can this node take work?" signal.
  char name[64];
  unsigned long val = 0, mem_total = 0, mem_avail = 0, mem_free = 0;
  while (fgets(buff, sizeof(buff), fd)) {
    if (sscanf(buff, "%63s %lu", name, &val) != 2) continue;
    if (strcmp(name, "MemTotal:") == 0) mem_total = val;
    else if (strcmp(name, "MemAvailable:") == 0) mem_avail = val;
    else if (strcmp(name, "MemFree:") == 0) mem_free = val;
  }
  fclose(fd);
  if (mem_total == 0) return 1.0;
  unsigned long usable = mem_avail ? mem_avail : mem_free;  // fallback if no MemAvailable
  return static_cast<double>(usable) / static_cast<double>(mem_total);
}

inline char* GetTime() {
  time_t now = time(0);

  char* dt = ctime(&now);

  tm* gmtm = gmtime(&now);
  dt = asctime(gmtm);
  return dt;
}
