// [ft 2026-11-04] Cooperative crash points for the failure-tolerance cells (migration_mechansim_exp FT_IMPLEMENTATION.md).
// A process reads the control file named by the environment variable FT_CRASH_FILE at every point it passes; when the
// file's content equals the point's name it deletes the file and _exit(137)s -- the same end as kill -9, but at an
// exact protocol position (between the Storage-CP claim and the MANIFEST write, after the first release of a purge, ...)
// that an external kill cannot hit. Unset variable or absent file = one open() per point and nothing else.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace ft_crash {

inline void Check(const char* point) {
  static const char* file = std::getenv("FT_CRASH_FILE");
  if (file == nullptr || *file == '\0') return;
  FILE* f = std::fopen(file, "r");
  if (f == nullptr) return;
  char buf[128];
  size_t n = std::fread(buf, 1, sizeof buf - 1, f);
  std::fclose(f);
  buf[n] = '\0';
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = '\0';
  if (std::strcmp(buf, point) != 0) return;
  ::unlink(file);
  std::fprintf(stderr, "[ft-crash] point=%s pid=%d: _exit(137)\n", point, static_cast<int>(::getpid()));
  std::fflush(stderr);
  ::_exit(137);
}

}  // namespace ft_crash
