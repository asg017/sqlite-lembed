#include "sqlite3.h"
#include "sqlite-vec.h"
#include "sqlite-lembed.h"
#include <stdio.h>
int core_init(const char *dummy) {
  int rc;
  rc = sqlite3_auto_extension((void *)sqlite3_vec_init);
  if(rc == SQLITE_OK) {
    rc = sqlite3_auto_extension((void *)sqlite3_lembed_init);
  }
  return rc;
}
