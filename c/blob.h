#ifndef PONY_ML_BLOB_H
#define PONY_ML_BLOB_H

#include <stddef.h>
#include <stdint.h>

typedef struct Buf {
  uint8_t* p;
  size_t n;
  size_t cap;
} Buf;

typedef struct Reader {
  const uint8_t* p;
  size_t n;
  size_t i;
} Reader;

int buf_ok(Buf* b);
void buf_free(Buf* b);
int buf_u32(Buf* b, uint32_t v);
int buf_u64(Buf* b, uint64_t v);
int buf_i64(Buf* b, int64_t v);
int buf_raw(Buf* b, const void* p, size_t n);

int rd_u32(Reader* r, uint32_t* v);
int rd_u64(Reader* r, uint64_t* v);
int rd_i64(Reader* r, int64_t* v);
const uint8_t* rd_take(Reader* r, size_t n);

#endif
