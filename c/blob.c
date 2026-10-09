#include "blob.h"

#include <stdlib.h>
#include <string.h>

int buf_ok(Buf* b) {
  b->p = NULL;
  b->n = 0;
  b->cap = 0;
  return 0;
}

void buf_free(Buf* b) {
  free(b->p);
  b->p = NULL;
  b->n = 0;
  b->cap = 0;
}

static int grow(Buf* b, size_t extra) {
  if (extra > (SIZE_MAX - b->n)) return -1;
  size_t need = b->n + extra;
  if (need <= b->cap) return 0;
  size_t cap = b->cap == 0 ? 256 : b->cap;
  while (cap < need) {
    if (cap > (SIZE_MAX / 2)) return -1;
    cap *= 2;
  }
  uint8_t* p = realloc(b->p, cap);
  if (p == NULL) return -1;
  b->p = p;
  b->cap = cap;
  return 0;
}

int buf_raw(Buf* b, const void* p, size_t n) {
  if (grow(b, n)) return -1;
  if (n > 0) memcpy(b->p + b->n, p, n);
  b->n += n;
  return 0;
}

int buf_u32(Buf* b, uint32_t v) { return buf_raw(b, &v, 4); }
int buf_u64(Buf* b, uint64_t v) { return buf_raw(b, &v, 8); }
int buf_i64(Buf* b, int64_t v) { return buf_raw(b, &v, 8); }

static int take(Reader* r, size_t n, const uint8_t** out) {
  if (n > (r->n - r->i)) return -1;
  *out = r->p + r->i;
  r->i += n;
  return 0;
}

int rd_u32(Reader* r, uint32_t* v) {
  const uint8_t* p = NULL;
  if (take(r, 4, &p)) return -1;
  memcpy(v, p, 4);
  return 0;
}

int rd_u64(Reader* r, uint64_t* v) {
  const uint8_t* p = NULL;
  if (take(r, 8, &p)) return -1;
  memcpy(v, p, 8);
  return 0;
}

int rd_i64(Reader* r, int64_t* v) {
  const uint8_t* p = NULL;
  if (take(r, 8, &p)) return -1;
  memcpy(v, p, 8);
  return 0;
}

const uint8_t* rd_take(Reader* r, size_t n) {
  const uint8_t* p = NULL;
  if (take(r, n, &p)) return NULL;
  return p;
}
