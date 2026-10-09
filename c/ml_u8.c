#include <stddef.h>
#include <stdint.h>
#include <string.h>

uint8_t ml_u8(const uint8_t* p, size_t i) {
  return p[i];
}

uint32_t ml_u32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

uint64_t ml_u64(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

int64_t ml_i64(const uint8_t* p) {
  int64_t v;
  memcpy(&v, p, 8);
  return v;
}

float ml_f32(const uint8_t* p) {
  float v;
  memcpy(&v, p, 4);
  return v;
}

double ml_f64(const uint8_t* p) {
  double v;
  memcpy(&v, p, 8);
  return v;
}

int32_t ml_i32(const uint8_t* p) {
  int32_t v;
  memcpy(&v, p, 4);
  return v;
}

void ml_put_u32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
void ml_put_u64(uint8_t* p, uint64_t v) { memcpy(p, &v, 8); }
void ml_put_i64(uint8_t* p, int64_t v) { memcpy(p, &v, 8); }
void ml_put_f32(uint8_t* p, float v) { memcpy(p, &v, 4); }

void ml_copy(void* dst, const void* src, size_t n) {
  if (n > 0) memcpy(dst, src, n);
}

void ml_fill_f32(uint8_t* p, size_t n, float v) {
  for (size_t i = 0; i < n; i++) memcpy(p + (i * 4), &v, 4);
}
