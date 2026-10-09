#ifndef ORT_SHIM_H
#define ORT_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef struct OrtShimSession OrtShimSession;

void ort_version(char* out, size_t out_len);
void ort_providers(char* out, size_t out_len);

/* Opens the customer file on GPU 0 and reads the model inputs and outputs. */
OrtShimSession* ort_open(const char* model_path, char* err, size_t err_len);

/* which 0 = inputs, 1 = outputs. The blob is left in ort_result_ptr. */
int ort_describe(OrtShimSession* session, int which, char* err, size_t err_len);

/*
 * Request: u32 n, repeated { u32 rank, i64 dims[rank], u64 nbytes, bytes }.
 * Response uses the same layout as describe, with the bytes of each output.
 */
int ort_run(
  OrtShimSession* session,
  const uint8_t* request,
  size_t request_len,
  char* err,
  size_t err_len);

const uint8_t* ort_result_ptr(const OrtShimSession* session);
size_t ort_result_len(const OrtShimSession* session);

void ort_close(OrtShimSession* session);

#endif
