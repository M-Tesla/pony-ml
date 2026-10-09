#ifndef TF_SHIM_H
#define TF_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef struct TfShimSession TfShimSession;

void tf_version(char* out, size_t out_len);

/* Reads the serving_default signature of the SavedModel. A NULL tag uses "serve". */
TfShimSession* tf_open(const char* export_dir, const char* tag, char* err, size_t err_len);

int tf_describe(TfShimSession* session, int which, char* err, size_t err_len);

int tf_run(
  TfShimSession* session,
  const uint8_t* request,
  size_t request_len,
  char* err,
  size_t err_len);

const uint8_t* tf_result_ptr(const TfShimSession* session);
size_t tf_result_len(const TfShimSession* session);

void tf_close(TfShimSession* session);

#endif
