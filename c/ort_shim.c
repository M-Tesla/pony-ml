#include "ort_shim.h"
#include "blob.h"

#include <onnxruntime_c_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IO 64
#define MAX_RANK 16

typedef struct IoInfo {
  char* name;
  int32_t dtype;
  uint32_t kind;
  uint32_t elem;
  int32_t rank;
  int64_t dims[MAX_RANK];
} IoInfo;

struct OrtShimSession {
  const OrtApi* api;
  OrtEnv* env;
  OrtSessionOptions* options;
  OrtSession* session;
  OrtMemoryInfo* memory;
  IoInfo inputs[MAX_IO];
  IoInfo outputs[MAX_IO];
  size_t n_in;
  size_t n_out;
  uint8_t* result;
  size_t result_len;
};

static void copy_err(char* err, size_t n, const char* msg) {
  if ((err == NULL) || (n == 0)) return;
  if (msg == NULL) msg = "onnx runtime error";
  snprintf(err, n, "%s", msg);
}

static int fail_status(const OrtApi* api, OrtStatus* status, char* err, size_t n) {
  if (status == NULL) return 0;
  copy_err(err, n, api->GetErrorMessage(status));
  api->ReleaseStatus(status);
  return -1;
}

static int kind_of(int dtype, uint32_t* kind, uint32_t* elem) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: *kind = 0; *elem = 4; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: *kind = 1; *elem = 8; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: *kind = 2; *elem = 4; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: *kind = 3; *elem = 8; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: *kind = 4; *elem = 2; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: *kind = 5; *elem = 1; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: *kind = 6; *elem = 1; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: *kind = 7; *elem = 2; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: *kind = 8; *elem = 4; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: *kind = 9; *elem = 8; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: *kind = 10; *elem = 1; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: *kind = 11; *elem = 2; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: *kind = 12; *elem = 2; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: *kind = 13; *elem = 8; return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: *kind = 14; *elem = 16; return 0;
    default: return -1;
  }
}

static int product(const int64_t* shape, size_t rank, size_t* out) {
  size_t n = 1;
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] < 0) return -1;
    size_t d = (size_t)shape[i];
    if ((d != 0) && (n > (SIZE_MAX / d))) return -1;
    n *= d;
  }
  *out = n;
  return 0;
}

static void stash(OrtShimSession* s, Buf* b) {
  free(s->result);
  s->result = b->p;
  s->result_len = b->n;
  b->p = NULL;
  b->n = 0;
  b->cap = 0;
}

static int write_io_list(const IoInfo* items, size_t count, int with_empty, Buf* b) {
  if (buf_u32(b, (uint32_t)count)) return -1;
  for (size_t i = 0; i < count; i++) {
    size_t len = strlen(items[i].name);
    if (buf_u32(b, (uint32_t)len)) return -1;
    if (buf_raw(b, items[i].name, len)) return -1;
    if (buf_u32(b, items[i].kind)) return -1;
    if (buf_u32(b, items[i].elem)) return -1;
    if (buf_u32(b, (uint32_t)items[i].rank)) return -1;
    for (int d = 0; d < items[i].rank; d++) {
      if (buf_i64(b, items[i].dims[d])) return -1;
    }
    if (with_empty && buf_u64(b, 0)) return -1;
  }
  return 0;
}

static int load_io(OrtShimSession* s, int output, char* err, size_t err_len) {
  const OrtApi* api = s->api;
  size_t count = 0;
  OrtStatus* st = output
    ? api->SessionGetOutputCount(s->session, &count)
    : api->SessionGetInputCount(s->session, &count);
  if (fail_status(api, st, err, err_len)) return -1;
  if (count > MAX_IO) {
    copy_err(err, err_len, "the model has too many inputs or outputs");
    return -1;
  }
  IoInfo* dest = output ? s->outputs : s->inputs;
  if (output) s->n_out = count;
  else s->n_in = count;

  OrtAllocator* allocator = NULL;
  if (fail_status(api, api->GetAllocatorWithDefaultOptions(&allocator), err, err_len))
    return -1;

  for (size_t i = 0; i < count; i++) {
    char* owned = NULL;
    st = output
      ? api->SessionGetOutputName(s->session, i, allocator, &owned)
      : api->SessionGetInputName(s->session, i, allocator, &owned);
    if (fail_status(api, st, err, err_len)) return -1;
    dest[i].name = malloc(strlen(owned) + 1);
    if (dest[i].name == NULL) {
      allocator->Free(allocator, owned);
      copy_err(err, err_len, "out of memory");
      return -1;
    }
    memcpy(dest[i].name, owned, strlen(owned) + 1);
    allocator->Free(allocator, owned);

    OrtTypeInfo* type_info = NULL;
    st = output
      ? api->SessionGetOutputTypeInfo(s->session, i, &type_info)
      : api->SessionGetInputTypeInfo(s->session, i, &type_info);
    if (fail_status(api, st, err, err_len)) return -1;

    enum ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
    if (fail_status(api, api->GetOnnxTypeFromTypeInfo(type_info, &onnx_type), err, err_len)) {
      api->ReleaseTypeInfo(type_info);
      return -1;
    }
    if (onnx_type != ONNX_TYPE_TENSOR) {
      api->ReleaseTypeInfo(type_info);
      copy_err(err, err_len, "the model has an input or output that is not a tensor");
      return -1;
    }
    const OrtTensorTypeAndShapeInfo* info = NULL;
    if (fail_status(api, api->CastTypeInfoToTensorInfo(type_info, &info), err, err_len)) {
      api->ReleaseTypeInfo(type_info);
      return -1;
    }
    enum ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (fail_status(api, api->GetTensorElementType(info, &dtype), err, err_len)) {
      api->ReleaseTypeInfo(type_info);
      return -1;
    }
    size_t rank = 0;
    if (fail_status(api, api->GetDimensionsCount(info, &rank), err, err_len)) {
      api->ReleaseTypeInfo(type_info);
      return -1;
    }
    if (rank > MAX_RANK) {
      api->ReleaseTypeInfo(type_info);
      copy_err(err, err_len, "tensor rank is too high");
      return -1;
    }
    if ((rank > 0) && fail_status(api, api->GetDimensions(info, dest[i].dims, rank), err, err_len)) {
      api->ReleaseTypeInfo(type_info);
      return -1;
    }
    api->ReleaseTypeInfo(type_info);
    if (kind_of((int)dtype, &dest[i].kind, &dest[i].elem)) {
      copy_err(err, err_len, "tensor type has no fixed size");
      return -1;
    }
    dest[i].dtype = (int32_t)dtype;
    dest[i].rank = (int32_t)rank;
  }
  return 0;
}

static const OrtApi* api_or_null(void) {
  const OrtApiBase* base = OrtGetApiBase();
  if (base == NULL) return NULL;
  return base->GetApi(ORT_API_VERSION);
}

void ort_version(char* out, size_t out_len) {
  const char* v = "";
  const OrtApiBase* base = OrtGetApiBase();
  if (base != NULL) {
    const char* got = base->GetVersionString();
    if (got != NULL) v = got;
  }
  copy_err(out, out_len, v);
}

void ort_providers(char* out, size_t out_len) {
  if ((out == NULL) || (out_len == 0)) return;
  out[0] = '\0';
  const OrtApi* api = api_or_null();
  if (api == NULL) {
    copy_err(out, out_len, "OrtApi is missing");
    return;
  }
  char** providers = NULL;
  int count = 0;
  if (fail_status(api, api->GetAvailableProviders(&providers, &count), out, out_len))
    return;
  size_t used = 0;
  for (int i = 0; i < count; i++) {
    size_t len = strlen(providers[i]);
    size_t extra = len + (used > 0 ? 1 : 0);
    if (used + extra + 1 > out_len) break;
    if (used > 0) out[used++] = ' ';
    memcpy(out + used, providers[i], len);
    used += len;
    out[used] = '\0';
  }
  OrtStatus* released = api->ReleaseAvailableProviders(providers, count);
  if (released != NULL) api->ReleaseStatus(released);
}

OrtShimSession* ort_open(const char* model_path, char* err, size_t err_len) {
  OrtShimSession* s = calloc(1, sizeof(*s));
  if (s == NULL) {
    copy_err(err, err_len, "out of memory");
    return NULL;
  }
  const OrtApiBase* base = OrtGetApiBase();
  if (base == NULL) {
    copy_err(err, err_len, "OrtGetApiBase returned null");
    free(s);
    return NULL;
  }
  s->api = base->GetApi(ORT_API_VERSION);
  if (s->api == NULL) {
    copy_err(err, err_len, "this library does not provide the requested OrtApi");
    free(s);
    return NULL;
  }
  const OrtApi* api = s->api;
  if (fail_status(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "pony-ml", &s->env), err, err_len))
    goto fail;
  if (fail_status(api, api->CreateSessionOptions(&s->options), err, err_len))
    goto fail;
  if (fail_status(api, api->SetIntraOpNumThreads(s->options, 1), err, err_len))
    goto fail;

  OrtCUDAProviderOptionsV2* cuda = NULL;
  if (fail_status(api, api->CreateCUDAProviderOptions(&cuda), err, err_len))
    goto fail;
  {
    const char* keys[] = { "device_id" };
    const char* values[] = { "0" };
    OrtStatus* cuda_status = api->UpdateCUDAProviderOptions(cuda, keys, values, 1);
    if (cuda_status == NULL)
      cuda_status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(s->options, cuda);
    api->ReleaseCUDAProviderOptions(cuda);
    if (fail_status(api, cuda_status, err, err_len))
      goto fail;
  }

  if (fail_status(api, api->CreateSession(s->env, model_path, s->options, &s->session), err, err_len))
    goto fail;
  if (fail_status(api, api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeCPU, &s->memory), err, err_len))
    goto fail;
  if (load_io(s, 0, err, err_len) || load_io(s, 1, err, err_len))
    goto fail;
  return s;

fail:
  ort_close(s);
  return NULL;
}

int ort_describe(OrtShimSession* session, int which, char* err, size_t err_len) {
  if (session == NULL) {
    copy_err(err, err_len, "null session");
    return -1;
  }
  Buf b;
  buf_ok(&b);
  const IoInfo* items = which ? session->outputs : session->inputs;
  size_t count = which ? session->n_out : session->n_in;
  if (write_io_list(items, count, 1, &b)) {
    buf_free(&b);
    copy_err(err, err_len, "out of memory");
    return -1;
  }
  stash(session, &b);
  return 0;
}

int ort_run(
  OrtShimSession* session,
  const uint8_t* request,
  size_t request_len,
  char* err,
  size_t err_len)
{
  if ((session == NULL) || (request == NULL)) {
    copy_err(err, err_len, "null session or request");
    return -1;
  }
  const OrtApi* api = session->api;
  Reader r = { request, request_len, 0 };
  uint32_t n = 0;
  if (rd_u32(&r, &n) || (n != session->n_in)) {
    copy_err(err, err_len, "request input count does not match the model");
    return -1;
  }

  int rc = -1;
  int64_t shapes[MAX_IO][MAX_RANK];
  const void* data[MAX_IO];
  OrtValue* inputs[MAX_IO];
  OrtValue* outputs[MAX_IO];
  const char* in_names[MAX_IO];
  const char* out_names[MAX_IO];
  const OrtValue* in_ptrs[MAX_IO];
  Buf b;
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  buf_ok(&b);

  for (uint32_t i = 0; i < n; i++) {
    uint32_t rank = 0;
    if (rd_u32(&r, &rank) || (rank != (uint32_t)session->inputs[i].rank) || (rank > MAX_RANK)) {
      copy_err(err, err_len, "rank does not match the model");
      goto cleanup;
    }
    for (uint32_t d = 0; d < rank; d++) {
      if (rd_i64(&r, &shapes[i][d])) {
        copy_err(err, err_len, "truncated request");
        goto cleanup;
      }
      int64_t declared = session->inputs[i].dims[d];
      if ((declared >= 0) && (declared != shapes[i][d])) {
        copy_err(err, err_len, "shape does not match the model");
        goto cleanup;
      }
      if (shapes[i][d] < 0) {
        copy_err(err, err_len, "negative dimension in the request");
        goto cleanup;
      }
    }
    uint64_t nbytes = 0;
    if (rd_u64(&r, &nbytes)) {
      copy_err(err, err_len, "truncated request");
      goto cleanup;
    }
    size_t count = 0;
    if (product(shapes[i], rank, &count)) {
      copy_err(err, err_len, "invalid shape");
      goto cleanup;
    }
    if (nbytes != (uint64_t)count * session->inputs[i].elem) {
      copy_err(err, err_len, "byte size does not match the shape");
      goto cleanup;
    }
    const uint8_t* bytes = rd_take(&r, (size_t)nbytes);
    if ((nbytes > 0) && (bytes == NULL)) {
      copy_err(err, err_len, "truncated request");
      goto cleanup;
    }
    data[i] = bytes;
    in_names[i] = session->inputs[i].name;
    if (fail_status(api, api->CreateTensorWithDataAsOrtValue(
          session->memory, (void*)data[i], (size_t)nbytes, shapes[i], rank,
          (enum ONNXTensorElementDataType)session->inputs[i].dtype, &inputs[i]), err, err_len))
      goto cleanup;
    in_ptrs[i] = inputs[i];
  }

  for (size_t i = 0; i < session->n_out; i++) out_names[i] = session->outputs[i].name;

  if (fail_status(api, api->Run(
        session->session, NULL, in_names, in_ptrs, n, out_names, session->n_out, outputs),
        err, err_len))
    goto cleanup;
  if (buf_u32(&b, (uint32_t)session->n_out)) {
    copy_err(err, err_len, "out of memory");
      goto cleanup;
  }
  for (size_t i = 0; i < session->n_out; i++) {
    OrtTensorTypeAndShapeInfo* info = NULL;
    if (fail_status(api, api->GetTensorTypeAndShape(outputs[i], &info), err, err_len))
      goto cleanup;
    enum ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    int64_t dims[MAX_RANK];
    int bad = fail_status(api, api->GetTensorElementType(info, &dtype), err, err_len)
      || fail_status(api, api->GetDimensionsCount(info, &rank), err, err_len);
    if (!bad && (rank > MAX_RANK)) {
      copy_err(err, err_len, "output rank is too high");
      bad = 1;
    }
    if (!bad && (rank > 0))
      bad = fail_status(api, api->GetDimensions(info, dims, rank), err, err_len);
    size_t count = 0;
    if (!bad) bad = fail_status(api, api->GetTensorShapeElementCount(info, &count), err, err_len);
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (bad) goto cleanup;

    uint32_t kind = 0;
    uint32_t elem = 0;
    if (kind_of((int)dtype, &kind, &elem)) {
      copy_err(err, err_len, "output type has no fixed size");
      goto cleanup;
    }
    void* raw = NULL;
    if (fail_status(api, api->GetTensorMutableData(outputs[i], &raw), err, err_len))
      goto cleanup;
    size_t name_len = strlen(session->outputs[i].name);
    if (buf_u32(&b, (uint32_t)name_len)
        || buf_raw(&b, session->outputs[i].name, name_len)
        || buf_u32(&b, kind)
        || buf_u32(&b, elem)
        || buf_u32(&b, (uint32_t)rank)) {
      copy_err(err, err_len, "out of memory");
      goto cleanup;
    }
    for (size_t d = 0; d < rank; d++) {
      if (buf_i64(&b, dims[d])) {
        copy_err(err, err_len, "out of memory");
        goto cleanup;
      }
    }
    uint64_t nbytes = (uint64_t)count * elem;
    if (buf_u64(&b, nbytes) || buf_raw(&b, raw, (size_t)nbytes)) {
      copy_err(err, err_len, "out of memory");
      goto cleanup;
    }
  }
  stash(session, &b);
  rc = 0;

cleanup:
  if (rc != 0) buf_free(&b);
  for (size_t i = 0; i < session->n_out; i++) {
    if (outputs[i] != NULL) api->ReleaseValue(outputs[i]);
  }
  for (uint32_t i = 0; i < n; i++) {
    if (inputs[i] != NULL) api->ReleaseValue(inputs[i]);
  }
  return rc;
}

const uint8_t* ort_result_ptr(const OrtShimSession* session) {
  return (session == NULL) ? NULL : session->result;
}

size_t ort_result_len(const OrtShimSession* session) {
  return (session == NULL) ? 0 : session->result_len;
}

void ort_close(OrtShimSession* session) {
  if (session == NULL) return;
  const OrtApi* api = session->api;
  for (size_t i = 0; i < session->n_in; i++) free(session->inputs[i].name);
  for (size_t i = 0; i < session->n_out; i++) free(session->outputs[i].name);
  free(session->result);
  if (api != NULL) {
    if (session->memory != NULL) api->ReleaseMemoryInfo(session->memory);
    if (session->session != NULL) api->ReleaseSession(session->session);
    if (session->options != NULL) api->ReleaseSessionOptions(session->options);
    if (session->env != NULL) api->ReleaseEnv(session->env);
  }
  free(session);
}
