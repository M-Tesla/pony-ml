#include "tf_shim.h"
#include "blob.h"

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/tf_buffer.h>
#include <tensorflow/c/tf_tensor.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IO 64
#define MAX_RANK 16

typedef struct Pb {
  const uint8_t* p;
  size_t n;
  size_t i;
} Pb;

typedef struct TfIo {
  char* name;
  char* op;
  int index;
  TF_DataType dtype;
  uint32_t kind;
  uint32_t elem;
  int32_t rank;
  int64_t dims[MAX_RANK];
} TfIo;

struct TfShimSession {
  TF_Graph* graph;
  TF_Session* session;
  TF_Status* status;
  TfIo inputs[MAX_IO];
  TfIo outputs[MAX_IO];
  size_t n_in;
  size_t n_out;
  uint8_t* result;
  size_t result_len;
};

static void copy_err(char* err, size_t n, const char* msg) {
  if ((err == NULL) || (n == 0)) return;
  if (msg == NULL) msg = "tensorflow error";
  snprintf(err, n, "%s", msg);
}

static int status_failed(TF_Status* status, char* err, size_t n) {
  if (TF_GetCode(status) == TF_OK) return 0;
  copy_err(err, n, TF_Message(status));
  return -1;
}

static int kind_of(TF_DataType dt, uint32_t* kind, uint32_t* elem) {
  size_t n = TF_DataTypeSize(dt);
  if ((n == 0) || (n > 16)) return -1;
  *elem = (uint32_t)n;
  switch (dt) {
    case TF_FLOAT: *kind = 0; break;
    case TF_DOUBLE: *kind = 1; break;
    case TF_INT32: *kind = 2; break;
    case TF_INT64: *kind = 3; break;
    case TF_INT16: *kind = 4; break;
    case TF_INT8: *kind = 5; break;
    case TF_UINT8: *kind = 6; break;
    case TF_UINT16: *kind = 7; break;
    case TF_UINT32: *kind = 8; break;
    case TF_UINT64: *kind = 9; break;
    case TF_BOOL: *kind = 10; break;
    case TF_HALF: *kind = 11; break;
    case TF_BFLOAT16: *kind = 12; break;
    case TF_COMPLEX64: *kind = 13; break;
    case TF_COMPLEX128: *kind = 14; break;
    default: *kind = 255; break;
  }
  return 0;
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

static void stash(TfShimSession* s, Buf* b) {
  free(s->result);
  s->result = b->p;
  s->result_len = b->n;
  b->p = NULL;
  b->n = 0;
  b->cap = 0;
}

static int pb_varint(Pb* b, uint64_t* out) {
  uint64_t v = 0;
  int shift = 0;
  while ((b->i < b->n) && (shift <= 63)) {
    uint8_t byte = b->p[b->i++];
    v |= (uint64_t)(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) {
      *out = v;
      return 0;
    }
    shift += 7;
  }
  return -1;
}

static int pb_skip(Pb* b, uint32_t wire) {
  uint64_t len = 0;
  if (wire == 0) return pb_varint(b, &len);
  if (wire == 1) {
    if ((b->n - b->i) < 8) return -1;
    b->i += 8;
    return 0;
  }
  if (wire == 5) {
    if ((b->n - b->i) < 4) return -1;
    b->i += 4;
    return 0;
  }
  if (wire != 2) return -1;
  if (pb_varint(b, &len) || (len > (b->n - b->i))) return -1;
  b->i += (size_t)len;
  return 0;
}

static int pb_bytes(Pb* b, Pb* out) {
  uint64_t len = 0;
  if (pb_varint(b, &len) || (len > (b->n - b->i))) return -1;
  out->p = b->p + b->i;
  out->n = (size_t)len;
  out->i = 0;
  b->i += (size_t)len;
  return 0;
}

static char* pb_strdup(Pb s) {
  char* p = malloc(s.n + 1);
  if (p == NULL) return NULL;
  if (s.n > 0) memcpy(p, s.p, s.n);
  p[s.n] = '\0';
  return p;
}

static int pb_is(Pb s, const char* lit) {
  size_t n = strlen(lit);
  return (s.n == n) && (memcmp(s.p, lit, n) == 0);
}

static void clear_io(TfIo* items, size_t n) {
  for (size_t i = 0; i < n; i++) {
    free(items[i].name);
    free(items[i].op);
    items[i].name = NULL;
    items[i].op = NULL;
  }
}

static int split_port(Pb full, char** op, int* index) {
  size_t cut = full.n;
  for (size_t i = 0; i < full.n; i++) {
    if (full.p[i] == ':') cut = i;
  }
  Pb name = { full.p, cut, 0 };
  *op = pb_strdup(name);
  if (*op == NULL) return -1;
  *index = 0;
  if (cut < full.n) {
    int v = 0;
    for (size_t i = cut + 1; i < full.n; i++) {
      if ((full.p[i] < '0') || (full.p[i] > '9')) break;
      v = (v * 10) + (full.p[i] - '0');
    }
    *index = v;
  }
  return 0;
}

static int parse_shape(Pb msg, int64_t* dims, int* rank, char* err, size_t err_len) {
  *rank = 0;
  while (msg.i < msg.n) {
    uint64_t key = 0;
    if (pb_varint(&msg, &key)) return -1;
    uint32_t field = (uint32_t)(key >> 3);
    uint32_t wire = (uint32_t)(key & 7);
    if ((field == 3) && (wire == 0)) {
      uint64_t v = 0;
      if (pb_varint(&msg, &v)) return -1;
      if (v) {
        copy_err(err, err_len, "tensor has unknown rank");
        return -1;
      }
    } else if ((field == 2) && (wire == 2)) {
      Pb dim;
      if (pb_bytes(&msg, &dim)) return -1;
      int64_t size = -1;
      int saw = 0;
      while (dim.i < dim.n) {
        uint64_t dk = 0;
        if (pb_varint(&dim, &dk)) return -1;
        uint32_t df = (uint32_t)(dk >> 3);
        uint32_t dw = (uint32_t)(dk & 7);
        if ((df == 1) && (dw == 0)) {
          uint64_t sv = 0;
          if (pb_varint(&dim, &sv)) return -1;
          size = (int64_t)sv;
          saw = 1;
        } else if (pb_skip(&dim, dw)) return -1;
      }
      if (*rank >= MAX_RANK) {
        copy_err(err, err_len, "tensor rank is too high");
        return -1;
      }
      dims[(*rank)++] = saw ? size : -1;
    } else if (pb_skip(&msg, wire)) return -1;
  }
  return 0;
}

static int parse_tensor_info(Pb msg, TfIo* io, char* err, size_t err_len) {
  memset(io, 0, sizeof(*io));
  Pb tensor_name = { NULL, 0, 0 };
  int saw_name = 0;
  uint64_t dtype = 0;
  int saw_dtype = 0;
  while (msg.i < msg.n) {
    uint64_t key = 0;
    if (pb_varint(&msg, &key)) return -1;
    uint32_t field = (uint32_t)(key >> 3);
    uint32_t wire = (uint32_t)(key & 7);
    if ((field == 1) && (wire == 2)) {
      if (pb_bytes(&msg, &tensor_name)) return -1;
      saw_name = 1;
    } else if ((field == 2) && (wire == 0)) {
      if (pb_varint(&msg, &dtype)) return -1;
      saw_dtype = 1;
    } else if ((field == 3) && (wire == 2)) {
      Pb shape;
      if (pb_bytes(&msg, &shape)) return -1;
      int rank = 0;
      if (parse_shape(shape, io->dims, &rank, err, err_len)) return -1;
      io->rank = rank;
    } else if (pb_skip(&msg, wire)) return -1;
  }
  if (!saw_name || !saw_dtype) {
    copy_err(err, err_len, "signature is missing a name or type");
    return -1;
  }
  if (split_port(tensor_name, &io->op, &io->index)) {
    copy_err(err, err_len, "out of memory");
    return -1;
  }
  io->dtype = (TF_DataType)dtype;
  if (kind_of(io->dtype, &io->kind, &io->elem)) {
    copy_err(err, err_len, "tensor type has no fixed size");
    return -1;
  }
  return 0;
}

static int take_entry(Pb entry, TfIo* io, char* err, size_t err_len) {
  Pb display = { NULL, 0, 0 };
  Pb info = { NULL, 0, 0 };
  int saw_info = 0;
  while (entry.i < entry.n) {
    uint64_t ek = 0;
    if (pb_varint(&entry, &ek)) return -1;
    uint32_t ef = (uint32_t)(ek >> 3);
    uint32_t ew = (uint32_t)(ek & 7);
    if ((ef == 1) && (ew == 2)) {
      if (pb_bytes(&entry, &display)) return -1;
    } else if ((ef == 2) && (ew == 2)) {
      if (pb_bytes(&entry, &info)) return -1;
      saw_info = 1;
    } else if (pb_skip(&entry, ew)) return -1;
  }
  if (!saw_info) {
    copy_err(err, err_len, "signature entry has no tensor");
    return -1;
  }
  if (parse_tensor_info(info, io, err, err_len)) return -1;
  io->name = pb_strdup(display);
  if (io->name == NULL) {
    copy_err(err, err_len, "out of memory");
    return -1;
  }
  return 0;
}

static int parse_signature(Pb msg, TfIo* inputs, size_t* n_in, TfIo* outputs, size_t* n_out,
                           char* err, size_t err_len) {
  *n_in = 0;
  *n_out = 0;
  while (msg.i < msg.n) {
    uint64_t key = 0;
    if (pb_varint(&msg, &key)) return -1;
    uint32_t field = (uint32_t)(key >> 3);
    uint32_t wire = (uint32_t)(key & 7);
    if (((field == 1) || (field == 2)) && (wire == 2)) {
      Pb entry;
      if (pb_bytes(&msg, &entry)) return -1;
      TfIo* dest = (field == 1) ? inputs : outputs;
      size_t* count = (field == 1) ? n_in : n_out;
      if (*count >= MAX_IO) {
        copy_err(err, err_len, "signature has too many tensors");
        return -1;
      }
      if (take_entry(entry, &dest[*count], err, err_len)) return -1;
      (*count)++;
    } else if (pb_skip(&msg, wire)) return -1;
  }
  return 0;
}

static int parse_meta(const uint8_t* data, size_t len, TfShimSession* s, char* err, size_t err_len) {
  Pb msg = { data, len, 0 };
  int filled = 0;
  while (msg.i < msg.n) {
    uint64_t key = 0;
    if (pb_varint(&msg, &key)) return -1;
    uint32_t field = (uint32_t)(key >> 3);
    uint32_t wire = (uint32_t)(key & 7);
    if ((field == 5) && (wire == 2)) {
      Pb entry;
      if (pb_bytes(&msg, &entry)) return -1;
      Pb sig_key = { NULL, 0, 0 };
      Pb sig = { NULL, 0, 0 };
      int saw_sig = 0;
      while (entry.i < entry.n) {
        uint64_t ek = 0;
        if (pb_varint(&entry, &ek)) return -1;
        uint32_t ef = (uint32_t)(ek >> 3);
        uint32_t ew = (uint32_t)(ek & 7);
        if ((ef == 1) && (ew == 2)) {
          if (pb_bytes(&entry, &sig_key)) return -1;
        } else if ((ef == 2) && (ew == 2)) {
          if (pb_bytes(&entry, &sig)) return -1;
          saw_sig = 1;
        } else if (pb_skip(&entry, ew)) return -1;
      }
      if (!saw_sig) continue;
      int is_default = pb_is(sig_key, "serving_default");
      if (filled && !is_default) continue;
      if (filled) {
        clear_io(s->inputs, s->n_in);
        clear_io(s->outputs, s->n_out);
        s->n_in = 0;
        s->n_out = 0;
      }
      if (parse_signature(sig, s->inputs, &s->n_in, s->outputs, &s->n_out, err, err_len))
        return -1;
      filled = 1;
      if (is_default) break;
    } else if (pb_skip(&msg, wire)) return -1;
  }
  if (!filled || (s->n_in == 0) || (s->n_out == 0)) {
    copy_err(err, err_len, "SavedModel has no signature with an input and an output");
    return -1;
  }
  return 0;
}

static int write_io_list(const TfIo* items, size_t count, Buf* b) {
  if (buf_u32(b, (uint32_t)count)) return -1;
  for (size_t i = 0; i < count; i++) {
    size_t len = strlen(items[i].name);
    if (buf_u32(b, (uint32_t)len) || buf_raw(b, items[i].name, len)) return -1;
    if (buf_u32(b, items[i].kind) || buf_u32(b, items[i].elem)) return -1;
    if (buf_u32(b, (uint32_t)items[i].rank)) return -1;
    for (int d = 0; d < items[i].rank; d++) {
      if (buf_i64(b, items[i].dims[d])) return -1;
    }
    if (buf_u64(b, 0)) return -1;
  }
  return 0;
}

void tf_version(char* out, size_t out_len) {
  const char* v = TF_Version();
  copy_err(out, out_len, (v == NULL) ? "" : v);
}

TfShimSession* tf_open(const char* export_dir, const char* tag, char* err, size_t err_len) {
  if ((tag == NULL) || (tag[0] == '\0')) tag = "serve";
  TfShimSession* s = calloc(1, sizeof(*s));
  if (s == NULL) {
    copy_err(err, err_len, "out of memory");
    return NULL;
  }
  s->status = TF_NewStatus();
  s->graph = TF_NewGraph();
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Buffer* meta = TF_NewBuffer();
  if ((s->status == NULL) || (s->graph == NULL) || (options == NULL) || (meta == NULL)) {
    copy_err(err, err_len, "out of memory for the session");
    TF_DeleteSessionOptions(options);
    TF_DeleteBuffer(meta);
    tf_close(s);
    return NULL;
  }
  const char* tags[1] = { tag };
  s->session = TF_LoadSessionFromSavedModel(
    options, NULL, export_dir, tags, 1, s->graph, meta, s->status);
  TF_DeleteSessionOptions(options);
  int bad = status_failed(s->status, err, err_len);
  if (bad || (s->session == NULL)) {
    if (!bad) copy_err(err, err_len, "TF_LoadSessionFromSavedModel returned null");
    TF_DeleteBuffer(meta);
    tf_close(s);
    return NULL;
  }
  bad = parse_meta((const uint8_t*)meta->data, meta->length, s, err, err_len);
  TF_DeleteBuffer(meta);
  if (bad) {
    tf_close(s);
    return NULL;
  }
  return s;
}

int tf_describe(TfShimSession* session, int which, char* err, size_t err_len) {
  if (session == NULL) {
    copy_err(err, err_len, "null session");
    return -1;
  }
  Buf b;
  buf_ok(&b);
  const TfIo* items = which ? session->outputs : session->inputs;
  size_t count = which ? session->n_out : session->n_in;
  if (write_io_list(items, count, &b)) {
    buf_free(&b);
    copy_err(err, err_len, "out of memory");
    return -1;
  }
  stash(session, &b);
  return 0;
}

int tf_run(
  TfShimSession* session,
  const uint8_t* request,
  size_t request_len,
  char* err,
  size_t err_len)
{
  if ((session == NULL) || (request == NULL)) {
    copy_err(err, err_len, "null session or request");
    return -1;
  }
  Reader r = { request, request_len, 0 };
  uint32_t n = 0;
  if (rd_u32(&r, &n) || (n != session->n_in)) {
    copy_err(err, err_len, "request input count does not match the model");
    return -1;
  }

  int rc = -1;
  int64_t shapes[MAX_IO][MAX_RANK];
  TF_Tensor* in_tensors[MAX_IO];
  TF_Tensor* out_tensors[MAX_IO];
  TF_Output in_ports[MAX_IO];
  TF_Output out_ports[MAX_IO];
  memset(in_tensors, 0, sizeof(in_tensors));
  memset(out_tensors, 0, sizeof(out_tensors));

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
    if (product(shapes[i], rank, &count) || (nbytes != (uint64_t)count * session->inputs[i].elem)) {
      copy_err(err, err_len, "byte size does not match the shape");
      goto cleanup;
    }
    const uint8_t* bytes = rd_take(&r, (size_t)nbytes);
    if ((nbytes > 0) && (bytes == NULL)) {
      copy_err(err, err_len, "truncated request");
      goto cleanup;
    }
    TF_Operation* op = TF_GraphOperationByName(session->graph, session->inputs[i].op);
    if (op == NULL) {
      copy_err(err, err_len, "input operation is not in the graph");
      goto cleanup;
    }
    in_ports[i].oper = op;
    in_ports[i].index = session->inputs[i].index;
    in_tensors[i] = TF_AllocateTensor(session->inputs[i].dtype, shapes[i], (int)rank, (size_t)nbytes);
    if (in_tensors[i] == NULL) {
      copy_err(err, err_len, "TF_AllocateTensor failed");
      goto cleanup;
    }
    if (nbytes > 0) memcpy(TF_TensorData(in_tensors[i]), bytes, (size_t)nbytes);
  }

  for (size_t i = 0; i < session->n_out; i++) {
    TF_Operation* op = TF_GraphOperationByName(session->graph, session->outputs[i].op);
    if (op == NULL) {
      copy_err(err, err_len, "output operation is not in the graph");
      goto cleanup;
    }
    out_ports[i].oper = op;
    out_ports[i].index = session->outputs[i].index;
  }

  TF_SessionRun(
    session->session, NULL,
    in_ports, in_tensors, (int)n,
    out_ports, out_tensors, (int)session->n_out,
    NULL, 0, NULL, session->status);
  if (status_failed(session->status, err, err_len)) goto cleanup;

  Buf b;
  buf_ok(&b);
  if (buf_u32(&b, (uint32_t)session->n_out)) {
    copy_err(err, err_len, "out of memory");
    goto cleanup_buf;
  }
  for (size_t i = 0; i < session->n_out; i++) {
    if (out_tensors[i] == NULL) {
      copy_err(err, err_len, "TF_SessionRun returned no tensor");
      goto cleanup_buf;
    }
    TF_DataType dt = TF_TensorType(out_tensors[i]);
    uint32_t kind = 0;
    uint32_t elem = 0;
    if (kind_of(dt, &kind, &elem)) {
      copy_err(err, err_len, "output type has no fixed size");
      goto cleanup_buf;
    }
    int rank = TF_NumDims(out_tensors[i]);
    if ((rank < 0) || (rank > MAX_RANK)) {
      copy_err(err, err_len, "output rank is too high");
      goto cleanup_buf;
    }
    int64_t dims[MAX_RANK];
    for (int d = 0; d < rank; d++) dims[d] = TF_Dim(out_tensors[i], d);
    size_t nbytes = TF_TensorByteSize(out_tensors[i]);
    size_t name_len = strlen(session->outputs[i].name);
    if (buf_u32(&b, (uint32_t)name_len)
        || buf_raw(&b, session->outputs[i].name, name_len)
        || buf_u32(&b, kind) || buf_u32(&b, elem)
        || buf_u32(&b, (uint32_t)rank)) {
      copy_err(err, err_len, "out of memory");
      goto cleanup_buf;
    }
    for (int d = 0; d < rank; d++) {
      if (buf_i64(&b, dims[d])) {
        copy_err(err, err_len, "out of memory");
        goto cleanup_buf;
      }
    }
    if (buf_u64(&b, nbytes) || buf_raw(&b, TF_TensorData(out_tensors[i]), nbytes)) {
      copy_err(err, err_len, "out of memory");
      goto cleanup_buf;
    }
  }
  stash(session, &b);
  rc = 0;
  goto cleanup;

cleanup_buf:
  buf_free(&b);
cleanup:
  for (uint32_t i = 0; i < n; i++) {
    if (in_tensors[i] != NULL) TF_DeleteTensor(in_tensors[i]);
  }
  for (size_t i = 0; i < session->n_out; i++) {
    if (out_tensors[i] != NULL) TF_DeleteTensor(out_tensors[i]);
  }
  return rc;
}

const uint8_t* tf_result_ptr(const TfShimSession* session) {
  return (session == NULL) ? NULL : session->result;
}

size_t tf_result_len(const TfShimSession* session) {
  return (session == NULL) ? 0 : session->result_len;
}

void tf_close(TfShimSession* session) {
  if (session == NULL) return;
  if (session->session != NULL) {
    TF_Status* status = session->status;
    int own = 0;
    if (status == NULL) {
      status = TF_NewStatus();
      own = 1;
    }
    if (status != NULL) {
      TF_CloseSession(session->session, status);
      TF_DeleteSession(session->session, status);
      if (own) TF_DeleteStatus(status);
    }
  }
  if (session->graph != NULL) TF_DeleteGraph(session->graph);
  if (session->status != NULL) TF_DeleteStatus(session->status);
  clear_io(session->inputs, session->n_in);
  clear_io(session->outputs, session->n_out);
  free(session->result);
  free(session);
}
