use "path:../lib"
use "path:../vendor/libtensorflow/lib"
use "lib:tf_shim"
use "lib:tensorflow"
use "lib:tensorflow_framework"

use @tf_version[None](out: Pointer[U8] tag, out_len: USize)
use @tf_open[Pointer[None] tag](
  export_dir: Pointer[U8] tag,
  saved_tag: Pointer[U8] tag,
  err: Pointer[U8] tag,
  err_len: USize)
use @tf_describe[I32](session: Pointer[None] tag, which: I32, err: Pointer[U8] tag, err_len: USize)
use @tf_run[I32](
  session: Pointer[None] tag,
  request: Pointer[U8] tag,
  request_len: USize,
  err: Pointer[U8] tag,
  err_len: USize)
use @tf_result_ptr[Pointer[U8] tag](session: Pointer[None] tag)
use @tf_result_len[USize](session: Pointer[None] tag)
use @tf_close[None](session: Pointer[None] tag)

primitive Tf
  fun version(): String =>
    let buf = Array[U8].init(0, 128)
    @tf_version(buf.cpointer(), buf.size())
    CString.copy_filled(buf)

  fun open(export_dir: String, saved_tag: String = "serve"): (TfSession | String) =>
    let err = Array[U8].init(0, 2048)
    let ptr = @tf_open(export_dir.cstring(), saved_tag.cstring(), err.cpointer(), err.size())
    if ptr.is_null() then
      CString.copy_filled(err)
    else
      TfSession(ptr)
    end

class TfSession
  let _ptr: Pointer[None] tag
  let _maker: Maker = Maker

  new create(ptr: Pointer[None] tag) =>
    _ptr = ptr

  fun ref inputs(): (Array[Tensor] | String) =>
    fetch(0)

  fun ref outputs(): (Array[Tensor] | String) =>
    fetch(1)

  fun ref run(items: Array[Tensor] box): (Array[Tensor] | String) =>
    let err = Array[U8].init(0, 2048)
    let request = _maker.request(items)
    if @tf_run(_ptr, request.cpointer(), request.size(), err.cpointer(), err.size()) != 0 then
      CString.copy_filled(err)
    else
      take()
    end

  fun ref close() =>
    @tf_close(_ptr)

  fun ref fetch(which: I32): (Array[Tensor] | String) =>
    let err = Array[U8].init(0, 2048)
    if @tf_describe(_ptr, which, err.cpointer(), err.size()) != 0 then
      CString.copy_filled(err)
    else
      take()
    end

  fun ref take(): (Array[Tensor] | String) =>
    let n = @tf_result_len(_ptr)
    let buf = Array[U8]
    buf.undefined(n)
    if n > 0 then
      @ml_copy(buf.cpointer(), @tf_result_ptr(_ptr), n)
    end
    _maker.parse(buf)
