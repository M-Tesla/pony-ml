use "path:../lib"
use "path:../vendor/onnxruntime-linux-x64-gpu_cuda13-1.30.0/lib"
use "lib:ort_shim"
use "lib:onnxruntime"

use @ort_version[None](out: Pointer[U8] tag, out_len: USize)
use @ort_providers[None](out: Pointer[U8] tag, out_len: USize)
use @ort_open[Pointer[None] tag](path: Pointer[U8] tag, err: Pointer[U8] tag, err_len: USize)
use @ort_describe[I32](session: Pointer[None] tag, which: I32, err: Pointer[U8] tag, err_len: USize)
use @ort_run[I32](
  session: Pointer[None] tag,
  request: Pointer[U8] tag,
  request_len: USize,
  err: Pointer[U8] tag,
  err_len: USize)
use @ort_result_ptr[Pointer[U8] tag](session: Pointer[None] tag)
use @ort_result_len[USize](session: Pointer[None] tag)
use @ort_close[None](session: Pointer[None] tag)

primitive Ort
  fun version(): String =>
    let buf = Array[U8].init(0, 128)
    @ort_version(buf.cpointer(), buf.size())
    CString.copy_filled(buf)

  fun providers(): String =>
    let buf = Array[U8].init(0, 512)
    @ort_providers(buf.cpointer(), buf.size())
    CString.copy_filled(buf)

  fun open(path: String): (OrtSession | String) =>
    let err = Array[U8].init(0, 2048)
    let ptr = @ort_open(path.cstring(), err.cpointer(), err.size())
    if ptr.is_null() then
      CString.copy_filled(err)
    else
      OrtSession(ptr)
    end

class OrtSession
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
    if @ort_run(_ptr, request.cpointer(), request.size(), err.cpointer(), err.size()) != 0 then
      CString.copy_filled(err)
    else
      take()
    end

  fun ref close() =>
    @ort_close(_ptr)

  fun ref fetch(which: I32): (Array[Tensor] | String) =>
    let err = Array[U8].init(0, 2048)
    if @ort_describe(_ptr, which, err.cpointer(), err.size()) != 0 then
      CString.copy_filled(err)
    else
      take()
    end

  fun ref take(): (Array[Tensor] | String) =>
    let n = @ort_result_len(_ptr)
    let buf = Array[U8]
    buf.undefined(n)
    if n > 0 then
      @ml_copy(buf.cpointer(), @ort_result_ptr(_ptr), n)
    end
    _maker.parse(buf)
