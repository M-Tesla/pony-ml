use @ml_u32[U32](p: Pointer[U8] tag)
use @ml_u64[U64](p: Pointer[U8] tag)
use @ml_i64[I64](p: Pointer[U8] tag)
use @ml_f32[F32](p: Pointer[U8] tag)
use @ml_put_u32[None](p: Pointer[U8] tag, v: U32)
use @ml_put_u64[None](p: Pointer[U8] tag, v: U64)
use @ml_put_i64[None](p: Pointer[U8] tag, v: I64)
use @ml_put_f32[None](p: Pointer[U8] tag, v: F32)
use @ml_copy[None](dst: Pointer[U8] tag, src: Pointer[U8] tag, n: USize)
use @ml_fill_f32[None](p: Pointer[U8] tag, n: USize, v: F32)

class Tensor
  let name: String
  var kind: U32 = 0
  var elem: U32 = 0
  let shape: Array[I64] = Array[I64]
  let data: Array[U8] = Array[U8]

  new create(name': String) =>
    name = name'

  fun kind_name(): String =>
    match kind
    | 0 => "f32"
    | 1 => "f64"
    | 2 => "i32"
    | 3 => "i64"
    | 4 => "i16"
    | 5 => "i8"
    | 6 => "u8"
    | 7 => "u16"
    | 8 => "u32"
    | 9 => "u64"
    | 10 => "bool"
    | 11 => "f16"
    | 12 => "bf16"
    else
      "bytes"
    end

  fun f32(index: USize): F32 =>
    @ml_f32(data.cpointer().offset(index * 4))

  fun i64(index: USize): I64 =>
    @ml_i64(data.cpointer().offset(index * 8))

class Maker
  fun ref f32(label: String, dims: Array[I64] box, values: Array[F32] box): Tensor =>
    let t = Tensor(label)
    t.kind = 0
    t.elem = 4
    for d in dims.values() do
      t.shape.push(d)
    end
    t.data.undefined(values.size() * 4)
    var i = USize(0)
    while i < values.size() do
      try @ml_put_f32(t.data.cpointer().offset(i * 4), values(i)?) end
      i = i + 1
    end
    consume t

  fun ref f32_fill(label: String, dims: Array[I64] box, count: USize, value: F32): Tensor =>
    let t = Tensor(label)
    t.kind = 0
    t.elem = 4
    for d in dims.values() do
      t.shape.push(d)
    end
    t.data.undefined(count * 4)
    @ml_fill_f32(t.data.cpointer(), count, value)
    consume t

  fun ref i64(label: String, dims: Array[I64] box, values: Array[I64] box): Tensor =>
    let t = Tensor(label)
    t.kind = 3
    t.elem = 8
    for d in dims.values() do
      t.shape.push(d)
    end
    t.data.undefined(values.size() * 8)
    var i = USize(0)
    while i < values.size() do
      try @ml_put_i64(t.data.cpointer().offset(i * 8), values(i)?) end
      i = i + 1
    end
    consume t

  fun ref request(items: Array[Tensor] box): Array[U8] =>
    var bytes = USize(4)
    for t in items.values() do
      bytes = bytes + 4 + (t.shape.size() * 8) + 8 + t.data.size()
    end
    let buf = Array[U8]
    buf.undefined(bytes)
    @ml_put_u32(buf.cpointer(), items.size().u32())
    var i = USize(4)
    for t in items.values() do
      @ml_put_u32(buf.cpointer().offset(i), t.shape.size().u32())
      i = i + 4
      var d = USize(0)
      while d < t.shape.size() do
        try @ml_put_i64(buf.cpointer().offset(i), t.shape(d)?) end
        i = i + 8
        d = d + 1
      end
      @ml_put_u64(buf.cpointer().offset(i), t.data.size().u64())
      i = i + 8
      if t.data.size() > 0 then
        @ml_copy(buf.cpointer().offset(i), t.data.cpointer(), t.data.size())
      end
      i = i + t.data.size()
    end
    consume buf

  fun ref parse(buf: Array[U8] box): (Array[Tensor] | String) =>
    if buf.size() < 4 then return "short response" end
    let n = @ml_u32(buf.cpointer()).usize()
    var i = USize(4)
    let out = Array[Tensor]
    var k = USize(0)
    while k < n do
      if (i + 4) > buf.size() then return "short response" end
      let name_len = @ml_u32(buf.cpointer().offset(i)).usize()
      i = i + 4
      if (i + name_len) > buf.size() then return "short response" end
      let built = String
      var c = USize(0)
      while c < name_len do
        built.push(@ml_u8(buf.cpointer().offset(i + c), 0))
        c = c + 1
      end
      i = i + name_len
      let owned = built.clone()
      let t = Tensor(recover val consume owned end)
      if (i + 12) > buf.size() then return "short response" end
      t.kind = @ml_u32(buf.cpointer().offset(i))
      i = i + 4
      t.elem = @ml_u32(buf.cpointer().offset(i))
      i = i + 4
      let rank = @ml_u32(buf.cpointer().offset(i)).usize()
      i = i + 4
      var d = USize(0)
      while d < rank do
        if (i + 8) > buf.size() then return "short response" end
        t.shape.push(@ml_i64(buf.cpointer().offset(i)))
        i = i + 8
        d = d + 1
      end
      if (i + 8) > buf.size() then return "short response" end
      let nbytes = @ml_u64(buf.cpointer().offset(i)).usize()
      i = i + 8
      if (i + nbytes) > buf.size() then return "short response" end
      if nbytes > 0 then
        t.data.undefined(nbytes)
        @ml_copy(t.data.cpointer(), buf.cpointer().offset(i), nbytes)
      end
      i = i + nbytes
      out.push(consume t)
      k = k + 1
    end
    consume out
