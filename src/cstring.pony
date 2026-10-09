use "path:../lib"
use "lib:ml_u8"

use @ml_u8[U8](p: Pointer[U8] tag, i: USize)

primitive CString
  fun copy_filled(buf: Array[U8] box): String =>
    let ptr = buf.cpointer()
    let n = buf.size()
    recover val
      let s = String
      var i: USize = 0
      while i < n do
        let b = @ml_u8(ptr, i)
        if b == 0 then break end
        s.push(b)
        i = i + 1
      end
      s
    end
