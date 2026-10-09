interface Runner
  fun ref inputs(): (Array[Tensor] | String)
  fun ref outputs(): (Array[Tensor] | String)
  fun ref run(items: Array[Tensor] box): (Array[Tensor] | String)

actor Main
  let maker: Maker = Maker

  new create(env: Env) =>
    env.out.print("onnxruntime " + Ort.version())
    env.out.print("tensorflow " + Tf.version())
    var failed = false
    let providers = Ort.providers()
    env.out.print("ort providers " + providers)
    if not providers.contains("CUDAExecutionProvider") then
      env.out.print("ort failed: CUDAExecutionProvider missing")
      failed = true
    end

    let root = "models/"
    match Ort.open(root + "multi_io.onnx")
    | let session: OrtSession =>
      if not file_case(env, "ort", session) then failed = true end
      session.close()
    | let err: String =>
      env.out.print("ort multi failed: " + err)
      failed = true
    end

    match Ort.open(root + "ids.onnx")
    | let session: OrtSession =>
      if not ids_case(env, "ort", session) then failed = true end
      session.close()
    | let err: String =>
      env.out.print("ort ids failed: " + err)
      failed = true
    end

    match Ort.open(root + "gemm.onnx")
    | let session: OrtSession =>
      if not gemm_case(env, session) then failed = true end
      session.close()
    | let err: String =>
      env.out.print("ort gemm failed: " + err)
      failed = true
    end

    match Tf.open(root + "multi")
    | let session: TfSession =>
      if not file_case(env, "tf", session) then failed = true end
      session.close()
    | let err: String =>
      env.out.print("tf multi failed: " + err)
      failed = true
    end

    match Tf.open(root + "ids")
    | let session: TfSession =>
      if not ids_case(env, "tf", session) then failed = true end
      session.close()
    | let err: String =>
      env.out.print("tf ids failed: " + err)
      failed = true
    end

    if failed then env.exitcode(1) end

  fun ref file_case(env: Env, label: String, session: Runner): Bool =>
    match session.inputs()
    | let spec: Array[Tensor] =>
      show_io(env, label + " inputs", spec)
      match session.outputs()
      | let declared: Array[Tensor] =>
        show_io(env, label + " outputs", declared)
        let batch = [I64(3); I64(2)]
        let given = Array[Tensor]
        given.push(maker.f32(
          "input_1", batch,
          [F32(1); F32(10); F32(2); F32(20); F32(3); F32(30)]))
        given.push(maker.f32(
          "input_2", batch,
          [F32(4); F32(1); F32(5); F32(2); F32(6); F32(3)]))
        match align(spec, given)
        | let ready: Array[Tensor] =>
          match session.run(ready)
          | let outs: Array[Tensor] =>
            let sum_ok = expect_f32(
              env, label, outs, "out_sum",
              [F32(5); F32(11); F32(7); F32(22); F32(9); F32(33)])
            let diff_ok = expect_f32(
              env, label, outs, "out_diff",
              [F32(-3); F32(9); F32(-3); F32(18); F32(-3); F32(27)])
            (outs.size() == declared.size()) and sum_ok and diff_ok
          | let err: String =>
            env.out.print(label + " run failed: " + err)
            false
          end
        | let err: String =>
          env.out.print(label + " " + err)
          false
        end
      | let err: String =>
        env.out.print(label + " outputs failed: " + err)
        false
      end
    | let err: String =>
      env.out.print(label + " inputs failed: " + err)
      false
    end

  fun ref ids_case(env: Env, label: String, session: Runner): Bool =>
    match session.inputs()
    | let spec: Array[Tensor] =>
      show_io(env, label + " inputs", spec)
      let given = Array[Tensor]
      given.push(maker.i64("ids", [I64(3)], [I64(10); I64(20); I64(30)]))
      match align(spec, given)
      | let ready: Array[Tensor] =>
        match session.run(ready)
        | let outs: Array[Tensor] =>
          expect_i64(env, label, outs, "ids_out", [I64(11); I64(21); I64(31)])
        | let err: String =>
          env.out.print(label + " ids failed: " + err)
          false
        end
      | let err: String =>
        env.out.print(label + " " + err)
        false
      end
    | let err: String =>
      env.out.print(label + " inputs failed: " + err)
      false
    end

  fun ref gemm_case(env: Env, session: OrtSession): Bool =>
    let cells = USize(1024) * USize(1024)
    match session.inputs()
    | let spec: Array[Tensor] =>
      let given = Array[Tensor]
      given.push(maker.f32_fill("x", [I64(1024); I64(1024)], cells, F32(1)))
      match align(spec, given)
      | let ready: Array[Tensor] =>
        var rounds = USize(0)
        var ok = true
        while rounds < 40 do
          match session.run(ready)
          | let outs: Array[Tensor] =>
            match named(outs, "y")
            | let y: Tensor =>
              let n = y.data.size() / 4
              let first = y.f32(0)
              let last = y.f32(n - 1)
              if (n != cells) or (first != F32(1024)) or (last != F32(1024)) then
                env.out.print("ort gemm failed " + first.string() + " " + n.string())
                ok = false
                break
              end
            else
              env.out.print("ort gemm missing y")
              ok = false
              break
            end
          | let err: String =>
            env.out.print("ort gemm failed: " + err)
            ok = false
            break
          end
          rounds = rounds + 1
        end
        if ok then env.out.print("ort gemm 1024 ok") end
        ok
      | let err: String =>
        env.out.print("ort gemm " + err)
        false
      end
    | let err: String =>
      env.out.print("ort gemm inputs failed: " + err)
      false
    end

  fun ref align(spec: Array[Tensor], given: Array[Tensor]): (Array[Tensor] | String) =>
    let out = Array[Tensor]
    for declared in spec.values() do
      match named(given, declared.name)
      | let item: Tensor => out.push(item)
      else
        return "missing input " + declared.name
      end
    end
    out

  fun ref named(items: Array[Tensor], name: String): (Tensor | None) =>
    for item in items.values() do
      if item.name == name then return item end
    end
    None

  fun ref show_io(env: Env, label: String, items: Array[Tensor]) =>
    env.out.write(label)
    for t in items.values() do
      env.out.write(" ")
      env.out.write(t.name)
      env.out.write(":")
      env.out.write(t.kind_name())
      env.out.write("[")
      var d = USize(0)
      while d < t.shape.size() do
        if d > 0 then env.out.write(", ") end
        try env.out.write(t.shape(d)?.string()) end
        d = d + 1
      end
      env.out.write("]")
    end
    env.out.print("")

  fun ref expect_f32(
    env: Env,
    label: String,
    items: Array[Tensor],
    name: String,
    expect: Array[F32] box)
    : Bool
  =>
    match named(items, name)
    | let t: Tensor =>
      env.out.write(label + " out." + t.name)
      let n = t.data.size() / 4
      var ok = (t.kind == 0) and (n == expect.size())
      var i = USize(0)
      while i < n do
        let v = t.f32(i)
        env.out.write(" ")
        env.out.write(v.string())
        if i < expect.size() then
          try ok = ok and (v == expect(i)?) end
        end
        i = i + 1
      end
      if ok then
        env.out.print(" ok")
      else
        env.out.print(" failed")
      end
      ok
    else
      env.out.print(label + " missing out." + name)
      false
    end

  fun ref expect_i64(
    env: Env,
    label: String,
    items: Array[Tensor],
    name: String,
    expect: Array[I64] box)
    : Bool
  =>
    match named(items, name)
    | let t: Tensor =>
      env.out.write(label + " out." + t.name)
      let n = t.data.size() / 8
      var ok = (t.kind == 3) and (n == expect.size())
      var i = USize(0)
      while i < n do
        let v = t.i64(i)
        env.out.write(" ")
        env.out.write(v.string())
        if i < expect.size() then
          try ok = ok and (v == expect(i)?) end
        end
        i = i + 1
      end
      if ok then
        env.out.print(" ok")
      else
        env.out.print(" failed")
      end
      ok
    else
      env.out.print(label + " missing out." + name)
      false
    end
