import struct
from pathlib import Path

root = Path(__file__).resolve().parent


def varint(n):
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field, wire):
    return varint((field << 3) | wire)


def pint(field, n):
    return key(field, 0) + varint(n)


def pstr(field, text):
    raw = text.encode()
    return key(field, 2) + varint(len(raw)) + raw


def pmsg(field, body):
    return key(field, 2) + varint(len(body)) + body


def pfloats(field, values):
    raw = struct.pack("<" + ("f" * len(values)), *values)
    return key(field, 2) + varint(len(raw)) + raw


def pfill(field, value, count):
    raw = struct.pack("<f", value) * count
    return key(field, 2) + varint(len(raw)) + raw


def onnx_shape(dims):
    return b"".join(pmsg(1, pint(1, d)) for d in dims)


def onnx_value(name, dims, elem=1):
    tensor = pint(1, elem) + pmsg(2, onnx_shape(dims))
    return pstr(1, name) + pmsg(2, pmsg(1, tensor))


def onnx_dim_param(name):
    return pstr(2, name)


def onnx_dim_fixed(n):
    return pint(1, n)


def onnx_value_dims(name, dim_msgs, elem=1):
    shape = b"".join(pmsg(1, d) for d in dim_msgs)
    tensor = pint(1, elem) + pmsg(2, shape)
    return pstr(1, name) + pmsg(2, pmsg(1, tensor))


def onnx_model():
    node = pstr(1, "x") + pstr(1, "b") + pstr(2, "y") + pstr(3, "add") + pstr(4, "Add")
    init = pint(1, 4) + pint(2, 1) + pfloats(4, [1.0, 1.0, 1.0, 1.0]) + pstr(8, "b")
    graph = (
        pmsg(1, node)
        + pstr(2, "add")
        + pmsg(5, init)
        + pmsg(11, onnx_value("x", [4]))
        + pmsg(12, onnx_value("y", [4]))
    )
    opset = pstr(1, "") + pint(2, 13)
    return pint(1, 7) + pstr(2, "pony-ml") + pmsg(7, graph) + pmsg(8, opset)


def gemm_model(n=1024):
    node = pstr(1, "x") + pstr(1, "b") + pstr(2, "y") + pstr(3, "gemm") + pstr(4, "MatMul")
    init = pint(1, n) + pint(1, n) + pint(2, 1) + pfill(4, 1.0, n * n) + pstr(8, "b")
    graph = (
        pmsg(1, node)
        + pstr(2, "gemm")
        + pmsg(5, init)
        + pmsg(11, onnx_value("x", [n, n]))
        + pmsg(12, onnx_value("y", [n, n]))
    )
    opset = pstr(1, "") + pint(2, 13)
    return pint(1, 7) + pstr(2, "pony-ml") + pmsg(7, graph) + pmsg(8, opset)


def tf_shape(dims):
    return b"".join(pmsg(2, pint(1, d)) for d in dims)


def tf_attr(name, value):
    return pmsg(5, pstr(1, name) + pmsg(2, value))


def tf_node(name, op, inputs, attrs):
    body = pstr(1, name) + pstr(2, op)
    for item in inputs:
        body += pstr(3, item)
    return body + attrs


def tf_saved_model():
    x = tf_node(
        "x",
        "Placeholder",
        [],
        tf_attr("dtype", pint(6, 1)) + tf_attr("shape", pmsg(7, tf_shape([4]))),
    )
    ones = pint(1, 1) + pmsg(2, tf_shape([4])) + pfloats(5, [1.0, 1.0, 1.0, 1.0])
    b = tf_node(
        "b",
        "Const",
        [],
        tf_attr("dtype", pint(6, 1)) + tf_attr("value", pmsg(8, ones)),
    )
    y = tf_node("y", "Add", ["x", "b"], tf_attr("T", pint(6, 1)))
    graph = pmsg(1, x) + pmsg(1, b) + pmsg(1, y) + pmsg(4, pint(1, 1994))
    info = pstr(1, "x:0") + pint(2, 1) + pmsg(3, tf_shape([4]))
    out_info = pstr(1, "y:0") + pint(2, 1) + pmsg(3, tf_shape([4]))
    signature = (
        pmsg(1, pstr(1, "x") + pmsg(2, info))
        + pmsg(2, pstr(1, "y") + pmsg(2, out_info))
        + pstr(3, "tensorflow/serving/predict")
    )
    meta = (
        pmsg(1, pstr(4, "serve") + pstr(5, "2.18.0"))
        + pmsg(2, graph)
        + pmsg(5, pstr(1, "serving_default") + pmsg(2, signature))
    )
    return pint(1, 1) + pmsg(2, meta)


def multi_io_model():
    add = pstr(1, "input_1") + pstr(1, "input_2") + pstr(2, "out_sum") + pstr(4, "Add")
    sub = pstr(1, "input_1") + pstr(1, "input_2") + pstr(2, "out_diff") + pstr(4, "Sub")
    batch = [onnx_dim_param("N"), onnx_dim_fixed(2)]
    graph = (
        pmsg(1, add)
        + pmsg(1, sub)
        + pstr(2, "multi")
        + pmsg(11, onnx_value_dims("input_1", batch))
        + pmsg(11, onnx_value_dims("input_2", batch))
        + pmsg(12, onnx_value_dims("out_sum", batch))
        + pmsg(12, onnx_value_dims("out_diff", batch))
    )
    opset = pstr(1, "") + pint(2, 13)
    return pint(1, 7) + pstr(2, "pony-ml") + pmsg(7, graph) + pmsg(8, opset)


def ids_model():
    node = pstr(1, "ids") + pstr(1, "one") + pstr(2, "ids_out") + pstr(4, "Add")
    one = struct.pack("<q", 1)
    init = pint(2, 7) + pmsg(9, one) + pstr(8, "one")
    batch = [onnx_dim_param("N")]
    graph = (
        pmsg(1, node)
        + pstr(2, "ids")
        + pmsg(5, init)
        + pmsg(11, onnx_value_dims("ids", batch, 7))
        + pmsg(12, onnx_value_dims("ids_out", batch, 7))
    )
    opset = pstr(1, "") + pint(2, 13)
    return pint(1, 7) + pstr(2, "pony-ml") + pmsg(7, graph) + pmsg(8, opset)


def tf_tensor(dtype, dims, raw):
    return pint(1, dtype) + pmsg(2, tf_shape(dims)) + pmsg(4, raw)


def tf_saved(nodes, sig_inputs, sig_outputs):
    graph = b"".join(pmsg(1, node) for node in nodes) + pmsg(4, pint(1, 1994))
    signature = b""
    for key, tensor_name, dtype, dims in sig_inputs:
        info = pstr(1, tensor_name) + pint(2, dtype) + pmsg(3, tf_shape(dims))
        signature += pmsg(1, pstr(1, key) + pmsg(2, info))
    for key, tensor_name, dtype, dims in sig_outputs:
        info = pstr(1, tensor_name) + pint(2, dtype) + pmsg(3, tf_shape(dims))
        signature += pmsg(2, pstr(1, key) + pmsg(2, info))
    signature += pstr(3, "tensorflow/serving/predict")
    meta = (
        pmsg(1, pstr(4, "serve") + pstr(5, "2.18.0"))
        + pmsg(2, graph)
        + pmsg(5, pstr(1, "serving_default") + pmsg(2, signature))
    )
    return pint(1, 1) + pmsg(2, meta)


def tf_multi():
    x = tf_node(
        "input_1", "Placeholder", [],
        tf_attr("dtype", pint(6, 1)) + tf_attr("shape", pmsg(7, tf_shape([-1, 2]))),
    )
    y = tf_node(
        "input_2", "Placeholder", [],
        tf_attr("dtype", pint(6, 1)) + tf_attr("shape", pmsg(7, tf_shape([-1, 2]))),
    )
    add = tf_node("out_sum", "Add", ["input_1", "input_2"], tf_attr("T", pint(6, 1)))
    sub = tf_node("out_diff", "Sub", ["input_1", "input_2"], tf_attr("T", pint(6, 1)))
    ports = [
        ("input_1", "input_1:0", 1, [-1, 2]),
        ("input_2", "input_2:0", 1, [-1, 2]),
    ]
    outs = [
        ("out_sum", "out_sum:0", 1, [-1, 2]),
        ("out_diff", "out_diff:0", 1, [-1, 2]),
    ]
    return tf_saved([x, y, add, sub], ports, outs)


def tf_ids():
    ids = tf_node(
        "ids", "Placeholder", [],
        tf_attr("dtype", pint(6, 9)) + tf_attr("shape", pmsg(7, tf_shape([-1]))),
    )
    one = tf_node(
        "one", "Const", [],
        tf_attr("dtype", pint(6, 9)) + tf_attr("value", pmsg(8, tf_tensor(9, [], struct.pack("<q", 1)))),
    )
    add = tf_node("ids_out", "Add", ["ids", "one"], tf_attr("T", pint(6, 9)))
    return tf_saved(
        [ids, one, add],
        [("ids", "ids:0", 9, [-1])],
        [("ids_out", "ids_out:0", 9, [-1])],
    )


def main():
    onnx_path = root / "add.onnx"
    saved_path = root / "add" / "saved_model.pb"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model())
    gemm_path = root / "gemm.onnx"
    gemm_path.write_bytes(gemm_model())
    saved_path.write_bytes(tf_saved_model())
    multi_path = root / "multi_io.onnx"
    ids_path = root / "ids.onnx"
    multi_saved = root / "multi" / "saved_model.pb"
    ids_saved = root / "ids" / "saved_model.pb"
    multi_saved.parent.mkdir(parents=True, exist_ok=True)
    ids_saved.parent.mkdir(parents=True, exist_ok=True)
    multi_path.write_bytes(multi_io_model())
    ids_path.write_bytes(ids_model())
    multi_saved.write_bytes(tf_multi())
    ids_saved.write_bytes(tf_ids())
    for path in (onnx_path, gemm_path, saved_path, multi_path, ids_path, multi_saved, ids_saved):
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
