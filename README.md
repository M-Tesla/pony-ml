# pony-ml

A Pony program that opens an ONNX file or a TensorFlow SavedModel. It reads the names, dtypes, and shapes from the file, accepts several rows in one request, and returns one output for each name.

ONNX Runtime is the GPU build with CUDA 13. TensorFlow is the 2.18 C library for CPU. That release does not publish a GPU C library.

## Code

`src/` is the program. `c/` talks to the C libraries. `models/make.py` generates the models used by the test. Downloaded libraries live in `vendor/` and are not part of the git history.

## How to run

`build.sh` looks for this tree next to the script:

- `vendor/0.74.0-95ad180/bin/ponyc`
- `vendor/onnxruntime-linux-x64-gpu_cuda13-1.30.0`
- `vendor/libtensorflow`
- `vendor/cuda/lib`, with the CUDA 13 libraries the GPU provider requires

```
sh build.sh
```

## Tests run on this machine

Machine: Intel Xeon E5-2666 v3, NVIDIA GeForce RTX 3060 12 GB. ONNX Runtime 1.30.0 with the CUDA provider on device 0. TensorFlow 2.18.0 C library, CPU.

`multi_io` has inputs `input_1:f32[-1, 2]` and `input_2:f32[-1, 2]`. Three rows were sent:

- `input_1`: 1 10, 2 20, 3 30
- `input_2`: 4 1, 5 2, 6 3

ONNX and TensorFlow both returned:

- `out.out_sum`: 5 11, 7 22, 9 33
- `out.out_diff`: -3 9, -3 18, -3 27

`ids` has input `ids:i64[-1]`. The row 10 20 30 returned `out.ids_out` 11 21 31 on both runtimes.

The ONNX MatMul of a 1024 by 1024 matrix of ones returned 1024 in every checked cell. During that run the provider list included `CUDAExecutionProvider`.

SqueezeNet 1.1 (`squeezenet1.1-7.onnx`, from the ONNX model zoo) declares `data:f32[1, 3, 224, 224]` and `squeezenet0_flatten0_reshape0:f32[1, 1000]`. The input was zeros of that shape. After 50 warmup calls, 200 full calls were timed. A full call builds the request, runs the model, and reads the named output. The times were min 1.064 ms, median 1.121 ms, p99 1.505 ms. While those runs were in progress, GPU utilization peaked at 60% and memory moved from 745 MiB to 958 MiB.
