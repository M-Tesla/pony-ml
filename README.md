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

The test checks `multi_io` and `ids` on both runtimes. On ONNX it also checks a 1024 MatMul of ones, whose result is 1024.
