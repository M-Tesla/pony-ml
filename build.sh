#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ort="$root/vendor/onnxruntime-linux-x64-gpu_cuda13-1.30.0"
cuda="$root/vendor/cuda/lib"
tf="$root/vendor/libtensorflow"
ponyc="$root/vendor/0.74.0-95ad180/bin/ponyc"

mkdir -p "$root/lib" "$root/build"
python3 "$root/models/make.py"

gcc -std=c11 -shared -fPIC -O2 \
  -o "$root/lib/libml_u8.so" \
  "$root/c/ml_u8.c"

gcc -std=c11 -shared -fPIC -O2 \
  -I"$ort/include" \
  -o "$root/lib/libort_shim.so" \
  "$root/c/ort_shim.c" "$root/c/blob.c" \
  -L"$ort/lib" -lonnxruntime \
  -Wl,-rpath,"$ort/lib" -Wl,-rpath,"$cuda"

gcc -std=c11 -shared -fPIC -O2 \
  -I"$tf/include" \
  -o "$root/lib/libtf_shim.so" \
  "$root/c/tf_shim.c" "$root/c/blob.c" \
  -L"$tf/lib" -ltensorflow \
  -Wl,-rpath,"$tf/lib"

"$ponyc" --bin-name pony-ml -o "$root/build" "$root/src"

export LD_LIBRARY_PATH="$cuda:$root/lib:$ort/lib:$tf/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$root"
"$root/build/pony-ml"
