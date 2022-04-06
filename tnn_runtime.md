## TopFormer-TNNRuntime

The doc introduces how to convert models into TNN.

**step 1**, convert your model to ONNX, run:

```
python3 tools/convert2onnx.py <config-file> --input-img <img-dir> --shape 512 512 --checkpoint <model-ckpt>
```

**step 2**, clone the TNN:

```
git clone https://github.com/Tencent/TNN.git
```

**step 3**, covert model to TNN following [convert2tnn](https://github.com/Tencent/TNN/blob/master/doc/en/user/convert_en.md).

- install the dependencies of convert2tnn
- compile the tools.

```
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh
```

then run:

```
python3 converter.py onnx2tnn <onnx-model-path> -optimize -v=v3.0 -o <output-file-name>
```

**step 4**, compile for Android following [tnn-compile](https://github.com/Tencent/TNN/blob/master/doc/en/user/compile_en.md).

- cmake（version 3.6 or higher）
- NDK configuration

then run:

```
./build_android.sh
```

**step 5**, models benchmark.

push all benchmark models to android device_dir:

```
/data/local/tmp/benchmark-model
```
then run:
```
./benchmark_models.sh -32 -t cpu -bs -f
```

you will get all model benchmark cost time info.
