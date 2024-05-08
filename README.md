## Installation

### Build
```bash
make start force-build=true use-gpu=false
```
* `force-build=true` - rebuilds the image
* `use-gpu=false` - disables GPU support

### Run
```
make exec
```

```bash
 python /home/workdir/assets/compile_toy_model.py --onnx_path=/home/workdir/assets/toy_model_opset11.onnx --artifacts_folder=/home/workdir/assets/artifacts compile
python /home/workdir/assets/compile_toy_model.py --onnx_path=/home/workdir/assets/toy_model_opset11.onnx --artifacts_folder=/home/workdir/assets/artifacts inference
```

### Stop
```
make stop
```

python /home/workdir/assets/compile_max_pool.py --onnx_path=/home/workdir/assets/max_pool_model_3_3.onnx --artifacts_folder=/home/workdir/assets/artifacts_max_pool compile