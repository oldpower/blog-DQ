# 【CUDA-BEVFusion】tool/build_trt_engine.sh 文件解读

## `build_trt_engine.sh`

```bash
# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet50/resnet50-int8/swint-tiny
base=model/$DEBUG_MODEL

# fp16/int8
precision=$DEBUG_PRECISION

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

function get_onnx_number_io(){

    # $1=model
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    # $5: extra_flags
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    extra_flags=$5
    result_save_directory=$base/build
    onnx=$base/$name.onnx

    if [ -f "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan already build 🙋🙋🙋.
        return
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    # echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    input_flags=${input_flags%?}
    output_flags=${output_flags%?}

    cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} ${extra_flags} \
        --saveEngine=${result_save_directory}/$name.plan \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"

    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model ${result_save_directory}/$name.plan.
        echo You can check the error message by ${result_save_directory}/$name.log 
        exit 1
    fi
}

# maybe int8 / fp16
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1

# fp16 only
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1

# for myelin layernorm head.bbox, may occur a tensorrt bug at layernorm fusion but faster
compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6

# for layernorm version head.bbox.onnx, accurate but slower
# compile_trt_model "head.bbox.layernormplugin" "$trtexec_fp16_flags" 1 6 "--plugins=libcustom_layernorm.so"
```
---
这段代码是一个用于编译 TensorRT 模型的 Bash 脚本。TensorRT 是 NVIDIA 提供的一个高性能深度学习推理库，能够优化和加速深度学习模型的推理过程。

### 1. **环境配置**
   - `tool/environment.sh` 脚本被调用来配置环境。这个脚本可能设置了必要的环境变量、路径等。
   - 如果环境配置失败（`ConfigurationStatus` 不等于 "Success"），脚本会输出错误信息并退出。

### 2. **TensorRT 版本检查**
   - 注释中提到可以通过 `trtexec` 命令获取 TensorRT 版本，但这段代码被注释掉了，可能是为了减少依赖或简化流程。

### 3. **模型和精度设置**
   - `base=model/$DEBUG_MODEL`：设置模型的基础路径，`DEBUG_MODEL` 可能是一个环境变量，指定了模型的类型（如 `resnet50`、`resnet50-int8`、`swint-tiny` 等）。
   - `precision=$DEBUG_PRECISION`：设置模型的精度（如 `fp16` 或 `int8`），`DEBUG_PRECISION` 也是一个环境变量。

### 4. **精度标志**
   - `trtexec_fp16_flags` 和 `trtexec_dynamic_flags` 是根据精度设置的 TensorRT 编译标志。
   - 如果精度是 `int8`，则 `trtexec_dynamic_flags` 会包含 `--int8` 标志。

### 5. **函数 `get_onnx_number_io`**
   - 这个函数用于获取 ONNX 模型的输入和输出数量。
   - 它使用 Python 脚本加载 ONNX 模型并解析其输入输出数量。
   - 如果模型文件不存在，函数会输出错误信息并返回。

### 6. **函数 `compile_trt_model`**
   - 这个函数用于编译 TensorRT 模型。
   - 参数：
     - `name`：模型名称。
     - `precision_flags`：精度标志（如 `--fp16` 或 `--int8`）。
     - `number_of_input` 和 `number_of_output`：模型的输入和输出数量。
     - `extra_flags`：额外的编译标志。
   - 函数首先检查是否已经存在编译好的模型文件（`.plan` 文件），如果存在则跳过编译。
   - 然后根据输入输出数量生成 `input_flags` 和 `output_flags`，这些标志用于指定输入输出的格式。
   - 使用 `trtexec` 命令编译模型，生成 `.plan` 文件，并将日志输出到 `.log` 文件中。
   - 如果编译失败，脚本会输出错误信息并退出。

### 7. **模型编译**
   - 脚本最后调用 `compile_trt_model` 函数编译多个模型：
     - `camera.backbone` 和 `fuser` 模型使用动态精度标志（可能是 `fp16` 或 `int8`）。
     - `camera.vtransform` 模型只使用 `fp16` 精度。
     - `head.bbox` 模型也使用 `fp16` 精度，但注释中提到可能存在 TensorRT 的 bug，因此提供了两种编译方式：
       - 一种是不使用 `layernorm` 插件，可能会更快但不够准确。
       - 另一种是使用 `layernorm` 插件，可能会更准确但速度较慢。

### 8. **总结**
   - 这个脚本的主要功能是根据指定的模型和精度，使用 TensorRT 编译 ONNX 模型为 `.plan` 文件，以便在 NVIDIA 硬件上进行高效的推理。
   - 脚本通过检查环境配置、模型输入输出数量、精度标志等，确保编译过程的正确性和高效性。
   - 如果编译失败，脚本会输出详细的错误信息，帮助用户排查问题。
