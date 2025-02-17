# 【TensorRT】trtexec的参数说明

以下是 `trtexec` 工具的详细参数说明：

## 1、参数
### `Model Options`

| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--uff=<file>` | 指定 UFF 模型文件 | `--uff=model.uff` | 用于加载 UFF 格式的模型。 |
| `--onnx=<file>` | 指定 ONNX 模型文件 | `--onnx=model.onnx` | 用于加载 ONNX 格式的模型。 |
| `--model=<file>` | 指定 Caffe 模型文件（默认 = 无模型，使用随机权重） | `--model=model.caffemodel` | 用于加载 Caffe 格式的模型文件。 |
| `--deploy=<file>` | 指定 Caffe 的 prototxt 文件 | `--deploy=deploy.prototxt` | 用于加载 Caffe 模型的网络结构文件。 |
| `--output=<name>[,<name>]*` | 指定输出名称（可以多次指定）；UFF 和 Caffe 模型至少需要一个输出 | `--output=output1,output2` | 用于指定模型的输出节点名称。 |
| `--uffInput=<name>,X,Y,Z` | 指定输入 blob 名称及其维度（X,Y,Z = C,H,W），可以多次指定；UFF 模型至少需要一个输入 | `--uffInput=input1,3,224,224` | 用于指定 UFF 模型的输入名称和维度。 |
| `--uffNHWC` | 设置输入是否为 NHWC 布局而不是 NCHW（在 `--uffInput` 中使用 X,Y,Z = H,W,C 顺序） | `--uffNHWC` | 用于指定 UFF 模型的输入布局格式。 |
### `Build Options`

| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--maxBatch` | 设置最大批处理大小并构建隐式批处理引擎（默认值与 `--batch` 相同） | `--maxBatch=32` | 当输入模型为 ONNX 或提供动态形状时，不应使用此选项。 |
| `--minShapes=spec` | 使用提供的最小形状配置文件构建动态形状 | `--minShapes=input0:1x3x256x256,input1:1x3x128x128` | 必须同时提供 `minShapes`、`optShapes` 和 `maxShapes`。 |
| `--optShapes=spec` | 使用提供的最优形状配置文件构建动态形状 | `--optShapes=input0:1x3x256x256,input1:1x3x128x128` | 如果仅提供 `optShapes`，则 `minShapes` 和 `maxShapes` 将设置为与 `optShapes` 相同的值。 |
| `--maxShapes=spec` | 使用提供的最大形状配置文件构建动态形状 | `--maxShapes=input0:1x3x256x256,input1:1x3x128x128` | 输入名称可以用转义的单引号包裹（例如：`'Input:0'`）。 |
| `--minShapesCalib=spec` | 使用提供的最小形状配置文件校准动态形状 | `--minShapesCalib=input0:1x3x256x256,input1:1x3x128x128` | 必须同时提供 `minShapesCalib`、`optShapesCalib` 和 `maxShapesCalib`。 |
| `--optShapesCalib=spec` | 使用提供的最优形状配置文件校准动态形状 | `--optShapesCalib=input0:1x3x256x256,input1:1x3x128x128` | 如果仅提供 `optShapesCalib`，则 `minShapesCalib` 和 `maxShapesCalib` 将设置为与 `optShapesCalib` 相同的值。 |
| `--maxShapesCalib=spec` | 使用提供的最大形状配置文件校准动态形状 | `--maxShapesCalib=input0:1x3x256x256,input1:1x3x128x128` | 输入名称可以用转义的单引号包裹（例如：`'Input:0'`）。 |
| `--inputIOFormats=spec` | 指定每个输入张量的类型和格式（默认 = 所有输入为 `fp32:chw`） | `--inputIOFormats=fp16:chw,fp32:chw` | 如果指定此选项，请为所有输入设置逗号分隔的类型和格式，顺序与网络输入 ID 相同。 |
| `--outputIOFormats=spec` | 指定每个输出张量的类型和格式（默认 = 所有输出为 `fp32:chw`） | `--outputIOFormats=fp16:chw,fp32:chw` | 如果指定此选项，请为所有输出设置逗号分隔的类型和格式，顺序与网络输出 ID 相同。 |
| `--workspace=N` | 设置工作空间大小（MiB） | `--workspace=1024` | 工作空间大小以 MiB 为单位。 |
| `--memPoolSize=poolspec` | 指定指定内存池的大小约束（MiB） | `--memPoolSize=workspace:1024,dlaSRAM:64` | 接受小数大小（例如 `0.25MiB`），并向下舍入到最接近的整数字节。 |
| `--profilingVerbosity=mode` | 指定分析详细程度 | `--profilingVerbosity=detailed` | 模式包括 `layer_names_only`、`detailed` 和 `none`（默认 = `layer_names_only`）。 |
| `--minTiming=M` | 设置内核选择中使用的最小迭代次数（默认 = 1） | `--minTiming=5` | 用于内核选择的最小迭代次数。 |
| `--avgTiming=M` | 设置每次迭代中用于内核选择的平均次数（默认 = 8） | `--avgTiming=10` | 用于内核选择的平均次数。 |
| `--refit` | 将引擎标记为可重构 | `--refit` | 允许检查引擎中的可重构层和权重。 |
| `--versionCompatible, --vc` | 将引擎标记为版本兼容 | `--versionCompatible` | 允许引擎在相同主机操作系统上与新版本的 TensorRT 一起使用。 |
| `--useRuntime=runtime` | 指定用于执行引擎的 TensorRT 运行时 | `--useRuntime=lean` | 运行时选项包括 `full`、`lean` 和 `dispatch`。 |
| `--leanDLLPath=<file>` | 在版本兼容模式下使用的外部精简运行时 DLL | `--leanDLLPath=lean.dll` | 仅适用于版本兼容模式。 |
| `--excludeLeanRuntime` | 在启用 `--versionCompatible` 时，此标志表示生成的引擎不应包含嵌入式精简运行时 | `--excludeLeanRuntime` | 用户必须显式指定有效的精简运行时以加载引擎。 |
| `--sparsity=spec` | 控制稀疏性（默认 = 禁用） | `--sparsity=enable` | 选项包括 `disable`、`enable` 和 `force`。 |
| `--noTF32` | 禁用 TF32 精度（默认启用 TF32，同时启用 FP32） | `--noTF32` | 默认情况下启用 TF32。 |
| `--fp16` | 启用 FP16 精度，同时启用 FP32（默认 = 禁用） | `--fp16` | 默认情况下禁用 FP16。 |
| `--int8` | 启用 INT8 精度，同时启用 FP32（默认 = 禁用） | `--int8` | 默认情况下禁用 INT8。 |
| `--fp8` | 启用 FP8 精度，同时启用 FP32（默认 = 禁用） | `--fp8` | 默认情况下禁用 FP8。 |
| `--best` | 启用所有精度以实现最佳性能（默认 = 禁用） | `--best` | 默认情况下禁用。 |
| `--directIO` | 避免在网络边界处重新格式化（默认 = 禁用） | `--directIO` | 默认情况下禁用。 |
| `--precisionConstraints=spec` | 控制精度约束设置（默认 = `none`） | `--precisionConstraints=obey` | 选项包括 `none`、`obey` 和 `prefer`。 |
| `--layerPrecisions=spec` | 控制每层精度约束 | `--layerPrecisions=conv1:fp16,conv2:fp32` | 仅在 `precisionConstraints` 设置为 `obey` 或 `prefer` 时有效。 |
| `--layerOutputTypes=spec` | 控制每层输出类型约束 | `--layerOutputTypes=conv1:fp16,conv2:fp32` | 仅在 `precisionConstraints` 设置为 `obey` 或 `prefer` 时有效。 |
| `--layerDeviceTypes=spec` | 指定每层设备类型 | `--layerDeviceTypes=conv1:GPU,conv2:DLA` | 如果未指定设备类型，则层将选择默认设备类型。 |
| `--calib=<file>` | 读取 INT8 校准缓存文件 | `--calib=calib.cache` | 用于 INT8 校准的缓存文件。 |
| `--safe` | 启用构建安全认证引擎 | `--safe` | 如果启用了 DLA，则自动指定 `--buildDLAStandalone`。 |
| `--buildDLAStandalone` | 启用构建 DLA 独立可加载文件 | `--buildDLAStandalone` | 启用此选项时，`--allowGPUFallback` 被禁止，`--skipInference` 默认启用。 |
| `--allowGPUFallback` | 当启用 DLA 时，允许 GPU 回退以处理不受支持的层（默认 = 禁用） | `--allowGPUFallback` | 默认情况下禁用。 |
| `--consistency` | 对安全认证引擎执行一致性检查 | `--consistency` | 用于安全认证引擎的一致性检查。 |
| `--restricted` | 启用安全范围检查 | `--restricted` | 使用 `kSAFETY_SCOPE` 构建标志启用安全范围检查。 |
| `--saveEngine=<file>` | 保存序列化引擎 | `--saveEngine=engine.plan` | 保存生成的引擎文件。 |
| `--loadEngine=<file>` | 加载序列化引擎 | `--loadEngine=engine.plan` | 加载现有的引擎文件。 |
| `--tacticSources=tactics` | 指定要使用的策略 | `--tacticSources=-CUDNN,+CUBLAS` | 通过添加（+）或删除（-）策略来指定使用的策略。 |
| `--noBuilderCache` | 禁用构建器中的计时缓存（默认启用计时缓存） | `--noBuilderCache` | 默认情况下启用计时缓存。 |
| `--heuristic` | 启用构建器中的策略选择启发式（默认禁用启发式） | `--heuristic` | 默认情况下禁用启发式。 |
| `--timingCacheFile=<file>` | 保存/加载序列化的全局计时缓存 | `--timingCacheFile=cache.timing` | 用于保存或加载计时缓存文件。 |
| `--preview=features` | 指定要使用的预览功能 | `--preview=+fasterDynamicShapes0805` | 通过添加（+）或删除（-）预览功能来指定使用的功能。 |
| `--builderOptimizationLevel` | 设置构建器优化级别（默认 = 3） | `--builderOptimizationLevel=5` | 较高的级别允许 TensorRT 花费更多时间进行优化。 |
| `--hardwareCompatibilityLevel=mode` | 使引擎文件与其他 GPU 架构兼容（默认 = `none`） | `--hardwareCompatibilityLevel=ampere+` | 选项包括 `none` 和 `ampere+`。 |
| `--tempdir=<dir>` | 覆盖 TensorRT 创建临时文件时使用的默认临时目录 | `--tempdir=/tmp` | 指定临时文件目录。 |
| `--tempfileControls=controls` | 控制 TensorRT 创建临时可执行文件时的权限 | `--tempfileControls=in_memory:allow,temporary:deny` | 控制是否允许创建内存中或文件系统中的临时文件。 |
| `--maxAuxStreams=N` | 设置每个推理流的最大辅助流数量 | `--maxAuxStreams=4` | 用于并行运行内核的辅助流数量，设置为 0 以实现最佳内存使用。 |
### `Inference Options`

| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--batch=N` | 设置隐式批处理引擎的批处理大小（默认 = 1） | `--batch=16` | 当引擎从 ONNX 模型构建或引擎构建时提供了动态形状时，不应使用此选项。 |
| `--shapes=spec` | 设置动态形状推理输入的输入形状 | `--shapes=input0:1x3x256x256,input1:1x3x128x128` | 输入名称可以用转义的单引号包裹（例如：`'Input:0'`）。 |
| `--loadInputs=spec` | 从文件加载输入值（默认 = 生成随机输入） | `--loadInputs=input0:input0.bin,input1:input1.bin` | 输入名称可以用单引号包裹（例如：`'Input:0'`）。 |
| `--iterations=N` | 运行至少 N 次推理迭代（默认 = 10） | `--iterations=100` | 用于控制推理运行的迭代次数。 |
| `--warmUp=N` | 在测量性能之前运行 N 毫秒进行预热（默认 = 200） | `--warmUp=500` | 用于预热 GPU 或其他硬件。 |
| `--duration=N` | 运行性能测量至少 N 秒的挂钟时间（默认 = 3） | `--duration=10` | 如果指定为 `-1`，推理将持续运行，除非手动停止。 |
| `--sleepTime=N` | 延迟推理启动，在启动和计算之间设置 N 毫秒的间隔（默认 = 0） | `--sleepTime=100` | 用于模拟延迟启动的场景。 |
| `--idleTime=N` | 在两次连续迭代之间休眠 N 毫秒（默认 = 0） | `--idleTime=50` | 用于控制迭代之间的间隔时间。 |
| `--infStreams=N` | 实例化 N 个引擎以并发运行推理（默认 = 1） | `--infStreams=4` | 用于并发推理的场景。 |
| `--exposeDMA` | 序列化与设备之间的 DMA 传输（默认 = 禁用） | `--exposeDMA` | 用于调试或分析 DMA 传输。 |
| `--noDataTransfers` | 禁用与设备之间的 DMA 传输（默认 = 启用） | `--noDataTransfers` | 用于测试无数据传输的场景。 |
| `--useManagedMemory` | 使用托管内存而不是单独的主机和设备分配（默认 = 禁用） | `--useManagedMemory` | 用于简化内存管理。 |
| `--useSpinWait` | 主动同步 GPU 事件。此选项可能会减少同步时间，但会增加 CPU 使用率和功耗（默认 = 禁用） | `--useSpinWait` | 用于优化同步性能。 |
| `--threads` | 启用多线程以使用独立线程驱动引擎或加速重构（默认 = 禁用） | `--threads` | 用于多线程推理或重构优化。 |
| `--useCudaGraph` | 使用 CUDA 图捕获引擎执行，然后启动推理（默认 = 禁用） | `--useCudaGraph` | 如果图捕获失败，此标志可能会被忽略。 |
| `--timeDeserialize` | 测量反序列化网络所需的时间并退出 | `--timeDeserialize` | 用于测量反序列化性能。 |
| `--timeRefit` | 测量推理前重构引擎所需的时间 | `--timeRefit` | 用于测量重构性能。 |
| `--separateProfileRun` | 不在基准测试运行中附加分析器；如果启用了分析，将执行第二次分析运行（默认 = 禁用） | `--separateProfileRun` | 用于分离性能测量和分析。 |
| `--skipInference` | 在引擎构建完成后退出并跳过推理性能测量（默认 = 禁用） | `--skipInference` | 用于仅构建引擎而不运行推理的场景。 |
| `--persistentCacheRatio` | 以比率设置持久缓存限制，0.5 表示最大持久 L2 大小的一半（默认 = 0） | `--persistentCacheRatio=0.5` | 用于控制持久缓存的大小。 |
### `Build and Inference Batch Options`

| 说明 | 备注 |
|------|------|
| 使用隐式批处理时，如果未指定引擎的最大批处理大小，则将其设置为推理批处理大小。 | 适用于隐式批处理模式。 |
| 使用显式批处理时，如果仅指定了推理的形状，则这些形状也将用作构建配置文件中的最小/最优/最大形状。 | 适用于显式批处理模式。 |
| 如果仅指定了构建的形状，则最优形状也将用于推理。 | 适用于显式批处理模式。 |
| 如果同时指定了推理和构建的形状，则它们必须兼容。 | 适用于显式批处理模式。 |
| 如果启用了显式批处理但未指定任何形状，则模型必须为所有输入提供完整的静态维度，包括批处理大小。 | 适用于显式批处理模式。 |
| 使用 ONNX 模型时，自动强制启用显式批处理。 | ONNX 模型不支持隐式批处理。 |

### `Reporting Options`

| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--verbose` | 启用详细日志记录（默认 = `false`） | `--verbose` | 用于输出更详细的日志信息。 |
| `--avgRuns=N` | 报告连续 N 次迭代的平均性能测量结果（默认 = 10） | `--avgRuns=20` | 用于控制性能测量的平均次数。 |
| `--percentile=P1,P2,P3,...` | 报告 P1,P2,P3,... 百分比的性能（0≤P_i≤100，0 表示最大性能，100 表示最小性能；默认 = 90,95,99%） | `--percentile=50,90,99` | 用于分析性能的百分位数分布。 |
| `--dumpRefit` | 打印可重构引擎中的可重构层和权重 | `--dumpRefit` | 用于调试或分析可重构引擎的结构。 |
| `--dumpOutput` | 打印最后一次推理迭代的输出张量（默认 = 禁用） | `--dumpOutput` | 用于查看推理结果的输出。 |
| `--dumpRawBindingsToFile` | 将最后一次推理迭代的输入/输出张量打印到文件（默认 = 禁用） | `--dumpRawBindingsToFile` | 用于保存推理的输入/输出数据。 |
| `--dumpProfile` | 打印每层的分析信息（默认 = 禁用） | `--dumpProfile` | 用于查看每层的性能分析数据。 |
| `--dumpLayerInfo` | 将引擎的层信息打印到控制台（默认 = 禁用） | `--dumpLayerInfo` | 用于查看引擎的层结构信息。 |
| `--exportTimes=<file>` | 将计时结果写入 JSON 文件（默认 = 禁用） | `--exportTimes=times.json` | 用于保存性能计时数据。 |
| `--exportOutput=<file>` | 将输出张量写入 JSON 文件（默认 = 禁用） | `--exportOutput=output.json` | 用于保存推理的输出结果。 |
| `--exportProfile=<file>` | 将每层的分析信息写入 JSON 文件（默认 = 禁用） | `--exportProfile=profile.json` | 用于保存每层的性能分析数据。 |
| `--exportLayerInfo=<file>` | 将引擎的层信息写入 JSON 文件（默认 = 禁用） | `--exportLayerInfo=layer_info.json` | 用于保存引擎的层结构信息。 |

### `System Options`

| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--device=N` | 选择 CUDA 设备 N（默认 = 0） | `--device=1` | 用于指定运行推理的 GPU 设备。 |
| `--useDLACore=N` | 选择支持 DLA 的层的 DLA 核心 N（默认 = 无） | `--useDLACore=0` | 用于指定使用 DLA 核心的编号。 |
| `--staticPlugins` | 静态加载插件库（.so）（可多次指定） | `--staticPlugins=libplugin.so` | 用于在构建时静态加载插件库。 |
| `--dynamicPlugins` | 动态加载插件库（.so），如果包含在 `--setPluginsToSerialize` 中，则可能与引擎一起序列化（可多次指定） | `--dynamicPlugins=libplugin.so` | 用于在运行时动态加载插件库。 |
| `--setPluginsToSerialize` | 与引擎一起序列化的插件库（.so）（可多次指定） | `--setPluginsToSerialize=libplugin.so` | 用于指定需要与引擎一起序列化的插件库。 |
| `--ignoreParsedPluginLibs` | 默认情况下，当构建版本兼容引擎时，ONNX 解析器指定的插件库会隐式地与引擎一起序列化（除非指定了 `--excludeLeanRuntime`）并动态加载。启用此标志以忽略这些插件库。 | `--ignoreParsedPluginLibs` | 用于忽略 ONNX 解析器指定的插件库。 |

### `Help`
| 参数 | 说明 | 示例 | 备注 |
|------|------|------|------|
| `--help`, `-h` | 打印帮助信息 | `--help` 或 `-h` | 用于显示所有可用参数及其说明。 |

## 2、参考
### 说明：
- `trtexec` 是 TensorRT 提供的一个命令行工具，用于优化和推理深度学习模型。
- 参数的具体使用取决于模型类型、硬件配置和推理需求。
- 更多参数和详细说明可参考官方文档或运行 `trtexec --help` 查看帮助信息。

### 示例：
```bash
trtexec --onnx=model.onnx --batch=8 --fp16 --saveEngine=engine.trt --verbose
```
此命令将加载一个ONNX模型，设置批量大小为8，启用FP16精度，保存生成的TensorRT引擎文件，并启用详细日志输出。
