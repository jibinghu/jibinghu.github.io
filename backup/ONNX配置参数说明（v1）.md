ONNX配置参数说明（v1）

torch Version 2.1.0 的 export参数：
``` python
def export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    export_params: bool = True,
    verbose: bool = False,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
    autograd_inlining: Optional[bool] = True,
)
```

model: pytorch模型
args: 第一个参数model的输入数据，因为模型的输入可能不止一个，因此采用元组作为参数
export_params: 导出的onnx模型文件可以包含网络结构与权重参数，如果设置该参数为False，则导出的onnx模型文件只包含网络结构，因此，一般保持默认为True即可
verbose: 该参数如果指定为True，则在导出onnx的过程中会打印详细的导出过程信息

opset_version: ONNX的算子集版本，默认为11。
f: 导出的onnx模型文件路径
input_names: 为输入节点指定名称，因为输入节点可能多个，因此该参数是一个列表
output_names: 为输出节点指定名称，因为输出节点可能多个，因此该参数是一个列表
dynamic_axes: 指定输入输出的张量，哪些维度是动态的，通过用字典的形式进行指定，如果某个张量的某个维度被指定为字符串或者-1，则认为该张量的该维度是动态的，但是一般建议只对batch维度指定动态，这样可提高性能，具体的格式见下面的代码

keep_initializers_as_inputs: 如果为 True，则所有初始化器（通常对应为参数）也将作为输入导出，添加到计算图中。 如果为 False，则初始化器不会作为输入导出，不添加到计算图中，仅将非参数输入添加到计算图中。
