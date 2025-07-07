# MiniCPM4 部署教程

## MiniCPM4 模型介绍

MiniCPM 4 是一个极致高效的端侧大模型，从模型架构、学习算法、训练数据与推理系统四个层面进行了高效优化，实现了极致的效率提升。

- 🏗️ 高效模型架构：
  - InfLLM v2 -- 可训练的稀疏注意力机制：采用可训练的稀疏注意力机制架构，在 128K 长文本处理中，每个词元仅需与不足 5% 的词元进行相关性计算，显著降低长文本的计算开销
- 🧠 高效学习算法：
  - 模型风洞 2.0 -- 高效 Predictable Scaling：引入下游任务的 Scaling 预测方法，实现更精准的模型训练配置搜索
  - BitCPM -- 极致的三值量化：将模型参数位宽压缩至 3 值，实现模型位宽 90% 的极致瘦身
  - 高效训练工程优化：采用 FP8 低精度计算技术，结合多词元预测（Multi-token Prediction）训练策略
- 📚 高知识密度训练数据：
  - UltraClean -- 高质量预训练数据的清洗与合成：构建基于高效验证的迭代式数据清洗策略，开源高质量中英文预训练数据集 [UltraFineweb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)
  - UltraChat v2 -- 高质量有监督微调数据合成：构建大规模高质量有监督微调数据集，涵盖知识密集型数据、推理密集型数据、指令遵循数据、长文本理解数据、工具调用数据等多个维度
- ⚡ 高效推理系统：
  - CPM.cu -- 轻量级的高效CUDA推理框架：融合了稀疏注意力机制、模型量化与投机采样，充分体现MiniCPM4的效率优势
  - ArkInfer -- 跨平台部署系统：支持多后端环境的一键部署，提供灵活的跨平台适配能力



## 模型下载

MiniCPM4提供了两种参数量的模型：MiniCPM4-8B与MiniCPM4-0.5B以及其量化版本/工具版本/加速模型，本教程主要使用基础模型，详情可见下表：

| 模型类型     | 模型名称                           | 最低资源要求 | 推理支持                              | 下载地址                                                     |
| ------------ | ---------------------------------- | ------------ | ------------------------------------- | ------------------------------------------------------------ |
| 基础模型     | MiniCPM4-8B                        | 18G          | transformers / vLLM / SGLang / CPM.cu | [HF](https://huggingface.co/openbmb/MiniCPM4-8B) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B) |
|              | MiniCPM4-0.5B                      | 2G           | transformers / vLLM / SGLang          | [HF](https://huggingface.co/openbmb/MiniCPM4-0.5B) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-0.5B) |
| 加速模型     | MiniCPM4-8B-Eagle-FRSpec           | -            | CPM.cu / SGLang                       | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec) / [MS](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec) |
|              | MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu | -            | CPM.cu                                | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu) |
|              | MiniCPM4-8B-Eagle-vLLM             | -            | vLLM                                  | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) / [MS](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-vLLM) |
|              | MiniCPM4-8B-marlin-Eagle-vLLM      | -            | vLLM                                  | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) /[ MS](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-vLLM) |
| 量化版本     | MiniCPM4-8B-marlin-cpmcu           | 8G           | CPM.cu                                | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-cpmcu) / [MS](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B-marlin-cpmcu) |
|              | MiniCPM4-8B-marlin-vLLM            | 8G           | vLLM                                  | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-vLLM) / [MS](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B-marlin-vLLM) |
|              | MiniCPM4-8B-mlx                    | 6G           | MLX                                   | [HF](https://huggingface.co/openbmb/MiniCPM4-8B-mlx) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-mlx) |
| 三元量化版本 | BitCPM4-0.5B                       | 1G           | transformers                          | [HF](https://huggingface.co/openbmb/BitCPM4-0.5B) / [MS](https://www.modelscope.cn/models/OpenBMB/BitCPM4-0.5B) |
|              | BitCPM4-1B                         | 2G           | transformers                          | [HF](https://huggingface.co/openbmb/BitCPM4-1B) / [MS](https://www.modelscope.cn/models/OpenBMB/BitCPM4-1B) |
| 工具版本     | MiniCPM4-Survey                    | 18G          | transformers / vLLM / SGLang / CPM.cu | [HF](https://huggingface.co/openbmb/MiniCPM4-Survey) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-Survey) |
|              | MiniCPM4-MCP                       | 18G          | transformers / vLLM / SGLang / CPM.cu | [HF](https://huggingface.co/openbmb/MiniCPM4-MCP) / [MS](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-MCP) |





## Transformers推理

### 环境准备

在autodl平台中租一个**单卡4090等24G**显存的显卡机器，如下图所示镜像选择PyTorch-->2.5.1-->3.12(ubuntu22.04)-->12.4 接下来打开刚刚租用服务器的JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行演示。

![01-1](E:\CodePrograms\self-llm\models\MiniCPM4\images\01-1.png)

使用`modelscope`库进行模型下载

```bash
pip install modelscope
modelscope download openbmb/MiniCPM4-8B --local-dir ./MiniCPM4-8B
```

### 代码准备

在你的本地路径下新建infer.py文件并在其中输入以下内容：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer  # 从transformers库导入所需的类
import torch  # 导入torch库，用于深度学习相关操作

torch.manual_seed(0)  # 设置随机种子以确保结果的可复现性

path = 'openbmb/MiniCPM4-8B' # 定义模型路径,可修改为您本地模型地址

device = "cuda" # 使用CUDA

tokenizer = AutoTokenizer.from_pretrained(path) # 从模型路径加载分词器

# 从模型路径加载模型，设置为使用bfloat16精度以优化性能，并将模型部署到支持CUDA的GPU上,trust_remote_code=True允许加载远程代码
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# 使用模型进行聊天，提出问题并设置生成参数
messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]

# 使用 tokenizer 的 apply_chat_template 方法将 messages 转换为模型可接受的输入格式
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

# 使用模型进行文本生成，并设置参数，如temperature、top_p值
model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    top_p=0.7,
    temperature=0.7
)

# 提取出生成的新 token ID，排除掉原始输入部分
output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

# 使用 tokenizer 将生成的 token ID 解码为自然语言文本
# skip_special_tokens=True：跳过特殊标记（如 <s>, </s>, <|assistant|> 等）
responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]

# 输出模型生成的回复内容
print(responses)
```

### 部署

在终端输入以下命令运行infer.py，即实现MiniCPM4的Transformers部署调用

```bash
python infer.py
```

