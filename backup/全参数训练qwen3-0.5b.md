``` python 
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab

# 设置 SwanLab 项目名称
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-dialog"

# 定义提示（PROMPT）和最大序列长度
PROMPT = "你是一个对话助手，你需要根据用户的问题，给出相应的回答。"
MAX_LENGTH = 2048

# 更新 SwanLab 配置
swanlab.config.update({
    "model": "Qwen/Qwen3-0.5B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

# 数据集格式转换函数
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            message = {
                "instruction": PROMPT,
                "input": data["question"],
                "output": data["answer"],
            }
            messages.append(message)
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 数据预处理函数
def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 推理函数
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 模型路径
model_dir = "/tmp/workspace/model/.cache/huggingface/download/naive"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点支持

# 数据集路径
train_dataset_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/train.jsonl"
test_dataset_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/val.jsonl"
train_jsonl_new_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/train_format.jsonl"
test_jsonl_new_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/val_format.jsonl"

# 转换数据集格式
if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 加载并处理训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 加载并处理验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# 设置训练参数
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-0.5B",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-0.5B",
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()

# 测试模型输出
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

# 记录测试结果并结束实验
swanlab.log({"Prediction": test_text_list})
swanlab.finish()
```

---

# 解释：

---

## 1. 基本环境与依赖导入

```python
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
```

* **json、pandas**：用于读取和处理 JSONL 格式的数据集。
* **torch**：PyTorch 底层库，用于张量运算和模型训练。
* **datasets.Dataset**：来自 Hugging Face 的 `datasets` 库，用于将 pandas DataFrame 封装成能被 Trainer 处理的数据集对象。
* **transformers** 相关：

  * `AutoTokenizer`：加载与模型对应的 tokenizer。
  * `AutoModelForCausalLM`：加载因果语言模型（Causal LM），用于对话/文本生成类任务。
  * `TrainingArguments`、`Trainer`：Hugging Face 官方的训练框架，用于管理训练超参、训练循环、保存模型等。
  * `DataCollatorForSeq2Seq`：对齐（padding）和构建 batch，适用于 Seq2Seq 或 CausalLM 之类的任务。
* **os**：主要用来配置环境变量和检查文件路径。
* **swanlab**：看起来是上传训练过程指标和日志到 SwanLab 平台的 SDK。

---

## 2. 设置 SwanLab 项目与全局配置

```python
# 设置 SwanLab 项目名称
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-dialog"
```

* 这一行将环境变量 `SWANLAB_PROJECT` 设为 `"qwen3-sft-dialog"`，表示后续所有通过 `swanlab.log()`、`swanlab.finish()` 上传的日志都归属于这个项目。

```python
# 定义提示（PROMPT）和最大序列长度
PROMPT = "你是一个对话助手，你需要根据用户的问题，给出相应的回答。"
MAX_LENGTH = 2048

# 更新 SwanLab 配置
swanlab.config.update({
    "model": "Qwen/Qwen3-0.5B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})
```

* `PROMPT` 里定义了一个“系统提示”（system prompt），即对话模型在训练和推理时的最初上下文。
* `MAX_LENGTH=2048`：指定输入+输出的最大 token 数目。
* `swanlab.config.update(...)`：将模型名、提示语和最大序列长度一并上传到 SwanLab，让后台记录这一配置。

---

## 3. 数据集格式转换

### 3.1 原始数据假设

* 原始训练集和验证集都是 JSONL 格式，每行都包含如下字段（示例）：

  ```json
  {
    "question": "用户的问题文本",
    "answer": "对应的回答文本"
  }
  ```
* 代码中给出了 `train_dataset_path` 和 `test_dataset_path`，例如：

  ```
  train_dataset_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/train.jsonl"
  test_dataset_path  = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/val.jsonl"
  ```

### 3.2 转换成带 “instruction/input/output” 的格式

```python
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            message = {
                "instruction": PROMPT,
                "input": data["question"],
                "output": data["answer"],
            }
            messages.append(message)
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
```

* 这段函数会把原始的 `question`、`answer` 字段提取出来，然后写成新的 JSONL，每行长这样：

  ```json
  {
    "instruction": "你是一个对话助手，你需要根据用户的问题，给出相应的回答。",
    "input": "<原来的 question 文本>",
    "output": "<原来的 answer 文本>"
  }
  ```

* 生成之后的文件路径是：

  * `train_format.jsonl`（训练集处理后）
  * `val_format.jsonl`（验证集处理后）

* 接下来的代码检查这两个新文件是否已经存在，如果不存在就调用上面的函数去生成：

  ```python
  train_jsonl_new_path = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/train_format.jsonl"
  test_jsonl_new_path  = "/tmp/workspace/RussianEnglishDialogue/Dataset/format/val_format.jsonl"

  if not os.path.exists(train_jsonl_new_path):
      dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
  if not os.path.exists(test_jsonl_new_path):
      dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)
  ```

---

## 4. 数据预处理函数（Tokenize & 构造 labels）

在对话或 SFT（Supervised Fine-Tuning）场景下，需要手动拼接“提示”“用户输入”“模型输出”三部分，并生成 `input_ids, attention_mask, labels`。labels 的构造方式是让模型只惩罚（loss）属于“回答”部分，而不惩罚“提示+用户输入”那段。

```python
def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 把 instruction 和 response 的 ids 拼接起来，末尾多一个 pad_token_id，用于强制生成结束
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # labels：前面 instruction 的部分都标成 -100（表示这个位置的 token 不计算 loss），
    # 后面才是真正要让模型去预测的回复 token，最后一位 pad_token_id 也参与计算（可以算作一个结束标记）。
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 如果长度超过了 MAX_LENGTH，就进行截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

* `<|im_start|>system`、`<|im_end|>` 等是 QQE（Qwen Prompt）里约定的特殊分隔符，用来标记对话角色。
* 整个 `input_ids` 里先包含 system+user，然后紧跟 response。
* 训练时，模型只有在 “response” 部分才会计算交叉熵损失，前面的 instruction/user 填成 `-100`，这样就不会对它们算 loss。
* 最后强制在序列末尾加一个 `pad_token_id`，用作生成结束的标志。

---

## 5. 推理（Inference）函数

```python
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    # 这里去掉输入部分，只保留模型“新生成的” token
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
```

* `messages` 的格式如：

  ```python
  [
    {"role": "system",    "content": PROMPT},
    {"role": "user",      "content": "<用户输入>"},
    # 这里函数内部会自动在末尾加上一个 <|im_start|>assistant> 的生成提示
  ]
  ```
* 先用 `apply_chat_template` 拼成一个完整的对话字符串（带角色分隔）交给 tokenizer 编码。
* 调用 `.generate(...)` 开始生成，`max_new_tokens=MAX_LENGTH` 表示“最多再生成这么多 token”。
* 生成后把完整的 `[input_ids + generated_ids]` 切割，只保留“模型后来新生成的那段”去解码。

---

## 6. 预训练模型加载：`model_dir`

```python
model_dir = "/tmp/workspace/model/.cache/huggingface/download/naive"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点支持
```

* `model_dir` 指向一个本地目录，里面应该已经缓存好了预训练的基础模型权重（这里是 “Qwen/Qwen3-0.5B”）。
* 先用 `AutoTokenizer.from_pretrained(model_dir)` 把对应的 tokenizer 加载进来。
* 再用 `AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)` 把模型加载到显卡上，并以 bfloat16 的方式存储参数，以便减少显存占用。
* `model.enable_input_require_grads()` 用来启用梯度检查点（gradient checkpointing），在训练大模型时可以节省显存，不过会稍微牺牲一部分计算效率。

---

## 7. 构建 Dataset 对象

```python
# --- 训练集 ---
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# --- 验证集 ---
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)
```

* `pd.read_json(..., lines=True)`：把新格式的 JSONL 文件读成一个 pandas DataFrame，DataFrame 列名是 `["instruction","input","output"]`。
* `Dataset.from_pandas(...)`：Hugging Face 的 `Dataset` 定义，用它可以把 pandas DataFrame 转换成一个能被 `Trainer` 直接消费的 dataset 对象。
* 然后 `.map(process_func, remove_columns=...)`：对每个样本都调用前面定义的 `process_func`，生成 `input_ids, attention_mask, labels` 三个字段，并删除原先的 `instruction,input,output` 列。
* 最终，`train_dataset` 和 `eval_dataset` 都是已经做过 tokenizer 和 label 构造的形式，且字段名固定为 `input_ids`、`attention_mask`、`labels`。

---

## 8. 设置训练参数（`TrainingArguments`）

```python
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-0.5B",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-0.5B",
)
```

重点参数说明：

* `output_dir="/root/autodl-tmp/output/Qwen3-0.5B"`

  * **这是训练好的模型、检查点（checkpoint）和配置等最终保存的路径。**
  * 训练过程中每隔 `save_steps=400` 会在这个目录下生成一次检查点，命名为 `checkpoint-400/`、`checkpoint-800/` 等等，并且训练结束后 Trainer 会把最终模型权重（和 tokenizer 配置）写到 `output_dir` 下。
* `per_device_train_batch_size=1, per_device_eval_batch_size=1`

  * 每张 GPU 上的 batch size 都是 1。
* `gradient_accumulation_steps=4`

  * 由于 batch size 太小，通过梯度累积把“等价 batch size”放大 4 倍，相当于每 4 步算一次梯度并更新一次模型。
* `eval_strategy="steps", eval_steps=100`

  * 每训练 100 步，就跑一次 eval。
* `logging_steps=10`

  * 每 10 步输出一次日志（loss、learning rate 等）。
* `num_train_epochs=2`

  * 总共训练数据迭代 2 个 epoch。
* `learning_rate=1e-4`

  * 学习率。
* `save_on_each_node=True`

  * 如果使用分布式训练，每个节点都会保存模型副本。
* `gradient_checkpointing=True` 与 `model.enable_input_require_grads()` 配合表示启用梯度检查点，节省显存。
* `report_to="swanlab", run_name="qwen3-0.5B"`

  * 将日志上报到 SwanLab，并且给这次实验起个名字。

---

## 9. 初始化 Trainer 并开始训练

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()
```

* `Trainer` 会根据上面传入的 `model`、`args`、`train_dataset`、`eval_dataset` 拆分训练循环、eval 循环，并且自动调用 `save_steps` 时保存检查点。
* `data_collator=DataCollatorForSeq2Seq(...)` 负责在一个 batch 中把不同长度的 `input_ids` 填充到相同长度、生成对应的 `attention_mask` 和 `labels`，以便 Hugging Face 可以直接把它送进模型做前向 / 反向传播。

**训练过程**：

1. 把 `train_dataset` 按照 `per_device_train_batch_size=1` 拆成一条条输入，4 步累积一次梯度。
2. 每 100 步跑一次 `eval_dataset` 测试，并把结果打印出来。
3. 每 400 步把当前 model state（包括模型权重、optimizer 状态、lr scheduler 状态等）保存到 `output_dir/checkpoint-400/`，以此类推。
4. 2 个 epoch 结束后，`trainer.train()` 会把最终的模型（等同于 `model.save_pretrained(output_dir)`）自动写到 `output_dir`，覆盖之前的权重文件。

---

## 10. 测试模型输出并将结果通过 SwanLab 上报

训练完成后，我们用同样的验证集前 3 条数据做一次简单的推理，看看模型的回答与真实答案有何差距，并把推理结果也上传到 SwanLab。

```python
# 读取验证集前三条
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    messages = [
        {"role": "system",    "content": f"{instruction}"},
        {"role": "user",      "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)
```

* `predict(...)` 函数会把 `instruction` + `user input` 拼进去，调用 `model.generate` 生成回答。
* 把每条测试结果包装成 `swanlab.Text(...)`（SwanLab 定义的文本对象），放到列表里。
* 训练结束后，调用：

  ```python
  swanlab.log({"Prediction": test_text_list})
  swanlab.finish()
  ```

  * 这意味着把测试结果一起上传到 SwanLab 平台，最后调用 `finish()` 报告实验结束。

---

## 11. “训练完的模型放在哪里了？”

* **关键在 `TrainingArguments` 里的 `output_dir`**，代码中指定为：

  ```python
  output_dir="/root/autodl-tmp/output/Qwen3-0.5B"
  ```

* 在训练过程中，`Trainer` 会在该目录下自动保存多个 checkpoint，例如：

  ```
  /root/autodl-tmp/output/Qwen3-0.5B/checkpoint-400/
  /root/autodl-tmp/output/Qwen3-0.5B/checkpoint-800/
  ...
  ```

* **训练结束后**，`Trainer.train()` 默认还会把当前的最优（或者最后一步）模型 `save_pretrained` 到 `output_dir`，并且将 tokenizer 相关配置也存入同一目录。最终目录结构可能类似如下：

  ```
  /root/autodl-tmp/output/Qwen3-0.5B/
  ├── config.json                 # 模型配置文件
  ├── pytorch_model.bin           # (或) pytorch_model.safetensors，模型最终权重
  ├── tokenizer_config.json       # tokenizer 配置
  ├── special_tokens_map.json     # 如果用了特殊 token
  ├── vocab.json / merges.txt     # 根据对应 tokenizer 类型
  ├── training_args.bin           # 保存了训练参数的二进制文件
  ├── checkpoint-400/             # 若中途保存了 checkpoint，会看到这样若干子文件夹
  │   ├── config.json
  │   ├── ...（各 checkpoint 权重、optimizer 状态）
  ├── checkpoint-800/
  │   └── ...
  └── last_checkpoint/  (或类似的)
      └── ...
  ```

* **也就是说**，你只要进到 `/root/autodl-tmp/output/Qwen3-0.5B` 这个文件夹，就能找到：

  1. 每次 `save_steps` 保存的中间 checkpoint
  2. 训练结束后最终写入的模型权重和 tokenizer 等文件

---

## 12. 小结

1. **代码流程**：

   * 设置环境（SwanLab、prompt、最大长度）
   * 把原始的 QA 数据转换成带 `instruction` / `input` / `output` 格式的 JSONL
   * 定义 `process_func`，拼接 system/user/assistant 三段，生成 `input_ids, attention_mask, labels`
   * 加载预训练模型和 tokenizer
   * 用 `Dataset.from_pandas` + `.map(process_func)` 得到能用于 Hugging Face Trainer 的格式化数据集
   * 构造 `TrainingArguments`，指定 `output_dir`、训练超参、将日志上报给 SwanLab
   * 初始化 `Trainer` 并执行 `.train()`，训练完成后会把模型保存到 `output_dir`
   * 最后用前三条验证集数据跑一次 `.generate()`，把预测结果上传到 SwanLab

2. **训练结束后模型存放的位置**：

   * 最终模型（以及所有中间 checkpoint）都存放在 `TrainingArguments` 中 `output_dir` 指定的目录下。
   * 本例中是：

     ```
     /root/autodl-tmp/output/Qwen3-0.5B
     ```
   * 进入该文件夹后，你会看到 `config.json, pytorch_model.bin, tokenizer_config.json, …` 等一系列文件，以及若干 `checkpoint-XXX` 子文件夹。

只要在训练完毕后，通过文件系统浏览或脚本 `ls /root/autodl-tmp/output/Qwen3-0.5B`，就能确认模型确实保存在哪个子目录下。
