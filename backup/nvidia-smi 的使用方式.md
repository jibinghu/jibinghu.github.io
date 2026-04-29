`nvidia-smi` 是 NVIDIA 提供的一个命令行工具，用于监控和管理 NVIDIA GPU 设备。它提供了丰富的参数选项，可以帮助用户获取 GPU 的详细信息、监控性能、管理任务等。以下是一些常用的额外参数选项及其功能：

---

### 1. **基本查询选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi` | 显示 GPU 的基本信息（默认行为）。 |
| `nvidia-smi -q` | 显示 GPU 的详细信息（包括温度、功耗、显存使用等）。 |
| `nvidia-smi -q -i <GPU_ID>` | 显示指定 GPU 的详细信息。 |
| `nvidia-smi -L` | 列出系统中所有 GPU 的简要信息。 |

---

### 2. **监控选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi dmon` | 监控 GPU 的性能指标（如功耗、温度、显存使用等）。 |
| `nvidia-smi dmon -s <metric>` | 监控指定指标（如 `p` 表示功耗，`u` 表示利用率）。 |
| `nvidia-smi pmon` | 监控 GPU 上的进程信息（如进程 ID、显存使用等）。 |
| `nvidia-smi topo -m` | 显示 GPU 的拓扑结构（如 GPU 之间的连接方式）。 |

---

### 3. **任务管理选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi -i <GPU_ID> -r` | 重置指定 GPU。 |
| `nvidia-smi -i <GPU_ID> -pm <0/1>` | 启用或禁用持久模式（Persistence Mode）。 |
| `nvidia-smi -i <GPU_ID> -e <0/1>` | 启用或禁用 ECC（Error Correction Code）。 |
| `nvidia-smi -i <GPU_ID> -c <compute_mode>` | 设置 GPU 的计算模式（如 `DEFAULT`、`EXCLUSIVE_PROCESS`）。 |

---

### 4. **显存管理选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi -i <GPU_ID> -f <filename>` | 将 GPU 信息输出到指定文件。 |
| `nvidia-smi --query-gpu=memory.total,memory.used --format=csv` | 查询显存使用情况（以 CSV 格式输出）。 |
| `nvidia-smi --id=<GPU_ID> --gpu-reset` | 重置指定 GPU 的显存。 |

---

### 5. **性能监控选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi --loop=<seconds>` | 每隔指定秒数刷新 GPU 信息。 |
| `nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv` | 查询 GPU 的详细性能指标（以 CSV 格式输出）。 |
| `nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used --format=csv` | 查询 GPU 的简要性能指标（以 CSV 格式输出）。 |

---

### 6. **日志和事件选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi -q -d PERFORMANCE` | 查询 GPU 的性能事件。 |
| `nvidia-smi -q -d ECC` | 查询 GPU 的 ECC 错误信息。 |
| `nvidia-smi -q -d TEMPERATURE` | 查询 GPU 的温度信息。 |
| `nvidia-smi -q -d POWER` | 查询 GPU 的功耗信息。 |

---

### 7. **其他实用选项**

| 参数 | 功能 |
|------|------|
| `nvidia-smi --help` | 显示帮助信息。 |
| `nvidia-smi --version` | 显示 `nvidia-smi` 的版本信息。 |
| `nvidia-smi --list-gpus` | 列出系统中所有 GPU 的简要信息。 |
| `nvidia-smi --display=<DISPLAY_MODE>` | 设置显示模式（如 `MEMORY`、`UTILIZATION`）。 |

---

### 8. **示例用法**

#### **(1) 监控 GPU 性能**
```bash
nvidia-smi dmon -s puc
```
- 监控 GPU 的功耗（`p`）、利用率（`u`）和温度（`c`）。

#### **(2) 查询显存使用情况**
```bash
nvidia-smi --query-gpu=memory.total,memory.used --format=csv
```
- 以 CSV 格式输出显存使用情况。

#### **(3) 重置 GPU**
```bash
nvidia-smi -i 0 -r
```
- 重置 GPU 0。

#### **(4) 监控 GPU 进程**
```bash
nvidia-smi pmon
```
- 监控 GPU 上的进程信息。

#### **(5) 查询 GPU 详细信息**
```bash
nvidia-smi -q -i 0
```
- 查询 GPU 0 的详细信息。

---

### 9. **总结**

`nvidia-smi` 提供了丰富的参数选项，可以满足 GPU 监控、管理和调试的多种需求。通过合理使用这些选项，用户可以更好地了解 GPU 的状态、优化性能并解决问题。如果需要更详细的信息，可以随时使用 `nvidia-smi --help` 查看帮助文档。