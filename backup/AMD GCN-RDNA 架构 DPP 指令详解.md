# AMD GCN/RDNA 架构 DPP 指令详解
在 AMD GCN / RDNA 架构中，DPP（Data Parallel Primitives）是一类非常底层、极其关键的 lane-to-lane 数据重排指令，本质上等价于 NVIDIA 的 warp shuffle，但在功能粒度与可编程性上更细致，是在 MI250X / CDNA2 上做 stencil、前缀扫描、warp-level reduction 的性能核心。

下面按**指令类型、控制字段、典型模式、对比 CUDA shuffle** 四个层面系统整理。

## 一、DPP 指令本质
DPP 并不是独立 opcode，而是 **VOP1 / VOP2 / VOP3 + DPP 控制字段** 的组合形态，核心是实现 wavefront 内任意 lane ↔ lane 数据重排 + ALU 融合，指令格式如下：
```asm
v_add_f32_dpp v0, v1, v2, dpp_ctrl:..., row_mask:..., bank_mask:..., bound_ctrl:...
```

## 二、核心 DPP 指令家族（最常用）
DPP 核心优势：不单是 shuffle 数据搬运，而是**shuffle + ALU 融合执行**，一条指令完成“搬运 + 运算”，是 RDNA/CDNA 架构关键性能亮点。

| 类别       | 示例指令                                                                 |
|------------|--------------------------------------------------------------------------|
| 整数算术   | v_add_i32_dpp, v_sub_i32_dpp, v_and_b32_dpp, v_or_b32_dpp, v_xor_b32_dpp |
| 浮点算术   | v_add_f32_dpp, v_mul_f32_dpp, v_fma_f32_dpp, v_mac_f32_dpp               |
| 最值       | v_max_f32_dpp, v_min_f32_dpp, v_max_i32_dpp, v_min_i32_dpp               |
| 逻辑/选择  | v_cndmask_b32_dpp, v_mov_b32_dpp                                         |

## 三、dpp_ctrl 控制模式全集
dpp_ctrl 是 DPP 的灵魂，直接决定数据来源的 lane 映射关系，以下是核心控制模式分类：

### 1.  基础偏移（shift）
核心实现行内数据偏移，对应 CUDA 的 shfl_up/shfl_down 语义
| 控制码                | 功能                  |
|-----------------------|-----------------------|
| DPP_ROW_SL1 ~ DPP_ROW_SL15 | 行内左移 1~15         |
| DPP_ROW_SR1 ~ DPP_ROW_SR15 | 行内右移 1~15         |
| DPP_ROW_RL1 ~ DPP_ROW_RL15 | 行内循环左移 1~15     |
| DPP_ROW_RR1 ~ DPP_ROW_RR15 | 行内循环右移 1~15     |

对应 CUDA 指令：
```cuda
__shfl_up
__shfl_down
```

### 2.  跨行操作（row broadcast）
实现行内指定 lane 数据广播
| 控制码                | 含义                  |
|-----------------------|-----------------------|
| DPP_ROW_BCAST0        | 每行第 0 lane 广播    |
| DPP_ROW_BCAST15       | 每行第 15 lane 广播   |

### 3.  绝对 lane 读取
固定读取本行指定 lane 数据，灵活性强
| 控制码        | 功能                  |
|---------------|-----------------------|
| DPP_ROW_SHARE(x) | 所有 lane 读取本行 lane x |

### 4.  四象限交换（适配 stencil halo 场景）
专为 2D 数据邻域交互设计，适配 stencil 算子 halo 数据传递
| 控制码                      | 含义                  |
|-----------------------------|-----------------------|
| DPP_QUAD_PERM(0,1,2,3)      | quad 内默认重排       |
| DPP_QUAD_PERM(1,0,3,2)      | quad 横向交换         |
| DPP_QUAD_PERM(2,3,0,1)      | quad 纵向交换         |

### 5.  完整查表置换（最强灵活度）
AMD DPP 独有优势，NVIDIA warp shuffle 不支持
| 控制码       | 功能说明                  |
|--------------|---------------------------|
| DPP_LUT1 ~ DPP_LUT4 | 支持 wave 内任意 32/64 lane 自定义映射 |

## 四、mask 与边界控制
DPP 指令的辅助控制字段，精准管控数据交互范围与越界行为
| 字段        | 核心功能                  |
|-------------|---------------------------|
| row_mask    | 控制哪些 lane 参与数据交互 |
| bank_mask   | 控制 LDS bank 访问权限    |
| bound_ctrl  | 越界数据是否置 0（1=置0，0=不处理） |

完整指令示例：
```asm
v_add_f32_dpp v0, v1, v2,
dpp_ctrl:DPP_ROW_SL1,
row_mask:0xF,
bank_mask:0xF,
bound_ctrl:1
```

## 五、典型 stencil / reduction 用法（性能核心场景）
### 行内前缀和（无 LDS/barrier，极致低延迟）
核心优势：无需 LDS 搬运，无需 barrier 同步，延迟 < 4 cycle
```asm
v_add_f32_dpp v0, v0, v0, dpp_ctrl:DPP_ROW_SL1
v_add_f32_dpp v0, v0, v0, dpp_ctrl:DPP_ROW_SL2
v_add_f32_dpp v0, v0, v0, dpp_ctrl:DPP_ROW_SL4
v_add_f32_dpp v0, v0, v0, dpp_ctrl:DPP_ROW_SL8
```

## 六、DPP vs CUDA Shuffle 本质差异
| 特性         | AMD DPP                  | NVIDIA warp shuffle      |
|--------------|--------------------------|--------------------------|
| 指令融合     | shuffle + ALU 一条完成   | shuffle + ALU 两条指令   |
| 映射能力     | 支持任意 LUT 级置换      | 仅支持 index/up/down 映射|
| 跨 quad 支持 | 原生支持                 | 不支持                   |
| 延迟         | 1 cycle 级               | 3–5 cycles               |

## 七、MI250X 实战最佳实践（FV3 stencil 场景）
传统 stencil 用 LDS 传递 halo 数据效率极低，优先用 DPP 替代，核心映射关系：
```
# x 方向 halo 数据传递
x 方向 halo：用 DPP_ROW_SL1/SR1

# y 方向 halo 数据传递
y 方向 halo：用 DPP_QUAD_PERM

# z 方向规约计算
z 方向规约：用 DPP_LUTx
```
核心收益：可实现 5-point stencil 算子完全消除 LDS 依赖，大幅提升吞吐