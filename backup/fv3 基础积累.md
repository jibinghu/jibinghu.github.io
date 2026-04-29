## 一、 `test_case`：

`test_case` 是要运行的理想化测试用例编号，需要 `warm_srart = .false` 即需要冷启动理想化初始化。

1. 浅水方程（Shallow-Water, 2D 动力核基准测试）

> 主要用于：水平动量离散、科氏项、地形项、守恒性、波动色散，对应 test_case = -1 … 9

序号 | 中文名称 | 主要测试目的
-- | -- | --
-1 | 散度守恒测试 | 验证离散算子对散度/质量守恒的保持性
0 | 理想化非线性形变流 | 测试形变流下的数值扩散与输运稳定性
1 | 余弦钟形平流 | 经典标量平流精度与形状保持测试
2 | 纬向地转平衡流 | 检验地转平衡保持（well-balanced）能力
3 | 非旋转势流 | 隔离科氏项后的纯势流一致性
4 | Rankine 涡合并（热带气旋） | 强涡度场、涡旋合并与数值耗散
5 | 孤立山地上的地转平衡流 | 地形–压力梯度–动量耦合一致性
6 | Rossby 波（波数 4） | 行星尺度 Rossby 波传播与色散
7 | 正压不稳定 | 不稳定增长与涡旋生成
8 | 赤道双涡“孤子”传播 | 相干结构传播与耗散
9 | 极涡测试 | 极区强涡度与球面几何处理

2. 三维静力/准静力理想化大气测试（3D Hydrostatic / Large-Scale）

> 主要用于：大尺度动力核、平衡初始化、斜压不稳定、地形波，对应 test_case = 10–19, 12/13/-13, 14

序号 | 中文名称 | 主要测试目的
-- | -- | --
10 | 含理想化山地的三维静力平衡 | 检验 3D 静力平衡与地形处理
11 | 含 USGS 地形的气候冷启动 | 气候模式冷启动流程
12 | Jablonowski–Williamson 斜压稳态 | 斜压平衡保持
13 | J–W 斜压扰动实验 | 斜压不稳定发展（黄金基准）
-13 | DCMIP2016 斜压波 | 社区对标用 J–W 测试
14 | 水行星（Aqua-planet）冷启动 | 无地形理想化气候实验
15 | 小地球密度流 | 密度流/冷池动力学
16 | 非旋转三维重力波 | 垂直传播与静力波动
17 | 旋转惯性–重力波 | 科氏力与重力波耦合
18 | 地形激发 Rossby 波 | 地形–大尺度行星波响应
19 | 无旋转密度流 | 密度流的对照实验

3. 三维非静力与地形/对流专项测试（Cloud-Resolving / Non-Hydrostatic）

> 主要用于：非静力解算、深对流、山波、风暴尺度

序号 | 中文名称 | 主要测试目的
-- | -- | --
20 | 非静力背风涡（无旋转） | 山地下游涡旋生成
21 | 非静力背风涡（有旋转） | 旋转–地形–非静力耦合
30 | 超级单体风暴（曲线风廓线，无旋转） | 深对流动力学
31 | 超级单体风暴（曲线风廓线，有旋转） | 旋转风暴结构
32 | 超级单体风暴（直线风廓线） | 对流对称性
33 | HIWPP Schär 山波（脊状山） | 经典山波基准
34 | HIWPP Schär 山波（圆形山） | 地形几何效应
35 | HIWPP Schär 山波（剪切风） | 剪切–山波耦合
36 | HIWPP 超级单体（无热扰动） | 对流自发性
37 | HIWPP 超级单体（给定热扰动） | 对流触发机制

4. 输运 / 守恒 / 理想化专项测试

> 不直接代表“真实天气”，但对数值正确性极其关键

序号 | 中文名称 | 主要测试目的
-- | -- | --
44 | 球面锁交换 | 密度前沿、守恒性
45 | 新测试（占位） | 依版本而异
51 | 三维 tracer 形变平流 | 输运核与 halo 处理
52 | 静止大气 + 地形 | 地形诱导数值噪声
55 | 热带气旋（TC） | TC 初始化与演化
-55 | DCMIP2016 TC | 社区对标 TC
101 | 三维非静力 LES（hybrid_z） | 大涡模拟（可选）

5. 周期边界理想化盒模型测试（Doubly-Periodic / Box Model）

> 一类不应与前面的 test_case 混为一谈，它们使用的是 周期边界、盒模型物理假设

序号 | 中文名称 | 主要用途
-- | -- | --
2 | 1.5 km 山地下的静止流 | 周期地形响应
14 | 周期水行星 + 暖泡 | 对流触发
15 | 等温暖泡 | 对流上升
16 | 等温冷泡 | 冷池传播
17 | 对称超级单体 | 理想化风暴
18 | 非对称超级单体 | 复杂风廓线
19 | 改进型超级单体 | 多 sounding/bubble
102 | 稳定边界层 LES | SBL 基准
103 | DYCOMS-II 层积云 | 云层 LES

## 二、`input.nml` 输入设置

``` shell
 &main_nml
     days   = 1,
     dt_atmos = 1800 /

 &fms_io_nml
     threading_write = 'multi'
     fileset_write = 'multi' /

 &fms_nml
     clock_grain = "LOOP",
     domains_stack_size = 900000 ! 内部栈设置大小
     print_memory_usage = .true.  / ! 打印内存使用信息

 &fv_core_nml
     layout   = 2,2
     npx      = 97,
     npy      = 97,
     npz      = 26,
     uniform_vert_spacing = .false., ! 均匀垂直间距
     ntiles   = 6,
     do_Held_Suarez = .true. ! 启用 Held–Suarez 理想化物理强迫
     adiabatic = .false. ! 允许加入 Held–Suarez 的加热/摩擦等“非绝热倾向”
     print_freq = 0,
     grid_type = 0
     warm_start = .false. ! 冷启动，从 test_case 生成，而不是从 restart 文件续
     io_layout = 1,1 /

 &fv_grid_nml
     grid_name = 'Conformal' / ! 选择 conformal cubed-sphere（保角映射）网格几何。它相对传统 gnomonic 网格，在某些指标（角度/面积畸变分布、拼接处行为等）上更有优势或更平滑，常用于改善网格几何误差在拼接区域的表现。

 &test_case_nml
     test_case = 14 ! Aqua-planet 冷启动
     alpha = 0.00 /

```

<img width="694" height="419" alt="Image" src="https://github.com/user-attachments/assets/3e4b2fa5-2f58-4fa7-b22b-642c1cdf2494" />

<img width="570" height="485" alt="Image" src="https://github.com/user-attachments/assets/89315d97-7633-4b25-84ac-ab99caef5edc" />





