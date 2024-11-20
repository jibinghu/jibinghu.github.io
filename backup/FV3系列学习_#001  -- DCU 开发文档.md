
---

FV3 源码：

编译流程：

0. module 检测可用包

``` bash
module avail
module list
#  需要 intel 编译器
module unload mpi/hpcx/2.12.0/gcc-8.3.1
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.3.0
```
> 在 Makefile 里已经行首部署了 export ：

``` bash
CC = icc
MPICC = mpiicc
```

2. 编译时环境变量设置：

``` bash
export I_MPI_F90=ifort
export I_MPI_CC=icc
export I_MPI_CXX=icc
```

3. 根据`ics_231207_checked/exp/exec.amd64` 目录下的 Makefile 文件进行项目构建
    1. 注意替换目录和相关依赖，可以在vim中使用`:%s/\/public\/home\/lihuiyuan\/fv3\//\/home\/myuser\/myproject\//g`批量替换
    2. 注意对intel编译器进行module load

4. 运行时命令：

```
export LD_LIBRARY_PATH=/public/software/mathlib/netcdf/intel/4.7.4/lib/:$LD_LIBRARY_PATH
./fms.x # 需要input.nml配置文件
```

---

## BUG && FAQ：

1. 编译阶段 DEBUG
> 在 Fortran 文件 mpp_io.F90 的第 335 行，编译器尝试寻找一个名为 **netcdf.inc** 的包含文件，但未能找到。

- 解决方法：
   - `-I/public/software/mathlib/netcdf/intel/4.7.4/include`
- 添加：` NETCDF_INC = -I/public/software/mathlib/netcdf/intel/4.7.4/include`
- 并对Makefile 中的 $(FC) (Fortran 编译器) 添加 NETCDF_INC：
   - `FFLAGS += $(NETCDF_INC)`

完整添加：

```
NETCDF_INC = -I/public/software/mathlib/netcdf/intel/4.7.4/include
NETCDF_LIB = -L/public/software/mathlib/netcdf/intel/4.7.4/lib/

FFLAGS += $(NETCDF_INC)
CFLAGS += $(NETCDF_INC) $(NETCDF_LIB)
LDFLAGS += -L/public/software/compiler/dtk/24.04.2/lib -L/public/software/mathlib/netcdf/intel/4.7.4/lib/ -L/public/software/mpi/intelmpi/2021.3.0/lib
```

2. 链接阶段 DEBUG

> [!IMPORTANT]
> ld: cannot find -lnetcdf
ld: cannot find -lnetcdff
ld: cannot find -lamdhip64
ld: cannot find -lmpifort


``` bash
mpif90 a2b_edge_cpu.o fv_mp_mod_cpu.o trid_d_sw_gpu.o sw_gpu.o d_sw_gpu.o tp_core_cpu.o sw_cpu.o trid_d_sw_cpu.o d_sw_cpu.o  mpp_data.o tp_core.o sw_core.o gradient_c2l.o atmosphere.o fms_io.o tracer_manager.o mpp_memutils.o fv_phys.o diag_util.o mosaic_util.o a2b_edge.o external_ic.o time_interp.o mosaic.o memuse.o memutils.o threadloc.o horiz_interp_bicubic.o diag_axis.o diag_manager.o fv_grid_utils.o mpp_utilities.o mpp_pset.o fms.o test_mpp_io.o create_xgrid.o interp.o diag_output.o fv_io.o fv_nudge.o horiz_interp_conserve.o fv_update_phys.o mpp_parameter.o axis_utils.o external_sst.o platform.o amip_interp.o mpp_domains.o fv_dynamics.o diag_grid.o fv_mapz.o test_cases.o time_manager.o fv_sg.o hswf.o sorted_index.o fv_arrays.o atmos_model.o mpp_io.o fv_control.o sim_nc_mod.o fv_tracer2d.o horiz_interp_spherical.o init_hydro.o fv_restart.o fv_timing.o horiz_interp.o gradient.o dyn_core.o fv_diagnostics.o nh_core.o fv_mp_mod.o fm_util.o horiz_interp_bilinear.o diag_data.o grid.o read_mosaic.o constants.o fv_grid_tools.o fv_fill.o test_mpp_pset.o test_fms_io.o fv_eta.o field_manager.o nsclock.o mpp.o fv_surf_map.o horiz_interp_type.o test_mpp_domains.o test_mpp.o lin_cloud_microphys.o -o fms.x  -L/public/software/mathlib/netcdf/4.6.2/intel/lib -L/public/software/compiler/rocm/dtk-22.04.2/lib -lnetcdf -lnetcdff -lmpi -lstdc++  -lamdhip64  
```

- compiler/dtk/24.04.2->/public/software/compiler/dtk/24.04.2/lib
- -L/public/software/mathlib/netcdf/intel/4.7.4/lib/


3. 最后的 DEBUG：

```
[ac0w1vpw3p@login01 exec.amd64]$ make
make: Circular fv_mp_mod_cpu.o <- fv_mp_mod_cpu.o dependency dropped.
make: Circular trid_d_sw_gpu.o <- trid_d_sw_gpu.o dependency dropped.
make: Circular tp_core_cpu.o <- tp_core_cpu.o dependency dropped.
make: Circular sw_cpu.o <- sw_cpu.o dependency dropped.
make: Circular trid_d_sw_cpu.o <- trid_d_sw_cpu.o dependency dropped.
make: Circular d_sw_cpu.o <- d_sw_cpu.o dependency dropped.
make: Nothing to be done for 'all'.
```

---

DCU 编程及作业调度：

https://developer.sourcefind.cn/gitbook/dcu_developer/OperationManual/3_CommandsAndSchedule/CommandsAndSchedule.html

---

### 概念：
> FV3（Finite-Volume Cubed-Sphere Dynamical Core）是由美国地球流体动力学实验室（GFDL）开发的动力内核，广泛应用于天气预报和气候模拟领域。其核心创新包括立方体球网格（Cubed-Sphere Grid）的均匀分布、拉格朗日垂直坐标的高效数值离散，以及质量守恒和物理一致性的有限体积格式。FV3以其卓越的计算效率、灵活的分辨率适配和强大的物理-动力耦合能力，被广泛采用于全球天气模型、区域气象预报和古气候模拟中，同时也支持多尺度模拟和新兴计算架构的优化。

---

### 论文学习：A Scientific Description of the GFDL Finite-Volume Cubed-Sphere Dynamical Core

这篇论文《A Scientific Description of the GFDL Finite-Volume Cubed-Sphere Dynamical Core》主要描述了美国地球流体动力学实验室（GFDL）开发的FV3动力内核的科学设计和技术实现。以下是其脉络和场景的详细解释：

1. 背景和意义

论文首先介绍了FV3（Finite-Volume Cubed-Sphere Dynamical Core）的历史背景和应用场景：
- 动力内核的需求：FV3起源于90年代初，目的是改进当时大气化学模型中存在的质量守恒、数值噪声等问题。
- 技术基础：基于有限体积方法（finite-volume methods），特别是质量守恒和高精度数值格式。
- 应用范围：FV3已经被广泛用于地球气候模型（如CESM、CMIP）、全球天气预报系统（如美国国家气象局的统一预报系统），以及其他国际机构的模型中。

2. FV3的核心创新

FV3的核心在于其科学设计，论文详细探讨了以下几个方面：

a. 立方体球网格 (Cubed-Sphere Grid)

- 优点：解决传统经纬度网格在极地区域收敛的问题，提供更均匀的分辨率分布。
- 技术实现：采用等边投影构建立方体球网格，优化了计算效率和数值稳定性。

b. 有限体积格式 (Finite-Volume Formulation)

- 质量守恒和数值精度：基于Lin和Rood的质量守恒对流格式，该格式通过一维和二维耦合方法实现高效计算。
- 适配不同分辨率：从低分辨率的古气候模拟到高分辨率的对流模拟（<100米）。

c. 拉格朗日垂直坐标 (Vertically-Lagrangian Coordinates)

- 核心优势：通过拉格朗日垂直离散化方法，显著提高了数值计算效率和精度。
- 适用性：支持无水动力学与含水动力学的无缝集成。

d. 湍流耗散与能量守恒

- 扩散处理：在保持能量守恒的同时引入适量的数值扩散，增强模型稳定性。
- 能量一致性：特别在模拟对流云系统中，通过热力学一致性改善模型性能。

4. 模型耦合与应用场景

FV3不仅是一个动力内核，还因其耦合性强而被广泛应用：
- 物理-动力耦合：支持不同的物理参数化方案，与现有气候和天气模型无缝衔接。
- 数据同化：能够与数据同化系统结合，提高初始场的准确性。
- 区域与全球模式统一：支持网格嵌套与局部加密，为区域天气预报（如热带气旋模拟）提供高分辨率支持。

6. 性能与可扩展性

- 计算效率：通过OpenMP和MPI并行化优化，使其在现代超级计算机上具有高效扩展能力。
- 未来架构适配：论文还提到FV3向GPU和领域特定语言（DSL，例如GT4py）的移植，以适应新兴的计算架构。

7. 未来发展方向

论文总结了FV3当前的局限性及其改进方向：
- 网格优化：开发Duo-Grid等新型网格，进一步消除数值噪声。
- 深大气动力学：扩展至深大气（如离子层）和其他行星大气的模拟。
- 物理-动力深度集成：将物理过程嵌入动力核心，减少计算误差并提高效率。

8. 学术与工程价值

这篇论文不仅是FV3的技术描述，还为领域内的数值天气预报和气候模拟提供了理论指导：
- 理论贡献：细致阐述了有限体积方法在流体动力学中的应用。
- 实用价值：为未来开发高性能、可扩展的动力模型奠定了基础。
