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
# 动态库路径
export LD_LIBRARY_PATH=/public/software/mathlib/netcdf/intel/4.7.4/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/hpc/software/compiler/intel/intel-compiler-2021.3.0/compiler/lib/intel64:$LD_LIBRARY_PATH

# Fabric 和 MPI 配置
export FI_PROVIDER=verbs
export FI_MLX_IFACE=mlx5_0
export I_MPI_FABRICS=shm:ofi
export I_MPI_DEBUG=5
export FI_PSM3_DISABLE=1


# 执行程序
mpirun -np 4 ./fms.x

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

---

Makefile 全文附上：

``` Makefile
[ac0w1vpw3p@login01 exec.amd64]$ cat Makefile 
# Makefile created by mkmf $Id: mkmf,v 18.0 2010/03/02 23:26:08 fms Exp $ 

CC = icc
MPICC = mpiicc

SRCROOT = /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/src/

CPPDEFS = -Duse_libMPI -Duse_netCDF -DSPMD


include /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../bin/mkmf.template.amd64

NETCDF_INC = -I/public/software/mathlib/netcdf/intel/4.7.4/include
NETCDF_LIB = -L/public/software/mathlib/netcdf/intel/4.7.4/lib/

FFLAGS += $(NETCDF_INC)
CFLAGS += $(NETCDF_INC) $(NETCDF_LIB)
LDFLAGS += -L/public/software/compiler/dtk/24.04.2/lib -L/public/software/mathlib/netcdf/intel/4.7.4/lib/ -L/public/software/mpi/intelmpi/2021.3.0/lib


.DEFAULT:
        -echo $@ does not exist.
all: fms.x
trid_d_sw_gpu.o: $(SRCROOT)atmos_cubed_sphere/model/trid_d_sw_gpu.cu trid_d_sw_gpu.o
        $(NVCC) $(NVCCFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/trid_d_sw_gpu.cu
trid_d_sw_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/trid_d_sw_cpu.c trid_d_sw_cpu.o
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/trid_d_sw_cpu.c
a2b_edge_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/a2b_edge_cpu.c
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/a2b_edge_cpu.c
fv_mp_mod_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/fv_mp_mod_cpu.c fv_mp_mod_cpu.o
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/fv_mp_mod_cpu.c
d_sw_gpu.o: $(SRCROOT)atmos_cubed_sphere/model/d_sw_gpu.cu
        $(NVCC) $(NVCCFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/d_sw_gpu.cu
d_sw_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/d_sw_cpu.c d_sw_cpu.o
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/d_sw_cpu.c
sw_gpu.o: $(SRCROOT)atmos_cubed_sphere/model/sw_gpu.cu
        $(NVCC) $(NVCCFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/sw_gpu.cu
sw_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/sw_cpu.c sw_cpu.o
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/sw_cpu.c
tp_core_cpu.o: $(SRCROOT)atmos_cubed_sphere/model/tp_core_cpu.c tp_core_cpu.o
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)atmos_cubed_sphere/model/tp_core_cpu.c
a2b_edge.o: $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90 fv_grid_utils.o fv_grid_tools.o fv_mp_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90
amip_interp.o: $(SRCROOT)shared/amip_interp/amip_interp.F90 time_interp.o time_manager.o horiz_interp.o fms.o fms_io.o constants.o platform.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/amip_interp/amip_interp.F90
atmos_model.o: $(SRCROOT)atmos_solo/atmos_model.F90 atmosphere.o time_manager.o fms.o fms_io.o mpp_domains.o mpp_io.o diag_manager.o field_manager.o tracer_manager.o memutils.o constants.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_solo/atmos_model.F90
atmosphere.o: $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90 constants.o fms.o time_manager.o mpp_domains.o fv_arrays.o fv_control.o fv_phys.o fv_diagnostics.o fv_timing.o fv_restart.o fv_dynamics.o fv_grid_tools.o lin_cloud_microphys.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90
axis_utils.o: $(SRCROOT)shared/axis_utils/axis_utils.F90 mpp_io.o mpp.o fms.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/axis_utils/axis_utils.F90
constants.o: $(SRCROOT)shared/constants/constants.F90
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/constants/constants.F90
create_xgrid.o: $(SRCROOT)shared/mosaic/create_xgrid.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic  $(SRCROOT)shared/mosaic/create_xgrid.c
diag_axis.o: $(SRCROOT)shared/diag_manager/diag_axis.F90 mpp_domains.o fms.o diag_data.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/diag_manager/diag_axis.F90
diag_data.o: $(SRCROOT)shared/diag_manager/diag_data.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h time_manager.o mpp_domains.o mpp_io.o fms.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/diag_manager/diag_data.F90
diag_grid.o: $(SRCROOT)shared/diag_manager/diag_grid.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h constants.o fms.o mpp.o mpp_domains.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/diag_manager/diag_grid.F90
diag_manager.o: $(SRCROOT)shared/diag_manager/diag_manager.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h time_manager.o mpp_io.o fms.o mpp.o diag_axis.o diag_util.o diag_data.o diag_output.o diag_grid.o constants.o mpp_domains.o fms_io.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/diag_manager/diag_manager.F90
diag_output.o: $(SRCROOT)shared/diag_manager/diag_output.F90 mpp_io.o mpp_domains.o mpp.o diag_axis.o diag_data.o time_manager.o fms.o platform.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/diag_manager/diag_output.F90
diag_util.o: $(SRCROOT)shared/diag_manager/diag_util.F90 diag_data.o diag_axis.o diag_output.o diag_grid.o fms.o fms_io.o mpp_domains.o time_manager.o mpp_io.o mpp.o constants.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/diag_manager/diag_util.F90
dyn_core.o: $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90 mpp_domains.o mpp_parameter.o fv_mp_mod.o fv_control.o sw_core.o a2b_edge.o nh_core.o fv_grid_tools.o fv_grid_utils.o fv_timing.o fv_diagnostics.o fv_nudge.o test_cases.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90
external_ic.o: $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90 fms.o fms_io.o mpp.o mpp_parameter.o mpp_domains.o tracer_manager.o field_manager.o constants.o external_sst.o fv_arrays.o fv_diagnostics.o fv_grid_tools.o fv_grid_utils.o fv_io.o fv_mapz.o fv_mp_mod.o fv_surf_map.o fv_timing.o init_hydro.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90
external_sst.o: $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90 amip_interp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90
field_manager.o: $(SRCROOT)shared/field_manager/field_manager.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager/parse.inc mpp.o mpp_io.o fms.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager    $(SRCROOT)shared/field_manager/field_manager.F90
fm_util.o: $(SRCROOT)shared/field_manager/fm_util.F90 field_manager.o fms.o mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/field_manager/fm_util.F90
fms.o: $(SRCROOT)shared/fms/fms.F90 mpp.o mpp_domains.o mpp_io.o fms_io.o memutils.o constants.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/fms/fms.F90
fms_io.o: $(SRCROOT)shared/fms/fms_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_2d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_3d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_4d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/write_data.inc mpp_io.o mpp_domains.o mpp.o platform.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms      $(SRCROOT)shared/fms/fms_io.F90
fv_arrays.o: $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp_domains.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90
fv_control.o: $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90 constants.o field_manager.o mpp.o mpp_domains.o tracer_manager.o fv_io.o fv_restart.o fv_arrays.o fv_grid_utils.o fv_grid_tools.o fv_mp_mod.o test_cases.o fv_timing.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90
fv_diagnostics.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90 constants.o fms_io.o time_manager.o mpp_domains.o diag_manager.o fv_arrays.o fv_mapz.o fv_mp_mod.o fv_eta.o fv_grid_tools.o fv_grid_utils.o a2b_edge.o fv_surf_map.o fv_sg.o tracer_manager.o field_manager.o mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90
fv_dynamics.o: $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90 constants.o dyn_core.o fv_mapz.o fv_tracer2d.o fv_grid_tools.o fv_control.o fv_grid_utils.o fv_mp_mod.o fv_timing.o diag_manager.o fv_diagnostics.o mpp_domains.o field_manager.o tracer_manager.o fv_sg.o tp_core.o time_manager.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90
fv_eta.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90 constants.o fv_mp_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90
fv_fill.o: $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90
fv_grid_tools.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90 constants.o fv_arrays.o fv_grid_utils.o fv_timing.o fv_mp_mod.o sorted_index.o mpp.o mpp_domains.o mpp_io.o mpp_parameter.o fms.o fms_io.o mosaic.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90
fv_grid_utils.o: $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o mpp_domains.o mpp_parameter.o external_sst.o fv_arrays.o fv_eta.o fv_mp_mod.o fv_timing.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90
fv_io.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90 fms.o fms_io.o mpp.o mpp_domains.o tracer_manager.o field_manager.o external_sst.o fv_arrays.o fv_mapz.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90
fv_mapz.o: $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90 constants.o fv_grid_tools.o fv_grid_utils.o fv_fill.o fv_mp_mod.o mpp_domains.o mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90
fv_mp_mod.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90 fms.o mpp.o mpp_domains.o mpp_parameter.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90
fv_nudge.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90 constants.o fms.o fms_io.o mpp.o mpp_domains.o time_manager.o external_sst.o fv_control.o fv_grid_utils.o fv_grid_tools.o fv_diagnostics.o tp_core.o fv_mapz.o fv_mp_mod.o fv_timing.o sim_nc_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90
fv_phys.o: $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90 time_manager.o fv_update_phys.o fv_timing.o hswf.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90
fv_restart.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90 constants.o fv_arrays.o fv_io.o fv_grid_tools.o fv_grid_utils.o fv_diagnostics.o init_hydro.o mpp_domains.o mpp.o test_cases.o fv_mp_mod.o fv_surf_map.o tracer_manager.o field_manager.o external_ic.o fv_eta.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90
fv_sg.o: $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90 constants.o fv_mp_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90
fv_surf_map.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90 fms.o mpp.o mpp_domains.o constants.o fms_io.o fv_grid_utils.o fv_mp_mod.o fv_timing.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90
fv_timing.o: $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90 fv_mp_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90
fv_tracer2d.o: $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90 tp_core.o fv_grid_tools.o fv_grid_utils.o fv_mp_mod.o mpp_domains.o fv_timing.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90
fv_update_phys.o: $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90 constants.o field_manager.o mpp_domains.o mpp_parameter.o mpp.o time_manager.o tracer_manager.o fv_arrays.o fv_control.o fv_mp_mod.o fv_eta.o fv_grid_utils.o fv_grid_tools.o fv_timing.o fv_diagnostics.o fv_nudge.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90
gradient.o: $(SRCROOT)shared/mosaic/gradient.F90 mpp.o constants.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/mosaic/gradient.F90
gradient_c2l.o: $(SRCROOT)shared/mosaic/gradient_c2l.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/gradient_c2l.h
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic  $(SRCROOT)shared/mosaic/gradient_c2l.c
grid.o: $(SRCROOT)shared/mosaic/grid.F90 constants.o fms.o mosaic.o mpp_domains.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/mosaic/grid.F90
horiz_interp.o: $(SRCROOT)shared/horiz_interp/horiz_interp.F90 fms.o mpp.o constants.o horiz_interp_type.o horiz_interp_conserve.o horiz_interp_bilinear.o horiz_interp_bicubic.o horiz_interp_spherical.o mpp_io.o mpp_domains.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp.F90
horiz_interp_bicubic.o: $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90 mpp.o fms.o horiz_interp_type.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90
horiz_interp_bilinear.o: $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90 mpp.o fms.o constants.o horiz_interp_type.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90
horiz_interp_conserve.o: $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90 mpp.o fms.o constants.o horiz_interp_type.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90
horiz_interp_spherical.o: $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90 mpp.o fms.o constants.o horiz_interp_type.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90
horiz_interp_type.o: $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90 mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90
hswf.o: $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90 constants.o mpp_domains.o time_manager.o diag_manager.o lin_cloud_microphys.o fv_grid_tools.o fv_grid_utils.o fv_mp_mod.o fv_diagnostics.o fv_timing.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90
init_hydro.o: $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90 constants.o fv_grid_utils.o fv_grid_tools.o fv_mp_mod.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90
interp.o: $(SRCROOT)shared/mosaic/interp.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/interp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic  $(SRCROOT)shared/mosaic/interp.c
lin_cloud_microphys.o: $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90 mpp.o diag_manager.o time_manager.o constants.o fms.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90
memuse.o: $(SRCROOT)shared/memutils/memuse.c
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)shared/memutils/memuse.c
memutils.o: $(SRCROOT)shared/memutils/memutils.F90 mpp.o mpp_io.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/memutils/memutils.F90
mosaic.o: $(SRCROOT)shared/mosaic/mosaic.F90 mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/mosaic/mosaic.F90
mosaic_util.o: $(SRCROOT)shared/mosaic/mosaic_util.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic  $(SRCROOT)shared/mosaic/mosaic_util.c
mpp.o: $(SRCROOT)shared/mpp/mpp.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/system_clock.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_a.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_s.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_a.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_s.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_int.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_scalar.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum.h mpp_parameter.o mpp_data.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include      $(SRCROOT)shared/mpp/mpp.F90
mpp_data.o: $(SRCROOT)shared/mpp/mpp_data.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_nocomm.inc mpp_parameter.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include      $(SRCROOT)shared/mpp/mpp_data.F90
mpp_domains.o: $(SRCROOT)shared/mpp/mpp_domains.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_define.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_misc.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_update_domains2D.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_update.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_updateV.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_check.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_checkV.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_redistribute.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_get_boundary.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_get_boundary.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_reduce.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_reduce.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum_tl.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_field.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_global_field.h mpp_parameter.o mpp_data.o mpp.o mpp_memutils.o mpp_pset.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include      $(SRCROOT)shared/mpp/mpp_domains.F90
mpp_io.o: $(SRCROOT)shared/mpp/mpp_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_misc.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_connect.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_read.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_read_2Ddecomp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_write.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write_2Ddecomp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write.h mpp_parameter.o mpp.o mpp_domains.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include      $(SRCROOT)shared/mpp/mpp_io.F90
mpp_memutils.o: $(SRCROOT)shared/mpp/mpp_memutils.F90 mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/mpp/mpp_memutils.F90
mpp_parameter.o: $(SRCROOT)shared/mpp/mpp_parameter.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/mpp_parameter.F90
mpp_pset.o: $(SRCROOT)shared/mpp/mpp_pset.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/mpp_pset.F90
mpp_utilities.o: $(SRCROOT)shared/mpp/mpp_utilities.F90 mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/mpp/mpp_utilities.F90
nh_core.o: $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90 fms.o constants.o fv_control.o tp_core.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90
nsclock.o: $(SRCROOT)shared/mpp/nsclock.c
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)shared/mpp/nsclock.c
platform.o: $(SRCROOT)shared/platform/platform.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/platform/platform.F90
read_mosaic.o: $(SRCROOT)shared/mosaic/read_mosaic.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/read_mosaic.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic  $(SRCROOT)shared/mosaic/read_mosaic.c
sim_nc_mod.o: $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90 mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90
sorted_index.o: $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90
sw_core.o: $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90 fv_mp_mod.o fv_grid_tools.o tp_core.o fv_grid_utils.o test_cases.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90
test_cases.o: $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90 constants.o init_hydro.o fv_mp_mod.o fv_grid_utils.o fv_surf_map.o fv_grid_tools.o fv_eta.o mpp.o mpp_domains.o mpp_parameter.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90
test_fms_io.o: $(SRCROOT)shared/fms/test_fms_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o mpp_domains.o mpp_io.o fms_io.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/fms/test_fms_io.F90
test_mpp.o: $(SRCROOT)shared/mpp/test_mpp.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/test_mpp.F90
test_mpp_domains.o: $(SRCROOT)shared/mpp/test_mpp_domains.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o mpp_domains.o mpp_memutils.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/test_mpp_domains.F90
test_mpp_io.o: $(SRCROOT)shared/mpp/test_mpp_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o mpp_domains.o mpp_io.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/test_mpp_io.F90
test_mpp_pset.o: $(SRCROOT)shared/mpp/test_mpp_pset.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h mpp.o mpp_pset.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c -I/work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include $(SRCROOT)shared/mpp/test_mpp_pset.F90
threadloc.o: $(SRCROOT)shared/mpp/threadloc.c
        $(CC) $(CPPDEFS) $(CPPFLAGS) $(CFLAGS) $(OTHERFLAGS) -c $(SRCROOT)shared/mpp/threadloc.c
time_interp.o: $(SRCROOT)shared/time_interp/time_interp.F90 time_manager.o fms.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/time_interp/time_interp.F90
time_manager.o: $(SRCROOT)shared/time_manager/time_manager.F90 constants.o fms.o fms_io.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/time_manager/time_manager.F90
tp_core.o: $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90 fv_mp_mod.o fv_grid_utils.o fv_grid_tools.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90
tracer_manager.o: $(SRCROOT)shared/tracer_manager/tracer_manager.F90 mpp.o mpp_io.o fms.o field_manager.o
        $(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) -c     $(SRCROOT)shared/tracer_manager/tracer_manager.F90
./external_ic.F90: $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90 .
./tp_core.F90: $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90 .
./mpp_sum_mpi.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_mpi.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_mpi.h .
./fv_sg.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90 .
./create_xgrid.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h .
./fv_grid_utils.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90 .
./fv_update_phys.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90 .
./mpp_pset.F90: $(SRCROOT)shared/mpp/mpp_pset.F90
        cp $(SRCROOT)shared/mpp/mpp_pset.F90 .
./mpp_data_sma.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_sma.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_sma.inc .
./test_mpp_pset.F90: $(SRCROOT)shared/mpp/test_mpp_pset.F90
        cp $(SRCROOT)shared/mpp/test_mpp_pset.F90 .
./fms_io.F90: $(SRCROOT)shared/fms/fms_io.F90
        cp $(SRCROOT)shared/fms/fms_io.F90 .
./mpp_do_global_field.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_global_field.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_global_field.h .
./mpp_do_update.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_update.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_update.h .
./mpp_data_nocomm.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_nocomm.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_nocomm.inc .
./fv_grid_tools.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90 .
./fv_io.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90 .
./read_mosaic.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/read_mosaic.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/read_mosaic.h .
./platform.F90: $(SRCROOT)shared/platform/platform.F90
        cp $(SRCROOT)shared/platform/platform.F90 .
./mosaic.F90: $(SRCROOT)shared/mosaic/mosaic.F90
        cp $(SRCROOT)shared/mosaic/mosaic.F90 .
./read_data_2d.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_2d.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_2d.inc .
./mpp_error_a_s.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_s.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_s.h .
./fv_dynamics.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90 .
./mpp_do_updateV.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_updateV.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_updateV.h .
./hswf.F90: $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90
        cp $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90 .
./test_cases.F90: $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90 .
./fv_arrays.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90 .
./fv_timing.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90 .
./mpp_util.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util.inc .
./mpp_data.F90: $(SRCROOT)shared/mpp/mpp_data.F90
        cp $(SRCROOT)shared/mpp/mpp_data.F90 .
./amip_interp.F90: $(SRCROOT)shared/amip_interp/amip_interp.F90
        cp $(SRCROOT)shared/amip_interp/amip_interp.F90 .
./sw_core.F90: $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90 .
./mpp_io_misc.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_misc.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_misc.inc .
./gradient.F90: $(SRCROOT)shared/mosaic/gradient.F90
        cp $(SRCROOT)shared/mosaic/gradient.F90 .
./horiz_interp_type.F90: $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90 .
./fv_tracer2d.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90 .
./mpp_global_sum_tl.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum_tl.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum_tl.h .
./fv_surf_map.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90 .
./mpp_comm_mpi.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_mpi.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_mpi.inc .
./mpp_utilities.F90: $(SRCROOT)shared/mpp/mpp_utilities.F90
        cp $(SRCROOT)shared/mpp/mpp_utilities.F90 .
./parse.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager/parse.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager/parse.inc .
./axis_utils.F90: $(SRCROOT)shared/axis_utils/axis_utils.F90
        cp $(SRCROOT)shared/axis_utils/axis_utils.F90 .
./fms_platform.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h .
./time_manager.F90: $(SRCROOT)shared/time_manager/time_manager.F90
        cp $(SRCROOT)shared/time_manager/time_manager.F90 .
./atmosphere.F90: $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90
        cp $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90 .
./test_mpp_domains.F90: $(SRCROOT)shared/mpp/test_mpp_domains.F90
        cp $(SRCROOT)shared/mpp/test_mpp_domains.F90 .
./read_data_4d.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_4d.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_4d.inc .
./mpp_transmit.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit.inc .
./mpp_reduce_sma.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_sma.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_sma.h .
./diag_axis.F90: $(SRCROOT)shared/diag_manager/diag_axis.F90
        cp $(SRCROOT)shared/diag_manager/diag_axis.F90 .
./constant.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h .
./mpp_write.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write.h .
./mpp_global_reduce.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_reduce.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_reduce.h .
./mpp_global_sum.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum.h .
./mpp_util_nocomm.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_nocomm.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_nocomm.inc .
./mosaic_util.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h .
./horiz_interp_conserve.F90: $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90 .
./dyn_core.F90: $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90 .
./init_hydro.F90: $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90 .
./nh_core.F90: $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90 .
./mpp_util_sma.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_sma.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_sma.inc .
./mpp_reduce_nocomm.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_nocomm.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_nocomm.h .
./diag_util.F90: $(SRCROOT)shared/diag_manager/diag_util.F90
        cp $(SRCROOT)shared/diag_manager/diag_util.F90 .
./diag_manager.F90: $(SRCROOT)shared/diag_manager/diag_manager.F90
        cp $(SRCROOT)shared/diag_manager/diag_manager.F90 .
./fv_fill.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90 .
./horiz_interp_bicubic.F90: $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90 .
./fms.F90: $(SRCROOT)shared/fms/fms.F90
        cp $(SRCROOT)shared/fms/fms.F90 .
./fv_restart.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90 .
./mpp_domains_comm.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_comm.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_comm.inc .
./mpp_update_domains2D.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_update_domains2D.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_update_domains2D.h .
./field_manager.F90: $(SRCROOT)shared/field_manager/field_manager.F90
        cp $(SRCROOT)shared/field_manager/field_manager.F90 .
./mpp_do_redistribute.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_redistribute.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_redistribute.h .
./fm_util.F90: $(SRCROOT)shared/field_manager/fm_util.F90
        cp $(SRCROOT)shared/field_manager/fm_util.F90 .
./mpp_comm_nocomm.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_nocomm.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_nocomm.inc .
./mpp_io.F90: $(SRCROOT)shared/mpp/mpp_io.F90
        cp $(SRCROOT)shared/mpp/mpp_io.F90 .
./sim_nc_mod.F90: $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90 .
./mpp.F90: $(SRCROOT)shared/mpp/mpp.F90
        cp $(SRCROOT)shared/mpp/mpp.F90 .
./mpp_domains_reduce.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_reduce.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_reduce.inc .
./memuse.c: $(SRCROOT)shared/memutils/memuse.c
        cp $(SRCROOT)shared/memutils/memuse.c .
./mpp_sum_sma.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_sma.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_sma.h .
./mpp_domains_misc.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_misc.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_misc.inc .
./gradient_c2l.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/gradient_c2l.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/gradient_c2l.h .
./mpp_util_mpi.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_mpi.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_mpi.inc .
./mpp_sum.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum.inc .
./mpp_chksum_scalar.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_scalar.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_scalar.h .
./gradient_c2l.c: $(SRCROOT)shared/mosaic/gradient_c2l.c
        cp $(SRCROOT)shared/mosaic/gradient_c2l.c .
./test_mpp_io.F90: $(SRCROOT)shared/mpp/test_mpp_io.F90
        cp $(SRCROOT)shared/mpp/test_mpp_io.F90 .
./diag_grid.F90: $(SRCROOT)shared/diag_manager/diag_grid.F90
        cp $(SRCROOT)shared/diag_manager/diag_grid.F90 .
./mpp_domains.F90: $(SRCROOT)shared/mpp/mpp_domains.F90
        cp $(SRCROOT)shared/mpp/mpp_domains.F90 .
./mpp_domains_define.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_define.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_define.inc .
./interp.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/interp.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/interp.h .
./diag_data.F90: $(SRCROOT)shared/diag_manager/diag_data.F90
        cp $(SRCROOT)shared/diag_manager/diag_data.F90 .
./a2b_edge.F90: $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90 .
./mpp_data_mpi.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_mpi.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_mpi.inc .
./mpp_memutils.F90: $(SRCROOT)shared/mpp/mpp_memutils.F90
        cp $(SRCROOT)shared/mpp/mpp_memutils.F90 .
./mpp_parameter.F90: $(SRCROOT)shared/mpp/mpp_parameter.F90
        cp $(SRCROOT)shared/mpp/mpp_parameter.F90 .
./mpp_chksum.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum.h .
./mpp_io_util.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_util.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_util.inc .
./mpp_do_check.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_check.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_check.h .
./mpp_error_a_a.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_a.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_a.h .
./lin_cloud_microphys.F90: $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90
        cp $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90 .
./mpp_io_read.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_read.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_read.inc .
./test_fms_io.F90: $(SRCROOT)shared/fms/test_fms_io.F90
        cp $(SRCROOT)shared/fms/test_fms_io.F90 .
./fv_diagnostics.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90 .
./horiz_interp_spherical.F90: $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90 .
./memutils.F90: $(SRCROOT)shared/memutils/memutils.F90
        cp $(SRCROOT)shared/memutils/memutils.F90 .
./write_data.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/write_data.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/write_data.inc .
./mpp_global_field.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_field.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_field.h .
./constants.F90: $(SRCROOT)shared/constants/constants.F90
        cp $(SRCROOT)shared/constants/constants.F90 .
./tracer_manager.F90: $(SRCROOT)shared/tracer_manager/tracer_manager.F90
        cp $(SRCROOT)shared/tracer_manager/tracer_manager.F90 .
./mpp_read_2Ddecomp.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_read_2Ddecomp.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_read_2Ddecomp.h .
./test_mpp.F90: $(SRCROOT)shared/mpp/test_mpp.F90
        cp $(SRCROOT)shared/mpp/test_mpp.F90 .
./mpp_error_s_s.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_s.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_s.h .
./mpp_io_write.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_write.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_write.inc .
./mpp_reduce_mpi.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_mpi.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_mpi.h .
./horiz_interp_bilinear.F90: $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90 .
./mosaic_util.c: $(SRCROOT)shared/mosaic/mosaic_util.c
        cp $(SRCROOT)shared/mosaic/mosaic_util.c .
./read_mosaic.c: $(SRCROOT)shared/mosaic/read_mosaic.c
        cp $(SRCROOT)shared/mosaic/read_mosaic.c .
./fv_mapz.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90 .
./fv_nudge.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90 .
./interp.c: $(SRCROOT)shared/mosaic/interp.c
        cp $(SRCROOT)shared/mosaic/interp.c .
./sorted_index.F90: $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90 .
./mpp_domains_util.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_util.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_util.inc .
./threadloc.c: $(SRCROOT)shared/mpp/threadloc.c
        cp $(SRCROOT)shared/mpp/threadloc.c .
./mpp_transmit_sma.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_sma.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_sma.h .
./mpp_comm_sma.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_sma.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_sma.inc .
./mpp_get_boundary.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_get_boundary.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_get_boundary.h .
./mpp_write_2Ddecomp.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write_2Ddecomp.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write_2Ddecomp.h .
./mpp_do_checkV.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_checkV.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_checkV.h .
./fv_phys.F90: $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90
        cp $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90 .
./read_data_3d.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_3d.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_3d.inc .
./time_interp.F90: $(SRCROOT)shared/time_interp/time_interp.F90
        cp $(SRCROOT)shared/time_interp/time_interp.F90 .
./fv_eta.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90 .
./horiz_interp.F90: $(SRCROOT)shared/horiz_interp/horiz_interp.F90
        cp $(SRCROOT)shared/horiz_interp/horiz_interp.F90 .
./diag_output.F90: $(SRCROOT)shared/diag_manager/diag_output.F90
        cp $(SRCROOT)shared/diag_manager/diag_output.F90 .
./fv_control.F90: $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90
        cp $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90 .
./mpp_error_s_a.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_a.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_a.h .
./mpp_transmit_mpi.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_mpi.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_mpi.h .
./nsclock.c: $(SRCROOT)shared/mpp/nsclock.c
        cp $(SRCROOT)shared/mpp/nsclock.c .
./mpp_transmit_nocomm.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_nocomm.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_nocomm.h .
./grid.F90: $(SRCROOT)shared/mosaic/grid.F90
        cp $(SRCROOT)shared/mosaic/grid.F90 .
./mpp_comm.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm.inc .
./system_clock.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/system_clock.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/system_clock.h .
./atmos_model.F90: $(SRCROOT)atmos_solo/atmos_model.F90
        cp $(SRCROOT)atmos_solo/atmos_model.F90 .
./external_sst.F90: $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90 .
./mpp_sum_nocomm.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_nocomm.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_nocomm.h .
./mpp_do_get_boundary.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_get_boundary.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_get_boundary.h .
./fv_mp_mod.F90: $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90
        cp $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90 .
./create_xgrid.c: $(SRCROOT)shared/mosaic/create_xgrid.c
        cp $(SRCROOT)shared/mosaic/create_xgrid.c .
./mpp_chksum_int.h: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_int.h
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_int.h .
./mpp_io_connect.inc: /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_connect.inc
        cp /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_connect.inc .
SRC = $(SRCROOT)shared/mpp/mpp_data.F90 $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90 $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90 $(SRCROOT)shared/mosaic/gradient_c2l.c $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90 $(SRCROOT)shared/fms/fms_io.F90 $(SRCROOT)shared/tracer_manager/tracer_manager.F90 $(SRCROOT)shared/mpp/mpp_memutils.F90 $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90 $(SRCROOT)shared/diag_manager/diag_util.F90 $(SRCROOT)shared/mosaic/mosaic_util.c $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90 $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90 $(SRCROOT)shared/time_interp/time_interp.F90 $(SRCROOT)shared/mosaic/mosaic.F90 $(SRCROOT)shared/memutils/memuse.c $(SRCROOT)shared/memutils/memutils.F90 $(SRCROOT)shared/mpp/threadloc.c $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90 $(SRCROOT)shared/diag_manager/diag_axis.F90 $(SRCROOT)shared/diag_manager/diag_manager.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90 $(SRCROOT)shared/mpp/mpp_utilities.F90 $(SRCROOT)shared/mpp/mpp_pset.F90 $(SRCROOT)shared/fms/fms.F90 $(SRCROOT)shared/mpp/test_mpp_io.F90 $(SRCROOT)shared/mosaic/create_xgrid.c $(SRCROOT)shared/mosaic/interp.c $(SRCROOT)shared/diag_manager/diag_output.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90 $(SRCROOT)shared/mpp/mpp_parameter.F90 $(SRCROOT)shared/axis_utils/axis_utils.F90 $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90 $(SRCROOT)shared/platform/platform.F90 $(SRCROOT)shared/amip_interp/amip_interp.F90 $(SRCROOT)shared/mpp/mpp_domains.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90 $(SRCROOT)shared/diag_manager/diag_grid.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90 $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90 $(SRCROOT)shared/time_manager/time_manager.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90 $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90 $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90 $(SRCROOT)atmos_solo/atmos_model.F90 $(SRCROOT)shared/mpp/mpp_io.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90 $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90 $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90 $(SRCROOT)shared/horiz_interp/horiz_interp.F90 $(SRCROOT)shared/mosaic/gradient.F90 $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90 $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90 $(SRCROOT)shared/field_manager/fm_util.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90 $(SRCROOT)shared/diag_manager/diag_data.F90 $(SRCROOT)shared/mosaic/grid.F90 $(SRCROOT)shared/mosaic/read_mosaic.c $(SRCROOT)shared/constants/constants.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90 $(SRCROOT)shared/mpp/test_mpp_pset.F90 $(SRCROOT)shared/fms/test_fms_io.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90 $(SRCROOT)shared/field_manager/field_manager.F90 $(SRCROOT)shared/mpp/nsclock.c $(SRCROOT)shared/mpp/mpp.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90 $(SRCROOT)shared/mpp/test_mpp_domains.F90 $(SRCROOT)shared/mpp/test_mpp.F90 $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_update_domains2D.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_redistribute.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_reduce.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_global_field.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_update.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/read_mosaic.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_misc.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_scalar.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/gradient_c2l.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_2d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_s.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_define.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/interp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_check.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_updateV.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_a.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_read.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/write_data.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_field.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_read_2Ddecomp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_misc.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_write.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_s.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum_tl.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager/parse.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write_2Ddecomp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_get_boundary.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_checkV.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_4d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_3d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_reduce.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_a.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_mpi.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/system_clock.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_get_boundary.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_int.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_connect.inc
OBJ =a2b_edge_cpu.o fv_mp_mod_cpu.o trid_d_sw_gpu.o sw_gpu.o d_sw_gpu.o tp_core_cpu.o sw_cpu.o trid_d_sw_cpu.o d_sw_cpu.o  mpp_data.o tp_core.o sw_core.o gradient_c2l.o atmosphere.o fms_io.o tracer_manager.o mpp_memutils.o fv_phys.o diag_util.o mosaic_util.o a2b_edge.o external_ic.o time_interp.o mosaic.o memuse.o memutils.o threadloc.o horiz_interp_bicubic.o diag_axis.o diag_manager.o fv_grid_utils.o mpp_utilities.o mpp_pset.o fms.o test_mpp_io.o create_xgrid.o interp.o diag_output.o fv_io.o fv_nudge.o horiz_interp_conserve.o fv_update_phys.o mpp_parameter.o axis_utils.o external_sst.o platform.o amip_interp.o mpp_domains.o fv_dynamics.o diag_grid.o fv_mapz.o test_cases.o time_manager.o fv_sg.o hswf.o sorted_index.o fv_arrays.o atmos_model.o mpp_io.o fv_control.o sim_nc_mod.o fv_tracer2d.o horiz_interp_spherical.o init_hydro.o fv_restart.o fv_timing.o horiz_interp.o gradient.o dyn_core.o fv_diagnostics.o nh_core.o fv_mp_mod.o fm_util.o horiz_interp_bilinear.o diag_data.o grid.o read_mosaic.o constants.o fv_grid_tools.o fv_fill.o test_mpp_pset.o test_fms_io.o fv_eta.o field_manager.o nsclock.o mpp.o fv_surf_map.o horiz_interp_type.o test_mpp_domains.o test_mpp.o lin_cloud_microphys.o
OFF = $(SRCROOT)atmos_cubed_sphere/tools/external_ic.F90 $(SRCROOT)atmos_cubed_sphere/model/tp_core.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_mpi.h $(SRCROOT)atmos_cubed_sphere/model/fv_sg.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/create_xgrid.h $(SRCROOT)atmos_cubed_sphere/model/fv_grid_utils.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_update_phys.F90 $(SRCROOT)shared/mpp/mpp_pset.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_sma.inc $(SRCROOT)shared/mpp/test_mpp_pset.F90 $(SRCROOT)shared/fms/fms_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_global_field.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_update.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_nocomm.inc $(SRCROOT)atmos_cubed_sphere/tools/fv_grid_tools.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_io.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/read_mosaic.h $(SRCROOT)shared/platform/platform.F90 $(SRCROOT)shared/mosaic/mosaic.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_2d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_s.h $(SRCROOT)atmos_cubed_sphere/model/fv_dynamics.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_updateV.h $(SRCROOT)atmos_cubed_sphere/driver/solo/hswf.F90 $(SRCROOT)atmos_cubed_sphere/tools/test_cases.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_arrays.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_timing.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util.inc $(SRCROOT)shared/mpp/mpp_data.F90 $(SRCROOT)shared/amip_interp/amip_interp.F90 $(SRCROOT)atmos_cubed_sphere/model/sw_core.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_misc.inc $(SRCROOT)shared/mosaic/gradient.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_type.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_tracer2d.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum_tl.h $(SRCROOT)atmos_cubed_sphere/tools/fv_surf_map.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_mpi.inc $(SRCROOT)shared/mpp/mpp_utilities.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/field_manager/parse.inc $(SRCROOT)shared/axis_utils/axis_utils.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/include/fms_platform.h $(SRCROOT)shared/time_manager/time_manager.F90 $(SRCROOT)atmos_cubed_sphere/driver/solo/atmosphere.F90 $(SRCROOT)shared/mpp/test_mpp_domains.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_4d.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_sma.h $(SRCROOT)shared/diag_manager/diag_axis.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/constant.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_reduce.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_sum.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_nocomm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/mosaic_util.h $(SRCROOT)shared/horiz_interp/horiz_interp_conserve.F90 $(SRCROOT)atmos_cubed_sphere/model/dyn_core.F90 $(SRCROOT)atmos_cubed_sphere/tools/init_hydro.F90 $(SRCROOT)atmos_cubed_sphere/model/nh_core.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_nocomm.h $(SRCROOT)shared/diag_manager/diag_util.F90 $(SRCROOT)shared/diag_manager/diag_manager.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_fill.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_bicubic.F90 $(SRCROOT)shared/fms/fms.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_restart.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_update_domains2D.h $(SRCROOT)shared/field_manager/field_manager.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_redistribute.h $(SRCROOT)shared/field_manager/fm_util.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_nocomm.inc $(SRCROOT)shared/mpp/mpp_io.F90 $(SRCROOT)atmos_cubed_sphere/tools/sim_nc_mod.F90 $(SRCROOT)shared/mpp/mpp.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_reduce.inc $(SRCROOT)shared/memutils/memuse.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_misc.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/gradient_c2l.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_util_mpi.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_scalar.h $(SRCROOT)shared/mosaic/gradient_c2l.c $(SRCROOT)shared/mpp/test_mpp_io.F90 $(SRCROOT)shared/diag_manager/diag_grid.F90 $(SRCROOT)shared/mpp/mpp_domains.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_define.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mosaic/interp.h $(SRCROOT)shared/diag_manager/diag_data.F90 $(SRCROOT)atmos_cubed_sphere/model/a2b_edge.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_data_mpi.inc $(SRCROOT)shared/mpp/mpp_memutils.F90 $(SRCROOT)shared/mpp/mpp_parameter.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_util.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_check.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_a_a.h $(SRCROOT)atmos_param/lin_cloud_microphys/lin_cloud_microphys.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_read.inc $(SRCROOT)shared/fms/test_fms_io.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_diagnostics.F90 $(SRCROOT)shared/horiz_interp/horiz_interp_spherical.F90 $(SRCROOT)shared/memutils/memutils.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/write_data.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_global_field.h $(SRCROOT)shared/constants/constants.F90 $(SRCROOT)shared/tracer_manager/tracer_manager.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_read_2Ddecomp.h $(SRCROOT)shared/mpp/test_mpp.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_s.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_write.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_reduce_mpi.h $(SRCROOT)shared/horiz_interp/horiz_interp_bilinear.F90 $(SRCROOT)shared/mosaic/mosaic_util.c $(SRCROOT)shared/mosaic/read_mosaic.c $(SRCROOT)atmos_cubed_sphere/model/fv_mapz.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_nudge.F90 $(SRCROOT)shared/mosaic/interp.c $(SRCROOT)atmos_cubed_sphere/tools/sorted_index.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_domains_util.inc $(SRCROOT)shared/mpp/threadloc.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_sma.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm_sma.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_get_boundary.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_write_2Ddecomp.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_checkV.h $(SRCROOT)atmos_cubed_sphere/driver/solo/fv_phys.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/fms/read_data_3d.inc $(SRCROOT)shared/time_interp/time_interp.F90 $(SRCROOT)atmos_cubed_sphere/tools/fv_eta.F90 $(SRCROOT)shared/horiz_interp/horiz_interp.F90 $(SRCROOT)shared/diag_manager/diag_output.F90 $(SRCROOT)atmos_cubed_sphere/model/fv_control.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_error_s_a.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_mpi.h $(SRCROOT)shared/mpp/nsclock.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_transmit_nocomm.h $(SRCROOT)shared/mosaic/grid.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_comm.inc /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/system_clock.h $(SRCROOT)atmos_solo/atmos_model.F90 $(SRCROOT)atmos_cubed_sphere/tools/external_sst.F90 /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_sum_nocomm.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_do_get_boundary.h $(SRCROOT)atmos_cubed_sphere/tools/fv_mp_mod.F90 $(SRCROOT)shared/mosaic/create_xgrid.c /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_chksum_int.h /work/home/ac0w1vpw3p/jibinghu_fv3/fv3_source/ics_231207_checked/exp/../src/shared/mpp/include/mpp_io_connect.inc
clean: neat
        -rm -f .fms.x.cppdefs $(OBJ) fms.x
neat:
        -rm -f $(TMPFILES)
localize: $(OFF)
        cp $(OFF) .
TAGS: $(SRC)
        etags $(SRC)
tags: $(SRC)
        ctags $(SRC)
fms.x: $(OBJ) 
        $(LD) $(OBJ) -o fms.x  $(LDFLAGS)
[ac0w1vpw3p@login01 exec.amd64]$ 
```
