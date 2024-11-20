FV3（Finite-Volume Cubed-Sphere）是美国国家海洋和大气管理局（NOAA）地球流体动力学实验室（GFDL）开发的下一代全球预报系统的动力核心模型。该模型采用立方球有限体积方法，旨在提高天气预报的准确性和效率。以下是关于FV3的论文、文档和代码资源：

论文：
1. Lin, S.-J. and Rood, R. B. (1996): “Multidimensional Flux-Form Semi-Lagrangian Transport Schemes.” Monthly Weather Review, 124(9), 2046–2070.
2. Lin, S.-J. (2004): “A ‘Vertically Lagrangian’ Finite-Volume Dynamical Core for Global Models.” Monthly Weather Review, 132(10), 2293–2307.
3. Putman, W. M. and Lin, S.-J. (2007): “Finite-Volume Transport on Various Cubed-Sphere Grids.” Journal of Computational Physics, 227(1), 55–78.

文档：
1. FV3: Finite-Volume Cubed-Sphere Dynamical Core：GFDL提供的FV3模型的详细介绍，包括其发展历史、基本算法和应用等。https://www.gfdl.noaa.gov/fv3/ ￼
2. FV3 Documentation and References：该页面汇总了FV3的科学文档、技术说明和相关参考资料，供研究人员深入了解FV3模型。https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/

代码：
FV3的源代码在GitHub上公开，您可以通过以下链接获取：
- [NOAA-GFDL/GFDL_atmos_cubed_sphere](https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere)：该仓库包含FV3动力核心的源代码，以及相关的构建和使用指南。 ￼
- [NOAA-EMC/fv3atm](https://github.com/NOAA-EMC/fv3atm)：该仓库包含NOAA统一预报系统（UFS）中大气分量的驱动程序和关键子组件，包括FV3动力核心。 ￼

Held, I. M., & Suarez, M. J. (1994).
A proposal for the intercomparison of the dynamical cores of atmospheric general circulation models.
Bulletin of the American Meteorological Society, 75(10), 1825–1830.
[DOI:10.1175/1520-0477(1994)075<1825:APFTIO>2.0.CO;2](https://doi.org/10.1175/1520-0477(1994)075%3C1825:APFTIO%3E2.0.CO;2)

论文概要

这篇论文提出了一种用于评估和比较大气环流模型（GCMs）动力核心的测试方法，后来被称为 Held-Suarez 测试。测试设计为理想化的数值实验，以消除物理过程的复杂性，从而专注于模型的动力核心性能。

测试的核心思想

	1.	理想化大气：采用简单的恒定强迫，忽略了许多复杂的物理过程（如湿物理和辐射），将模型的重点放在动力学上。
	2.	主要物理假设：
	•	干燥大气：忽略水汽及其相关的过程。
	•	恒温层：对流层设定恒温分布，顶层设为恒温层。
	•	辐散制约：采用线性加热与冷却强迫，模拟热量分布。
	3.	用来分析的指标：
	•	通过平衡态的速度场、温度分布、能量收支等指标，评估模型的动力学性能。

Held-Suarez 测试的意义

	•	标准化对比工具：为不同动力核心的对比提供了统一的标准，广泛用于新型动力核心（如 FV3）的验证。
	•	开发工具：帮助模型开发人员专注于改进动力学核心，而不受其他复杂物理参数化的影响。