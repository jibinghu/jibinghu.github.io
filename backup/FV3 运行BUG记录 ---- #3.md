``` bash
[acox11ust1@login02 pubrel]$ ./fms.x mpirun -np=24
[0] MPI startup(): FI_PSM3_UUID was not generated, please set it to avoid possible resources ownership conflicts between MPI processes
login02:pid12135.fms.x: unknown link speed 0x80
login02:pid12135.fms.x: Failed to modify UD QP to INIT on mlx5_0: Operation not permitted
Abort(1615503) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init: Other MPI error, error stack:
MPIR_Init_thread(138)........: 
MPID_Init(1169)..............: 
MPIDI_OFI_mpi_init_hook(1807): 
create_endpoint(2473)........: OFI endpoint open failed (ofi_init.c:2473:create_endpoint:Invalid argument)
```

> 遇事不决先问GPT：

---

1. FI_PSM3_UUID 问题
   错误信息中提到 FI_PSM3_UUID 没有生成。FI_PSM3_UUID 是用于 InfiniBand 网络的标识符。如果没有正确配置它，可能会导致 MPI 进程之间的资源冲突。
   
   解决方法：
   1. 设置 FI_PSM3_UUID 环境变量。可以通过以下命令来设置它：
      ```bash
      export FI_PSM3_UUID=<your_uuid>
      ```
   2. 如果不清楚如何生成 UUID，您可以参考相关的网络配置文档，或者联系您的系统管理员来获取该值。

2. 未知链接速度
   错误信息 unknown link speed 0x80 提示 MPI 无法识别链接的速度。
   
   解决方法：
   1. 确认你的网络硬件（如 InfiniBand 卡）是否正确配置并且驱动程序已安装。
   2. 检查 InfiniBand 设备的状态和驱动程序，可以通过以下命令检查：
      ```bash
      ibv_devinfo
      ```
   3. 如果硬件或驱动出现问题，可以尝试重新启动硬件或更新驱动。

3. 权限问题：Failed to modify UD QP to INIT on mlx5_0
   错误信息 Failed to modify UD QP to INIT on mlx5_0: Operation not permitted 表明程序没有足够的权限来修改 InfiniBand 设备的配置。
   
   解决方法：
   1. 使用管理员权限（sudo）来执行 MPI 程序：
      ```bash
      sudo ./fms.x mpirun -np 24
      ```
   2. 如果您没有管理员权限，联系系统管理员，确认您的用户是否有访问 InfiniBand 设备的权限。

4. OFI 端点创建失败
   错误信息 OFI endpoint open failed 表示在初始化过程中尝试创建 OFI（OpenFabrics Interfaces）端点时失败，通常是因为参数无效。
   
   解决方法：
   1. 检查 MPI 和 OFI 库是否正确安装和配置。确保你使用的版本是兼容的。
   2. 查看 OFI 的配置文件和日志，确保所有必要的配置都已经正确设置。
   3. 可以尝试更新 MPI 或 OFI 版本，或者切换到其他适配的网络库。

5. MPI 环境配置
   如果问题持续存在，可能需要检查 MPI 环境配置，包括网络和库的配置。
   
   解决方法：
   1. 重新配置和编译 MPI 程序，确保选用正确的网络插件（如 psm3、ofi 等）。
   2. 确认 MPI 程序的参数和选项是否正确，尤其是关于网络接口的配置。

综上所述，解决步骤可以包括：
1. 设置 FI_PSM3_UUID 环境变量。
2. 检查并更新 InfiniBand 驱动和配置。
3. 确认程序是否有足够的权限来操作硬件。
4. 检查并更新 MPI 和 OFI 配置，确保使用兼容的网络库。

---

检查存在 Infiniband：
```
[acox11ust1@login02 pubrel]$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      e612:e861:1ff9:2c9a
        sys_image_guid:                 e612:e861:1ff9:2c9a
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       SUM0000000002
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 1042
                        port_lid:               318
                        port_lmc:               0x00
                        link_layer:             InfiniBand

hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      e612:e861:1ff9:2c9c
        sys_image_guid:                 e612:e861:1ff9:2c9a
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       SUM0000000002
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 1042
                        port_lid:               109
                        port_lmc:               0x00
                        link_layer:             InfiniBand

hca_id: mlx5_2
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      e612:e861:1ff9:2c9e
        sys_image_guid:                 e612:e861:1ff9:2c9a
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       SUM0000000002
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 1042
                        port_lid:               110
                        port_lmc:               0x00
                        link_layer:             InfiniBand

hca_id: mlx5_3
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      e612:e861:1ff9:2ca0
        sys_image_guid:                 e612:e861:1ff9:2c9a
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       SUM0000000002
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 1042
                        port_lid:               111
                        port_lmc:               0x00
                        link_layer:             InfiniBand
```