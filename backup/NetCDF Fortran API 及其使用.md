NetCDF（Network Common Data Form）是一种用于科学数据存储和共享的自描述、网络透明、直接访问和可扩展的文件格式。它支持多种编程语言，包括 Fortran。在 Fortran 中使用 NetCDF 主要通过 netCDF-Fortran 库，该库提供了对 NetCDF C 库的 Fortran 接口。

> sudo apt-get install libnetcdff-dev

常用 NetCDF Fortran API 函数

1. 文件操作

创建文件：nf90_create
打开文件：nf90_open
关闭文件：nf90_close

2. 定义模式（Define Mode）

定义维度：nf90_def_dim
定义变量：nf90_def_var
定义属性：nf90_put_att
结束定义模式：nf90_enddef

3. 数据操作

写入变量数据：nf90_put_var
读取变量数据：nf90_get_var
写入属性数据：nf90_put_att
读取属性数据：nf90_get_att



示例：

``` fortran
program netcdf_example
  use netcdf
  implicit none

  integer :: ncid, varid, status
  integer, parameter :: NX = 10, NY = 10
  integer :: data(NX, NY)
  integer :: i, j

  ! 创建 NetCDF 文件
  status = nf90_create('example.nc', nf90_clobber, ncid)
  if (status /= nf90_noerr) stop 'Error creating file'

  ! 定义维度
  status = nf90_def_dim(ncid, 'x', NX, dimid_x)
  status = nf90_def_dim(ncid, 'y', NY, dimid_y)

  ! 定义变量
  status = nf90_def_var(ncid, 'data', nf90_int, [dimid_x, dimid_y], varid)

  ! 结束定义模式
  status = nf90_enddef(ncid)

  ! 写入数据
  data = reshape([(i-1)*NY + j, i=1, NX, j=1, NY])
  status = nf90_put_var(ncid, varid, data)

  ! 关闭文件
  status = nf90_close(ncid)

  ! 重新打开文件读取数据
  status = nf90_open('example.nc', nf90_nowrite, ncid)
  status = nf90_inq_varid(ncid, 'data', varid)
  status = nf90_get_var(ncid, varid, data)
  status = nf90_close(ncid)

  ! 输出读取的数据
  print *, 'Data from NetCDF file:'
  do i = 1, NX
     print *, data(i, :)
  end do

end program netcdf_example
```

`gfortran -o netcdf_example netcdf_example.f90 -I/usr/local/include -L/usr/local/lib -lnetcdff`
