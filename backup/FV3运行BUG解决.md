``` bash
ERROR: Unable to locate a modulefile for 'compiler/cmake/3.22.0-rc1'
ERROR: Unable to locate a modulefile for 'apps/anaconda3/5.2.0'
ERROR: Unable to locate a modulefile for 'mathlib/fftw/3.3.8/double/intel'
ERROR: Unable to locate a modulefile for 'compiler/rocm/dtk-22.04.2'
ERROR: Unable to locate a modulefile for 'mathlib/netcdf/4.6.2/intel'
ERROR: Unable to locate a modulefile for 'mathlib/hdf5/1.8.12/intel'
srun: error: a01r4n05: task 3: Segmentation fault (core dumped)
srun: launch/slurm: _step_signal: Terminating StepId=2417.0
slurmstepd: error: *** STEP 2417.0 ON a01r4n02 CANCELLED AT 2024-12-04T22:23:57 ***
srun: error: a01r4n02: task 0: Segmentation fault (core dumped)
srun: error: a01r4n04: task 2: Segmentation fault (core dumped)
srun: error: a01r4n03: task 1: Segmentation fault (core dumped)
[mpiexec@a01r4n02] wait_proxies_to_terminate (../../../../../src/pm/i_hydra/mpiexec/intel/i_mpiexec.c:537): downstream from host a01r4n02 exited with status 139
[mpiexec@a01r4n02] main (../../../../../src/pm/i_hydra/mpiexec/mpiexec.c:2125): assert (pg->intel.exitcodes != NULL) failed
[mpiexec@a01r4n02] HYD_sock_write (../../../../../src/pm/i_hydra/libhydra/sock/hydra_sock_intel.c:360): write error (Bad file descriptor)
[mpiexec@a01r4n02] HYD_sock_write (../../../../../src/pm/i_hydra/libhydra/sock/hydra_sock_intel.c:360): write error (Bad file descriptor)
[mpiexec@a01r4n02] HYD_sock_write (../../../../../src/pm/i_hydra/libhydra/sock/hydra_sock_intel.c:360): write error (Bad file descriptor)
[mpiexec@a01r4n02] HYD_sock_write (../../../../../src/pm/i_hydra/libhydra/sock/hydra_sock_intel.c:360): write error (Bad file descriptor)
```