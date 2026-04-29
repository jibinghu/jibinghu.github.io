查看所有节点的分区信息：

``` bash
[acox11ust1@login02 pubrel]$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
work         up   infinite      3 drain* a01r4n[35,38-39]
work         up   infinite      1   drng a01r4n33
work         up   infinite      3  drain a01r4n[25,34,37]
work         up   infinite      1    mix a01r4n26
work         up   infinite      8  alloc a01r4n[00-01,27-32]
work         up   infinite     24   idle a01r4n[02-24,36]
[acox11ust1@login02 pubrel]$ Web console: https://a01r2n00:9090/
```

查看当前节点的分区信息：

`sinfo -N -n $(hostname)`