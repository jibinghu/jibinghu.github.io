`find . -type f -name "input.nml"`
可以加一个反向过滤：
`find . -type f -name "*搜索的内容*" | grep -r "删除的内容" > reserve.log`
