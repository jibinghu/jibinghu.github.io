查找特定文件/文件夹，筛选所选信息并将重点信息标红：

`find / -name "* search_content*" 2>/dev/null | grep --color=always "search_content"`


`find / -name "*search_content*" -printf "\033[1;31m%p\033[0m\n" 2>/dev/null`

这里将 find 的输出高亮设置为红色 (\033[1;31m)，可以根据需要修改颜色。