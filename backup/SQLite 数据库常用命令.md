SQLite 数据库常用命令：

``` sql
-- 1. 打开数据库文件
sqlite3 database.db

-- 2. 显示所有表
.tables

-- 3. 查看数据库结构（所有表的定义）
.schema

-- 4. 查看某张表的结构
.schema table_name

-- 5. 查看某张表的全部数据
SELECT * FROM table_name;

-- 6. 查看前 N 行数据
SELECT * FROM table_name LIMIT N;

-- 7. 统计表中的记录数
SELECT COUNT(*) FROM table_name;

-- 8. 条件查询
SELECT * FROM table_name WHERE column_name = 'value';

-- 9. 排序查询
SELECT * FROM table_name ORDER BY column_name ASC;   -- 升序
SELECT * FROM table_name ORDER BY column_name DESC;  -- 降序

-- 10. 分组统计
SELECT column_name, COUNT(*) FROM table_name GROUP BY column_name;

-- 11. 修改数据
UPDATE table_name SET column_name = 'new_value' WHERE condition;

-- 12. 删除数据
DELETE FROM table_name WHERE condition;

-- 13. 导出查询结果到 CSV 文件
.mode csv
.output output_file.csv
SELECT * FROM table_name;
.output stdout  -- 恢复输出到屏幕

-- 14. 导入 CSV 文件到表
.mode csv
.import input_file.csv table_name

-- 15. 设置输出格式
.mode column  -- 默认文本输出
.mode csv     -- CSV 格式输出
.mode table   -- 表格格式输出

-- 16. 显示查询结果的标题
.headers on

-- 17. 查看当前数据库文件
.databases

-- 18. 清屏命令（适用于支持的终端）
.system clear

-- 19. 显示帮助信息
.help

-- 20. 退出 SQLite
.exit
.quit```