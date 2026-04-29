fs_cli

`show dialplan`
> 展示拨号计划（dialplan）模块
```
输出：
freeswitch@wanren4090> show dialplan
type,name,ikey
dialplan,LUA,mod_lua
dialplan,XML,mod_dialplan_xml
dialplan,XUI,mod_xui
dialplan,inline,mod_dptools

4 total.
```

`show modules`
> 展示已加载的（modules）模块
```
输出（部分）：
xswitch>  show modules
type,name,ikey,filename
api,...,mod_commands,/usr/local/freeswitch/mod/mod_commands.so
api,acl,mod_commands,/usr/local/freeswitch/mod/mod_commands.so
api,ai,mod_ai,/usr/local/freeswitch/mod/mod_ai.so
api,ali_token,mod_ali,/usr/local/freeswitch/mod/mod_ali.so
api,alias,mod_commands,/usr/local/freeswitch/mod/mod_commands.so
api,av,mod_av,/usr/local/freeswitch/mod/mod_av.so
```