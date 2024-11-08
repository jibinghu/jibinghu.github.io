但正常使用 `git push origin` 向 repo 推送时发现报错如下：

``` bash
[main da5bafe] 11/8/01
 3 files changed, 55 insertions(+)
 create mode 100644 cpp_prac/Cpython_bind/Cpython.cpp
 create mode 100644 cpp_prac/Cpython_bind/Cpython.py
 create mode 100755 cpp_prac/Cpython_bind/demo_module.so
(py310) binghu@iscashpc:~/leetcode$ git push origin
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1006/vscode-git-8f5a0a4c40.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1607:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1006/vscode-git-8f5a0a4c40.sock'
}
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1006/vscode-git-8f5a0a4c40.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1607:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1006/vscode-git-8f5a0a4c40.sock'
}
remote: No anonymous write access.
fatal: Authentication failed for 'https://github.com/jibinghu/leetcode/'
```

显然是身份验证不通过，但使用 `ssh -T git@github.com` remote github时返回：

`Hi jibinghu! You've successfully authenticated, but GitHub does not provide shell access.`

说明密钥是正确的，`git remote -v` 检查：

``` bash
origin  https://github.com/jibinghu/leetcode (fetch)
origin  https://github.com/jibinghu/leetcode (push)
```

> 身份验证已经通过。如果你在使用 Git 时遇到问题，可能是因为你在尝试通过 HTTPS 而不是 SSH 进行推送。为了使用 SSH，确保你的远程仓库 URL 使用的是 git@github.com 格式，而不是 https://github.com 格式。

`git remote set-url origin git@github.com:jibinghu/leetcode.git` 后再 `git push origin main`，access！ 