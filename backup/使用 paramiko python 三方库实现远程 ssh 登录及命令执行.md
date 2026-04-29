``` python
import os
import paramiko
from typing import Optional, Tuple

def run_ssh_cmd_key(
    host: str,
    port: int,
    username: str,
    cmd: str,
    *,
    key_path: Optional[str] = None,         
    key_passphrase: Optional[str] = None, 
    timeout: int = 15,
    use_sudo: bool = False,
    sudo_password: Optional[str] = None, 
    allow_agent: bool = True,         
    look_for_keys: bool = True,
) -> Tuple[int, str, str]:
    """
    通过 SSH 私钥执行远程命令。
    返回 (exit_code, stdout_str, stderr_str)
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 关键改动：不手动加载 PKey，交给 Paramiko 自动识别
    key_filename = os.path.expanduser(key_path) if key_path else None

    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            key_filename=key_filename,
            passphrase=key_passphrase, 
            timeout=10,
            allow_agent=allow_agent,
            look_for_keys=look_for_keys,
            compress=True,
        )

        real_cmd = cmd
        if use_sudo:
            # 更安全：目标机 /etc/sudoers 配置 NOPASSWD；否则这里要喂密码
            real_cmd = f"sudo -S -p '' {cmd}"

        get_pty = bool(use_sudo and sudo_password)
        stdin, stdout, stderr = client.exec_command(real_cmd, timeout=timeout, get_pty=get_pty)

        if use_sudo and sudo_password:
            stdin.write(sudo_password + "\n")
            stdin.flush()

        out = stdout.read().decode(errors="ignore")
        err = stderr.read().decode(errors="ignore")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        client.close()


if __name__ == "__main__":
    host = "192.168.0.1"
    port = 22
    user = "pi"

    rc, out, err = run_ssh_cmd_key(
        host, port, user, "uname -a",
        key_path="~/.ssh/id_rsa.pub",
        key_passphrase=None,
        use_sudo=False,
    )
    print("rc=", rc)
    print("stdout:", out)
    print("stderr:", err)
```