### Firewall-cmd 简介

`firewall-cmd` 是一个用于管理 Linux 系统上的 Firewalld 防火墙的命令行工具。Firewalld 是一种动态防火墙管理工具，基于 **zones（区域）** 和 **services（服务）** 模型，允许用户轻松定义规则来控制网络流量的进出。

---

### 基本概念

1. **Zones（区域）**  
   Firewalld 使用区域来定义网络接口的信任级别。例如，可以有 `public` 区域、`home` 区域等，每个区域有不同的网络访问控制策略。

2. **Services（服务）**  
   Firewalld 通过定义服务（如 HTTP、SSH、FTP 等）来管理不同应用的网络访问。

3. **Direct rules（直接规则）**  
   这些规则可以绕过区域模型，允许用户直接配置防火墙规则。

---

### 常用 `firewall-cmd` 命令

#### 1. 查看防火墙状态
- **查看当前防火墙状态（是否启用）**  
  ```bash
  firewall-cmd --state
  ```
  **输出**：`running` 表示防火墙正在运行。

- **查看防火墙的规则和区域**  
  ```bash
  firewall-cmd --list-all
  ```

---

#### 2. 查看当前区域
- **查看当前活动的区域**  
  ```bash
  firewall-cmd --get-default-zone
  ```

- **查看某个区域的规则**  
  ```bash
  firewall-cmd --zone=public --list-all
  ```

---

#### 3. 修改防火墙配置
- **永久更改防火墙配置并使其生效**  
  ```bash
  firewall-cmd --zone=public --add-service=http --permanent
  ```

- **临时添加服务**  
  ```bash
  firewall-cmd --zone=public --add-service=http
  ```

- **删除服务**  
  ```bash
  firewall-cmd --zone=public --remove-service=http --permanent
  ```

---

#### 4. 启用和禁用防火墙服务
- **启用防火墙服务**  
  ```bash
  systemctl start firewalld
  ```

- **禁用防火墙服务**  
  ```bash
  systemctl stop firewalld
  ```

- **设置防火墙开机启动**  
  ```bash
  systemctl enable firewalld
  ```

- **禁止防火墙开机启动**  
  ```bash
  systemctl disable firewalld
  ```

---

#### 5. 查看所有已开启的服务
- **查看所有已开启的服务（临时）**  
  ```bash
  firewall-cmd --list-services
  ```

- **查看所有已开启的服务（永久）**  
  ```bash
  firewall-cmd --permanent --list-services
  ```

---

#### 6. 添加和删除端口
- **临时开放端口**  
  ```bash
  firewall-cmd --zone=public --add-port=8080/tcp
  ```

- **永久开放端口**  
  ```bash
  firewall-cmd --zone=public --add-port=8080/tcp --permanent
  ```

- **临时关闭端口**  
  ```bash
  firewall-cmd --zone=public --remove-port=8080/tcp
  ```

- **永久关闭端口**  
  ```bash
  firewall-cmd --zone=public --remove-port=8080/tcp --permanent
  ```

---

#### 7. 重新加载防火墙配置
- **重新加载防火墙配置**  
  ```bash
  firewall-cmd --reload
  ```

---

#### 8. 查看区域配置
- **查看默认区域**  
  ```bash
  firewall-cmd --get-default-zone
  ```

- **列出所有区域**  
  ```bash
  firewall-cmd --list-all-zones
  ```

---

#### 9. 设定区域和接口
- **临时将接口（例如 eth0）分配到某个区域**  
  ```bash
  firewall-cmd --zone=public --change-interface=eth0
  ```

- **永久将接口分配到某个区域**  
  ```bash
  firewall-cmd --zone=public --change-interface=eth0 --permanent
  ```

---

#### 10. 查看直通规则（直接规则）
- **列出直接规则**  
  ```bash
  firewall-cmd --direct --get-all-rules
  ```

- **添加直接规则**  
  ```bash
  firewall-cmd --direct --add-rule ipv4 filter INPUT 0 -s 192.168.1.1 -j ACCEPT --permanent
  ```

---

#### 11. 禁用和启用防火墙（临时）
- **临时禁用防火墙（不会改变防火墙的启动设置）**  
  ```bash
  firewall-cmd --zone=public --remove-service=ssh
  ```

---

#### 12. 验证防火墙规则
- **验证某项规则是否生效**  
  ```bash
  firewall-cmd --zone=public --query-service=http
  ```
  **输出**：如果该服务存在，则返回 `yes`，否则返回 `no`。

---

### 总结
- `firewall-cmd` 是一个管理 Firewalld 防火墙的命令行工具，支持添加/删除服务、端口、区域和直接规则。
- 通过 `--permanent` 参数可以使更改永久生效，`--reload` 用于重新加载配置。
- 通过 **区域（zones）** 和 **服务（services）** 管理不同的流量策略，能够灵活设置防火墙的访问控制。

---

# 辨析：

### Firewalld、iptables 和 UFW 对比

Firewalld、iptables 和 UFW 都是 Linux 系统中用于管理网络流量和防火墙规则的工具，但它们在设计理念、使用方式和配置灵活性上有显著区别。以下是它们的主要区别：

---

### 1. **iptables**

iptables 是一个低级别的、命令行的防火墙工具，直接操作 Linux 内核中的 Netfilter 框架来过滤网络数据包。

#### 特点：
- **底层控制**：iptables 提供了非常细粒度的控制，允许用户配置具体的网络规则（如允许或拒绝某个 IP 地址或端口的访问）。
- **规则定义**：用户通过命令行规则定义来设置防火墙。规则按顺序检查，直到匹配成功。它有多个表（如 `filter`、`nat`、`mangle`）和链（如 `INPUT`、`FORWARD`、`OUTPUT`）。
- **复杂性**：iptables 适合需要深度自定义规则的高级用户，规则的管理和维护可能比较复杂。
- **不支持动态管理**：配置更改后需要重启防火墙服务，且更改往往需要保存到规则文件，否则会丢失。

#### 示例：
```bash
# 允许端口 22（SSH）上的入站连接
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
```

#### 适用场景：
- 高级用户需要对防火墙进行精细化配置，或在复杂的环境中使用。
- 系统需要进行定制化防火墙规则管理。

---

### 2. **UFW（Uncomplicated Firewall）**

UFW 是一个更为简单和用户友好的防火墙工具，通常作为 iptables 的前端。它封装了 iptables 的复杂性，提供了易用的命令行接口。

#### 特点：
- **简化接口**：UFW 提供了简单易用的命令，使得管理防火墙规则变得直观和简单。对于大多数用户，UFW 提供了足够的功能。
- **默认策略**：UFW 默认会拒绝所有入站流量，允许所有出站流量，用户可以通过规则来更改这一默认策略。
- **易于配置**：用户不需要记住复杂的 iptables 语法，可以使用简洁的命令来管理规则。
- **不如 iptables 灵活**：UFW 是 iptables 的封装器，因此它没有 iptables 那样底层的灵活性，更多适合个人使用或者简单的防火墙需求。

#### 示例：
```bash
# 允许 SSH 连接
ufw allow ssh

# 开启防火墙
ufw enable
```

#### 适用场景：
- 对于普通用户或没有复杂需求的场景，UFW 提供了易用的接口来配置防火墙。
- 适合用于桌面系统、开发环境和简单服务器。

---

### 3. **Firewalld**

Firewalld 是一个现代的防火墙管理工具，基于 iptables 和 nftables（新的防火墙框架），它引入了区域（zones）和服务（services）模型，旨在简化防火墙的管理，同时保留一定的灵活性。

#### 特点：
- **动态管理**：Firewalld 支持动态规则更新，无需重启防火墙服务，能实时应用更改。这意味着你可以随时修改防火墙规则，而不会中断现有连接。
- **区域和服务模型**：Firewalld 使用“区域”来定义不同的网络信任级别，每个区域可以有不同的规则。此外，Firewalld 使用服务的概念（如 HTTP、SSH 等），使得规则更加高层次的抽象。
- **简易的命令行接口**：相比 iptables，Firewalld 提供了更易于使用的命令行工具（`firewall-cmd`）。
- **支持 nftables**：Firewalld 可以与 nftables 配合工作，提供比 iptables 更高效的包过滤性能（但仍然保持对 iptables 的兼容）。

#### 示例：
```bash
# 允许端口 22（SSH）上的入站连接
firewall-cmd --zone=public --add-port=22/tcp --permanent
firewall-cmd --reload
```

#### 适用场景：
- 需要较为简便且灵活的防火墙配置和管理，特别适用于系统管理员和中小型企业服务器。
- 支持动态规则更新的场景。
- 适合需要基于区域和服务进行管理的场景（例如企业级应用、数据中心环境）。

---

### 总结对比

| **特性**           | **iptables**                          | **UFW**                                | **Firewalld**                          |
|--------------------|---------------------------------------|----------------------------------------|----------------------------------------|
| **配置复杂度**      | 高，灵活性强                          | 低，简化了 iptables 的规则管理          | 中等，灵活性较高，但比 iptables 简单   |
| **管理方式**        | 直接操作 Netfilter，规则复杂           | 简单的命令行接口，封装了 iptables       | 动态管理，基于区域和服务的抽象模型     |
| **规则存储方式**    | 静态，修改后需保存                    | 静态，修改后需保存                     | 动态，实时生效，但也可以永久配置       |
| **适用对象**        | 高级用户，网络管理员                  | 普通用户，桌面和小型服务器             | 系统管理员，中小型企业服务器，动态管理 |
| **是否支持动态更新**| 不支持动态更新，重启服务才生效        | 不支持动态更新，重启服务才生效         | 支持动态更新，实时生效                 |
| **底层支持**        | 基于 iptables 和 nftables             | 封装 iptables                          | 基于 iptables 和 nftables              |

---

### 选择哪个工具？
- **iptables**：如果你需要精细控制、防火墙配置非常复杂，或者你是在老旧的系统上工作，iptables 仍然是一个很强大的选择。
- **UFW**：如果你只是需要一个简单的防火墙，尤其是在个人电脑或不需要复杂配置的情况下，UFW 是一个非常简单、易用的工具。
- **Firewalld**：如果你希望有更现代的防火墙管理工具，支持动态更新并且能够使用区域和服务模型进行配置，Firewalld 是一个理想的选择，特别适用于中大型企业和数据中心。