# Linux / RDK X5 / Windows 静态 IP 与静态路由配置教程

> 适用目标：把设备网卡配置到 `172.29.254.0/24` 网段，并添加到 `10.18.0.0/16` 的静态路由。  
> 示例路由等价于 Windows 命令：  
> `route add 10.18.0.0 mask 255.255.0.0 172.29.254.1`

---

## 0. 示例网络规划

本文统一使用下面这组示例。实际使用时请按你的环境替换。

| 项目 | 示例值 | 说明 |
|---|---:|---|
| RDK X5 / Linux 网卡 | `eth0` | 开发板上的有线网卡 |
| RDK X5 / Linux IP | `172.29.254.5` | 开发板静态 IP |
| Windows 有线网卡 IP | `172.29.254.6` | 连接 RDK X5 的 Windows 网卡，避免和开发板冲突 |
| 子网掩码 | `255.255.255.0` | 等价 CIDR：`/24` |
| 下一跳 / 网关 | `172.29.254.1` | 访问 `10.18.0.0/16` 时的下一跳 |
| 目标网段 | `10.18.0.0/16` | 等价掩码：`255.255.0.0` |

**重要提醒：同一个二层网络中不能有两个设备使用同一个 IP。**

例如：

- 如果 RDK X5 已经配置为 `172.29.254.5`；
- Windows 连接 RDK X5 的有线网卡就不要再配置成 `172.29.254.5`；
- Windows 可以配置成 `172.29.254.6`、`172.29.254.10` 等未被占用的地址。

---

## 1. 路由概念速查

### 1.1 Windows 命令和 Linux 命令的对应关系

Windows：

```bat
route add 10.18.0.0 mask 255.255.0.0 172.29.254.1
```

Linux：

```bash
sudo ip route add 10.18.0.0/16 via 172.29.254.1 dev eth0
```

含义相同：

```text
访问 10.18.0.0/16 这个网段时，下一跳走 172.29.254.1。
```

### 1.2 子网掩码和 CIDR 对照

| 子网掩码 | CIDR 写法 |
|---|---:|
| `255.255.255.0` | `/24` |
| `255.255.0.0` | `/16` |
| `255.0.0.0` | `/8` |

所以：

```text
172.29.254.5/24       = 172.29.254.5 + 255.255.255.0
10.18.0.0/16          = 10.18.0.0 + 255.255.0.0
```

---

# 第一部分：Linux / RDK X5 配置

---

## 2. Linux 临时配置方式：使用 `ip` 命令

这种方式 **立即生效**，但 **重启后会丢失**。适合测试网络是否可用。

### 2.1 查看当前网卡状态

```bash
# 查看 eth0 的 IP、MAC、链路状态等信息
ip addr show eth0

# 查看当前 IPv4 路由表
ip route
```

示例输出可能类似：

```text
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
    inet 192.168.127.10/24 brd 192.168.127.255 scope global eth0
```

---

### 2.2 方式 A：直接替换 eth0 的 IP

> 注意：如果你正在通过 `eth0` 的旧 IP 远程 SSH 登录，执行 `flush` 后 SSH 可能会立刻断开。

```bash
# 清除 eth0 上已有的 IPv4 地址
sudo ip addr flush dev eth0

# 给 eth0 添加新的静态 IP，/24 等价于 255.255.255.0
sudo ip addr add 172.29.254.5/24 dev eth0

# 确保 eth0 处于启用状态
sudo ip link set eth0 up

# 添加静态路由：访问 10.18.0.0/16 时走 172.29.254.1
sudo ip route add 10.18.0.0/16 via 172.29.254.1 dev eth0
```

如果提示路由已存在：

```text
RTNETLINK answers: File exists
```

可以使用 `replace` 覆盖：

```bash
# 如果路由存在则更新，不存在则添加
sudo ip route replace 10.18.0.0/16 via 172.29.254.1 dev eth0
```

---

### 2.3 方式 B：远程操作时更安全的写法

如果你是 SSH 远程操作，建议先 **追加新 IP**，确认能通后再删除旧 IP。

```bash
# 先给 eth0 追加一个新 IP，不删除旧 IP
sudo ip addr add 172.29.254.5/24 dev eth0

# 添加或更新静态路由
sudo ip route replace 10.18.0.0/16 via 172.29.254.1 dev eth0

# 查看 eth0 当前是否已经有 172.29.254.5/24
ip addr show eth0

# 确认网络可用后，再删除旧 IP，例如原来是 192.168.127.10/24
sudo ip addr del 192.168.127.10/24 dev eth0
```

---

### 2.4 删除临时配置

```bash
# 删除静态路由
sudo ip route del 10.18.0.0/16 via 172.29.254.1 dev eth0

# 删除 eth0 上的静态 IP
sudo ip addr del 172.29.254.5/24 dev eth0
```

---

## 3. RDK X5 永久配置方式：修改 NetworkManager 配置文件

RDK X5 官方文档推荐修改：

```text
/etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

也就是说，RDK X5 上优先使用 **NetworkManager 的 `.nmconnection` 文件**，不要优先去新建 `/etc/netplan/01-netcfg.yaml`。

---

### 3.1 备份原配置

```bash
# 先备份，改错时可以恢复
sudo cp /etc/NetworkManager/system-connections/netplan-eth0.nmconnection \
        /etc/NetworkManager/system-connections/netplan-eth0.nmconnection.bak
```

---

### 3.2 编辑配置文件

```bash
sudo vim /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

找到 `[ipv4]` 段。

原始示例可能是：

```ini
[ipv4]
address1=192.168.127.10/24,192.168.127.1
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
```

改成下面这样：

```ini
[ipv4]
address1=172.29.254.5/24,172.29.254.1
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
route1=10.18.0.0/16,172.29.254.1
```

字段解释：

| 字段 | 示例 | 说明 |
|---|---|---|
| `address1` | `172.29.254.5/24,172.29.254.1` | 配置 eth0 静态 IP 和默认网关 |
| `dns` | `8.8.8.8;8.8.4.4;` | DNS 服务器，多个 DNS 用分号分隔，末尾通常也带分号 |
| `method` | `manual` | 手动配置 IPv4，不使用 DHCP |
| `route-metric` | `700` | 路由优先级，数字越大优先级越低 |
| `route1` | `10.18.0.0/16,172.29.254.1` | 静态路由，访问 `10.18.0.0/16` 走 `172.29.254.1` |

---

### 3.3 推荐完整配置示例

> 下面是可复制版本。不要在实际 `.nmconnection` 文件里随便加入中文注释。
> 
> 注意：`uuid` 建议保留你原文件里的值，不要在多份连接配置中重复使用同一个 `uuid`。

```ini
[connection]
id=netplan-eth0
uuid=f6f8b5a7-9e23-49b2-a792-dc589b3d3e88
type=ethernet
interface-name=eth0

[ethernet]
wake-on-lan=0

[ipv4]
address1=172.29.254.5/24,172.29.254.1
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
route1=10.18.0.0/16,172.29.254.1

[ipv6]
addr-gen-mode=eui64
method=ignore

[proxy]
```

如果你想给静态路由单独加 metric，可以写成：

```ini
route1=10.18.0.0/16,172.29.254.1,5
```

其中最后的 `5` 是这条静态路由的 metric。

---

### 3.4 如果不想让 eth0 成为默认网关

某些场景下，你只希望 eth0 用于访问 `10.18.0.0/16`，不希望系统默认上网流量走 `172.29.254.1`。

这种情况下可以这样写：

```ini
[ipv4]
address1=172.29.254.5/24
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
never-default=true
route1=10.18.0.0/16,172.29.254.1
```

说明：

```text
address1=172.29.254.5/24
```

只配置 IP，不配置默认网关。

```text
never-default=true
```

表示这个连接不要生成默认路由。

```text
route1=10.18.0.0/16,172.29.254.1
```

只让访问 `10.18.0.0/16` 的流量走 `172.29.254.1`。

---

### 3.5 修正配置文件权限

NetworkManager 的连接配置文件权限不能太开放，否则可能被忽略。

```bash
# 设置属主为 root
sudo chown root:root /etc/NetworkManager/system-connections/netplan-eth0.nmconnection

# 设置权限为 600：只有 root 可读写
sudo chmod 600 /etc/NetworkManager/system-connections/netplan-eth0.nmconnection

# 查看权限，应该类似 -rw-------
ls -l /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

期望输出类似：

```text
-rw------- 1 root root  xxx  netplan-eth0.nmconnection
```

---

### 3.6 让配置生效

RDK X5 官方推荐：

```bash
# RDK X5 官方脚本：重新加载网络配置
sudo restart_network
```

如果该命令不可用，可以使用 NetworkManager 命令：

```bash
# 重新加载 NetworkManager 连接配置
sudo nmcli connection reload

# 断开 eth0 对应连接
sudo nmcli connection down netplan-eth0

# 重新启用 eth0 对应连接
sudo nmcli connection up netplan-eth0
```

如果你不确定连接名，可以查看：

```bash
nmcli connection show
```

---

### 3.7 检查 Linux / RDK X5 配置结果

```bash
# 查看 eth0 的 IP
ip addr show eth0

# 查看路由表
ip route

# 只查看 10.18 网段相关路由
ip route | grep 10.18
```

期望能看到：

```text
inet 172.29.254.5/24
```

以及：

```text
10.18.0.0/16 via 172.29.254.1 dev eth0
```

测试网关：

```bash
# 测试下一跳是否能 ping 通
ping -c 4 172.29.254.1
```

测试目标网段：

```bash
# 按实际目标设备替换 10.18.0.1
ping -c 4 10.18.0.1
```

查看访问路径：

```bash
# Linux 上查看到目标 IP 的路由路径
traceroute 10.18.0.1
```

如果没有 `traceroute`：

```bash
sudo apt update
sudo apt install traceroute
```

---

## 4. Ubuntu Netplan 方式，仅适用于系统确实使用 Netplan 的情况

如果 `/etc/netplan/` 原本是空的，而 RDK X5 官方文档明确要求改 NetworkManager 的 `.nmconnection` 文件，那么不建议强行新建 netplan 文件。

如果你的普通 Ubuntu 系统确实使用 netplan，可以参考本节。

---

### 4.1 判断系统网络管理方式

```bash
# 查看 NetworkManager 是否运行
systemctl is-active NetworkManager

# 查看 systemd-networkd 是否运行
systemctl is-active systemd-networkd

# 查看 netplan 配置文件
ls -l /etc/netplan/
```

如果 `NetworkManager` 是 `active`，netplan 中通常要写：

```yaml
renderer: NetworkManager
```

如果 `systemd-networkd` 是 `active`，netplan 中通常要写：

```yaml
renderer: networkd
```

---

### 4.2 Netplan + NetworkManager 示例

编辑文件：

```bash
sudo nano /etc/netplan/01-netcfg.yaml
```

内容：

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 172.29.254.5/24
      gateway4: 172.29.254.1
      routes:
        - to: 10.18.0.0/16
          via: 172.29.254.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

修正权限：

```bash
sudo chown root:root /etc/netplan/01-netcfg.yaml
sudo chmod 600 /etc/netplan/01-netcfg.yaml
```

应用配置：

```bash
# 推荐先 try，防止远程断网后无法恢复
sudo netplan try

# 确认没问题后正式应用
sudo netplan apply
```

---

### 4.3 Netplan + systemd-networkd 示例

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 172.29.254.5/24
      routes:
        - to: default
          via: 172.29.254.1
        - to: 10.18.0.0/16
          via: 172.29.254.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

启动 `systemd-networkd`：

```bash
sudo systemctl enable --now systemd-networkd
sudo netplan apply
```

---

# 第二部分：Windows 配置

---

## 5. Windows 配置前的注意事项

### 5.1 不要和 RDK X5 使用同一个 IP

如果 RDK X5 是：

```text
172.29.254.5
```

Windows 有线网卡建议使用：

```text
172.29.254.6
```

不要这样：

```text
RDK X5  = 172.29.254.5
Windows = 172.29.254.5
```

这会产生 IP 冲突，表现为：

- 一会儿能 ping 通，一会儿不能；
- SSH 偶发断开；
- ARP 表混乱；
- 路由看起来正确但实际访问异常。

---

### 5.2 和 Clash Verge Rev TUN 共存时的推荐原则

如果 Windows 开启了 Clash Verge Rev 的 TUN 模式，连接 RDK X5 的这块有线网卡建议：

```text
只配置静态 IP
不要配置默认网关
不要配置 DNS
只添加 10.18.0.0/16 的静态路由
把接口 metric 设置得高一些，例如 700
```

原因：

- TUN 模式会接管系统路由；
- 多网卡环境下，默认网关和 DNS 很容易互相抢；
- 如果 RDK 有线网卡也配置了默认网关，部分网站可能会走错出口；
- 只添加目标网段路由，可以让 `10.18.0.0/16` 走 RDK，有公网访问则继续走 Clash / Wi-Fi / 原上网网卡。

推荐逻辑：

```text
访问 10.18.0.0/16      -> 走 RDK 有线网卡 -> 172.29.254.1
访问普通网站           -> 走 Clash TUN / 正常上网网卡
```

---

## 6. Windows CMD 方式配置

以下命令需要在 **管理员 CMD** 或 **管理员 Windows Terminal** 中执行。

---

### 6.1 查看 Windows 网卡名称和编号

```bat
rem 查看 IPv4 接口列表
netsh interface ipv4 show interfaces
```

示例输出：

```text
Idx     Met         MTU          状态                名称
---  ----------  ----------  ------------  ---------------------------
 12          25        1500  connected     以太网
 18          35        1500  connected     WLAN
```

这里：

```text
Idx = 12
名称 = 以太网
```

后续示例用：

```text
网卡名：以太网
接口编号：12
```

---

## 7. Windows 推荐方式：只配置 IP，不配置默认网关

这是和 Clash Verge Rev TUN 共存时最推荐的方式。

### 7.1 设置 Windows 有线网卡静态 IP

```bat
rem 把“以太网”替换成你自己的网卡名称
rem 这里最后的 none 表示不配置默认网关
netsh interface ipv4 set address name="以太网" static 172.29.254.6 255.255.255.0 none
```

说明：

```text
172.29.254.6      Windows 有线网卡 IP
255.255.255.0     子网掩码
none              不设置默认网关
```

---

### 7.2 删除可能已经存在的错误默认路由

如果之前给这块网卡设置过 `172.29.254.1` 作为默认网关，建议删除：

```bat
rem 删除指向 172.29.254.1 的默认路由
route delete 0.0.0.0 mask 0.0.0.0 172.29.254.1
```

如果提示找不到路由，不用管。

---

### 7.3 删除旧的 10.18 静态路由

为了避免重复或指错接口，先删除旧路由：

```bat
rem 删除旧的 10.18.0.0/16 路由
route delete 10.18.0.0
```

如果提示找不到路由，也不用管。

---

### 7.4 添加永久静态路由

先通过 `route print` 确认接口编号：

```bat
rem 查看路由表和接口列表
route print -4
```

假设连接 RDK X5 的有线网卡接口编号是 `12`，则执行：

```bat
rem -p 表示永久保存，重启后仍然生效
rem metric 5 表示这条路由本身优先级较高
rem IF 12 表示强制指定从接口编号 12 这块网卡出去
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
```

如果你的接口编号不是 `12`，要替换成你自己的编号。

---

### 7.5 把 RDK 有线网卡接口 metric 调高

这一步用管理员 PowerShell 执行。

```powershell
# 把“以太网”替换成你的有线网卡名称
# InterfaceMetric 数字越大，默认优先级越低
Set-NetIPInterface -InterfaceAlias "以太网" -AddressFamily IPv4 -AutomaticMetric Disabled -InterfaceMetric 700
```

---

### 7.6 清空这块网卡的 DNS

管理员 PowerShell：

```powershell
# 重置“以太网”的 DNS，避免这块网卡参与 DNS 解析
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

检查 DNS：

```powershell
# 查看该网卡当前 DNS 配置
Get-DnsClientServerAddress -InterfaceAlias "以太网"
```

---

### 7.7 检查结果

CMD：

```bat
rem 查看 IP、网关、DNS
ipconfig /all

rem 查看 IPv4 路由表
route print -4
```

你应该看到：

```text
以太网:
  IPv4 地址: 172.29.254.6
  默认网关: 空
```

路由表里应该有：

```text
10.18.0.0    255.255.0.0    172.29.254.1
```

路由表里不应该有这条：

```text
0.0.0.0      0.0.0.0        172.29.254.1
```

测试：

```bat
rem 测试下一跳
ping 172.29.254.1

rem 测试目标网段，按实际地址替换 10.18.0.1
ping 10.18.0.1

rem 查看访问 10.18.0.1 的路径，第一跳应该是 172.29.254.1
tracert -d 10.18.0.1

rem 查看普通公网访问路径，正常不应该走 172.29.254.1
tracert -d 8.8.8.8
```

---

## 8. Windows 传统方式：配置 IP + 默认网关

如果你明确希望这块 Windows 有线网卡也使用 `172.29.254.1` 作为默认网关，可以这样设置。

> 注意：开启 Clash Verge Rev TUN 时，这种方式更容易造成部分网站断网，因此不作为首选。

```bat
rem 设置静态 IP、子网掩码和默认网关
netsh interface ipv4 set address name="以太网" static 172.29.254.6 255.255.255.0 172.29.254.1

rem 添加永久静态路由
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1
```

如果要指定接口编号：

```bat
rem 假设有线网卡接口编号是 12
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
```

---

## 9. Windows PowerShell 方式配置

以下命令需要在 **管理员 PowerShell** 中执行。

---

### 9.1 查看网卡

```powershell
# 查看网卡名称、状态、接口索引等信息
Get-NetAdapter
```

示例：

```text
Name     InterfaceDescription       ifIndex Status
----     --------------------       ------- ------
以太网   Realtek PCIe GbE             12    Up
WLAN     Intel Wi-Fi                  18    Up
```

---

### 9.2 推荐方式：只配置 IP，不配置默认网关

```powershell
# 给“以太网”配置静态 IP，不配置默认网关
New-NetIPAddress -InterfaceAlias "以太网" -IPAddress 172.29.254.6 -PrefixLength 24

# 添加到 10.18.0.0/16 的静态路由
New-NetRoute -DestinationPrefix "10.18.0.0/16" -InterfaceAlias "以太网" -NextHop "172.29.254.1" -RouteMetric 5

# 把这块网卡的接口 metric 调高，避免抢默认出口
Set-NetIPInterface -InterfaceAlias "以太网" -AddressFamily IPv4 -AutomaticMetric Disabled -InterfaceMetric 700

# 清空这块网卡的 DNS 配置
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

---

### 9.3 传统方式：配置 IP + 默认网关

```powershell
# 配置静态 IP 和默认网关
New-NetIPAddress -InterfaceAlias "以太网" -IPAddress 172.29.254.6 -PrefixLength 24 -DefaultGateway 172.29.254.1

# 添加静态路由
New-NetRoute -DestinationPrefix "10.18.0.0/16" -InterfaceAlias "以太网" -NextHop "172.29.254.1" -RouteMetric 5
```

---

### 9.4 PowerShell 删除旧 IP / 旧路由

如果重复执行 `New-NetIPAddress` 报错，可能是 IP 已存在。

查看当前 IP：

```powershell
Get-NetIPAddress -InterfaceAlias "以太网" -AddressFamily IPv4
```

删除指定 IP：

```powershell
# 删除 172.29.254.6 这个 IP
Remove-NetIPAddress -InterfaceAlias "以太网" -IPAddress 172.29.254.6 -Confirm:$false
```

查看旧路由：

```powershell
Get-NetRoute -DestinationPrefix "10.18.0.0/16"
```

删除旧路由：

```powershell
Remove-NetRoute -DestinationPrefix "10.18.0.0/16" -Confirm:$false
```

---

## 10. Clash Verge Rev TUN 模式下的专门建议

### 10.1 推荐网络结构

```text
Windows Wi-Fi / 主上网网卡
        ↓
    Clash Verge Rev TUN
        ↓
    访问公网网站

Windows 有线网卡 172.29.254.6/24
        ↓
    172.29.254.1
        ↓
    10.18.0.0/16
```

### 10.2 Windows 网卡推荐状态

连接 RDK X5 的有线网卡：

```text
IPv4 地址: 172.29.254.6
子网掩码: 255.255.255.0
默认网关: 空
DNS: 空或不参与解析
接口 metric: 700
静态路由: 10.18.0.0/16 -> 172.29.254.1
```

### 10.3 Clash / Mihomo TUN 配置建议

不同版本的 Clash Verge Rev 界面不完全一样，重点找这些选项：

```text
TUN Mode: 开启
Auto Route: 开启
DNS Override / DNS 复写: 开启
Auto Detect Interface: 避免自动选到 RDK 有线网卡
出口网卡: 优先指定真正上网的 Wi-Fi / WLAN
```

如果你的配置文件支持 `route-exclude-address`，可以排除 RDK 相关网段：

```yaml
tun:
  enable: true
  auto-route: true
  route-exclude-address:
    - 172.29.254.0/24
    - 10.18.0.0/16
```

含义：

```text
172.29.254.0/24 不让 TUN 接管
10.18.0.0/16    不让 TUN 接管
```

这样可以避免 Clash TUN 抢走访问 RDK / 目标内网的流量。

---

# 第三部分：故障排查

---

## 11. Linux / RDK X5 常见问题

### 11.1 `ip route add` 提示路由已存在

报错：

```text
RTNETLINK answers: File exists
```

解决：

```bash
# 用 replace 替代 add
sudo ip route replace 10.18.0.0/16 via 172.29.254.1 dev eth0
```

---

### 11.2 修改 `.nmconnection` 后不生效

检查 1：文件权限。

```bash
ls -l /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

应该类似：

```text
-rw------- 1 root root ... netplan-eth0.nmconnection
```

修复：

```bash
sudo chown root:root /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
sudo chmod 600 /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

检查 2：连接名是否正确。

```bash
nmcli connection show
```

检查 3：重新加载并启动连接。

```bash
sudo nmcli connection reload
sudo nmcli connection down netplan-eth0
sudo nmcli connection up netplan-eth0
```

RDK X5 优先尝试：

```bash
sudo restart_network
```

---

### 11.3 `netplan apply` 出现权限警告

警告类似：

```text
Permissions for /etc/netplan/01-netcfg.yaml are too open
```

修复：

```bash
sudo chown root:root /etc/netplan/01-netcfg.yaml
sudo chmod 600 /etc/netplan/01-netcfg.yaml
```

如果 RDK X5 官方要求使用 `.nmconnection`，而 `/etc/netplan/` 原来是空的，则建议把后来新建的 netplan 文件停用：

```bash
sudo mv /etc/netplan/01-netcfg.yaml /etc/netplan/01-netcfg.yaml.bak
```

---

### 11.4 能 ping 通 `172.29.254.1`，但不能访问 `10.18.0.0/16`

检查路由：

```bash
ip route | grep 10.18
```

应该有：

```text
10.18.0.0/16 via 172.29.254.1 dev eth0
```

如果没有：

```bash
sudo ip route replace 10.18.0.0/16 via 172.29.254.1 dev eth0
```

还需要确认：

- `172.29.254.1` 这台设备是否真的知道如何转发到 `10.18.0.0/16`；
- 目标 `10.18.x.x` 主机是否允许 ICMP / TCP 访问；
- 中间设备是否有防火墙策略；
- 回程路由是否正确。

---

## 12. Windows 常见问题

### 12.1 `route add` 提示对象已存在

可以先删除旧路由：

```bat
route delete 10.18.0.0
```

再重新添加：

```bat
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
```

---

### 12.2 `route add` 走错网卡

原因：Windows 有多块网卡时，可能自动选错接口。

解决：指定 `IF` 参数。

```bat
rem 先查接口编号
route print -4

rem 假设 RDK 有线网卡接口编号是 12
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
```

---

### 12.3 Clash Verge Rev 开 TUN 后部分网站断网

推荐检查：

```bat
ipconfig /all
route print -4
```

连接 RDK 的有线网卡应该满足：

```text
默认网关为空
DNS 不使用这块网卡
接口 metric 较高，例如 700
只有 10.18.0.0/16 指向 172.29.254.1
```

可以重新执行推荐配置：

```bat
netsh interface ipv4 set address name="以太网" static 172.29.254.6 255.255.255.0 none
route delete 0.0.0.0 mask 0.0.0.0 172.29.254.1
route delete 10.18.0.0
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
```

PowerShell：

```powershell
Set-NetIPInterface -InterfaceAlias "以太网" -AddressFamily IPv4 -AutomaticMetric Disabled -InterfaceMetric 700
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

---

### 12.4 Windows 恢复 DHCP

如果要恢复自动获取 IP：

```bat
rem 恢复 IP 为 DHCP
netsh interface ipv4 set address name="以太网" source=dhcp

rem 恢复 DNS 为 DHCP
netsh interface ipv4 set dnsservers name="以太网" source=dhcp

rem 删除静态路由
route delete 10.18.0.0
```

PowerShell：

```powershell
# 删除 10.18 静态路由
Remove-NetRoute -DestinationPrefix "10.18.0.0/16" -Confirm:$false

# 恢复 DNS
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

---

# 第四部分：推荐最终配置模板

---

## 13. RDK X5 推荐模板

注意：下面模板里的 `uuid` 只是沿用示例。实际操作时建议保留你原文件里的 `uuid`，只修改 `[ipv4]` 段。

文件：

```text
/etc/NetworkManager/system-connections/netplan-eth0.nmconnection
```

配置：

```ini
[connection]
id=netplan-eth0
uuid=f6f8b5a7-9e23-49b2-a792-dc589b3d3e88
type=ethernet
interface-name=eth0

[ethernet]
wake-on-lan=0

[ipv4]
address1=172.29.254.5/24,172.29.254.1
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
route1=10.18.0.0/16,172.29.254.1

[ipv6]
addr-gen-mode=eui64
method=ignore

[proxy]
```

应用：

```bash
sudo chown root:root /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
sudo chmod 600 /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
sudo restart_network
```

检查：

```bash
ip addr show eth0
ip route | grep 10.18
ping -c 4 172.29.254.1
```

---

## 14. Windows + Clash TUN 推荐模板

假设：

```text
Windows RDK 有线网卡名：以太网
Windows RDK 有线网卡接口编号：12
Windows RDK 有线网卡 IP：172.29.254.6
RDK / 下一跳：172.29.254.1
目标网段：10.18.0.0/16
```

管理员 CMD：

```bat
rem 查看接口，确认网卡名和接口编号
netsh interface ipv4 show interfaces
route print -4

rem 设置 Windows 有线网卡 IP，不设置默认网关
netsh interface ipv4 set address name="以太网" static 172.29.254.6 255.255.255.0 none

rem 删除可能冲突的旧默认路由和旧静态路由
route delete 0.0.0.0 mask 0.0.0.0 172.29.254.1
route delete 10.18.0.0

rem 添加永久静态路由，IF 12 替换为你的实际接口编号
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12

rem 检查路由表
route print -4
```

管理员 PowerShell：

```powershell
# 把 RDK 有线网卡优先级调低
Set-NetIPInterface -InterfaceAlias "以太网" -AddressFamily IPv4 -AutomaticMetric Disabled -InterfaceMetric 700

# 清掉 RDK 有线网卡 DNS，避免影响 Clash / 系统 DNS
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

检查：

```bat
ipconfig /all
route print -4
ping 172.29.254.1
tracert -d 10.18.0.1
tracert -d 8.8.8.8
```

期望结果：

```text
访问 10.18.0.0/16：走 172.29.254.1
访问普通公网网站：不走 172.29.254.1
```

---

## 15. 一页速查版

### Linux 临时生效

```bash
sudo ip addr add 172.29.254.5/24 dev eth0
sudo ip route replace 10.18.0.0/16 via 172.29.254.1 dev eth0
ip addr show eth0
ip route | grep 10.18
```

### RDK X5 永久生效

```bash
sudo vim /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
sudo chmod 600 /etc/NetworkManager/system-connections/netplan-eth0.nmconnection
sudo restart_network
```

`[ipv4]` 段：

```ini
[ipv4]
address1=172.29.254.5/24,172.29.254.1
dns=8.8.8.8;8.8.4.4;
method=manual
route-metric=700
route1=10.18.0.0/16,172.29.254.1
```

### Windows + Clash TUN 推荐

```bat
netsh interface ipv4 show interfaces
netsh interface ipv4 set address name="以太网" static 172.29.254.6 255.255.255.0 none
route delete 0.0.0.0 mask 0.0.0.0 172.29.254.1
route delete 10.18.0.0
route -p add 10.18.0.0 mask 255.255.0.0 172.29.254.1 metric 5 IF 12
route print -4
```

PowerShell：

```powershell
Set-NetIPInterface -InterfaceAlias "以太网" -AddressFamily IPv4 -AutomaticMetric Disabled -InterfaceMetric 700
Set-DnsClientServerAddress -InterfaceAlias "以太网" -ResetServerAddresses
```

---

## 16. 修改记录

| 日期 | 内容 |
|---|---|
| 2026-05-20 | 初版：整理 Linux、RDK X5、Windows、Clash TUN 共存配置 |
