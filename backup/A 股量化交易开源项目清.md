# 🇨🇳 **一、完全适用于 A 股的主流开源框架（强烈推荐）**

## **1️⃣ vn.py（国内最强，适用股票/期货/期权/CTA/HFT）**

📌 **Github**：[https://github.com/vnpy/vnpy](https://github.com/vnpy/vnpy)
📌 **语言**：Python
📌 **适合**：A 股 / 期货 / 期权 / CTA / 高频 / 实盘

### 📌 为什么推荐？

* 国内量化的 **头号开源项目**，几乎行业标准。
* 支持几乎所有国内券商 API（CTP、LTS、XTP 等）。
* **有模拟盘**（仿真交易）、本地行情回测引擎。
* 完整的实盘交易系统 + 策略开发框架 + 图形界面。
* 机构和私募大量使用。

### ✔ 你如果想入门 A 股量化，用这个最实在。

---

## **2️⃣ RiceQuant（米筐）——开源版 + 云平台（含模拟交易）**

📌 [https://github.com/ricequant/rqalpha](https://github.com/ricequant/rqalpha)
📌 **语言**：Python
📌 **适合**：股票多因子、事件驱动策略、回测、模拟盘

### 📌 特点

* 国内另一大主流框架，配套社区丰富。
* 数据结构、策略框架类似 Zipline，但适配中国市场。
* 有 **模拟交易（Paper Trading）**。
* 自带大量教学材料、新手友好。

---

## **3️⃣ QUANTAXIS（适合数据科研 + 回测体系）**

📌 [https://github.com/QUANTAXIS/QUANTAXIS](https://github.com/QUANTAXIS/QUANTAXIS)
📌 **语言**：Python
📌 **适合**：数据分析、数据仓库、多策略回测、模拟

### 📌 特点

* 完整的数据采集（股票/期货/加密等）。
* 内置多种回测引擎，含 **模拟交易**。
* 社区较活跃，很多插件。
* 非常适合你想深入理解“量化交易全流程”（数据→策略→回测→模拟→实盘）。

---

# 🇨🇳 **二、适配 A 股券商 API 的 C++/Python 框架**

## **4️⃣ XQuant / XTP API（中泰证券，适合实盘高频）**

📌 [https://github.com/xtpapi](https://github.com/xtpapi)
📌 **语言**：C++ / Python
📌 **适合**：A 股 高频交易（HFT） / 做市

### 📌 特点

* 适用于股票的毫秒级交易 API。
* 大量私募用来做 A 股日内 / 高频 / 做市策略。
* 需要券商开户申请 API 权限。

---

## **5️⃣ CTP / LTS（期货为主，但 A 股衍生品常用）**

📌 [https://github.com/openctp](https://github.com/openctp)
📌 **语言**：C++
📌 **适合**：CTA、期货、现货相关策略

### 📌 特点

虽非股票主力框架，但 A 股量化会用期现策略，需要 CTP。

---

# 🇨🇳 **三、偏学习型（新手友好、有教学资料）**

## **6️⃣ AkShare（数据神器）**

📌 [https://github.com/akfamily/akshare](https://github.com/akfamily/akshare)
📌 **语言**：Python
📌 **适合**：数据采集（A股股票、基金、期货、期权）

### 📌 特点

* 无门槛获取 A 股行情、财务、新闻等数据。
* 可搭配任何框架使用（vn.py / RQAlpha /量化研究）。

---

## **7️⃣ Backtrader（国外框架，但能跑 A 股数据）**

📌 [https://github.com/mementum/backtrader](https://github.com/mementum/backtrader)
📌 **语言**：Python
📌 **适合**：回测 / 教学 / 快速实现策略

### 📌 特点

* 写策略非常简单。
* 和 AkShare 配合可以做稳定的 A 股回测平台。
* 适合初学者理解“策略调度方式”。

---

# 📚 **四、如果你是新手，推荐你接受以下路线：**

## **入门路线：**

### 1. **数据 → AkShare**

学会获取 A 股行情、财报、K 线。

### 2. **回测 → backtrader 或 rqalpha**

理解：

* 调仓周期
* 交易事件
* 买卖触发

### 3. **模拟盘 → rqalpha / quantaxis**

跑“纸面交易”测试你的策略。

### 4. **实盘 → vn.py**

连接真实股票账户（需券商 API）。
你才能最终做到“真实交易”。

