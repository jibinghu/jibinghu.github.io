学习自：[人人都能看懂的 RL-PPO 理论知识](https://zhuanlan.zhihu.com/p/7461863937)


12.1 [更] 还是没能学完，

参考资料是Sutton的这本[强化学习导论](http://incompleteideas.net/book/the-book-2nd.html)。在现有的很多教材中，一般会按照这本导论的介绍方式，从MDP(马尔可夫决策过程）和价值函数定义介绍起，然后按照value-based，policy-based，actor-critic的顺序介绍。但是由于本文的重点是actor-critic，所以我在写文章时，按照自己的思考方式重新做了整理：

- 我们会先介绍policy-based下的优化目标。
- 然后再介绍价值函数的相关定义。
- 引入actor-critic，讨论在policy-based的优化目标中，对“价值”相关的部分如何做优化。
- 基于actor-critic的知识介绍PPO。

一、策略

策略就是智能体根据当前状态 𝑠 选择动作 𝑎 的规则。分为确定性策略和随机性策略。

1. 确定性策略：

就是一个函数：智能体在看到状态 $S_t$ 的情况下，确定地执行 $a_t$ 。

$$
a_t = μ_θ​ (s_t)
$$

只输出一个动作，而不是概率分布。

2. 随机性策略：

智能体在看到状态 $s_t$  的情况下，可能执行的动作符合概率分布。 

 $$
a_t \sim \pi_\theta(\,\cdot \mid s_t\,)
$$ 


这是又分为离散概率和连续概率，连续概率通常基于高斯分布。


二、奖励

奖励由当前状态、已经执行的动作和下一步的状态共同决定。即在当前状态 $s_t$ 下执行动作 $a_t$ 后能够得到多少收益。

1. 单步奖励

$$
r_t = R(s_t, a_t, s_{t+1})
$$

奖励和策略 $\pi$ 无关
用于评估当前动作的好坏，指导智能体的动作选择。

2. T 步累积奖励

T步累积奖励等于一条运动轨迹/一个回合/一个rollout后的单步奖励的累加。 (假设一个 episode 长度为 𝑇，那么从时刻 𝑡 开始的“未来总回报”定义)

$$
R(r) = \sum^{T-1}_{t=0} r_t
$$

表示从 t 时刻开始，能拿到的全部奖励。

3. 折扣奖励

为了使远期奖励对训练影响更小，引入折扣因子 $\gamma$ ∈ [0, 1]：

$$
R(\gamma) = \sum^\inf_{t=0} \gamma^t r_t
$$

这里折扣因子引入指数，远期的影响会更小。

三、运动轨迹（trajectory）和状态转移


智能体和环境做一系列/一回合交互后得到的state、action和reward的序列，所以运动轨迹也被称为 **episodes** 或者 **rollouts** ，这里我们假设智能体和环境交互了 $T$ 次。

四、Policy-based强化学习优化目标

抽象来说，强化学习的优化过程可以总结为：

价值评估：给定一个策略，如何准确评估当前策略的价值？
策略迭代：给定一个当前策略的价值评估，如何据此优化策略？

<img width="100" height="250" alt="Image" src="https://github.com/user-attachments/assets/eb2e8a5f-cf79-4746-9d8a-d28dab88123b" />

<img width="705" height="258" alt="Image" src="https://github.com/user-attachments/assets/8708aabf-3139-416d-9ffb-71fc0aeb9d28" />

policy-based下的强化学习优化目标：

<img width="480" height="38" alt="Image" src="https://github.com/user-attachments/assets/9784bd7b-2845-4f82-8a02-dafd4cc63d52" />

<img width="688" height="193" alt="Image" src="https://github.com/user-attachments/assets/123966dc-acda-4c4e-a143-385ac5097010" />


五、策略的梯度上升

这里 $\theta$ 即策略参数是自变量，要看在这个策略的情况下，期望对其求导。

<img width="487" height="513" alt="Image" src="https://github.com/user-attachments/assets/74252fb3-3c71-4eba-b9e6-5536417c813e" />

这里引入 log，可以把乘法变加法。

> “导数 = 概率 × log 概率的导数” 是数学恒等式。。。就是 log(x) 求导 = x^'/x

对 log 一项再展开推导：

因为：

轨迹 τ 在策略 πθ 下出现的概率（trajectory probability）

> 轨迹概率 = 初始状态概率 × 环境概率 × 策略概率

<img width="469" height="85" alt="Image" src="https://github.com/user-attachments/assets/1375a47b-ac96-4081-9f14-29a34d850973" />

有

<img width="696" height="138" alt="Image" src="https://github.com/user-attachments/assets/4dd1016e-a2c7-417e-bdff-65b03ce7446c" />

被约去的两项是因为这里我们是在对策略求梯度，而这两项和环境相关，不和策略相关。

综上，最终策略的梯度表达式为：

<img width="402" height="113" alt="Image" src="https://github.com/user-attachments/assets/de8ba20c-d18f-446a-b3b5-093759ed0004" />

所以：！！！

<img width="483" height="234" alt="Image" src="https://github.com/user-attachments/assets/4b9e73c9-f0a1-49be-84d0-81ea9253b469" />

<img width="746" height="91" alt="Image" src="https://github.com/user-attachments/assets/4fa6fa43-5d48-44d3-b28d-757e17b05864" />

在实践中，我们可以通过采样足够多的轨迹来估计这个期望。假设采样N条轨迹，N足够大，每条轨迹涵盖 $T_n$ 步，则上述优化目标可以再次被写成：

<img width="418" height="387" alt="Image" src="https://github.com/user-attachments/assets/d3ea9f64-f360-4c9a-82dc-29c425e01570" />

<img width="615" height="464" alt="Image" src="https://github.com/user-attachments/assets/cf47465d-123a-4d49-a4ec-0d2b82fba205" />

公式中连乘有两个部分，上述很重要。


---

接下来引入价值：

策略的梯度可以表示成：

<img width="440" height="150" alt="Image" src="https://github.com/user-attachments/assets/92b60320-b27d-4876-811e-81bcdc25765a" />

也就是说在长为 $T_n$ 的轨迹中，策略的选取以及环境的反馈以及奖励函数的整体期望得到最终基于策略的期望，但是环境只与采取的动作有关，（我理解是按照链式法则来看，整体的梯度求导只与策略的参数 $\theta$ 有关即可，与环境相关被剔除）。

<img width="728" height="305" alt="Image" src="https://github.com/user-attachments/assets/31bad809-e31e-4038-b31a-f8651dd5ec6d" />

<img width="725" height="234" alt="Image" src="https://github.com/user-attachments/assets/40026a71-de3b-499f-9e74-960d2b46b92b" />

所以把整条轨迹上的奖励函数替换成一个更可行的价值函： $\psi$ 。

<img width="720" height="275" alt="Image" src="https://github.com/user-attachments/assets/23e60a21-4b5c-4992-b08e-111f6ff65cea" />

接下来逐一进行说明：

<img width="462" height="202" alt="Image" src="https://github.com/user-attachments/assets/ecef3a4c-b91a-404c-a837-f688f110349f" />

<img width="723" height="180" alt="Image" src="https://github.com/user-attachments/assets/62c92010-c154-4ae3-9bc5-00b85bdafaf4" />

<img width="740" height="342" alt="Image" src="https://github.com/user-attachments/assets/0dc88feb-8fb7-4a6d-8fe0-c7c1a3a9a2ec" />

<img width="584" height="268" alt="Image" src="https://github.com/user-attachments/assets/6292bcbe-680b-4585-bfbb-845599a3fcb5" />

<img width="742" height="871" alt="Image" src="https://github.com/user-attachments/assets/62e37d19-ce2b-4a52-911c-7413ede710ba" />

这就说明，当优势越大时，说明一个动作比其他动作更好，因为已经减去了随机采样得到的基线（当然这里也可以选择其他方式来确定 baseline），所以这时候我们要提升这个动作的概率。

<img width="560" height="249" alt="Image" src="https://github.com/user-attachments/assets/182a8014-2bd6-4e99-b95a-b484e97c7111" />

注意，这时候我们已经可以知道 状态价值函数、动作价值函数的抽象概念是什么意思了(基于上述的说法)。


<img width="719" height="562" alt="Image" src="https://github.com/user-attachments/assets/b6eba9d7-5e35-4fb6-9762-55e876d15d7c" />


<img width="729" height="702" alt="Image" src="https://github.com/user-attachments/assets/301f4fde-961f-48e3-b8c4-e14c8d43063d" />

我觉得有必要自己推导一下整个过程：

<img width="242" height="54" alt="Image" src="https://github.com/user-attachments/assets/05e25f45-f0ad-4743-b2f6-3400a572f97d" />

<img width="425" height="48" alt="Image" src="https://github.com/user-attachments/assets/f1e63572-de4c-4920-8a90-5b0ad9e07305" />

<img width="280" height="42" alt="Image" src="https://github.com/user-attachments/assets/7fe0ba3f-dfd9-42b5-b30e-d6794c8119af" />

<img width="742" height="82" alt="Image" src="https://github.com/user-attachments/assets/9bafdfa5-9fbd-4ac4-b094-47d5e0ccbd3d" />

这时

<img width="746" height="296" alt="Image" src="https://github.com/user-attachments/assets/64cc1341-1ac5-4c55-9f46-fa3a0dbc008c" />

<img width="627" height="397" alt="Image" src="https://github.com/user-attachments/assets/88f1a5bf-6d06-427c-9a28-377534e9c313" />

基于贝尔曼方程：https://datawhalechina.github.io/easy-rl/#/chapter2/chapter2

<img width="749" height="398" alt="Image" src="https://github.com/user-attachments/assets/fcb097e3-c84f-414d-92ed-022c4a4b14f4" />

<img width="748" height="671" alt="Image" src="https://github.com/user-attachments/assets/b572a5fe-b862-4df0-9767-f4420c79bd33" />

<img width="473" height="674" alt="Image" src="https://github.com/user-attachments/assets/15151ed8-d096-435e-be0b-4ec5f0bc6f9a" />

<img width="733" height="672" alt="Image" src="https://github.com/user-attachments/assets/dc9de55c-5b02-4e3e-8f9f-0db6aed27841" />

<img width="715" height="212" alt="Image" src="https://github.com/user-attachments/assets/dacb0cdb-492d-474a-a750-649442761b47" />

<img width="634" height="257" alt="Image" src="https://github.com/user-attachments/assets/14af2134-7390-4827-9005-7e62ee6324d3" />














