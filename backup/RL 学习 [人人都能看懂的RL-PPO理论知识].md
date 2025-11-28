学习自：[人人都能看懂的 RL-PPO 理论知识](https://zhuanlan.zhihu.com/p/7461863937)

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
