# SOSRL 双层强化学习算法说明书

本文档说明当前 SOSRL 双层强化学习算法的输入、输出、状态、动作、奖励、经验回放、训练过程和评估方法。它用于回答“算法如何运行和复现”，不作为源码阅读教程或论文方法章节。

当前算法名称可表述为：

> 基于架构规则与调度规则动作抽象的同频双层 DQN。

算法在同一个 mission 的离线调度构造过程中，交替完成体系架构调整和 operation-system 分配。

---

## 1. 算法目标

给定：

- 一个由多个 task 组成的 mission；
- 每个 task 内具有固定前后序关系的 operations；
- 一个包含全部候选 component systems 的系统池；
- 每个系统的能力类型、成本和可用时间窗；
- 初始体系架构和预算 \(B\)。

算法需要输出：

1. mission 内每道 operation 对应的执行系统；
2. 每道 operation 的开始时间和完成时间；
3. 调度构造过程中发生的体系 ADD、REMOVE、REPLACE 或 KEEP 决策；
4. 最终 makespan、净成本、当前在役成本和体系变化次数。

算法同时优化：

- mission 可完成性；
- makespan；
- 体系净成本；
- 预算违规；
- 不必要的体系频繁变化。

当前实现采用标量化奖励，不计算显式 Pareto 前沿，也不是严格的约束强化学习。

---

## 2. 问题输入与约束

### 2.1 Mission

Mission 表示为：

\[
\mathcal M=\{\tau_1,\tau_2,\ldots,\tau_T\}.
\]

每个 task \(\tau_i\) 包含 \(O\) 道顺序执行的 operation：

\[
\tau_i=(o_{i,1},o_{i,2},\ldots,o_{i,O}).
\]

operation \(o_{i,j}\) 具有：

- 能力类型 \(g_{i,j}\)；
- 持续时间 \(p_{i,j}\)；
- 释放时间 \(r_{i,j}\)。

当前默认场景为：

| 参数                             |  默认值 |
| -------------------------------- | ------: |
| task 数量\(T\)                   |      30 |
| 每个 task 的 operation 数量\(O\) |       4 |
| operation 总数                   |     120 |
| operation duration               |  20–30 |
| 能力类型                         | S、D、I |

### 2.2 候选系统

候选系统全集为：

\[
\mathcal S=\{S_1,S_2,\ldots,S_N\}.
\]

系统 \(S_k\) 具有：

- 能力类型 \(g_k\)；
- 成本 \(c_k\)；
- 可用时间窗 \([l_k,u_k]\)。

当前配置包含 22 个候选系统。

### 2.3 可行性约束

operation 只能分配给能力匹配的系统：

\[
x_{i,j,k}=1\Rightarrow g_{i,j}=g_k.
\]

同一 task 内满足前序约束：

\[
s_{i,j+1}\ge f_{i,j}.
\]

系统不能同时执行两道 operation。若系统当前最早可用时间为 \(q_k\)，则：

\[
s_{i,j,k}=\max(q_k,r_{i,j}),
\]

\[
f_{i,j,k}=s_{i,j,k}+p_{i,j}.
\]

只有满足

\[
f_{i,j,k}\le u_k
\]

的分配才有效。

### 2.4 调度语义

算法执行的是离线列表调度。每个决策步向当前方案中追加一道 operation assignment，而不是让现实时间推进到下一个开始或完成事件。

已排定 operation 不回滚。系统被移除后：

- 历史 assignment 保留；
- 历史开始和完成时间保留；
- 历史系统 ready time 保留；
- 该系统只是不再承担未来 operation。

---

## 3. 双层决策结构

算法包含两个策略网络：

| 层级 | 策略             |     输入 |         输出 |
| ---- | ---------------- | -------: | -----------: |
| 上层 | Architecture DQN | 架构观测 | 6 条架构规则 |
| 下层 | Scheduler DQN    | 调度观测 | 4 条调度规则 |

每个调度构造步的执行顺序固定为：

```text
1. 生成 Architecture observation 和 action mask
2. Architecture DQN 选择一条架构规则
3. 规则解析器将其转换为具体 KEEP/ADD/REMOVE/REPLACE
4. 更新 active architecture 和成本
5. 生成 Scheduler observation 和 action mask
6. Scheduler DQN 选择 SPT/WINQ/CR/MS
7. 调度规则选择一道 operation
8. CSSA 选择具体系统
9. 环境执行一道 assignment
10. 计算两层奖励并记录 transition
```

两层共享同一份 mission、部分调度、系统状态和成本状态。

策略分解为：

\[
\pi(a_t^A,a_t^S\mid s_t)
========================

\pi_A(a_t^A\mid s_t^A)
\pi_S(a_t^S\mid s_t^S,z_t^+),
\]

其中 \(z_t^+\) 是执行上层动作后的 active system mask。上层动作先改变下层本步的可行分配集合。

本算法每个构造步都调用上层，`KEEP` 表示本步不修改体系。算法不使用第三个触发网络或 option termination network。

---

## 4. 环境内部动作

环境最终执行的动作是一个具体 assignment：

\[
a_t^{env}=(task_i,operation_j,system_k).
\]

其整数编码为：

\[
a_t^{env}=((i\times O)+j)\times N+k.
\]

默认配置下，环境 assignment 动作空间为：

\[
30\times4\times22=2640.
\]

两个 DQN 均不直接学习这 2640 个动作：

- Architecture DQN 将上层动作压缩为 6 条规则；
- Scheduler DQN 将下层动作压缩为 4 条规则；
- 确定性规则将两层输出转换为具体 assignment。

---

## 5. 上层 Architecture DQN

### 5.1 架构状态

时刻 \(t\) 的体系架构表示为：

\[
z_t=(z_{1,t},z_{2,t},\ldots,z_{N,t}),
\qquad z_{k,t}\in\{0,1\}.
\]

\(z_{k,t}=1\) 表示系统 \(S_k\) 当前可以承担未来 operation。

### 5.2 Architecture observation

上层观测由以下部分连接而成：

| 特征组                   |   维度 |
| ------------------------ | -----: |
| Scheduler 的完整调度状态 |     25 |
| active system mask       |  \(N\) |
| used system mask         |  \(N\) |
| system ready time        |  \(N\) |
| 每种能力的统计特征       | \(5F\) |
| 成本和进度标量           |      6 |

因此：

\[
d_A=25+3N+5F+6.
\]

当前 \(N=22,F=3\)，Architecture observation 为 112 维。

每种能力包含五个特征：

1. 剩余 operation duration；
2. active systems 剩余时间窗容量；
3. inactive 候选系统比例；
4. 当前 ready operations 阻塞率；
5. active systems 利用率。

六个全局标量为：

1. operation 完成进度；
2. 当前 makespan；
3. `active_cost / budget`；
4. `net_cost / budget`；
5. 超预算比例；
6. 距上次架构变化的步数比例。

### 5.3 Architecture action

Architecture DQN 输出 6 个 Q 值：

| 编号 | 动作                    | 解析结果                     |
| ---: | ----------------------- | ---------------------------- |
|    0 | `KEEP`                | 不改变 active architecture   |
|    1 | `ADD_CAPABILITY`      | 补充当前缺失能力             |
|    2 | `ADD_CAPACITY`        | 扩充压力最大的能力容量       |
|    3 | `ADD_WINDOW`          | 加入能够解除时间窗阻塞的系统 |
|    4 | `REMOVE_REDUNDANT`    | 移除不影响剩余覆盖的冗余系统 |
|    5 | `REPLACE_INEFFICIENT` | 用同能力系统替换低效系统     |

#### KEEP

当前 active architecture 存在至少一个可行 assignment 时有效。若当前体系无解但完整候选池能够解救，则 `KEEP` 被屏蔽。

#### ADD_CAPABILITY

检查当前 ready operations。若某 operation 所需能力在 active systems 中完全不存在，则从 inactive systems 中选择能够完成该 operation 的系统。

排序键为：

```text
(预计完成时间, 系统成本, system index)
```

#### ADD_CAPACITY

对能力类型 \(g\) 计算：

\[
pressure_g=
\frac{remaining\ demand_g}{remaining\ active\ capacity_g}.
\]

优先扩充压力最大的能力类型。该类型内部优先选择：

```text
(-剩余容量, 系统成本, system index)
```

#### ADD_WINDOW

针对“能力类型已经存在，但所有 active systems 都因时间窗无法完成”的 ready operation，加入预计完成时间最早的 inactive system。

#### REMOVE_REDUNDANT

只有同时满足以下条件的系统才能移除：

- 移除后所有剩余能力需求仍有系统覆盖；
- 移除后剩余时间窗容量仍覆盖剩余需求；
- 移除后当前仍存在可行 assignment。

候选系统按退款额从高到低选择。

#### REPLACE_INEFFICIENT

在相同能力类型内原子执行 REMOVE+ADD。替换价值为：

\[
v=
\frac-\bar f_}}
---------------

\frac{c_{new}-\rho c_{old}}{B}.
\]

只有 \(v>0\) 的候选替换才有效，并选择价值最大的组合。

### 5.4 Architecture action mask

每条规则执行前先进行确定性模拟。无法产生合法具体动作的规则被屏蔽。

若：

- 当前 active architecture 无可行 assignment；
- 完整候选池仍存在可行 assignment；

则状态需要架构救援，而不是 dead end。

只有完整候选池也无法安排任何当前 operation 时，才判定真正 dead end。

---

## 6. 下层 Scheduler DQN

### 6.1 Scheduler observation

Scheduler 使用固定 25 维状态。所有时间量使用

\[
M_{ref}=\sum_{i=1}^{T}\sum_{j=1}^{O}p_{i,j}
\]

归一化。

| 特征组         | 维度 | 内容                                                     |
| -------------- | ---: | -------------------------------------------------------- |
| task 数量      |    3 | 未完成、当前可调度、等待 task 比例                       |
| 当前 operation |    3 | duration 的和、均值、最小值                              |
| 剩余工作量     |    5 | 剩余 duration 和/均值/最大值，next-type load 均值/最小值 |
| 紧迫度         |    5 | TTD 均值/最小值，slack 均值/最小值，等待 task 最小 slack |
| 系统和进度     |    4 | ready delay、任务完成率、负 TTD 比例、负 slack 比例      |
| 异质性         |    5 | 五类统计量的变异系数                                     |

其中：

\[
TTD_i=due_i-estimated\ finish_i,
\]

\[
slack_i=TTD_i-remaining\ processing\ time_i.
\]

### 6.2 Scheduler action

Scheduler DQN 输出 4 个调度规则 Q 值：

| 编号 | 规则 | operation 选择指标，越小越优先   |
| ---: | ---- | -------------------------------- |
|    0 | SPT  | 当前 operation duration          |
|    1 | WINQ | 下一能力类型的平均 ready load    |
|    2 | CR   | `TTD / max(remaining_time, 1)` |
|    3 | MS   | slack                            |

指标相同时，使用以下顺序打破平局：

```text
(rule metric, earliest start, task index, operation index)
```

### 6.3 CSSA 系统选择

调度规则选出 operation 后，CSSA 在能力匹配、时间窗可行的 active systems 中按以下顺序选择系统：

```text
(start time, finish time, accumulated busy time, system index)
```

最终得到环境动作 `(task, operation, system)`。

---

## 7. 成本模型

### 7.1 成本变量

算法区分：

- `net_cost`：初始成本和 ADD 收费减去 REMOVE 退款；
- `active_cost`：当前 active systems 的标价之和；
- `gross_charge`：初始投入加全部 ADD 收费，不扣退款；
- `total_refund`：累计退款；
- `peak_net_cost`、`peak_active_cost`：mission 构造过程中曾达到的最高成本；
- `ever_over_budget`：任意构造步是否出现过 `net_cost > B`；
- `final_over_budget`：终止状态是否超预算。

环境在每个 episode 开始时保存 initial cost，在每次 ADD、REMOVE、REPLACE 后更新峰值和超预算标志。累计投入满足：

\[
gross\ charge=initial\ cost+\sum ADD\ cost.
\]

### 7.2 成本转移

退款率为 \(\rho\)，默认 \(\rho=0.8\)：

\[
\Delta C_t=
\begin{cases}
0,& KEEP,\\
c_k,& ADD(S_k),\\
-\rho c_k,& REMOVE(S_k),\\
c_{new}-\rho c_{old},& REPLACE(S_{old},S_{new}).
\end{cases}
\]

系统重新加入时再次收取 100% 成本。一次 ADD-REMOVE 循环产生：

\[
(1-\rho)c_k=0.2c_k
\]

的净损失。

---

## 8. 奖励函数

### 8.1 Scheduler reward

环境基础奖励为：

\[
r_t^{S,base}=
-\frac{makespan_{t+1}-makespan_t}{M_{ref}}.
\]

在双层训练中加入终止反馈：

\[
r_t^S=r_t^{S,base}+r_t^{terminal},
\]

其中：

\[
r_t^{terminal}=
\begin{cases}
+1,& mission\ success,\\
-2,& dead\ end,\\
0,& otherwise.
\end{cases}
\]

Scheduler 不直接接收成本奖励，其主要目标是 makespan 和任务完成。

### 8.2 Architecture reward

预算势函数为：

\[
\Phi(C)=20
\left[
\max\left(0,\frac{C}{B}-1\right)
\right]^2.
\]

Architecture 一步奖励为：

\[
r_t^A=
-10\frac{\Delta makespan_t}{M_{ref}}
-\frac{\Delta net\ cost_t}{B}
-\left[\Phi(C_{t+1})-\Phi(C_t)\right]
-0.01\mathbb I(a_t^A\ne KEEP)
+r_t^{terminal}.
\]

各项含义：

| 项                               | 作用                                 |
| -------------------------------- | ------------------------------------ |
| `-10 × makespan delta / Mref` | 防止上层为了节省成本显著破坏调度效率 |
| `-net cost delta / B`          | ADD 产生负奖励，REMOVE 产生正奖励    |
| `-[Phi(next)-Phi(old)]`        | 对进入或加剧超预算状态施加二次惩罚   |
| `-0.01 × changed`             | 抑制不必要的频繁体系变化             |
| terminal                         | 成功奖励和不可恢复失败惩罚           |

### 8.3 operation 失败如何反馈到上层

若上层动作执行后不存在任何 Scheduler rule action，则本步直接：

- 判定 dead end；
- 给 Architecture reward 加入 `-2`；
- 终止 Architecture transition。

若本步 operation 成功排定，但排定后完整候选池无法继续完成剩余 operation，则：

- Scheduler reward 加入 `-2`；
- Architecture reward 同样加入 `-2`；
- episode 终止。

若失败或成功发生在后续步骤，其影响通过 Architecture 的 5-step return 和 DQN bootstrap 向前传播。

### 8.4 预算奖励的解释限制

当前预算是软约束。规则不会因为 `next_net_cost > B` 被强制屏蔽。

预算使用势函数差，因此策略离开超预算区时会获得相反方向的势差反馈。该设计鼓励回到预算内，但不保证整个轨迹从未超过预算。

评估同时输出两种预算口径：

- `budget_violation` / `final_over_budget` 检查最终：

\[
net\ cost_{terminal}>B.
\]

- `ever_over_budget` 检查完整构造轨迹；
- `peak_net_cost` 和 `peak_active_cost` 给出轨迹峰值。

这些指标能够揭示“过程中超过 8000、最终通过 REMOVE 回到预算内”的 mission。它们仍然是评估指标；如果要求策略从不越界，还需要硬预算 action mask 或约束强化学习方法。

---

## 9. 网络结构与动作选择

两个策略都使用两层 MLP：

```text
Linear(obs_dim, hidden_dim)
ReLU
Linear(hidden_dim, hidden_dim)
ReLU
Linear(hidden_dim, action_dim)
```

| 网络             | 输入维度 | 输出维度 |
| ---------------- | -------: | -------: |
| Architecture DQN |      112 |        6 |
| Scheduler DQN    |       25 |        4 |

每层分别维护 online network 和 target network。target network 只是 TD 学习副本，不参与形成第三个策略。

训练和推理都应用 action mask：

\[
a_t=\arg\max_{a:m_t(a)=1}Q(s_t,a).
\]

探索采用带 mask 的 \(\epsilon\)-greedy，只在有效动作中随机采样。

---

## 10. Replay Buffer 与 TD 更新

### 10.1 Scheduler 一步 Replay

Scheduler transition 为：

\[
(s_t^S,a_t^S,r_t^S,s_{t+1}^S,d_t,m_{t+1}^S).
\]

TD target 为：

\[
y_t^S=r_t^S+
\gamma(1-d_t)
\max_{a:m_{t+1}^S(a)=1}Q_S^-(s_{t+1}^S,a).
\]

Scheduler 的下一状态必须在下一次 Architecture action 执行后才能确定，因为新 architecture 会改变 Scheduler 的可行集合。

因此实现采用延迟补全：

```text
本步 Scheduler 执行动作
    ↓
暂存 (state, action, reward)
    ↓
下一步先执行 Architecture action
    ↓
得到真正的 next scheduler state 和 next mask
    ↓
将完整 transition 写入 Scheduler replay
```

若 episode 已终止，直接用终止状态和全零 next mask 入库。

### 10.2 Architecture 5-step Replay

Architecture 使用默认 5-step return：

\[
R_t^{(n)}=
\sum_{i=0}^{n-1}\gamma^i r_{t+i}^A.
\]

Replay 中存储：

\[
(s_t^A,a_t^A,R_t^{(n)},s_{t+n}^A,d_t,m_{t+n}^A,\gamma^n).
\]

TD target 为：

\[
y_t^A=R_t^{(n)}+
\gamma^n(1-d_t)
\max_{a:m_{t+n}^A(a)=1}Q_A^-(s_{t+n}^A,a).
\]

episode 提前终止时，累积器将剩余不足 5 步的 transition 全部刷新进 replay。

### 10.3 参数更新

- 优化器：Adam；
- loss：均方 TD error；
- 梯度裁剪：10；
- target 更新：每 `target_update_interval` 次 learn 硬同步；
- 下一状态无有效动作时，bootstrap 项为 0。

---

## 11. 单个 episode 算法

```text
输入：mission、初始 architecture、Architecture DQN、Scheduler DQN
输出：完整或失败的调度方案、体系变化轨迹、成本与 makespan

初始化统一环境
初始化 Architecture 5-step accumulator
初始化 Scheduler pending transition

while mission 未完成：
    1. 生成 architecture_obs 和 architecture_mask

    2. if architecture_mask 全为 0：
           标记真正 dead end
           刷新未完成 transition
           break

    3. Architecture DQN 以 masked epsilon-greedy 选择规则
    4. 确定性解析并执行 KEEP/ADD/REMOVE/REPLACE

    5. 生成 schedule_obs 和 schedule_mask
    6. 用当前 schedule_obs 补全上一条 pending Scheduler transition

    7. if schedule_mask 全为 0：
           上层获得 dead-end reward
           写入终止 Architecture transition
           break

    8. Scheduler DQN 选择 SPT/WINQ/CR/MS
    9. 调度规则选择 operation，CSSA 选择 system
   10. 环境执行一道 assignment

   11. 计算 Scheduler reward
   12. 非终止时暂存 Scheduler transition；终止时直接入库

   13. 计算 Architecture reward
   14. 原始 Architecture transition 进入 5-step accumulator

   15. 根据当前训练阶段更新指定网络

返回 episode 指标
```

---

## 12. 训练方案

### 12.1 阶段一：Scheduler 预训练

目的：先获得一个能够在静态可行体系中进行调度的下层策略。

训练设置：

- architecture 在整个 episode 内固定；
- Scheduler 选择四条调度规则；
- 使用一步 replay；
- 主要奖励为负 makespan 增量；
- 随机生成静态可行 architecture 和 mission。

### 12.2 阶段二：Architecture 训练

目的：在固定 Scheduler 的条件下学习体系调整策略。

训练设置：

- 加载 Scheduler checkpoint；
- Scheduler 使用贪心动作，参数冻结；
- Architecture 使用 \(\epsilon\)-greedy；
- Architecture 每个构造步更新一次；
- 使用 5-step replay；
- 场景从四类初始缺陷中采样。

场景课程为：

| 场景类型     | 比例 | 初始体系处理                     |
| ------------ | ---: | -------------------------------- |
| 可行但非最优 |  50% | 向可行体系加入一个随机系统       |
| 容量紧张     |  20% | 每种能力尽量只保留一个短窗口系统 |
| 缺少能力     |  15% | 移除近期所需能力系统             |
| 冗余或超预算 |  15% | 加入最多两个高成本系统           |

### 12.3 阶段三：交替微调

每 10 个 episode 为一组：

- 前 8 个 episode 更新 Architecture；
- 后 2 个 episode 更新 Scheduler；
- 非更新网络继续参与决策；
- Scheduler 微调使用较小学习率；
- 当前实现中的微调 epsilon 固定为 `epsilon_end`。

这种交替方式用于降低两个策略同时快速变化造成的非平稳性。

---

## 13. 推理算法

推理时：

- 加载组合 checkpoint；
- 两层都设置 `epsilon=0`；
- 两层参数均不更新；
- 不向 replay 写入 transition；
- 架构规则和调度规则仍执行相同 action mask。

推理输出至少应包含：

- 是否成功完成 mission；
- 是否 dead end；
- makespan；
- 最终 `net_cost`；
- 最终 `active_cost`；
- `initial_net_cost` 与 `peak_net_cost`；
- `initial_active_cost` 与 `peak_active_cost`；
- `gross_charge`；
- `total_refund`；
- `ever_over_budget` 与 `final_over_budget`；
- architecture change count；
- 六条架构规则计数；
- 四条调度规则计数。

---

## 14. 评估协议

所有比较方法应使用：

- 相同 `eval_seed`；
- 相同 unseen missions；
- 相同初始 architecture；
- 相同 `scenario_hash` 配对。

当前评估方法包括：

| 方法                      | 含义                                      |
| ------------------------- | ----------------------------------------- |
| HRL                       | 两个训练后的 DQN                          |
| Static Initial            | 初始 architecture 固定，只运行 Scheduler  |
| Fixed Architecture Rules  | 能 KEEP 就 KEEP，否则选编号最小的救援规则 |
| Random Architecture Rules | 在有效架构规则中随机选择                  |
| Full System Reference     | 激活完整候选池后运行 Scheduler            |
| Flat IntDQN               | 直接选择 assignment 的扁平 DQN 对照       |

主要指标：

1. mission 成功率；
2. 成功 mission 的平均 makespan；
3. 平均最终净成本；
4. 最终超预算率；
5. 平均体系变化次数；
6. 平均退款额。

当前已经输出的全过程成本指标：

1. `peak_net_cost`；
2. `peak_active_cost`；
3. `ever_over_budget`；
4. `gross_charge`。

`Full System Reference` 只是资源全集下的启发式调度参考，不是精确优化得到的理论最优 makespan。

---

## 15. 默认超参数

| 参数                              | 默认值 |
| --------------------------------- | -----: |
| budget                            |   8000 |
| refund rate                       |    0.8 |
| gamma                             |   0.99 |
| Architecture n-step               |      5 |
| Architecture learning rate        |   1e-4 |
| Scheduler fine-tune learning rate |   1e-5 |
| batch size                        |     64 |
| replay capacity                   |  20000 |
| minimum replay size               |    500 |
| target update interval            |    100 |
| epsilon start                     |    1.0 |
| epsilon end                       |   0.05 |
| epsilon decay                     |  0.995 |
| hidden dimension                  |    256 |

长期 seed 实验脚本当前使用 `hidden_dim=128` 和 `buffer_size=50000`。checkpoint 会保存实际使用的网络和超参数配置。

---

## 16. 运行命令

### 16.1 预训练 Scheduler

```powershell
python -m sosrl train-scheduler `
  --episodes 1000 `
  --scenario-pool-size 100 `
  --rule-set standard `
  --cost-limit 8000 `
  --lr 0.001 `
  --hidden-dim 128 `
  --seed 4 `
  --device cuda `
  --output-dir runs/hrl_scheduler_seed4
```

### 16.2 训练 Architecture

```powershell
python -m sosrl train-architecture `
  --scheduler-checkpoint runs/hrl_scheduler_seed4/scheduler.pt `
  --episodes 1000 `
  --scenario-pool-size 100 `
  --budget 8000 `
  --refund-rate 0.8 `
  --gamma 0.99 `
  --n-step 5 `
  --architecture-lr 0.0001 `
  --batch-size 64 `
  --buffer-size 50000 `
  --min-buffer-size 500 `
  --target-update-interval 100 `
  --epsilon-start 1.0 `
  --epsilon-end 0.05 `
  --epsilon-decay 0.995 `
  --hidden-dim 128 `
  --seed 4 `
  --device cuda `
  --output-dir runs/hrl_architecture_seed4
```

也可以使用已有的兼容 Scheduler checkpoint：

```powershell
--scheduler-checkpoint runs/SIG1000_standard_seed4/model.pt
```

### 16.3 交替微调

```powershell
python -m sosrl finetune `
  --scheduler-checkpoint runs/hrl_scheduler_seed4/scheduler.pt `
  --architecture-checkpoint runs/hrl_architecture_seed4/architecture.pt `
  --episodes 500 `
  --scheduler-finetune-lr 0.00001 `
  --seed 4 `
  --device cuda `
  --output-dir runs/hrl_finetuned_seed4
```

### 16.4 配对评估

```powershell
python -m sosrl evaluate `
  --checkpoint runs/hrl_architecture_seed4/hrl.pt `
  --eval-episodes 100 `
  --eval-seed 20260724 `
  --device cuda `
  --output-dir runs/hrl_evaluation_seed4
```

### 16.5 扁平基线与 Scheduler 配对比较

```powershell
python -m sosrl train-flat --episodes 1000 --seed 1

python -m sosrl compare-schedulers `
  --model SIG=runs/SIG1000_standard_seed4/model.pt `
  --model MIG=runs/MIG1000/model.pt `
  --model MEG=runs/MEG1000/model.pt `
  --eval-episodes 100 `
  --eval-seed 20260724
```

---

## 17. 输出文件

### Scheduler 预训练

```text
scheduler.pt
train_history.csv
eval_results.csv
eval_schedule.csv
config.json
```

### Architecture 训练

```text
architecture.pt
hrl.pt
architecture_history.csv
hrl_config.json
```

### 交替微调

```text
architecture.pt
scheduler.pt
hrl.pt
finetune_history.csv
hrl_config.json
```

### 评估

```text
results.csv
summary.csv
evaluation_manifest.json
```

`hrl.pt` 是组合 checkpoint，包含两个策略的 online network、target network、optimizer、配置和训练进度。Replay Buffer 不写入 checkpoint。

---

## 18. 算法适用边界

当前算法适用于：

- 同一 mission 内离线逐步构造调度；
- 候选系统池已知；
- 系统具有单一能力类型、固定成本和固定时间窗；
- 已排定 operation 不回滚；
- 体系变化通过增、删、同能力替换表示。

当前算法不直接解决：

- 现实执行阶段的事件驱动在线重调度；
- 系统内部结构、接口或能力参数演化；
- operation 抢占和历史排程重排；
- 严格全过程预算可行性；
- 连续动作或大规模组合架构的端到端搜索；
- 显式 Pareto 多目标优化。

---

## 19. 实现对应表

| 算法组成                                      | 实现文件                                     |
| --------------------------------------------- | -------------------------------------------- |
| mission、system 与场景数据                    | `sosrl/domain.py`、`config.json`             |
| 统一静态/动态 HRL 环境                        | `sosrl/environment.py`                       |
| Scheduler、Huang 与 Architecture 规则         | `sosrl/rules/`                               |
| 网络、Replay、配置与 checkpoint               | `sosrl/rl/`                                  |
| 场景、episode、训练与评估                     | `sosrl/workflows/`                           |
| 扁平 IntDQN 对照                              | `sosrl/baselines/`                           |
| 六个统一命令                                  | `sosrl/cli.py`、`sosrl/__main__.py`          |
| 绘图和成本轨迹报告                            | `scripts/`                                   |
| 环境与算法测试                                | `tests/`                                     |

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

固定 seed4 行为与 CPU 性能回归：

```powershell
python -m scripts.benchmark_hrl `
  --checkpoint runs/hrl_budget20_seed4/architecture_1000/hrl.pt `
  --expected tests/fixtures/seed4_hrl_regression.json `
  --max-seconds 4.97
```

训练曲线和全过程成本事件报告分别由 `scripts.plot_history` 与 `scripts.report_cost_trajectory` 从 `runs/` 中重建；生成内容写入被 Git 忽略的 `reports/` 或 `outputs/`。
