# GP–BDQN 体系架构自适应与任务规划：文献定位、创新点与后续实验计划

> 版本：2026-08-23  
> 适用实现：上层 Direct GP 逐步选择具体 `KEEP/ADD/REMOVE/REPLACE`，下层冻结的受约束加性 Branching DQN 选择合法 `(task, system)` 对。

## 0. 执行摘要

当前方法最合适的学术定位是：

> **面向任务的 operation 级体系架构自适应—资源规划协同方法：通过遗传规划离线进化一个可解释的状态—动作评分函数，在线排序合法的具体架构变更；随后由冻结的、实体结构化的 Branching DQN 在变更后的可行域中完成任务—系统联合选择。**

当前方法不应表述为“首次将 GP 与强化学习结合”“首次联合优化重构和调度”或“首次用 AI 进行 SoS 架构设计”。这些宽泛优先权主张已经被相邻研究覆盖。真正有辨识度的组合在于：

1. SoS 架构不是 mission 前一次性选择，而是在每个前沿 operation 分配前允许 `KEEP/ADD/REMOVE/REPLACE`；
2. GP 不是选择人工宏规则，也不是生成完整架构，而是用带反事实增量的状态—动作特征直接排序当前合法的具体变更；
3. 上层符号策略与下层神经规划器异构解耦，GP 的端到端适应度由冻结 BDQN 的完整 rollout 给出；
4. 下层用实体编码、受约束的任务—系统 pair mask 和全局联合 argmax，避免直接输出固定的 (T\times N) 联合动作头；
5. 训练、验证、IID/OOD 测试、策略部署与下层参数冻结均有可审计产物。

现有结果可视为有希望的 pilot，但还不能作为最终论文证据。当前 `120×50×3` GP 实验仅绑定一个 seed-4 BDQN checkpoint，OOD 只改变了任务数、工时范围和预算；主对照中也缺少 Huang 风格静态外层搜索、小规模精确解和更强的 GP/联合规划基线。

## 1. 当前实现与证据审计

### 1.1 当前算法闭环

每个构造步执行：

1. 枚举当前合法的具体 `KEEP/ADD(j)/REMOVE(i)/REPLACE(i,j)`；
2. 为每个候选计算 39 维 `system_delta` 状态—动作特征；
3. 固定 GP 表达式输出候选分数并选择最小者；
4. 执行架构动作并刷新可行性和 `decision_version`；
5. 重新构造 BDQN 的任务实体、系统实体和 (T\times N) pair mask；
6. 冻结 BDQN 选择 `(task_idx, system_idx)`，执行该 task 当前前沿 operation。

当前动作全集为 203 个：1 个 KEEP、22 个 ADD、22 个 REMOVE、158 个同能力有向 REPLACE。GP 的候选过滤只保证动作后至少存在一个当前合法 pair，故它保证**即时可调度性**，不保证剩余任务的全局可完成性。

### 1.2 当前 pilot 结果

| 测试 | 方法 | 成功率 | 平均 makespan | 最终净成本 | 峰值净成本 | 曾超预算率 | 架构变化数 |
|---|---|---:|---:|---:|---:|---:|---:|
| IID, 500 | fixed + BDQN | 48.4% | 466.6 | 5382.2 | 5382.2 | 29.0% | 0.00 |
| IID, 500 | manual-6 DQN + BDQN | 100.0% | 524.5 | 3260.4 | 7137.9 | 32.6% | 15.52 |
| IID, 500 | direct GP + BDQN | 100.0% | 498.1 | 2895.5 | 7418.7 | 29.8% | 15.71 |
| OOD, 200 | fixed + BDQN | 31.5% | 583.1 | 5505.1 | 5505.1 | 31.0% | 0.00 |
| OOD, 200 | manual-6 DQN + BDQN | 99.0% | 729.7 | 4543.8 | 7687.0 | 50.0% | 15.47 |
| OOD, 200 | direct GP + BDQN | 98.0% | 700.1 | 3702.5 | 7729.3 | 33.0% | 15.18 |

相对 manual-6 DQN，GP 在 IID 上将平均 makespan 降低约 5.0%、最终净成本降低约 11.2%；在当前 OOD 上分别降低约 4.1% 和 18.5%。但是：

- OOD 成功率由 99% 降到 98%，必须做失败归因，不能只强调连续指标；
- fixed 的 makespan 只来自少数成功实例，不能用其较低均值宣称更优；
- GP 峰值净成本略高于 manual-6 DQN，说明最终成本优势可能来自后续退款，而非全轨迹预算控制；
- 当前 GP 使用 3 次独立进化而非预设的 10 次，且只使用一个 BDQN 训练随机种子；
- 500/200 个测试场景提高了场景层精度，但不能替代算法训练种子的独立重复。

### 1.3 当前 OOD 的实际强度

现有 OOD 仅包含：

- task 数从 30 增至 40；
- operation duration 从 `[20,30]` 变为 `[15,35]`；
- 预算从 8000 改为 6400 或 9600。

候选系统池、能力种类、场景构造逻辑和四类初始架构缺陷没有改变。因此当前 OOD 应称为**近分布规模—预算偏移**，不能称为跨系统池、跨约束结构或动态扰动泛化。

## 2. 最接近的文献与差异

### 2.1 SoS 架构选择与任务规划

| 文献 | 已有贡献 | 与本项目的关键差异 |
|---|---|---|
| Lin et al. (2023), *When architecture meets AI* | Actor–critic 直接完成 SoS 系统选择和任务分配，强调近实时与跨规模泛化 | 直接神经组合构造；没有 operation 级 ADD/REMOVE/REPLACE 轨迹，也没有符号 GP 上层和独立 BDQN 规划器 |
| Huang et al. (2023), *When architecture meets RL+EA* | 外层 SAEA/GA 搜索静态 SoS 架构，内层 DQN 通过调度规则生成 mission plan 并返回适应度 | 架构在完整下层规划期间固定；本项目在每个 assignment 前在线重构，并直接评分具体架构动作 |
| Fang et al. (2025), incremental SoS evolution path | PPO 选择多阶段架构演化路径，考虑不确定性、依赖和未来能力 | 阶段级架构路径与能力评估；没有 operation 级 task–system 调度器，也不是可部署的符号策略 |

结论：本项目不能声称首次“任务驱动 SoS 架构设计”，但可以检验其是否首次在该问题设定下实现**operation 粒度的符号架构动作排序与结构化神经规划耦合**。最终论文使用“据我们所检索的文献”而非绝对优先权措辞。

### 2.2 重构—调度并发优化

| 文献 | 已有贡献 | 对本项目的约束与启示 |
|---|---|---|
| Yang et al. (2023), CIRP-JMST | 两个协同 DRL agent 分别负责动态可重构 flow shop 的调度和重构，毫秒级决策 | 已覆盖“在线并发重构与调度”的宽泛主张；本项目差异必须落到 SoS 动作语义、符号 GP、pair-BDQN 和成本/退款轨迹 |
| Yang et al. (2024), AEI | 将 EDQN 扩展到分布式动态可重构车间，并报告跨配置归一化和泛化 | 要求本项目补充真正的跨系统池、跨规模与推断时间实验 |
| Guo et al. (2024), JMSY | GPHH 同时处理可重构制造单元能力、路由和排序，并考虑重构时间 | 已覆盖“GP 用于重构和调度”；本项目须证明异构 GP–BDQN 分解优于纯 GP 多规则协同及人工规则 |
| Guo et al. (2025), ESWA | 多群体 GP 协同进化工艺规划、单元重构、任务调度和资源分配规则 | 对本项目构成强纯 GP 基线；也提示需要报告 bloat、规则复杂度和离线求解成本 |

### 2.3 GP、RL 与分层调度的组合

| 文献 | 已有贡献 | 与本项目的关键差异 |
|---|---|---|
| Zhang et al. (2021), IEEE TCYB | GP 同时进化 dynamic FJSP 的路由与排序规则，并做特征选择以提高性能和可解释性 | 纯 GP 的两条规则协同；本项目由 GP 负责架构变更、BDQN 负责任务—系统联合选择 |
| Luo et al. (2021), C&IE | 两层 DQN：高层选临时优化目标，低层选复合调度规则 | 上层不改变下层物理资源可行域；本项目上层动作立即改变本步 pair mask |
| Kim et al. (2025), C&IE | VNS-GP 先产生高质量 priority rules，DQN 在每个调度点在线选择规则 | 这是最接近的 GP–DRL 混合之一，但 GP 与 DQN 的角色相反；没有 SoS 具体架构动作轨迹或 Branching task–system planner |
| Tian et al. (2025/2026), JEM | 上层改进 GA 生成批次方案，下层 PS-DDQN 调度并反馈适应度 | 与 Huang 类似，是“外层组合方案—完整下层评价”，不是 operation 级在线架构重构 |
| Hein et al. (2018), EAAI | 用 GP 学习可解释的符号 RL policy | 支撑“符号策略可部署”的方法动机，但不能替代本项目在 SoS/规划问题上的实证创新 |

### 2.4 大动作空间与可行域处理

Tavakoli et al. (2018) 的 Branching Dueling Q-Network 通过共享决策模块和动作分支使网络输出随动作维度线性增长。当前 SOSRL 借鉴该思想，但其下层不是原始 BDQ 的严格复现，而是：

\[
Q(s,i,k)=V(s)+\widetilde A_{task}(s,i)+\widetilde A_{sys}(s,k),
\]

然后在合法的 (T\times N) pair mask 上做联合 argmax。论文中应称为**受约束加性 Branching DQN**。该模型不含 task–system 显式交互项，这既是结构简化，也是一项必须消融的表达能力限制。

动作 masking 和状态依赖合法动作集已有充分研究，不能单独作为创新点；本项目可强调的是 mask 在两层之间的因果作用：GP 动作改变架构后重新计算 pair mask，再由 BDQN 决策。

## 3. 可辩护的创新点

### C1. 问题建模创新：operation 级架构自适应—规划耦合

将传统“先选定静态架构，再完整规划”的嵌套优化改为 operation 粒度的序列过程：每次分配前先进行一次可逆/有成本的体系变更，再在新可行域中规划。研究对象从单一终态架构变为：

\[
(z_0,a_0^A,z_1,a_0^S,\ldots,z_T),
\]

即架构演化轨迹与部分调度轨迹的联合结果。该创新是否成立，需要静态 Huang-like 基线和动态扰动实验共同证明。

### C2. 方法创新：带反事实增量的具体架构动作符号排序

GP 对每个当前合法的具体动作 (a\) 计算 (F_\theta(\phi(s,a))\)，而不是：

- 选择六个人工架构宏规则；
- 直接输出 22 维系统 mask；
- 记忆系统编号；
- 只根据当前状态输出一个固定类别动作。

39 维特征中的反事实增量直接描述动作对成本、能力覆盖、合法 pair、阻塞和前沿完成时间的局部影响。这一主张必须由 `system/system_demand/system_delta/op_context` 消融、删除 delta 特征消融和反事实探针实验支持。

### C3. 算法架构创新：符号上层与冻结结构化神经下层的异构模块化耦合

GP 的适应度由 BDQN 完整 rollout 反馈，但 BDQN 参数保持不变；线上 GP 分数不输入 BDQN，BDQN 也不参与架构选择。该设计同时提供：

- 上层策略可读、易验证、低延迟；
- 下层对可变任务/系统实体的表示能力；
- 两层可独立训练、替换和审计；
- 避免联合端到端训练的不稳定性。

这一创新必须通过“冻结 vs 微调”“跨 BDQN checkpoint 交换”和“GP 对单一下层策略是否过拟合”实验验证，不能只靠架构图声明模块化。

### C4. 支撑性创新：受约束加性 task–system 联合价值模型

下层用共享实体编码、分支优势和 pair mask 将输出规模从显式 (TN) logits 降为 (1+T+N)，同时仍在完整合法 pair 上做联合选择。该贡献属于对 BDQ 的问题特定改造，不宜作为首要理论创新。应与联合 pair MLP、低秩交互项和规则式 Scheduler DQN 比较。

### C5. 支撑性创新：可复现的演化、选模和部署协议

冻结场景 manifest、训练/验证/测试隔离、候选表达式去重、parsimony、策略 JSON、系统池/checkpoint hash、加载时 AST 白名单以及下层权重前后 hash，有助于把 GP 从一次性实验规则变成可部署策略。它是工程和实验可信度贡献，不宜替代算法效果贡献。

### 3.1 不应使用的创新措辞

- “首次将 GP 与 DQN/DRL 结合”；
- “首次同时优化重构和调度”；
- “首次使用 GP 生成调度/重构规则”；
- “首次使用分层 RL 做调度”；
- “首次使用 action masking/BDQN 处理大动作空间”；
- “GP 策略天然可解释”。可读公式只是可解释性的前提，仍需稳定性、敏感性和案例验证。

## 4. 研究问题与预注册假设

| RQ | 研究问题 | 可证伪假设 |
|---|---|---|
| RQ1 | GP–BDQN 是否优于静态架构、人工架构规则和 Huang-like 外层搜索？ | 在不降低成功率的前提下，proposed 的 all-scenario 代价、最终净成本和 both-success makespan 更低 |
| RQ2 | 改善来自 GP、反事实特征、具体动作空间还是合法候选过滤？ | 删除 delta 特征、改成宏规则或弱化可行性过滤会显著降低 OOD 成功率或增加代价 |
| RQ3 | GP 是否只适配了 seed-4 BDQN？ | 同一 GP 在未见 BDQN checkpoints 上仍保持主要优势；跨 checkpoint 性能下降有限 |
| RQ4 | 加性 BDQN 是否足以表达 task–system 关系？ | 加入低秩 pair 交互后若效果无显著改善，则加性模型是合理的效率—性能折中 |
| RQ5 | 方法能否跨任务规模、系统池、能力分布、时间窗和预算泛化？ | 在预定义 near/far OOD 上成功率下降和归一化代价增长受控，并优于对照 |
| RQ6 | operation 级架构调整在扰动下是否比静态或阶段级方法恢复更快？ | 系统失效、任务突发、预算削减后，proposed 的恢复步数、性能损失面积和重规划延迟更小 |
| RQ7 | 符号策略是否真正稳定、可理解和可部署？ | 不同 GP seeds 的关键特征与行为模式稳定；简化规则保持高决策一致率和相近性能 |
| RQ8 | 离线训练成本能否由在线收益摊销？ | 报告 break-even mission 数，并在目标部署频率下优于重复在线搜索 |

## 5. 评价指标与统计口径

### 5.1 主指标采用分层顺序

1. **Mission success/dead-end rate**：第一主指标；
2. **全实例综合代价**：沿用 failure-first 的 (J)，同时单独报告每项组成；
3. **both-success paired makespan**：只在两个方法都成功的同一实例上比较，并报告样本数；
4. **最终净成本、峰值净成本、gross charge、refund、ever/final budget violation**；
5. **架构变化数与抖动**：变化次数、ADD–REMOVE 循环数、同一系统重复加入数；
6. **效率**：离线训练环境步数/rollout 数、wall-clock、CPU/GPU、线上每步 p50/p95/p99 延迟、内存；
7. **最优性差距**：小规模实例相对 CP-SAT/MILP/动态规划最优解或下界的 gap。

禁止把失败实例排除后得到的低 makespan 当成整体优势。表格按“成功率 → 全实例代价 → both-success 连续指标”的顺序展示。

### 5.2 统计设计

- 所有方法使用相同 scenario IDs、初始 architecture 和随机扰动，实现 paired comparison；
- 场景不能冒充训练重复。统计抽样单元至少包含“训练得到的独立 stack”和“测试场景”两层；
- 主结果使用**分层 bootstrap**：先重采样独立训练 stack，再在每个 stack 内重采样成对场景，报告 95% CI；
- 成功/失败用 paired proportion difference，并以 McNemar 检验或 paired bootstrap 辅助；
- 连续差值报告均值/中位数、95% CI 和 paired effect size；
- 同一 RQ 内多重比较采用 Holm 校正；
- 跨场景族汇总同时报告 IQM 和 performance profile，避免单个均值主导；
- 所有超参数和候选模型只用训练/验证集决定，Test-IID/OOD 只运行一次锁定分析；
- 预先指定最小有意义差异，例如成功率 1 percentage point、makespan 3%、最终成本 5%，再做功效分析决定训练 stack 数。

## 6. 实验矩阵

### E0. 正确性与协议锁定（必须先完成）

目的：确认比较的差异来自算法而非实现语义。

- 架构候选枚举和特征提取前后环境 state hash 不变；
- 所有执行动作属于当前合法候选，`decision_version` 过期动作必被拒绝；
- GP 演化前后 BDQN online/target 参数 hash 一致；
- 固定 checkpoint、策略和 scenario manifest 时结果逐步可复现；
- 所有基线共享同一成本、退款、时间窗、dead-end 和终止语义；
- 记录每个失败的首个不可恢复原因：缺能力、时间窗耗尽、预算策略、局部可行但未来失效、下层 pair 选择失误或 provider invariant violation。

交付：一个 `protocol_manifest.json`、一致性测试报告和失败类型字典。

### E1. 主效果与强基线

| 编号 | 方法 | 回答的问题 | 优先级 |
|---|---|---|---:|
| B0 | 小规模 CP-SAT/MILP/DP | 最优性 gap 与环境正确性 | P0 |
| B1 | fixed architecture + BDQN | 是否需要在线架构适应 | P0，已有 |
| B2 | random legal concrete action + BDQN | 合法过滤本身能达到什么水平 | P0，已有 |
| B3 | myopic counterfactual heuristic + BDQN | GP 是否超过人工单步代价规则 | P0，需补 |
| B4 | manual-6 Architecture DQN + BDQN | 具体符号动作是否优于人工宏规则 | P0，已有 |
| B5 | Huang-like static GA/SAEA + frozen BDQN | operation 级适应是否优于 mission 前静态搜索 | P0，需补 |
| B6 | direct GP + rule Scheduler DQN/CSSA | 下层 BDQN 的增益 | P1 |
| B7 | co-evolved GP routing/sequencing/reconfiguration rules | 是否需要异构 GP–BDQN 分解 | P1/算力允许 |
| B8 | concurrent Architecture DQN + BDQN | 两个神经 agent 的并发重构—规划基线 | P0，可复用现有实现 |
| B9 | proposed direct GP + frozen BDQN | 主方法 | P0 |
| B10 | joint neural pair scorer/PPO | 端到端联合动作上界型基线 | P2/算力允许 |

公平性有两种口径，必须同时报告：

1. **训练/搜索预算匹配**：相同 environment rollout 数或相同 wall-clock；
2. **部署口径**：离线训练成本分开，比较锁定策略的在线决策时间和效果。

Huang-like GA 的个体是一份 mission 内固定的系统子集，适应度由同一 frozen BDQN 完整规划返回。这样可直接隔离“静态外层组合搜索”与“operation 级在线动作策略”的差异。

### E2. GP 表示、特征与 fitness 消融

#### E2.1 特征组

- `system`：21 维；
- `system_demand`：29 维；
- `system_delta`：39 维主方法；
- `op_context`：operation-heavy 对照；
- `system_delta - counterfactual deltas`；
- `system_delta - cost/budget`；
- `system_delta - demand/pressure`。

#### E2.2 动作和可行域

- 宏规则 GP：只选择 KEEP/ADD/REMOVE/REPLACE 类别，再用确定性 resolver；
- 具体动作 GP：当前主方法；
- 去除 REPLACE；
- 仅 KEEP+ADD；
- 当前“即时至少一个合法 pair”过滤；
- “剩余能力覆盖”保守过滤；
- (h=1/4) 步 BDQN look-ahead 过滤或风险特征；
- 软预算惩罚 vs 硬预算 mask。

“无合法过滤、靠惩罚学习”只做小规模诊断，避免大量无意义失败和不公平训练预算。

#### E2.3 演化和选模

- failure-first lexicographic fitness vs 单一加权和；
- 去掉峰值超预算项；
- 去掉架构变化项；
- parsimony 系数 `0/0.001/0.005`；
- 每代随机小批场景 vs 固定训练集；
- 无 anchor 复评；
- validation 的 1% 近优简约选择 vs 只选最低均值；
- 单树 vs 小型 ensemble（仅当单树 OOD 方差明显）。

每项消融只改变一个因素；先用 5 个 GP seeds 做筛选，再对关键消融使用完整重复。

### E3. 下层规划器与两层耦合

#### E3.1 BDQN 结构消融

- 四规则 Scheduler DQN；
- 当前加性 BDQN：(V+A_t+A_s)；
- 显式 joint pair MLP：对每个合法 `(task,system)` 拼接 embedding 打分；
- 低秩交互：(V+A_t+A_s+u_t^\top v_s)；
- 去掉 global encoder、task pooling 或 system pooling；
- pair mask vs 先独立选 task/系统再修复。

#### E3.2 冻结、微调与模块互换

- `G0+B0`：固定 GP + 原 BDQN；
- `G0+B1`：固定 GP 下微调 BDQN；
- `G1+B0`：重新进化 GP、BDQN 不变；
- `G1+B1`：交替/联合微调（探索性，不作为首篇主方法）。

#### E3.3 跨 checkpoint 矩阵

训练 (K) 个独立 BDQN checkpoints (B_1,ldots,B_K)，对每个 (B_i) 进化 GP (G_i)，评估所有 (G_i+B_j)。输出 (K\times K) heatmap：

- 对角线高、非对角线明显下降：GP 与特定下层共适应，模块化主张较弱；
- 非对角线稳定：符号上层具有 plug-compatible 泛化；
- 若共适应明显，下一版本可让 GP fitness 对多个 frozen BDQN checkpoint 求均值或 worst-case。

这是当前只绑定 seed-4 checkpoint 时最重要的补充实验之一。

### E4. 泛化与规模实验

#### E4.1 近分布 OOD

- (T\in\{20,30,40,50\})；
- (O\in\{3,4,5\})；
- duration 均值和方差分别偏移；
- budget 按“最小静态可行成本”的比例设为 `{0.8,1.0,1.2}`，避免绝对预算与规模混淆。

#### E4.2 远分布 OOD

- 新 system pool：系统数、成本、窗口、能力比例全部重新采样；
- (N\in\{11,22,33,44\})；
- 成本—窗口长度相关性从负相关、独立到正相关；
- time-window tightness 分为宽、训练内、窄三档；
- capability demand 用不同 Dirichlet 浓度生成，覆盖均衡和偏科任务；
- 对系统顺序做随机置换，验证模型没有 index leakage；
- 初始架构缺陷比例从训练的等量四类改为自然混合和极端单类。

当前 GP artifact 绑定 system-pool hash，跨系统池测试需要区分：

1. **特征/策略泛化实验**：允许加载同一表达式，但重新校验 schema；
2. **部署安全模式**：系统池 hash 不一致时拒绝加载。两者不能混为一个运行模式。

增加第四/第五能力类型需要修改当前固定三维 one-hot 和聚合 schema，属于方法扩展，不应在不改模型时宣称支持。

#### E4.3 设计方法

完整笛卡尔积成本过高。建议用预先生成的 30–50 个 scenario families 做分层 Latin hypercube/正交抽样，每族 100–200 个固定实例；另设 5–10 个极端 stress families。按 family 报告性能 profile，而不是只给一个总均值。

### E5. 动态扰动与韧性实验

现有环境本质上是离线构造式调度。要支持该实验，先增加可重放的 disturbance hook，并保证历史 assignment 不回滚。扰动在 mission 进度 25%、50%、75% 注入：

| 扰动 | 强度 | 需要记录 |
|---|---|---|
| active system 失效 | 1 个/同能力 30% | 被影响后续 operations、恢复动作、dead-end 原因 |
| 时间窗缩短/延迟到达 | 20%/40% | 新旧可行 pair 数与恢复步数 |
| operation 工时偏差 | ×`[0.8,1.2]` / `[0.6,1.5]` | makespan shock 与计划稳定性 |
| task burst/new mission tasks | +10%/+30% | 新任务等待、重规划延迟 |
| 临时预算削减 | -10%/-25% | 峰值违规、移除/替换代价 |
| 新系统到达/成本变化 | 1–3 个/±20% | 策略是否利用新能力而不过度抖动 |

韧性指标：扰动后成功率、归一化性能损失面积、恢复到扰动前 90% 可行水平的决策步数、额外架构变更、额外成本、重规划 p95 延迟。

第一篇论文可先完成系统失效、窗口缩短和预算削减三类；通信受限、部分可观测和中继依赖需要新的网络/观测模型，适合作为下一阶段，而不是当前算法的小消融。

### E6. 小规模精确解与下界

建立两个互补 oracle：

1. **固定架构精确调度**：CP-SAT/MILP 求 task–operation–system assignment 与 makespan，评估 BDQN 的规划 gap；
2. **静态架构选择 + 精确调度**：枚举或 MILP 选择系统子集并求最优调度，评估 Huang-like 与 proposed 终态方案；
3. **有限变更的动态 oracle**：在 (T\leq6,O\leq3,N\leq8) 且限制最多 1–2 次架构变更时，用动态规划/枚举求联合最优。

建议规模：

- `T={3,5,8}`，`O={2,3}`，`N={6,8,10}`；
- 每个配置 50–100 个实例；
- solver time limit 同时报告 60 s 和 600 s；
- 未证最优时报告 incumbent、best bound 和 gap，不把 incumbent 当最优解。

### E7. 可解释性与策略行为

当前胜出表达式主要使用 `added_cost_ratio`、`delta_best_frontier_finish_norm`、`budget_excess_after_ratio`、`target_type_pressure` 和 `delta_net_cost_ratio`。仅列出这些变量不足以证明可解释性，应增加：

- 10–30 个独立 GP runs 中特征出现频率、语义等价表达式和行为一致性；
- 节点数/高度与 validation/test 性能的 Pareto 图；
- 代数简化或语义简化前后的 action agreement、success 和代价；
- 对成本、压力、完成时间增量做单变量和双变量反事实扫描；
- 检查常识性单调性：其他条件不变时，更高超预算是否降低 ADD 倾向，完成时间改善是否提高动作偏好；
- 代表性成功/失败轨迹：每步 top-3 候选、分数 margin、架构动作、pair 数和 BDQN 动作；
- 与专家规则的 disagreement cases，由人解释 GP 是发现了有效非直觉关系还是利用了模拟器漏洞；
- 对输入加小扰动，统计 action flip rate，识别决策边界脆弱区域。

受保护、裁剪的算子意味着普通代数恒等变换未必保持语义；所有简化必须以全验证集候选评分/动作等价和测试性能复核。

### E8. 计算效率与摊销

分别报告：

- BDQN 训练：环境步数、wall-clock、硬件、参数量；
- GP 进化：候选表达式数、去重率、完整 rollout 数、worker 数、wall-clock；
- validation selection 和 test 的额外成本；
- 在线每步：候选枚举、特征构造、GP 打分、BDQN 编码/推断、mask/argmax 的分项时间；
- 随 (T,N,lvert\mathcal A(s)\rvert) 增长的 p50/p95 延迟和内存；
- 与静态 GA/SAEA 在线重搜相比的 break-even mission 数：

\[
N_{break-even}=
\frac{C_{offline}^{GP}+C_{offline}^{BDQN}}
{C_{online}^{baseline}-C_{online}^{GP+BDQN}}.
\]

同时给出“只算部署推断”和“包含全部离线训练”的两种结论。

## 7. 随机种子与算力方案

### 7.1 当前 pilot

- 1 个 BDQN seed；
- 3 个 GP runs；
- population 120、generation 50；
- IID 500、near-OOD 200。

只用于发现问题、估计效应和运行时间。

### 7.2 论文最低方案

- 5 个独立 BDQN training seeds；
- 每个 BDQN 上 5 个独立 GP runs，validation 选 1 个，得到 5 个独立 stacks；
- 每个 stack 测试 1000 IID、1000 near-OOD、每个 far-OOD family 100–200；
- 主基线至少在相同 5 个 BDQN checkpoints 和完全相同 scenarios 上运行；
- 对最关键比较做分层 bootstrap，不将数十万 episode 当作独立算法重复。

### 7.3 推荐确认性方案

- 10 个 BDQN seeds；
- 每个 BDQN 上 5 个 GP runs；若 pilot 显示 GP seed 方差大，再增至 10；
- 预先用 pilot 方差和最小有意义差异做功效分析；
- 对 `G_i+B_j` 做至少 (5\times5) 跨 checkpoint 矩阵；
- GP 超参数只在少量训练 seeds 上定一次，不能为每个 test family 单独调参。

当前 `120×50×3` 约产生 (120\times50\times16\times3=288,000) 个世代内场景 rollout（未扣表达式去重），已耗费数小时。直接把标准 `200×80×10` 再乘 10 个 BDQN seeds 会产生约 2560 万个世代内 rollout。建议先完成 E1/E2 pilot 和运行时建模，再锁定最终种子数；不要一开始盲目执行完整笛卡尔积。

## 8. 分阶段实施计划

### Phase 0：协议与缺口修复（3–5 天）

- 锁定主指标、失败定义、paired scenario manifests 和 test 冻结规则；
- 增加失败 taxonomy、top-k action trace 和分项计时；
- 核对 summary 中 makespan 的条件口径；
- 实现 hierarchical bootstrap、paired success difference、Holm correction；
- 输出当前 pilot 的标准化报告，作为功效分析输入。

退出条件：同一策略/场景可逐步重放；任何失败可定位到唯一首因；统计脚本通过合成数据测试。

### Phase 1：论文核心证据（1–2 周）

- 实现 myopic heuristic 和 Huang-like static GA + frozen BDQN；
- 训练至少 5 个 BDQN seeds；
- 每个 checkpoint 完成 5 个 GP runs；
- 运行 E1 主表、E2 关键特征消融和 E3 跨 checkpoint；
- 小规模实现固定架构精确调度 oracle。

退出条件：C1–C3 每项至少有一个直接消融或强基线支持；主结论在训练 stack 层有 CI。

### Phase 2：泛化、动态性与机制（2–3 周）

- 参数化 scenario generator，生成 near/far OOD families；
- 完成系统排列不变性、新 system pool、规模/预算/窗口压力测试；
- 增加系统失效、窗口缩短、预算削减 disturbance hooks；
- 完成低秩 pair interaction 与 freeze/fine-tune 消融；
- 运行可解释性反事实探针和失败案例分析。

退出条件：能够明确回答 GP 的收益来自何处、在哪些 OOD/扰动条件下失效、是否依赖特定 BDQN。

### Phase 3：确认性复验与论文产物（1–2 周）

- 根据预先功效分析补足 10 个 stacks 或所需样本；
- 锁定代码 commit、环境和所有 manifests 后执行一次 final test；
- 生成主表、消融表、性能 profile、Pareto 图、OOD heatmap、cross-checkpoint heatmap、典型轨迹图；
- 打包策略 JSON、checkpoints、配置、场景 hashes、硬件和完整命令。

## 9. 预期论文图表

1. **方法图**：GP 合法候选—反事实特征—具体架构动作—新 pair mask—BDQN 联合规划；
2. **文献差异表**：静态/动态架构、GP/RL 角色、动作粒度、下层规划、可解释性；
3. **主结果表**：success、all-scenario (J)、both-success makespan、成本/预算/变化、延迟；
4. **性能 profile**：跨 scenario families 的 IQM 与 threshold profile；
5. **特征/fitness 消融表**；
6. **OOD heatmap**：任务规模 × 系统规模 × 预算压力；
7. **cross-BDQN heatmap**：(G_i+B_j)；
8. **效率—质量 Pareto 图**：在线时延、离线训练成本与效果；
9. **符号策略解释图**：反事实曲面与代表性 action trace；
10. **失败归因图**：各方法在 IID/OOD/扰动下的失败类型比例。

## 10. 论文贡献表述建议

可以使用：

> We formulate task-oriented SoS adaptation and planning as an operation-level sequential process in which each concrete architecture modification changes the feasible task–system assignment set of the subsequent planning decision.

> We evolve a compact symbolic state–action scoring policy over legal KEEP/ADD/REMOVE/REPLACE candidates using counterfactual architecture and schedulability features, rather than selecting handcrafted macro-rules or emitting a full architecture vector.

> We couple the symbolic upper policy with a frozen, entity-structured additive Branching DQN planner and evaluate modularity through cross-checkpoint transfer, controlled fine-tuning, and paired IID/OOD/disturbance experiments.

中文概括：

> 本文将任务导向的体系架构选择由 mission 前静态组合优化推进为 operation 粒度的架构自适应—任务规划协同；提出基于反事实状态—动作特征的直接遗传规划策略，对合法具体架构变更进行符号排序，并与冻结的实体结构化受约束加性 BDQN 规划器解耦耦合。

## 11. 优先级结论

如果资源有限，先完成以下六项，其他实验可后置：

1. **Huang-like static GA + frozen BDQN**；
2. **5 个以上独立 BDQN seeds × 每个 5 个 GP runs**；
3. **`system_delta` 与无 delta 的特征消融**；
4. **跨 BDQN checkpoint 的 (G_i+B_j) 矩阵**；
5. **真正的 far-OOD：新 system pool + 规模/窗口/预算压力**；
6. **OOD 失败案例与“即时可行、未来 dead-end”归因**。

这六项直接决定 C1–C3 是否成立。动态扰动、低秩交互、纯 GP 多规则和完整精确动态 oracle 很有价值，但应在核心创新被证实后投入。

## 12. 主要参考文献

1. Lin, M., Chen, T., Chen, H., Ren, B., & Zhang, M. (2023). *When architecture meets AI: A deep reinforcement learning approach for system of systems design*. Advanced Engineering Informatics, 56, 101965. https://doi.org/10.1016/j.aei.2023.101965
2. Huang, Y., Luo, A., Chen, T., Zhang, M., Ren, B., & Song, Y. (2023). *When architecture meets RL+EA: A hybrid intelligent optimization approach for selecting combat system-of-systems architecture*. Advanced Engineering Informatics, 58, 102209. https://doi.org/10.1016/j.aei.2023.102209
3. Fang, Z., Chen, D., Ju, Q., & Wang, J. (2025). *Architecting Path Selection Method for Incremental Evolution in System-of-Systems*. IEEE Systems Journal, 19(2), 636–647. https://doi.org/10.1109/JSYST.2025.3553965
4. Yang, S., Wang, J., Xin, L., & Xu, Z. (2023). *Real-time and concurrent optimization of scheduling and reconfiguration for dynamic reconfigurable flow shop using deep reinforcement learning*. CIRP Journal of Manufacturing Science and Technology, 40, 243–252. https://doi.org/10.1016/j.cirpj.2022.12.001
5. Yang, S., Wang, J., & Xu, Z. (2024). *Learning to schedule dynamic distributed reconfigurable workshops using expected deep Q-network*. Advanced Engineering Informatics, 59, 102307. https://doi.org/10.1016/j.aei.2023.102307
6. Guo, H., Liu, J., Wang, Y., & Zhuang, C. (2024). *An improved genetic programming hyper-heuristic for the dynamic flexible job shop scheduling problem with reconfigurable manufacturing cells*. Journal of Manufacturing Systems, 74, 252–263. https://doi.org/10.1016/j.jmsy.2024.03.009
7. Guo, H., Li, K., Liu, J., Zhuang, C., & Pei, F. (2025). *Dynamic integrated process planning and scheduling under multi-resource constraints in workshops with reconfigurable manufacturing cells: a novel hyper-heuristic approach*. Expert Systems with Applications, 289, 128337. https://doi.org/10.1016/j.eswa.2025.128337
8. Kim, H.-I., Kim, Y.-R., & Lee, D.-H. (2025). *A genetic programming based reinforcement learning algorithm for dynamic hybrid flow shop scheduling with reworks under general queue time limits*. Computers & Industrial Engineering, 203, 111062. https://doi.org/10.1016/j.cie.2025.111062
9. Tian, J., Zhou, X., Leng, J., & Zhang, Y. (2025/2026). *A batch scheduling method for multi-production line hybrid flow shop integrating genetic algorithm and deep reinforcement learning*. Journal of Engineering Manufacture. https://doi.org/10.1177/09544054251400818
10. Zhang, F., Mei, Y., Nguyen, S., & Zhang, M. (2021). *Evolving Scheduling Heuristics via Genetic Programming With Feature Selection in Dynamic Flexible Job-Shop Scheduling*. IEEE Transactions on Cybernetics, 51(4), 1797–1811. https://doi.org/10.1109/TCYB.2020.3024849
11. Luo, S., Zhang, L., & Fan, Y. (2021). *Dynamic multi-objective scheduling for flexible job shop by deep reinforcement learning*. Computers & Industrial Engineering, 159, 107489. https://doi.org/10.1016/j.cie.2021.107489
12. Tavakoli, A., Pardo, F., & Kormushev, P. (2018). *Action Branching Architectures for Deep Reinforcement Learning*. AAAI-18. https://doi.org/10.1609/aaai.v32i1.11798
13. Hein, D., Udluft, S., & Runkler, T. A. (2018). *Interpretable policies for reinforcement learning by genetic programming*. Engineering Applications of Artificial Intelligence, 76, 158–169. https://doi.org/10.1016/j.engappai.2018.09.007
14. Li, L., Fu, X., Zhen, H.-L., Yuan, M., Wang, J., Lu, J., Tong, X., Zeng, J., & Schnieders, D. (2022). *Bilevel learning for large-scale flexible flow shop scheduling*. Computers & Industrial Engineering, 168, 108140. https://doi.org/10.1016/j.cie.2022.108140
15. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS 2021, 29304–29320. https://arxiv.org/abs/2108.13264
16. Patterson, A., Neumann, S., White, M., & White, A. (2024). *Empirical Design in Reinforcement Learning*. Journal of Machine Learning Research, 25(318), 1–63. https://www.jmlr.org/papers/v25/23-0183.html
