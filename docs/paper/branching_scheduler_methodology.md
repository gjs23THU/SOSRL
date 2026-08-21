# 受约束加性 Branching DQN 调度方法

## 1. 方法定位

本实现用冻结的 Architecture DQN 提供当前 active systems，下层直接选择
`(task_idx, sys_idx)`。其中 `task_idx` 对应一个未完成任务，实际 operation 由
`op_idx = task_op_indices[task_idx]` 唯一确定。它借鉴 Branching DQN 的共享状态
表示和分支优势头，但训练目标是联合动作的一步 Double DQN 目标，因此应称为
“无交互项的受约束加性 BDQN”，不是原论文中对各分支 TD 目标取均值的严格复现。

## 2. 价值函数与合法动作

网络输出公共状态价值、逐任务优势和逐系统优势：

\[
Q(s,i,k)=V(s)+\widetilde A_{task}(s,i)+\widetilde A_{sys}(s,k).
\]

优势只在至少存在一个合法配对的边际实体上做均值中心化。推理时广播生成
`T x N` 分数矩阵，再将 pair mask 为假的位置设为负无穷并做全局 argmax。
NumPy/PyTorch 的行优先顺序保证分数相同时先选较小 `task_idx`，再选较小
`sys_idx`。epsilon 探索从 `np.argwhere(pair_mask)` 中均匀抽取完整动作对，两个
分支不会独立随机。

网络输出规模为 `1+T+N`，而合法性扫描与精确联合 argmax 仍为 `O(TN)`。该加性
模型不表达 operation-system 的专属协同或冲突；若其 makespan 稳定落后联合动作
模型，应另行增加低秩交互项，而不是把该能力归因给当前版本。

## 3. 结构化观测

每个决策状态包含：

- 25 维现有全局调度特征；
- 每个任务 15 维特征：未完成标志、operation 进度、持续时间、ready time、剩余
  工作量、TTD、slack、下一能力负载、最早可行开始/完成时间、可行系统比例、
  due time 和三维能力 one-hot；
- 每个候选系统 16 维特征：active/used、ready time、时间窗、剩余窗口、
  busy/idle/utilization、可服务任务比例、可行任务的最小开始/最小完成/平均完成
  时间和三维能力 one-hot；
- task/system entity mask、`T x N` pair mask、任务前沿 operation 索引和
  `decision_version`。

所有时间量除以 mission 总处理时间 `M`，系统成本不进入下层观测。batch 内按最大
`T/N` padding，padding 实体不参与 pooling 或优势归一化。当前特征 schema 固定为
三种功能类型；网络和 collator 支持可变 `T/N`，当前环境仍使用 30 个任务和 22 个
候选系统。

## 4. 网络与学习

- Task Encoder：`15 -> 128 -> 64`；
- System Encoder：`16 -> 128 -> 64`；
- Global Encoder：`25 -> 128 -> 128`；
- task 与 active system embedding 分别 masked mean/max pooling；
- Context Encoder：`384 -> 256 -> 128`；
- 两个共享优势头：`192 -> 128 -> 1`；
- State Value Head：`128 -> 128 -> 1`。

隐藏层使用 ReLU。Replay 保存结构化状态副本、完整动作对、奖励、下一结构化状态和
终止标志。一步 Double DQN 先由 online network 在下一合法 pair 上选动作，再由
target network 评估；无下一合法动作时 bootstrap 为零。训练采用 Huber loss、梯度
裁剪 10、uniform replay 和每个 assignment 最多一次更新，target network 每 250
次学习硬同步。

## 5. 双层时序与奖励

每轮先让冻结的 Architecture DQN 以 `epsilon=0` 执行一个合法架构规则，再构造
BDQN 观测。非终止下层 transition 会保持 pending，直到下一轮 Architecture DQN
完成动作后才写入 replay，确保 replay 的 `next_state` 与实际下一次下层决策状态
一致。

下层奖励为：

\[
r^S=-\frac{\Delta makespan}{M}+r^{terminal},
\]

成功奖励为 `+1`，完整候选系统也无法继续时为 `-2`。成本、退款和架构变化奖励不
传入下层。若顶层执行合法动作后当前 pair mask 仍为空，但完整系统全集存在可行
assignment，则记录 `provider_invariant_violation` 并终止 episode。动作执行前还会
核对 `decision_version`、任务前沿和 pair 合法性，以拒绝 stale action。

## 6. Checkpoint 与评估

`branching_scheduler.pt` 是可供未来 GP 顶层直接加载的独立下层模型；
`architecture_branching.pt` 同时保存冻结 Architecture DQN 和 BDQN。checkpoint
记录算法 kind、schema 版本、15/16/25 维特征名、网络结构、online/target 权重、
optimizer、学习步数、训练场景 hash 和 architecture provider 元数据。组合加载器
按 scheduler kind 分派；没有 kind 字段的旧规则 HRL checkpoint 仍按四规则
Scheduler DQN 加载。

`evaluate` 对组合 checkpoint 自动选择规则 HRL 或 branching episode runner，并在
相同场景 hash 上输出成功率、成功 makespan、成本、架构变化、非法动作和 provider
invariant violation 等字段。评估还记录 online 推理网络参数量、逐次上/下层策略
推理时间和 CUDA peak allocated memory（CPU 运行时显存字段为 0）；学习曲线 AUC
可由逐 episode 的 `branching_history.csv` 在统一环境步横轴上计算。
