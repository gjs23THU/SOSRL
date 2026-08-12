# ADR-0001: 双策略网络共享单一 MissionEnv

## Status

Accepted

## Context

SOSRL 需要在同一 mission 的离线调度构造步中同时处理两类决策：调整 active component-system portfolio，以及把当前 operation 分配给 system。直接学习完整联合动作会产生过大的组合动作空间；增加触发网络、终止网络或双输出头又会偏离“两层、两个策略网络”的研究约束。

实现还必须满足以下非功能要求：

- Scheduler observation 固定为 25 维，已有 SIG、MIG、MEG checkpoint 可继续加载；
- Architecture 输出固定为 6，Scheduler 输出固定为 4；
- mission、部分调度、system ready time 和成本只能有一份权威状态；
- 固定 seed 的动作轨迹、makespan 和成本在结构重构前后保持一致；
- 三个标准 CPU episode 的评估耗时不超过 4.97 秒；
- checkpoint 只通过 `torch.load(..., weights_only=True)` 读取，不执行任意 pickle 对象；
- 训练结果、模型和本地报告不进入 Git。

## Decision

采用同频双层 DQN，并共享一个 `MissionEnv`：

1. 每个离线构造步先由单头 Architecture DQN 选择 6 条架构规则之一；
2. 确定性规则解析器将其转换为 KEEP、ADD、REMOVE 或 REPLACE，并更新 `active_system_mask`；
3. 单头 Scheduler DQN 在更新后的体系中选择 SPT、WINQ、CR 或 MS；
4. 调度规则和 CSSA 生成唯一的 `(task, operation, system)` 环境动作；
5. 两层使用独立 replay 和 target network，但共享 mission/environment state；
6. IntDQN 的独立环境仅作为扁平联合动作对照组，不参与 HRL 主流程。

代码按职责组织为：

- `sosrl/environment.py`：唯一 HRL 环境及成本轨迹；
- `sosrl/rules/`：两层确定性动作抽象；
- `sosrl/rl/`：网络、Replay、配置和 checkpoint；
- `sosrl/workflows/`：场景、episode、训练和评估；
- `sosrl/baselines/`：扁平对照；
- `sosrl/cli.py`：唯一正式命令入口。

Architecture rule simulation 共享按 `decision_version` 缓存的 current-operation/system finish matrix，避免为每个候选架构重复扫描完整 assignment 空间。

## Consequences

### Positive

- 策略网络严格保持两个，研究对象和论文表述一致；
- 上层动作先改变下层可行集合，因果顺序明确；
- 环境状态没有副本同步问题；
- 规则动作抽象把 2640 维 assignment 空间压缩为 6 与 4 个网络输出；
- 旧 checkpoint 的 state dict 和配置格式保持可加载；
- finish-matrix 缓存显著降低 Architecture action mask 成本。

### Negative

- 上层每个构造步都决策，不能表达持续多步 option；
- 六条人工规则限制了 Architecture DQN 能探索的调整集合；
- 预算仍是软约束，`ever_over_budget` 只能观测越界，不能阻止越界；
- Scheduler 的 next state 必须等下一次 Architecture action 后才能补全，Replay 编排比普通 DQN 更复杂。

### Neutral

- target network 是训练副本，不计作第三个策略网络；
- Flat IntDQN 保留独立环境是实验对照需要，不代表生产主流程存在第二套 HRL 状态；
- 本 ADR 不改变奖励权重、课程比例或训练超参数。

## Alternatives Considered

**单网络联合动作 DQN**

- 拒绝：动作规模和无效组合过大，且难以区分架构与调度信用分配。

**第三个触发/终止网络或标准 option-critic**

- 拒绝：增加策略网络并引入额外训练不稳定性，不符合当前两个网络约束。

**Architecture 网络使用两个输出头**

- 拒绝：会把“调整类型”和“具体 system”再次拆开，增加接口和 loss 设计；当前确定性解析器已经完成具体目标选择。

**Architecture 与 Scheduler 使用两个环境**

- 拒绝：需要同步历史 assignment、ready time、active mask 和成本，容易产生不一致状态。

## References

- [算法说明书](../../README.md)
- [双层 HRL 初始设计](../plans/2026-08-11-two-level-hrl-design.md)
- [论文方法章节](../paper/hrl_methodology.md)
