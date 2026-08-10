# SOSRL

SOSRL 是一个面向多类型系统任务调度的深度强化学习实验项目。任务由多道有前置关系的工序组成，每道工序要求特定 `func_type` 的机器执行；DQN 不直接选择任务或机器，而是在 SPT、WINQ、CR、MS 四条调度规则之间动态选择。

当前实现采用排班环境 v2：场景池在 episode 开始前给出不可变系统架构，`MissionEnv` 只接受工序到执行系统的 assignment。规则层完成“规则选择工序 → CSSA 从架构内选择执行系统 → 编码为环境动作”的转换。

## 调度流程

```text
25 维 SA 聚合状态
        ↓
DQN 选择规则（SPT / WINQ / CR / MS）
        ↓
rule.py 从全局可行工序中选择一道工序
        ↓
CSSA 从类型匹配的可行机器中选择具体机器
        ↓
encode_assignment() 编码为 MissionEnv assignment 动作
        ↓
MissionEnv.step() 执行调度并更新状态与奖励
```

具有相同 `func_type` 的机器只在统计下一机器类型负载时作为一个集合使用。

## 核心建模

### 任务与机器

- 一个 Task 包含固定顺序的多道 Operation。
- Operation 具有加工时长、释放时间和所需 `func_type`。
- ComponentSystem 表示候选机器，具有类型、成本和可用时间窗口。
- 同一时刻一台机器只能执行一道工序。
- 工序不可抢占，后续工序只能在前置工序完成后开始。
- 每个 episode 的系统架构固定；四规则不产生系统选择或移除动作。
- 架构选择与排班完全分离；`MissionEnv` 中不存在选择、移除系统或修改成本的动作。
- 排班采用离线列表调度：同一系统上的工序按决策顺序在尾部追加，不同系统允许排到早于上一决策的绝对时间。

默认场景由 [config.json](config.json) 定义：

- 30 个任务，每个任务 4 道工序；
- 工序类型为 `S`、`D`、`I`；
- 工序时长从 `[20, 30]` 随机生成；
- 每个任务分别从 `U(1, due_time_tightness)` 采样紧迫度，交期为 `release_time + total_duration × sampled_tightness`；
- 候选机器具有不同的成本和可用时间窗口。

### 25 维状态空间

`State.to_obs()` 输出固定 25 维 `float32` observation：

| 分组         | 维度 | 特征                                                           |
| ------------ | ---: | -------------------------------------------------------------- |
| 任务数量     |    3 | 未完成任务比例、可行候选比例、晚于排班前沿的等待任务比例       |
| 当前工序时间 |    3 | 总和、均值、最小值                                             |
| 剩余工作量   |    5 | 剩余加工时间总和/均值/最大值、下一机器类型负载均值/最小值      |
| 紧迫程度     |    5 | TTD 均值/最小值、slack 均值/最小值、等待候选最小 slack         |
| 系统状态     |    4 | 平均机器可用延迟、平均完成率、已拖期率、预计拖期率             |
| 异质性       |    5 | 当前加工时间、剩余加工时间、TTD、slack、下一类型负载的变异系数 |

其中：

```text
earliest_start(task) = 当前工序在所有可行架构内系统上的最早开工时间
TTD = due_time - earliest_start(task)
remaining_time = 当前工序及后续工序的加工时间之和
slack = TTD - remaining_time
CV = std / (abs(mean) + eps)
```

环境不维护全局 `current_time`。机器等待、TTD 和 slack 均从当前部分排程的 earliest-start 派生；时间特征使用当前场景全部工序总时长归一化，空集合统计值为 0。

### 四条规则

| 动作 | 规则 | 选择标准                                     |
| ---: | ---- | -------------------------------------------- |
|    0 | SPT  | 当前工序加工时间最小                         |
|    1 | WINQ | 下一道工序所需机器类型的平均剩余占用时间最小 |
|    2 | CR   | `TTD / max(remaining_time, eps)` 最小      |
|    3 | MS   | `slack` 最小                               |

WINQ 是适配当前无显式机器队列环境的代理指标。末道工序没有下一机器类型，其 WINQ 指标取 0。

规则选择工序时使用以下稳定排序键：

```text
(rule_metric, earliest_start_time, task_idx, op_idx)
```

CSSA 选择机器时使用：

```text
(start_time, finish_time, sys_busy_time, sys_idx)
```

### 动作与奖励

DQN 动作空间固定为 4，`MissionEnv.action_space` 只包含 assignment：

```text
Discrete(T × O × N)
```

`rule.py` 通过 `MissionEnv.encode_assignment()` 将规则结果编码为 assignment 动作，再交给环境执行。开始时间统一计算为：

```text
start = max(system_ready_time, operation_ready_time)
finish = start + duration
```

只有 `finish <= system.available_until` 的 assignment 才有效。

当前奖励为负 makespan 增量：

```text
reward = -(new_makespan - old_makespan) / total_processing_time
```

若任务尚未全部完成且没有任何可行 assignment，episode 以 dead end 结束。dead end 不追加额外奖励或惩罚。

## 项目结构

```text
SOSRL/
├── config.json                 # 任务类型、机器和场景生成配置
├── syn.py                      # Task、Operation、ComponentSystem 与随机场景生成
├── env.py                      # State、25维状态、动作掩码和底层环境转移
├── rule.py                     # SPT/WINQ/CR/MS、CSSA 和原动作编码
├── dqn.py                      # DQN、经验回放、场景池、训练与评估
├── main.py                     # 命令行训练入口与结果保存
└── tests/
    └── test_sa_rule_env.py     # 状态、规则、编码、奖励和 rollout 测试
```

## 环境准备

建议使用 Python 3.10 或兼容版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy gymnasium matplotlib
```

PyTorch 请根据本机 CPU/CUDA 环境按官方方式安装。例如已有可用 CUDA 版本时，确认：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- State 不持有 `arch`、`mission` 对象；
- 25 维 observation 的形状、类型和有限性；
- 下一机器类型负载计算；
- 四规则选择及原动作空间编码；
- CSSA 机器选择边界；
- makespan 增量奖励；
- 四条固定规则的完整 rollout。

## 开始训练

使用默认参数：

```powershell
python main.py
```

训练开始前会以 JSON 打印本次训练、评估、输出路径、环境和规则参数；训练期间使用 `tqdm` 显示 episode 进度以及 reward、makespan、已分配工序数、epsilon 和 loss。

一个较完整的训练示例：

```powershell
python main.py `
  --episodes 500 `
  --scenario-pool-size 40 `
  --shared-mission `
  --eval-episodes 20 `
  --selected-system-num "8,18" `
  --min-system-num 8 `
  --max-system-num 18 `
  --cost-limit 8000 `
  --lr 0.0005 `
  --buffer-size 20000 `
  --min-buffer-size 500 `
  --epsilon-decay 0.995 `
  --hidden-dim 128 `
  --seed 42 `
  --run-name sa_rule_500_seed42
```

常用参数：

| 参数                      | 含义                                    |     默认值 |
| ------------------------- | --------------------------------------- | ---------: |
| `--episodes`            | 训练 episode 数                         |        100 |
| `--scenario-pool-size`  | 训练场景池大小                          |         20 |
| `--scenario-order`      | 场景抽取方式：`random`/`sequential` | `random` |
| `--shared-mission`      | 所有训练架构复用同一个 mission          |       关闭 |
| `--eval-episodes`       | 独立评估场景数                          |          5 |
| `--eval-seed`           | 独立评估场景池随机种子                  | `20260724` |
| `--selected-system-num` | 固定数量或范围，如`12`、`8,18`      |   `none` |
| `--cost-limit`          | 架构成本上限                            |       8000 |
| `--gamma`               | 折扣因子                                |       0.99 |
| `--lr`                  | 学习率                                  |      0.001 |
| `--batch-size`          | 批大小                                  |         64 |
| `--buffer-size`         | Replay Buffer 容量                      |      10000 |
| `--min-buffer-size`     | 开始学习前的最小经验数                  |        500 |
| `--epsilon-decay`       | 每个 episode 的 epsilon 衰减            |      0.995 |
| `--hidden-dim`          | Q 网络隐藏层宽度                        |        128 |
| `--seed`                | Python、NumPy、PyTorch 随机种子         |          1 |

使用 `--shared-mission` 时，训练池只生成一次 mission，`--scenario-pool-size` 个场景分别采样不同 architecture。评估池始终独立生成，并同时重新采样 mission 和 architecture；固定 `--eval-seed` 后，不同训练设置会使用相同的评估场景。

比较 SIG、MIG、MEG 等多个 checkpoint 时，应让所有模型运行同一个、独立于各训练池的测试场景集合，不能直接汇总各自训练池或训练结束时产生的评估结果。仓库提供统一评估入口：

```powershell
python evaluate_independent.py `
  --eval-episodes 100 `
  --eval-seed 20260724
```

默认读取 `SIG1000_standard_seed4`、`MIG1000` 和 `MEG1000`；也可以重复使用 `--model 名称=checkpoint路径` 指定其他模型。结果写入 `runs/SIG_MIG_MEG_independent_eval/`，其中包含逐场景结果、总体汇总、配对比较、完整场景清单和评估清单。

架构生成时除检查各功能类型的总容量是否覆盖任务需求外，还会检查每种类型至少有一台机器满足 `available_until >= min_coverage_until`。该阈值在 `config.json` 中配置，当前为 600。

## 训练输出

结果默认保存在：

```text
runs/<run-name>/rule_sa/
├── config.json
├── model.pt
├── train_history.csv
├── eval_results.csv
└── eval_schedule.csv
```

- `train_history.csv`：每轮奖励、makespan、已分配工序数、dead end、loss、epsilon 和规则使用次数。
- `model.pt`：可复用的 DQN 检查点，包含在线/目标网络、优化器、网络维度、训练状态和环境配置。
- `eval_results.csv`：每个评估场景的完成情况与规则使用次数。
- `eval_schedule.csv`：已分配工序的任务、机器、开始时间和结束时间。
- `config.json`：本轮 DQN 训练参数。

后续可直接加载模型进行推理或继续训练：

```python
import dqn

agent, checkpoint = dqn.DQNAgent.load_checkpoint(
    "runs/seed_1/rule_sa/model.pt",
    device="cpu",
)
action = agent.select_action(observation, action_mask, epsilon=0.0)
```

`runs/` 已加入 `.gitignore`，不会默认提交到版本库。

## 历史实验结果（环境 v1）

使用上述 500 episode 配置、seed 42，在 20 个独立场景上得到：

| 策略       | 完整完成率 | 平均完成工序数 | 平均奖励 |
| ---------- | ---------: | -------------: | -------: |
| 四规则 DQN |        45% |         112.90 |   -0.454 |
| 固定 MS    |        35% |         111.85 |   -0.492 |
| 固定 WINQ  |         5% |          98.00 |   -0.976 |
| 固定 CR    |         5% |          97.40 |   -0.994 |
| 固定 SPT   |         0% |          64.00 |   -2.113 |

以下结果来自含全局 `current_time` 的旧环境，只作历史参考，不能与 v2 直接比较。正式比较应在 v2 上重新训练多个随机种子，并在完全相同的评估场景上报告均值和标准差。

## 已知限制

- 当前是全局可行工序上的规则选择，不是基于机器局部队列的事件驱动调度。
- WINQ 使用下一机器类型剩余占用时间作为代理，而非真实队列工作量。
- 场景池仅按各机器类型的总容量筛选架构，无法保证给定机器尾部追加顺序一定存在完整排程，因此仍可能出现真实 dead end。
- 当前奖励以 makespan 为目标，TTD 和 slack 用于状态与规则，但没有直接进入奖励函数。

## 后续方向

- 增加严格的训练/验证/测试场景划分和周期性 greedy 验证。
- 对场景生成器增加时序可行性筛选。
- 使用多随机种子评估并输出固定规则对照表。
- 根据研究目标考虑 tardiness/slack 奖励或多目标奖励。
- 如需更紧凑的离线排程，再独立实现同机历史空档插入；如需事件驱动调度，则作为不同环境版本实现。
