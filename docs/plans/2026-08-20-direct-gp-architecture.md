# 系统特征驱动的直接 GP 上层架构策略

## 目标与边界

本实现用一棵 Genetic Programming 表达式树替换上层“DQN 选择六个人工架构规则”的主方法。GP 在每个下层派工决策前评价所有当前合法的具体架构动作，直接执行得分最低者；下层仍使用已经训练并冻结的 Branching DQN。

本阶段不重新训练 BDQN，不使用 GP-guided DQN，不交替训练两层，也不让 GP 生成 22 维体系掩码。

## 固定时序

```text
固定场景 s_t
   ↓
枚举合法 KEEP / ADD(j) / REMOVE(i) / REPLACE(i,j)
   ↓
为每个候选计算 φ(s_t,a)
   ↓
GP 树输出 Score(s_t,a)，按确定性规则取 argmin
   ↓
执行一个具体架构动作，刷新 decision_version 和可行性缓存
   ↓
重新构造 BranchingObservation 与 T×N pair mask
   ↓
冻结 BDQN 以 epsilon=0 选择 (task_idx, system_idx)
   ↓
执行该 task 的当前前沿 operation
```

BDQN 不接收 GP score，也不参与架构动作选择。

## 动作空间

动作全集由 1 个 KEEP、22 个 ADD、22 个 REMOVE 和同功能类型的 158 个有向 REPLACE 组成，共 203 个。每一步只保留物理可执行且动作后至少存在一个下层合法 pair 的候选。

候选枚举只构造假设 active mask，不修改环境。执行时再次校验 `decision_version`，从而拒绝基于旧状态计算的动作。

## 特征集

特征schema版本为2。主方法 `system_delta` 使用39个终端：

- 9个当前体系特征：进度、工期、激活/使用比例、成本、预算、距上次变化步数和利用率；
- 12个动作目标特征：ADD/REMOVE flag，新旧系统成本、时间窗、历史使用和利用率；
- 8个需求上下文特征：能力覆盖、合法pair、阻塞前沿和目标功能类型压力；
- 10个反事实增量特征：净成本、覆盖、pair、阻塞和目标类型容量的变化，以及动作后前沿完成时间及其变化。可由“当前值＋变化量”严格重构的动作后特征不再重复注册。

时间统一除以场景总工作量尺度 `M`，成本除以预算 `B`，普通特征裁剪至 `[-10,10]`，flag保持0/1，无合法完成时间使用2.0哨兵。KEEP的动作目标与目标功能类型字段为0。

消融预设为 `system`（21）、`system_demand`（29）、`system_delta`（39）和 `op_context`（旧25维调度观测加12个动作目标特征）。

## GP表示与遗传参数

函数集为受保护的 `add/sub/mul/pdiv/minimum/maximum/negative/absolute`，并使用六位小数的 `U[-1,1]` ephemeral constant。分数越小越好。平分时依次比较变更系统数、动作类型 KEEP/ADD/REMOVE/REPLACE、旧系统编号和新系统编号。

标准参数：种群200、80代、10次独立运行、锦标赛5、精英2、交叉0.75、变异0.20、复制0.05。变异内部为子树0.50、同元数节点替换0.25、常数重采样0.25。初始化使用Half-and-Half深度2–4，树高不超过6，节点不超过40。

## Fitness

单场景代价为：

\[
J_e=10\frac{C_{\max}}M+\frac{C_{net}}B
+20\left[\max(0,C_{peak}/B-1)\right]^2
+0.01N_{change}+10R_e,
\]

成功时 `R_e=0`，失败时为未完成operation比例。DEAP双目标fitness为 `(failure_rate, mean(J)+0.001|tree|)`，按字典序最小化；同时单独保存未加树规模惩罚的 `raw_mean_j`。

## 数据隔离与选模

场景生成后固化为带SHA-256的JSON manifest。标准规模为Train 256、Validation 128、Test-IID 500、Test-OOD 200，四类场景等量。每代从Train按类别抽16个场景；64个固定Train场景仅用于每10代anchor复评。

所有独立运行的每代冠军、anchor候选和最终前10按表达式去重，再在完整Validation上复评。最终选择顺序为：最低失败率、最低raw mean J的1%范围、最少节点、最低树高、表达式字符串。Test只在规则锁定后使用。

## 产物与恢复

部署产物 `gp_policy.json` 只包含注册过的primitive、feature schema、表达式、前缀树、验证fitness、系统池hash和BDQN checkpoint hash。加载器用受限AST验证表达式，不执行未知节点。

训练恢复文件 `evolution_state.pkl` 只用于本地可信恢复。它每代通过临时文件原子替换写入，不参与线上部署。

## 验收检查

- 候选枚举和特征提取不改变环境；
- 每个执行动作都来自当前合法候选；
- GP动作后才构造BDQN pair mask；
- 旧 `decision_version` 动作被拒绝；
- BDQN在GP演化前后参数hash完全相同；
- JSON加载后的表达式、树规模、评分和动作保持一致；
- 固定seed的smoke产生相同冠军和Validation winner；
- 在线应用不包含交叉、变异或训练逻辑。
