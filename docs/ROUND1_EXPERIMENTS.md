# 第一轮 GP + BDQN 实验执行说明

schema-v1 保持原来的 `fixed / arch / g0` 三类 provider，不修改正在运行的实验。schema-v2 的正式顺序固定为 `fixed / ss / arch / g0`，其中 `ss` 是不经训练或验证集调参的 S/s-HCM 经验基线，固定阈值为 0.40/0.90。正式实验默认使用 seeds 1–8、200k BDQN 收敛预算、80k 迁移预算和 8 个 GP CPU workers。

## 1. 先做端到端 smoke

```powershell
python -m sosrl smoke-round1-study `
  --architecture-checkpoint <ARCHITECTURE_DQN.pt> `
  --gp-policy <G0_GP_POLICY.json> `
  --output-dir runs/round1_smoke
```

Smoke 会实际运行四种 provider、4×4 交叉评估、七条迁移路径和一个 `P=8, G=1` 的 GP 单元，但不会消费 Test-v2。完成标志为 `runs/round1_smoke/smoke_manifest.json`。

## 2. 初始化正式研究

从头建立 schema-v2：

```powershell
python -m sosrl init-round1-study `
  --architecture-checkpoint <ARCHITECTURE_DQN.pt> `
  --gp-policy <G0_GP_POLICY.json> `
  --output-dir runs/round1_formal_v2 `
  --ss-low-threshold 0.40 `
  --ss-high-threshold 0.90 `
  --device cuda
```

若 schema-v1 的 convergence 阶段已经完整结束，推荐直接创建独立的 v2 目录：

```powershell
python -m sosrl init-round1-study `
  --augment-from runs/round1_formal/study_manifest.json `
  --output-dir runs/round1_formal_v2 `
  --ss-low-threshold 0.40 `
  --ss-high-threshold 0.90
```

增量初始化会逐 seed 校验并引用 v1 的 fixed/arch/g0 cell、场景清单和共享初始权重，不写入或改动 v1 文件；v2 的 convergence 阶段只训练缺失的 `ss` cell，四类 provider 齐备后重新计算共同收敛步数 T0，并在 v2 目录内重新运行下游矩阵。

初始化会生成或继承六个互不重叠的场景 manifest，并完成以下预检：

- 每个场景的 `static_feasible_system_indices` 都能覆盖任务且预算内；
- Fixed provider 的确定性动作只有 KEEP；
- 每个 seed 的四组 BDQN 引用同一个初始权重文件，逐张量哈希一致；
- Architecture DQN、G0、S/s-HCM 配置、场景、初始模型和当前代码 commit 均登记 SHA-256；
- Test-IID-v2 和 Test-OOD-v2 保持锁定。

S/s-HCM 的 `budget_mode=soft`、`replace_mode=hierarchical`、策略版本和两个阈值都会写入 manifest 并纳入配置哈希；同一 v2 study 恢复时不允许变化。

主入口为 `runs/round1_formal_v2/study_manifest.json`。

## 3. 分阶段运行

建议在可保留进程输出的终端中逐段执行：

```powershell
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage convergence
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage cross
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage migration
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage hyper-screen
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage hyper-confirm
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage gp-discovery --workers 8
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage gp-confirm --workers 8
```

也可以用 `--stage all` 顺序执行全链。BDQN cell 在 checkpoint 处原子保存网络、优化器、replay、epsilon、场景采样位置以及 Python/NumPy/PyTorch 随机状态；进程中断后重发同一命令即可从最近 checkpoint 确定性恢复。已经完成的 cell 和 GP 配置会先校验输入哈希，然后复用产物。

## 4. 锁定后消费 Test-v2

只有 `hyper-confirm` 与 `gp-confirm` 均完成后，下列命令才会成功：

```powershell
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage final-test
python -m sosrl run-round1-study --study-manifest runs/round1_formal_v2/study_manifest.json --stage report
```

BDQN 训练、超参数选择和交叉评估接口会拒绝 split 名以 `test` 开头的 manifest。最终测试首次完成后，study manifest 会记录消费时间和最终 BDQN、GP、Test-v2 的 SHA-256；再次调用只返回已经锁定的结果。

## 5. 主要产物

- `bdqn/convergence/`：四 provider、八个 seed 的训练历史、checkpoint、原始 validation 结果及收敛判定；增量 v2 的旧三类 cell 通过 manifest 引用 v1；
- `bdqn/cross_matrix/`：完整 4×4 原始结果和汇总；
- `bdqn/migration/`：F→F、F→G、S→S、S→G、A→A、A→G、G→G；
- `bdqn/hyper_screen/`、`bdqn/hyper_confirm/`：H0–H10、Hsingle、Hstar 和最终 BDQN 选择；
- `gp/discovery/`：population、generation、等预算矩阵和轴向收敛；
- `gp/confirm/`：选中配置与 `200×80` 的 10-run 确认、代数收敛和累计 runs 收敛；
- `final_test/`：IID/OOD 最终锁定结果；
- `report/`：五类图、按 seed/类别分层 bootstrap、win/tie/loss 和可复现性索引。

每个 BDQN cell 都带 `cell_manifest.json`；每个 GP 目录都带 `gp_stack_manifest.json`。选择过程只读取 B-validation 或 G-validation，不读取 Test-v2。
