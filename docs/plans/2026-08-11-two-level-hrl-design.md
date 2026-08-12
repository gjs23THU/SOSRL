# Two-Level Architecture/Scheduling HRL

## Decision

The mission is planned offline by repeatedly appending one operation assignment.
Each construction cycle first invokes one architecture policy and then one
scheduling policy. The architecture policy has one six-action output head; the
scheduling policy keeps the existing four dispatching-rule actions. Target
networks are training copies, not additional policies.

## Environment contract

`MissionEnv(adaptive=True)` owns both the partial schedule and mutable system
portfolio. `active_system_mask` controls future assignments, while completed
assignments are never rolled back. Adding a system charges 100% of its cost and
removing it refunds 80%; re-adding charges the full cost again. The existing
25-dimensional scheduling observation remains checkpoint-compatible. A separate
fixed-width architecture observation exposes the active/used masks, ready times,
capability pressure, progress, and cost state from the same environment.

Architecture actions are abstract rules: KEEP, ADD_CAPABILITY, ADD_CAPACITY,
ADD_WINDOW, REMOVE_REDUNDANT, and REPLACE_INEFFICIENT. A deterministic resolver
maps a valid rule to concrete system indices. Invalid rules are masked. KEEP is
masked when the active portfolio cannot schedule a ready operation but another
portfolio can; the episode is a true dead end only when the full candidate pool
cannot schedule any remaining ready operation.

## Learning contract

The Scheduler DQN stores ordinary one-step transitions. Its next state is
captured after the next architecture action, because that action changes the
lower-level feasible set before the next scheduling decision. The Architecture
DQN stores five-step transitions including the realized bootstrap discount and
flushes short returns on terminal states.

Scheduler reward is normalized negative makespan increment. Architecture reward
uses 10-weighted makespan increment, net cost increment, a quadratic soft
budget potential, a small change penalty, and shared success/dead-end terminal
rewards. Training proceeds as scheduler pretraining, frozen-scheduler architecture
training, then 8:2 alternating architecture/scheduler fine-tuning.

## Commands

```text
python hrlmain.py pretrain-scheduler
python hrlmain.py train-architecture --scheduler-checkpoint PATH
python hrlmain.py finetune --scheduler-checkpoint PATH --architecture-checkpoint PATH
python hrlmain.py evaluate --checkpoint PATH
```

Evaluation uses one unseen, paired scenario pool for HRL, the static initial
architecture, fixed and random architecture rules, and the full-system upper
bound. Scenario hashes are emitted with every result.
