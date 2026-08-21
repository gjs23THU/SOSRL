# Direct Genetic Programming for System-Architecture Adaptation

## Problem formulation

At each constructive scheduling step, the upper policy observes the current mission and system state and chooses one concrete architecture action before the lower scheduler assigns a frontier operation. Let \(s_t\) denote the current partial schedule and architecture, and let \(\mathcal A(s_t)\) be the set of immediately executable concrete actions. The evolved policy is a single symbolic scoring function

\[
F_\theta:\phi(s_t,a)\mapsto \mathbb R,
\qquad
a_t^*=\arg\min_{a\in\mathcal A(s_t)}F_\theta(\phi(s_t,a)).
\]

Thus, GP does not select among handcrafted macro-rules and does not output an unconstrained 22-dimensional architecture. It ranks legal concrete KEEP, ADD, REMOVE, and same-type REPLACE decisions.

## Legal candidate construction

The fixed action universe contains 203 actions: one KEEP, 22 ADDs, 22 REMOVEs, and 158 ordered same-capability replacements. A state-dependent candidate is retained only when its physical preconditions hold and the hypothetical post-action active mask admits at least one feasible frontier task-system pair. The feasibility check is performed against cached candidate finish times without cloning or mutating the environment.

The filter deliberately checks immediate schedulability rather than enforcing complete future capability coverage. This keeps domain preference out of the candidate generator and lets evolution learn when temporary withdrawal and later re-addition are worthwhile.

## State-action representation

The primary representation contains 39 normalized terminals under feature schema version 2. Nine describe the current architecture and partial schedule; twelve describe the systems added or removed by the candidate; eight summarize demand, capability coverage, pair feasibility, frontier blockage, and target-type pressure; ten encode counterfactual deltas and post-action frontier finish statistics. Redundant post-action terminals that are exactly reconstructable from a current value and its delta are excluded.

Time quantities are divided by total mission workload \(M\), monetary quantities by budget \(B\), and ordinary values are clipped to \([-10,10]\). Missing feasible finish statistics use the normalized sentinel 2.0. System identifiers are excluded from the terminal set, preventing GP from memorizing index-specific decisions.

Three system-level presets support ablation: `system` (21 terminals), `system_demand` (29), and `system_delta` (39). A fourth preset, `op_context`, combines the legacy 25-dimensional scheduling observation with the twelve action-target features to test whether operation-heavy context generalizes less effectively than system-level context.

## Symbolic policy space

Individuals are single expression trees. The primitive set consists of clipped addition, subtraction and multiplication, protected division, numeric minimum and maximum, protected negation, and protected absolute value. Ephemeral constants are sampled uniformly from \([-1,1]\) and stored to six decimal places. Conditional branching, transcendental functions, random terminals, system indices, and multi-tree ensembles are excluded in the first version.

Trees are initialized using ramped half-and-half at depths 2–4 and are limited to height 6 and 40 nodes. One-point subtree crossover and all mutation operators are protected by both limits; an invalid offspring is replaced by the corresponding parent copy.

## Evolutionary objective

For scenario \(e\), the cost is

\[
J_e=10\frac{C_{\max,e}}{M_e}
+\frac{C_{net,e}}{B_e}
+20\left[\max\left(0,\frac{C_{peak,e}}{B_e}-1\right)\right]^2
+0.01N_{change,e}+10R_e,
\]

where \(R_e=0\) on success and otherwise equals the remaining-operation fraction. Each individual receives the lexicographically minimized DEAP fitness

\[
\left(
\frac{1}{K}\sum_e \mathbb I[e\text{ fails}],
\frac{1}{K}\sum_eJ_e+0.001|\theta|
\right).
\]

This formulation makes mission completion a hard empirical priority; makespan, net cost, peak budget violation and architecture churn distinguish policies only after failure rate.

## Evolution protocol

The standard experiment uses 200 individuals, 80 generations and 10 independent runs. Parent selection is tournament selection of size five, with two elites per generation. Variation probabilities are 0.75 one-point crossover, 0.20 mutation, and 0.05 reproduction. Conditional on mutation, subtree replacement, node replacement and constant resampling occur with probabilities 0.50, 0.25 and 0.25.

All individuals within one generation are evaluated on the same stratified batch of sixteen frozen training scenarios, four from each scenario category. Identical expression strings are evaluated once per generation. Every ten generations, the current top ten are re-evaluated on a fixed 64-scenario training anchor; anchor fitness is logged but is not mixed into generational selection.

## Coupling to the lower scheduler

The lower Branching DQN is loaded from one checkpoint, placed in evaluation mode, frozen with `requires_grad=False`, and queried with \(\epsilon=0\). After the GP provider executes its architecture action, all architecture-dependent masks and the decision version are refreshed. A new BranchingObservation is then constructed, and the BDQN selects a legal \((task,system)\) pair for the task's current frontier operation.

GP scores are never supplied to the BDQN. The BDQN never selects or modifies the architecture. Parameter hashes over both online and target networks are compared before and after evolution; any difference aborts training.

## Validation selection and test protocol

Per-generation champions, periodic anchor candidates and each final population's top ten are pooled across runs and deduplicated by expression. Every unique candidate is evaluated on the complete validation set. Selection first minimizes validation failure rate, then identifies the best unregularized mean cost, retains candidates within one percent of it, and chooses the smallest tree, breaking further ties by height and expression string.

Only after this rule is locked are Test-IID and Test-OOD evaluated. Fixed architecture, uniformly random concrete actions, the existing six-rule architecture DQN, and direct GP share the same frozen lower checkpoint and scenario hashes. Continuous paired differences and paired success/dead-end proportion differences are reported with bootstrap 95% intervals.

## Deployment

The deployed stack consists of a validated `gp_policy.json` and an unchanged Branching DQN checkpoint. JSON records the complete feature registry, primitive-set version, expression and prefix tree, score direction, deterministic tie-break rule, validation fitness, system-pool hash and checkpoint SHA-256. The online runtime only validates, compiles and evaluates the fixed expression; evolutionary operators and pickle recovery state are absent from deployment.
