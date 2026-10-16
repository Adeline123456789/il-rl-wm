# E4–E6 implementation contract (before implementation)

The first model uses the same raw state vector that the frozen BC and SAC Actor consume. A frozen training-only mean and standard deviation (floor 0.01) scale the inputs and state error. A two-layer 256-unit ReLU MLP predicts a residual next state and one reward logit. Its decoded output is a raw state vector, so neither policy requires retraining an encoder. There is no learned encoder, ensemble, uncertainty penalty, or separate training framework.

For two real transitions from one episode, initialize the model at the real first observation and recursively feed its own prediction into step two. Minimize the mean of normalized state MSE plus binary cross entropy for the two steps. Binary reward applies only to the three verified sparse tasks. A success reward is not a terminal event under our continuing-task protocol. Time limits remain recording boundaries and cannot be crossed by sampled training sequences.

Initial side training: fixed expert training episodes plus real SAC replay; 50% from each source, batch 512, Adam 3e-4, 10,000 updates, seed 0, fixed final checkpoint. The engineer smoke uses 200 updates. Split SAC replay by whole episode with seed 42; use the existing expert episode split. Keep validation out of both model optimization and normalization. Online SAC has already learned from its replay: these held-out episodes validate the model, not an independent end-to-end RL system. Save dataset/checkpoint hashes and exact episode IDs.

Report free rollout error at steps one and two, separately for expert and online data. For rewards report counts, Brier error and threshold-0.5 confusion counts for zero, nonzero, five-step pre-success windows, and unsuccessful episodes. Empty strata are unavailable, not zero error. These metrics measure prediction, not candidate ranking. Failed expert replay episodes remain excluded from learning but can be diagnosed separately.

## Pure task score and student continuation

Use a separate learned task Q head, rather than the SAC soft Q. Let mu be the current SAC mean action. Train Q_task on real replay using target:

    y = r + gamma * Q_task_target(s_next, mu(s_next))

The current protocol bootstraps after both task success and time limits, using final observation before reset. There is no entropy term in this head. Target parameters use Polyak tau=.01; actor and target gradients are stopped. This is an added evaluator and its parameters, updates and runtime must be counted. It estimates the deterministic student's continuing task return, subject to approximation and support errors; it does not evaluate the replanning controller's return.

For each fixed candidate a[0:H], recursively predict s_hat and reward, then score:

    score = sum(gamma**t * r_hat[t], t=0..H-1)
            + gamma**H * Q_task(s_hat[H], mu(s_hat[H]))

All candidate sources use the same head and formula. With gamma=.9, H=2, rewards [1,2], and terminal value 3, the expected score is 5.23. With no predicted rewards and terminal value 3 it is 2.43. Reward at t=0 is not discounted; terminal value is discounted exactly H times. Real diagnostic returns use the same continuation definition and explicitly account for any finite diagnostic horizon.

The initial planner batches all candidates on GPU: one IL and eight stochastic SAC trajectories, H=2. Each candidate starts from the same observation and its second action uses its own predicted state. No-IL evaluation uses nine SAC candidates. Execute the selected first action, except an independently sampled 0.2 exploration mask actually executes a new stochastic SAC action. Only the executed real transition enters replay. Candidate counts are not real exploration counts. Record generation/scoring latency and observed peak resources before claiming acceptable cost.

This file is a definition for later stages, not evidence that E4–E6 are implemented or tested. The score is a sequence-commitment estimate followed by the mean student; the separate first-action continuation diagnostic measures how much of that estimate the student can inherit.

Implementation staging update: independent E5 generation and the E6 score/evaluator code now exist; three CPU tests cover recursive policy input, a hand-computed discounted score, and entropy-free TD without actor gradients. Actual frozen-snapshot GPU tests remain queued. The value preheat is fixed at 5,000 Adam updates, batch 512 with equal expert/online sources, seed 0. It uses the first transition of each available H2 training sequence; the final transition of each episode is excluded only for this preheat. Validation reports Bellman consistency and predictions outside the known [0,100] return range, not a claim of calibrated value accuracy.
