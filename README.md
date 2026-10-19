# il-rl-wm
就是让rl以il为基础然后通过世界模型改进最后rl的性能超过wm


当前科研执行规则见 [PROMPT.md](PROMPT.md)、[RESEARCH_CHARTER.md](RESEARCH_CHARTER.md)；实际进展与原始实验索引见 [research_state.json](research_state.json) 和 [RESEARCH_LOG.md](RESEARCH_LOG.md)。首批尚在执行，开发seed0的完整U0末期退化已保留，不能将工程smoke当作成功结论。

环境、数据、便携权重、离线复算和最终测试入口见 [REPRODUCING.md](REPRODUCING.md)。当前仅使用ManiSkill官方BC/SAC主训练器；RLPD在独立JAX环境对照。U1只测试critic LayerNorm，预登记与边界见 [U1_CHANGE.md](U1_CHANGE.md)。所有任务保持真实稀疏奖励，最终部署包不包含IL。

用户要求的当前阶段上传入口：[PROGRESS_SNAPSHOT.md](PROGRESS_SNAPSHOT.md)。这是仍在训练中的进度快照，包含原始实验文件与便携权重，不是最终测试结论。

ACT→SAC独立研究进度（2026-10-19）见 [ACT_TRANSFER_REPORT.md](ACT_TRANSFER_REPORT.md)、[ACT_FAILURES.md](ACT_FAILURES.md) 和 [阶段权重复算](releases/ACT_REPRODUCTION.md)。保留原MLP-BC/U0/U1对照；教师阶段与首轮专家状态蒸馏已实际运行，发现技能损失及自动reset评测配对问题，正在固定模型核验。每个可验证小阶段后台增量更新研究分支，不自动合并main，尚无完整ACT→SAC→WM有效性结论。
