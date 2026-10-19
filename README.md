> 2026-10-19 补评完成：用户明确批准300秒后，106/106正式条件完成。U2均值/随机各0/100，旧预算中断记录保留；未增加训练或恢复自动化。最新表、补评账本和研究限制见[FINAL_BATCH_REPORT.md](FINAL_BATCH_REPORT.md)。下方旧104/106或未授权描述为此前历史。

# il-rl-wm

> 2026-10-19 终态更新：正式测试104/106条件完成，U2均值预算中断、随机未启动。详见[FINAL_BATCH_REPORT.md](FINAL_BATCH_REPORT.md)。共享总账23.647652小时；原最终cap已用完，未安排重试或新训练。下文开发记录/早期状态保留为历史，不能当作仍在运行。
就是让rl以il为基础然后通过世界模型改进最后rl的性能超过wm


当前规则见 [PROMPT.md](PROMPT.md)、[RESEARCH_CHARTER.md](RESEARCH_CHARTER.md)、[最新任务](ACT_NEXT_STAGE_TASK_2026-10-19.md)。实际作业与共享预算见 [research_state.json](research_state.json) 和 [RESEARCH_LOG.md](RESEARCH_LOG.md)，运行时以各manifest/status为准。

2026-10-19：ACT教师、D1行为蒸馏、C1独立参数迁移、同协议T2–T6在线对照及U2有限均值保持均已完成开发运行。T6专家混合fixed-final为34/40；其他纯在线方法与U2为0/40。T6收益不能直接归因ACT初始化；ACT-WM仍未验证。

- [U2完整验收](ACT_U2_FINAL_REVIEW.md)：闭环、均值/方差、真实奖励、Critic日志和真实状态梯度证据。
- [因果缺口与10问](ACT_TRANSFER_CAUSAL_GAPS.md)：区分强教师、初始化、专家回放和RL贡献。
- [下一批协议与预算草案](NEXT_RESEARCH_BATCH.md)：R0/R1/R3未获GPU授权，未启动。
- [完整历史报告](ACT_TRANSFER_REPORT.md)、[失败记录](ACT_FAILURES.md)、[文献/源码核验](ACT_RELATED_WORK.md)。

原106条件正式评测在方法冻结后后台执行，共享原7200秒限额；未完成条件保持空值。单卡24小时总预算不重置。阶段代码、配置与原始证据增量上传research/act-sac-transfer，保留Draft PR #2，不合并main。

[复算说明](REPRODUCING.md)、[ACT便携权重](releases/ACT_REPRODUCTION.md)、[阶段快照与历史](PROGRESS_SNAPSHOT.md)。保留全部MLP-BC/U0/U1及失败结果，ManiSkill官方BC/SAC仍是主训练底座；其他框架只读参考或隔离对照。核心目标仍是最终不依赖ACT的SAC+WM。
