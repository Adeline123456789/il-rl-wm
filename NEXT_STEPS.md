# 后续工作（首批执行中）

## 2026-10-19 ACT 阶段更新

T2–T6全部完整结束，fixed-final成功0/0/0/0/34（各40回合），真实10环境ACT入口短检通过。U2唯一有限均值保持对照已登记并后台运行：PID2935565，权威状态artifacts/act-u2-full-batch-001/status.json，完整262144交互、cap4500秒。先查实际PID和manifest，不重启；CPU后处理已依赖排队，避免重复导出/诊断。

1. U2结束后审查原始结果、15360次预期有效辅助更新及其后零系数、Critic/Actor初始化一致、所有数据仅expert-train。比较T5与U2单因素，以及T4与T6额外专家数据收益；失败也保存。
2. 两个局部假设额度已登记用完。冻结比较或明确关闭有限负结果阶段后，记录理由并释放configs/act_final_test_gate.json。用新队列ID运行configs/act_final_evaluation_plan.json：88历史+18ACT条件，共106条件，仍共享原7200秒总cap；不使用测试结果调参。若U2失败/未完成，显式保留未测试行，按runner的--ids选择实际已完成的模型，不能把缺失当零分。
3. 从原始逐回合记录汇总，回答ACT十二问题及PROMPT十问题，刷新并验证完整源码历史归档，后台上传并核对远端。保留全部失败与未完成项。ACT多种子/多任务和新WM尚未完成，不能用T6成功代替ACT迁移/WM贡献证据。

每个可验证小阶段后台增量更新研究分支与Draft PR #2，不合并main。 当前本地源码HEAD固定为9f1536c直到最终评测结束；归档及后续证据通过临时Git index的tree后台发布，正常index与HEAD不改。下次发布必须以最新已核验远端tree为base并只添加本阶段文件，不能从HEAD重建而丢失已发布归档；最后评测结束再做本地交付提交。详情见research_state.json.source_publication和最新github-act-stage-upload回执。旧训练和以下2026-10-16历史计划完整保留。真实剩余预算以全部GPU manifest加当前运行时间为准，7200秒最终评测额度不可挪作研究调参。

截至 2026-10-16，本文件记录未完成事项，不代表批次结项。实时任务以 `research_state.json` 和各管线 `status.json` 为准；总限额仍为 24 GPU 作业小时，其中保留 2 小时最终评测。只使用一个已登记研究改动 U1（SAC critic LayerNorm）。

1. 完成已排队的独立 PickCube 训练种子、隔离 RLPD 完整基线和 StackCube/PegInsertionSide 开发运行。Pick seeds1–4 分别从头训练 BC；U0/U1 与各自 C 续训对照按累计交互预算配对。失败或预算不足的条件保留，不以零分填充或按成绩删除。
2. 方法固定后执行 `configs/final_evaluation_plan.json` 的独立测试：固定末次模型，每条件 100 完整回合。当前 52 个模型/阶段、88 个方法条件尚未开始测试；使用 `scripts/run_final_evaluation.py`，按实际训练完成情况显式保留未完成项。
3. 用 `scripts/summarize_final_results.py` 生成逐条件表和训练种子间离散程度；开发 seed0 与独立确认分开报告，RLPD 等交互与额外计算分开报告。结合真实曲线、行为计数、F1 限制和文献证据回答 PROMPT.md 的全部十项结项问题。
4. 导出可携带分析权重和无 IL 推理包，保留原始逐回合指标、失败日志、哈希及复算命令；在最终代码冻结后生成并验证源码历史 bundle。统一上传新工作分支，验证远端内容，不合并 main。

首批尚不能回答的研究问题：U1 的稳定性改善是否跨独立种子成立；规划器是否胜过同预算稳定 SAC；是否能迁移到另外两个任务。开发 seed0 的 U1 无 IL 规划为 15/16 验证成功，裸 SAC 和 C-LN 续训为 16/16，因此不能把修复 U0 崩溃等同于规划收益。

四次 F1 诊断均缺少可排序候选对，本批已停止前缀搜索。后续若单独研究排序，最有信息量的是在预登记的接近/抓取转换/扰动状态覆盖下，验证同状态候选能产生可辨别的真实回报差异，再评价评分；不应继续重复成功平台或降低容限。该工作不在当前已排队实验内。

Stack/Peg 额外训练 seeds1/2 与 RLPD 额外种子仍未排队，是否能执行取决于实际剩余预算；视觉阶段未实施。Peg 只有 126 条 GPU 成功轨迹、88 个不同来源组，CPU/GPU 开环回放成功差异尚未消除，不能借用 CPU 标签或把重复组当独立演示。

2026-10-19 05:42 UTC：U2完整结束，CPU诊断和单因素审计exit0。比较冻结见artifacts/act-comparison-freeze-001/result.json，gate已放行。仅启动一次act-final-evaluation-launch-002/run.py；其批次输出act-final-evaluation-batch-002和汇总act-final-summary-002。不得重复启动或以final结果调参。所有106条件后更新报告/原始证据/源码commit覆盖并后台交付，之后暂停本批自动化。

2026-10-19：最终队列002已因旧导出计数字段失败；001/002失败证据保留。唯一当前队列003，PID3003841，先核验act-final-evaluation-launch-003/status.json与batch-003，不重启002。仅余68登记项，共享剩余7078.596346秒。自动汇总act-final-summary-003包含已完成002的两项。Stage16远端6de7192596f3feee1de2f2b09232644ca3468fb3，tree4f36d687fbd59776748b58dca772d173cb5bf1e4已核验。
