# 失败驱动创新挖掘（2026-10-20）

本轮由用户明确要求 ARIS + nature-academic-search，多源查新并由 DeepSeek V4-Pro 最多评审8轮。工作目录沿用 il-rl-wm，不进入另一个 lowdata 项目。只读取已有实验、做CPU复算和文献研究；新增GPU预算0，不重启R3，不恢复自动化，不修改原核心约束。

门槛预先固定：overall≥8、novelty≥8、importance≥8、mechanism/testability≥7、feasibility≥7，无未解决的直接先例碰撞或致命问题。评分是未校准的研究判断（calibration:none），通过只意味着候选值得申请实验，不意味着结果成立或论文可接受。8轮以内可以提前通过；最多3批不同机制候选，停滞或预算用尽记NO_QUALIFIED_IDEA/NEEDS_RETHINK。

证据锚点：本地HEAD 9f1536c；远端研究分支21935f29787dad569731831c65b956f2e5158e91，main仍7e678e3。GitHub网页抓取失败，已用已连接GitHub读取固定commit文件，并通过git ls-remote核实分支。

本轮发现的状态差异：R3 manifest记录2026-10-19 10:17:35 UTC退出，exit1，signal15，计费846.954124秒；PID71954不存在。旧队列和research_state仍写training。R3无完整结果，不解释SIGTERM的来源，不把中断当方法失败，不重试。R0/R1/R2仅单训练seed开发比较31/30/34（各40），新seed30000正式比较未完成。

无.codegraph/或graphify-out/；当前工具列表无codebase-memory图谱、academic-search MCP。代码发现无需进行；必要实证读取走精确已知路径。学术检索采用网页主来源+公开API；preflight: PubMed/CrossRef/arXiv三端点均可达。DeepSeek MCP可用，按本地reviewer contract处理，不用Codex自评替代。
