# 前人踩过的坑与本项目不能跳过的检验

检索日期2026-10-20；nature-academic-search multi-source-search。学术MCP未挂载，公开网页/arXiv/PMLR/作者项目页回退；OpenAlex补充调用HTTP429，未以空结果冒充不存在。arXiv/PMLR全文已取回者保存原文本与manifest；本文是针对候选的检索，不是穷尽证明。

|一手来源|前人问题与处理|对本项目的否定性约束|证据层级|
|---|---|---|---|
|[Q-Learning With World Models](https://arxiv.org/abs/2608.17163)，Dong/Jia/Finn/Sadigh，2026预印本|用WM展开搜索动作，Q/策略仍用真实转移训练|“WM只用于搜索、真实数据训练”已存在；本地不受想象TD标签污染，也仍会受错误选动作影响|arXiv v1全文；作者项目页称code coming soon，未声称可完整复现|
|[Imitation Bootstrapped RL](https://arxiv.org/abs/2311.02198)，Hu等|固定模仿策略与RL候选竞争；Soft IBRL保留低分动作被执行的机会|随机仲裁与双策略不是新；比较校正方案必须给等量真实执行对照|v4全文；已有本地作者源码核验|
|[MoDem-V2](https://arxiv.org/abs/2309.14236)，2023首发|专家先验附近的模型不能任意评价广泛动作；用受控交权、集成和保守评价|“不确定性扣分+从BC逐步接管”不能作为核心创新|v2全文|
|[TD-M(PC)²](https://arxiv.org/abs/2502.03550)，2025首发|规划行为与Actor不一致，约束策略与行为分布偏离|SAC-only bootstrap不保证Q查询在支持域内|v1全文，后续版本/会议身份不由此推断|
|[The Surprising Difficulty of Search in Model-Based RL](https://arxiv.org/abs/2601.21306)，Chang等，2026|搜索即使配合准确模型仍可伤害；MRS.Q用较大Q集成的最小值抑制高估|WM预测误差下降不保证决策好；只打败失稳TwinQ不足|v1全文；原文§5还含归一化等调整，不能把全成绩归于一行min|
|[Efficient Online RL Fine-Tuning Need Not Retain Offline Data / WSRL](https://arxiv.org/abs/2412.07762)，Zhou等|预训练策略真实预热，帮助在线分布适应|已有热启动，不应把4096预热换来源称原创；它含离线RL Critic，本项目fresh Critic不同|v2全文；既有作者源码commit核验|
|[What Makes Value Learning Efficient in Residual RL? / DAWN](https://proceedings.mlr.press/v306/ma26s.html)，Ma等，2026|研究残差RL的价值学习、数据锚定、归一化|支持调查数据支持，不证明必须额外长Critic预训练；依赖教师的残差部署不满足最终无IL|本轮出版方摘要；此前本地全文和代码审计见ACT_RELATED_WORK，未声称本轮重新复现|
|[Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem](https://arxiv.org/abs/2402.02868)，Wołczyk等|未重访的预训练能力会退化；比较保持方法|保持损失/遗忘不是新；教师均值MSE不约束学生方差，也不覆盖学生新状态|v3全文；本地已有NetHack源码审计不等同机器人SAC复现|
|[LOLS: Learning to Search Better Than Your Teacher](https://proceedings.mlr.press/v37/changb15.pdf)，Chang/Krishnamurthy/Agarwal/Daumé/Langford，ICML2015|在学习者roll-in状态比较动作，rollout混合参考/学生，改善参考与局部策略偏离的遗憾|“用同续接比首动作”已有基础；A2要有新的可验证交互发现或机制|PMLR正式全文；初次误取1502.02458被排除，正确arXiv1502.02206|
|[AggreVaTeD](https://proceedings.mlr.press/v70/sun17d/sun17d.pdf)，Sun等，ICML2017|利用专家cost-to-go教学生，比只看动作监督更丰富|专家后缀价值不是学生后缀价值；改变目标需明确|出版方全文网页核验，arXiv摘要存档|
|[THOR](https://arxiv.org/abs/1805.11240)，Sun/Bagnell/Boots|次优价值oracle的精度与所需规划时域相关|“多展开一点接价值”是已知权衡；不直接移植其reward shaping到稀疏主协议|摘要级，本轮未全文验证其定理|
|[Decoupled Q-Chunking](https://arxiv.org/abs/2512.10926)，Li/Park/Levine|长Critic与短策略解耦，处理反应性与多步传播|长段高分不推出首动作可继承；动作块不仅gamma幂次|v2 PDF全文提取；HTML404转PDF成功|
|[SEAR](https://arxiv.org/abs/2603.01891)，2026|多前缀动作块评价与重新规划|前缀价值本身不是新；其chunk目标不等同单步SAC|v2全文|
|[Beyond Visual Quality](https://arxiv.org/abs/2609.24745)，Yuan等，2026预印本|同状态oracle区分可用候选与能否识别；机会集中在少数决策|“先测候选覆盖再测排序”与“动作多样≠结果多样”已有直接研究|v1全文；作者项目页与arXiv标识更新不同，以arXiv记录为准|
|[SufficientPlan](https://arxiv.org/abs/2610.08350)，2026预印本|用配对闭环证据选择足够规划预算，并缓存不变上下文|通用非劣规划预算检验/缓存不是新；其后验标准不等同同时频率覆盖|v1全文|
|[Calibration-risk routing](https://arxiv.org/abs/2610.01001)，Zhang/Zheng，2026预印本|分离拟合、选择和校准数据，控制模型适应路由|校准切分原则已有；与动作候选仲裁仍不同，不能凭相近名称直接否定|v1全文|
|[Pass the Baton / Relay-OPD](https://arxiv.org/abs/2607.26057)，Xu等，2026预印本|教师与学生在失败前缀后的续接不对称；短教师接管后交回学生|“教师短接管再交回”与续接不对称已有LLM先例；机器人连续控制差异需要具体机制，领域不同不自动创新|v1全文，不使用其LLM结果证明机器人有效|
|[Smooth Exploration for Robotic RL / gSDE](https://arxiv.org/abs/2005.05719)，Raffin/Kober/Stulp|逐步独立探索导致抖动；状态相关并周期重采样的噪声|时间相关噪声和采样区间已有，A5不能仅改噪声就声称创新|arXiv摘要，会议引用由arXiv记录给出|
|[Using Common Random Numbers for Simulation-based Planning with Rollouts](https://arxiv.org/abs/2605.04732)，Yadav等，2026|共同随机数降低rollout相对效用方差|A3若只复用随机流是旧工具；尾段取消另需完整状态合流证据|arXiv摘要与投稿记录；全文补读前不援引具体证明|
|[Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)，Dudík/Langford/Li|用奖励预测和选择倾向纠正部分反馈评估|DR不是新；必须有真实随机化概率、正支持、固定/可追踪续接，不能把高斯密度当选择概率|arXiv摘要|

未核实项：[World Models Need Not Be Right to Rank Policies Right](https://openreview.net/forum?id=KIunPO9DWe)被浏览器challenge/API403阻挡，保留线索，不引用未读定理。该缺口意味着涉及“相对价值界”的候选仍有未解决的查新风险。

其他近邻线索如Contrastive Branch Policy Optimization、Prefix-GRPO、VH-GPO、TIDE，本轮只在检索中发现，未完成一手精读前不得作为确定碰撞，更不能把LLM agent结果当作机器人证据。
