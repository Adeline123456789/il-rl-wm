# IBRL：双候选与续接目标核对

核验日期：2026-10-16。Hengyuan Hu、Suvir Mirchandani、Dorsa Sadigh，*Imitation Bootstrapped Reinforcement Learning*，arXiv:2311.02198v6。本次阅读范围为[论文 IV-A–B 和 V-A](https://arxiv.org/html/2311.02198v6#S4)，不是全文复现。

论文用独立 IL 同时参与实际动作选择和 TD 目标构造；soft 版本针对贪心选择可能屏蔽动作的问题。其论文实验主干为 TD3，不能直接称为本项目的 SAC。论文中的目标续接策略仍允许 IL，因而也不能把其 Q 当成移除 IL 后学生的价值。

[作者源码 rl/q_agent.py](https://github.com/hengyuan-hu/ibrl/blob/3c457da3689cc90f881bcdf42a0a578f4a1f60aa/rl/q_agent.py) 的 `_act_ibrl`、`_act_ibrl_soft`、`update_critic` 已逐项核对：target critic 比较两个动作，softmax 后真实抽样，bootstrap 可继续使用 BC。没有世界模型递推。源码许可尚未核实，保持只读，不复制。

对本项目的意义是检查“候选评分对应谁续接”，以及是否因低分而持续缺少学生数据。这些机制与本项目 F1 的首动作/完整序列后统一学生续接诊断有关；它们不证明当前模型排序错误，也不证明增加 softmax 会有效。U0 已有 0.2 实际随机 SAC 执行，不能把额外软选择、价值改造和蒸馏一起加入。待完整 U0/F1 证据后再决定是否提出单一局部假设。

额外成本与边界：IBRL 需独立 BC 推理及候选 critic 评估；其实现还包含不同网络与环境。论文的任务、数据和奖励时序不同，不能用其分数与我们的 ManiSkill continuing 协议直接比较。这里仅借鉴目标核对，不安装另一套训练框架。

出版身份补核：2026-10-16直接读取RSS官方页面，确认IBRL为RSS2024、DOI10.15607/RSS.2024.XX.056，作者与既有arXiv记录一致。来源：https://roboticsproceedings.org/rss20/p056.html 。这只补充出版元数据，不扩大此前方法阅读范围或改变源码许可判断。
