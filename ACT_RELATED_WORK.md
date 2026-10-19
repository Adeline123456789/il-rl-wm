# ACT 迁移来源核验

2026-10-17。当前为针对教师和迁移接口的来源审计，非穷尽综述。所有代码只读参考或局部复用，不混装外部框架。详见 `literature/act_source_manifest.json` 与 `vendor/maniskill/act/SOURCE_MANIFEST.json`。

| 来源 | 已核实版本与用途 | 适用边界 |
|---|---|---|
| [ManiSkill ACT](https://github.com/mani-skill/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/act) | Apache-2.0；7个源文件blob逐项匹配；DETRVAE/Transformer原样保留，训练/评测局部适配 | state入口确实存在；本项目小配置、mask、末层、无EMA/采样及执行差异明示，不称完整原样复现 |
| [原始ACT](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488) | MIT；读取detr/models/detr_vae.py、transformer.py及LICENSE；[RSS2023](https://roboticsproceedings.org/rss19/p016.html)，Zhao/Kumar/Levine/Finn，DOI 10.15607/RSS.2023.XIX.016 | CVAE训练动作后验与推理零latent分开；动作块/时间集成已有，不是本项目原创；未接入ALOHA/MuJoCo |
| [LeRobot ACT](https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/act) | Apache-2.0及第三方MIT声明；读取modeling_act/configuration_act；n_action_steps、队列、ACTTemporalEnsembler | chunk长度与实际执行长度不同；集成要求每步查询并有历史，不能假定纯state标签；不安装其框架 |
| [PAC-ACT v1](https://arxiv.org/html/2607.09590v1) | Pang、Li，2026-07-10预印本；核对方法、实验及消融章节，算法1为chunk-level PPO，含先验保持项 | 不是标准单步SAC；主实验dense，sparse消融仍保留行为先验。129M级模型/A100实验不能证明本项目3090可行。当前未在论文链接和定向搜索中核实到作者官方代码；不伪称复现 |
| [SEAR v2](https://arxiv.org/abs/2603.01891v2) | 2026-08-13修订，Nagy等；摘要核验多前缀chunk critic与receding horizon/random replanning | 本轮不改chunk SAC；尚未源码复现 |
| [DQC v2](https://arxiv.org/abs/2512.10926v2) | Li/Park/Levine；摘要核验critic/policy chunk长度解耦；[作者源码](https://github.com/ColinQiyangLi/dqc)待后续锁定 | 目标/价值错配近邻，不能据摘要称完成方法复现 |

PAC-ACT已涉及ACT参数迁移，故不主张“首次ACT→Actor-Critic”。其推理随机性/CVAE表述与本项目锁定原ACT零latent实现需区分；本项目不照搬其PPO、奖励先验或chunk TD语义。

Policy Decorator：已阅读[论文v1方法§4及失败分析附录F相关段落](https://arxiv.org/html/2412.13630v1)，并核对[作者BeT残差训练/评测实现](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_bet_maniskill2.py)。小残差以受限幅度和逐渐增加的执行概率进行探索，教师参数冻结；evaluate仍调用教师，故不能作为最终无教师方案。论文报告随机critic导致技能退化的现象，可用于后续F-ACT-02的竞争解释，不证明本项目已发生。源码还保留BeT dropout、修改reward为r−1，训练协议不等于本项目。README声称MIT但LICENSE实际为空；只读研究、不复制或安装。来源hash见manifest。

继续阅读清单：DAWN价值预热/LN、Q2RL/QWM/MRS.Q/TD-M(PC)^2仅按具体失败阶段核查；在读到准确源码前不宣称已复用或复现。原RLPD/IBRL/TD-MPC2的已锁定SHA、许可与源码范围沿用 `UPSTREAMS.md`，不重新安装或更换锁定版本。未发现ACT闭环或迁移故障前，不引入新的ensemble、悲观评分或探索奖励。
