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

## 2026-10-19：F-ACT-01 定向补核（当前仅摘要/书目）

- Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011, PMLR15:627–635. https://proceedings.mlr.press/v15/ross11a.html 。出版方书目和摘要核验；策略动作影响下一状态，监督样本IID假设不适用。关联当前学生自身状态夹爪误差扩大，但这是候选解释，不是本实验因果结论。实施DAgger式扩展前还需方法/源码核验，不把验证states回灌训练。
- Rusu et al. Policy Distillation. https://arxiv.org/abs/1511.06295v2 ，v2于2016-01-07，页面注明Submitted to ICLR2016，未据此声称已录用。摘要证据是Atari/DQN策略压缩，不能作为ACT→连续标准SAC成功保证。本次未复用代码，无新增依赖。
- Czarnecki et al. Distilling Policy Distillation. https://arxiv.org/abs/1902.02186 。定位并打开元数据页；未完成方法精读，暂不据此设计新机制。

没有将行为蒸馏或ACT参数复用称为原创。下一阶段只比较参数移植，后续改进须另登记单一机制和实际数据权限。

## 2026-10-19 F-ACT-02：早期遗忘的定向检索与源码核验

沿用 nature-academic-search 的多来源检索流程；本会话未挂载其专用 search_papers/search_arxiv 工具，实际使用网页检索、arXiv/PMLR/作者项目页及已认证GitHub只读接口。检索词包括 SAC pretrained policy fine tuning catastrophic forgetting behavior cloning regularization、ACT SAC fine tuning policy preservation 2026、IBRL bc loss actor update。过滤无关LLM遗忘及论坛建议；不声称穷尽查新。

- **IBRL / RFT**：作者[项目页](https://ibrl.hengyuanhu.com/)将RFT明确列为BC预训练后加BC损失的既有基线。锁定`3c457da3689cc90f881bcdf42a0a578f4a1f60aa`的[rl/q_agent.py](https://github.com/hengyuan-hu/ibrl/blob/3c457da3689cc90f881bcdf42a0a578f4a1f60aa/rl/q_agent.py)中`_compute_actor_bc_loss`用零噪声动作对demo动作作平方误差；`update_actor_rft`将其加到Actor目标，动态版本按参考/当前Q比较调权。该Actor目标没有SAC熵项，不能整套照搬。许可仍未核实，代码只读，无复制、无新依赖。源文件hash与原记录一致。
- **AdaptiveBC**：Zhao/Boney/Ilin/Kannala/Pajarinen，[arXiv2210.13846v1](https://arxiv.org/html/2210.13846v1)，已读§III–V及算法1。BC约束可缓解初期下降，但过强会妨碍改进；方法使用TD3式目标及Q集成，不能当标准SAC。作者[代码](https://github.com/zhaoyi11/adaptive_bc/tree/d47226d4ffc9959ac0ed1337ddd7ebb787fc05e1)的Actor损失以绝对Q均值归一化，调权使用归一化回报、移动平均和回合长度；论文正文的归一化写法与代码不同。本项目不引入集成、自适应回报控制或Q归一化。四个只读文件逐Git blob匹配，Apache-2.0，详见`literature/act_forgetting_source_manifest.json`；不安装其Python3.8/Gym0.21/D4RL依赖。
- **Forgetting mitigation**：Wolczyk等，[ICML2024/PMLR235:53039–53078](https://proceedings.mlr.press/v235/wolczyk24a.html)，出版方书目/摘要核验。其关注微调初期未访问的状态子空间能力退化，可作为覆盖不足的竞争解释。PDF抓取报Internal Error，本次只据摘要记录，未宣称方法或作者源码已核验；不据此实现新模块。

这些来源说明行为保持、BC正则与遗忘分析已有先例，不构成本项目原创。AdaptiveBC的alpha是BC权重，不能与SAC温度alpha混称。Policy Decorator/PAC-ACT的既有核验继续有效，但残差教师依赖/chunk PPO均不替代本项目单步SAC。

## 2026-10-19：补读 forgetting mitigation 方法与作者源码

此前PDF访问失败，现已通过[arXiv v3 HTML](https://arxiv.org/html/2402.02868v3)阅读§2–5、B.1/B.3、C及F的相关段落；不把先前摘要级记录改写为当时已读全文。论文BC保持在预训练状态上计算teacher→student KL，KS则用学生在线状态；两者不能混称。机器人实验使用SAC，Actor保持而不正则Critic，但含阶段head、额外时间输入、成功/时限terminal及奖励修改，不能直接迁入本项目稀疏continuing协议。NetHack使用APPO，并伴随冻结encoder、熵设置等差异；这些不是本项目单一保持损失的效果证据。

论文直接链接的[作者仓库](https://github.com/BartekCupial/finetuning-RL-as-CL/tree/390d2041a2829a9ca609594d11e5567e0f1368d2)锁定`390d2041a2829a9ca609594d11e5567e0f1368d2`。实际读取NetHack `learner.py`的`_distillation_loss`、`_kickstarting_loss`和系数调度，以及`models/kickstarter.py`：categorical teacher→student KL；KS过滤invalid样本；teacher eval/no-grad，wrapper forward仍调用teacher并保留其RNN state，因此不能直接当作本项目无教师部署。代码还包含保持系数与探索系数的可选联动，本项目不复制它。三文件含MIT LICENSE逐Git blob验证，见`literature/forgetting_mitigation_source_manifest.json`。机器人SAC补充源码尚未核验，未安装NetHack或SampleFactory。

对F-ACT-02的下一步启发仍是待验证推断：区分专家训练状态上的技能保持与在线覆盖不足；本项目ACT为确定性零latent教师，不能直接定义其categorical/Gaussian KL。若测试均值MSE保持，必须说明它没有约束SAC方差；若测试初始随机SAC分布KL，应另立单机制对照，不能把二者一起加入。当前仅来源核验，未启动改进实验。

## 2026-10-19：WSRL / DAWN 定向核验（不新增训练机制）

**Observed problem**：T5缺乏真实成功转移，U2保持期仍无正奖励；T6专家混合可学习但初始化贡献未分离。下一候选仅改变4096预热动作来源，不复制整套算法，不叠加U2。

### WSRL

Zhou、Peng、Li、Levine、Kumar，*Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data*，ICLR2025；arXiv2412.07762预印本发表于2024。已核对[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html)、[方法/算法与附录F–I](https://arxiv.org/html/2412.07762v2)。研究起点含离线RL策略及价值函数；短期策略rollout支持在线价值重校准，论文中的初期冻结Actor实验也未消除解冻后的下降。这里没有预训练Critic，因此不能照搬“价值重校准成功”。

作者仓库锁定`ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2`：`finetune.py`302–341行始终由当前策略采集，warmup期间不更新，之后才进行在线更新；`sac.py:update_high_utd`将batch分成多个Critic子批，再一次Actor/temperature更新。实际antmaze启动脚本为5000 warmup、UTD4、batch1024，wsrl配置10个Q并取2、策略与Critic均LN；奖励也有scale/bias。这些都与当前单步双Q SAC不同，不移植。核验树未包含LICENSE，未推定授权，不复用源码。

### DAWN

Ma、Li、Wang、Liu、Bacon、Tao，*What Makes Value Learning Efficient in Residual Reinforcement Learning?*，ICML2026/PMLR306:84569–84601。[出版入口](https://proceedings.mlr.press/v306/ma26s.html)链接作者源码。正文3.1/3.2比较基础策略数据锚定、额外Critic预训练、归一化；方法针对冻结基础策略的有界残差RL。其默认不需要额外Critic-only阶段；这与“必须先长时间单练Critic”的解释不同。

作者仓库锁定`71122b7fa89568bc2d49831fed8cb1b01e5a91e8`：`run_dawn_maniskill.sh`默认learning-starts20000、warmup-strategy none（零残差，实际执行基础策略）、critic-warmup-updates0。训练入口535–559行实现预热，635–731行是可选Critic-only路径而非默认；299–315行最终evaluate仍调用基础策略。`nets/dawn.py`的LN Critic是残差块网络，不等于本项目原U1。LICENSE文件实际0字节；且入口依赖仓库树内未见的`utils.profiling`，未验证可直接运行。不安装框架、不复制代码。

### 有界适配与证据边界

论文和源码支持提出数据支持假设，不证明当前4096 uniform是唯一根因。未来只改预热动作来源，保留现有双LN Critic/更新/熵/预算；初始4096真实步计入总交互，ACT执行显式计费并在部署移除。额外Critic更新、UTD、残差、归一化改动均不捆绑。本轮只完成只读核验与计划，实验结果=未运行，结论=待检验，原创性=已有预热思想在此迁移接口下的受控验证。

11份作者源文件逐Git blob和SHA256验证，准确路径/版本/字节数见`literature/wsrl_dawn_source_manifest.json`。源文件只留本地.cache、未纳入训练或分发；复用文件=无，修改=无，新增训练依赖=无。DAWN论文PDF hash与获取记录见`artifacts/wsrl-dawn-literature-001/status.json`。网络PDF读取工具失败后从出版链接后台获取并本地提取；首次GitHub连接器返回的是解码文本而非API JSON，纠正解析后全部blob哈希匹配，未把失败当完成。
