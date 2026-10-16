Codex 下一阶段研究任务：ACT 教师、ACT→SAC 迁移与世界模型规划
目标仓库：https://github.com/Adeline123456789/il-rl-wm 基于分支：research/u0-batch-20261016（审查锚点 4e014a582be095eec4571495998f5cc433cc2664；开工先核对远端新提交）
你是科研工程执行者。请在单张 RTX 3090（24GB）上连续进行有界的实现→测试→实验→诊断→读论文和开源源码→局部改进。无需每个小模块等待网页 ChatGPT 审批；但涉及新增付费、超出用户已批准的总体预算、改变研究主干、不可恢复的操作时停止并征询用户。不要自行证明自己创新、不要篡改失败结果。完成里程碑后统一上传工作分支，保留旧代码和原始报告，不自动合并 main。
0. 先读资料、保护现有训练与证据
首先读取 AGENTS.md、PROMPT.md、RESEARCH_CHARTER.md、UPSTREAMS.md、RESEARCH_LOG.md、research_state.json、PROGRESS_SNAPSHOT.md、NEXT_STEPS.md、U1_CHANGE.md、diagnostics/F-U0-001.md、diagnostics/F1_batch_diagnosis.md、diagnostics/U1_seed0.md、configs/pickcube_protocol.json、configs/wm_u0_design.md 及本地最新状态。先查询 active PID、GPU 作业锁、剩余 GPU 作业时长及尚未入库的本地结果；不得因为新指令杀死、覆盖或重复启动现有训练。先保存当前工作区和历史 checkpoint 哈希。如果本地正在执行已登记 seed/最终测试，不抢同一张 GPU；可以先只读审计和 CPU 单测。
本研究主干不变：专家数据→训练并冻结独立 IL→把 IL 能力迁移给随机 SAC→训练时 IL/SAC 各产生候选序列→动作条件世界模型用短时真实奖励预测+独立纯任务长期价值统一评分→执行首动作并重新观测→真实环境 replay 与 SAC 探索/更新→最终彻底移除 IL，只保留 SAC 多候选+WM 规划。机器人操作只是检验通用算法的平台。
现有历史基线必须保留，不能事后改写：PickCube MLP-BC 闭环 0/40；旧 U0 seed0 末次裸 SAC / 含 IL 规划 / 无 IL 规划均 0/16；现有 LayerNorm U1 seed0 为 16/16、13/16、15/16；同等交互、无规划的 C/C-LN 续训均 16/16。上述只来自开发验证，正式多种子测试未完成。曾在固定诊断状态上做 4 次候选 F1（总计 496320 环境分支交互），但没有可辨识排序对，不能宣称 WM 排序好或坏。在线 WM 一步归一化状态 MSE 显著高于专家数据；U0 Critic 及任务价值曾数值失稳，U1 只测试了 Critic LayerNorm。这些问题不能因为换了 ACT 而消失。
新工作另建分支（如 research/act-sac-transfer），所有 ACT 方案标记为 ACT-T0 / ACT-D1 / ACT-W1；保留旧 U0 和旧 U1 的实验标签、数据、权重与源码。新教师与 SAC 桥接属于明确的结构变更，不将其成绩与旧 U0 直接归因于世界模型改善。
1. 锁定高质量源码，不要重写训练框架
主工程仍用已经锁定的 ManiSkill PyTorch 主训练器和相同 PhysX/Sparse 奖励协议。先核对该版本实际存在和许可，不自动升级主环境：
ManiSkill（唯一主环境/主 SAC/主 ACT 来源）：https://github.com/mani-skill/ManiSkill ；当前锁定 commit 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3；ACT 入口 examples/baselines/act/{README.md,train.py,act/detr/detr_vae.py,act/evaluate.py}；SAC 入口 examples/baselines/sac/sac.py。
原始 ACT：https://github.com/tonyzhaozh/act ；RSS 2023 https://roboticsproceedings.org/rss19/p016.html ；核对 CVAE、动作块、时间集成，不迁入它的 MuJoCo/ALOHA 环境。
LeRobot ACT：https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act ；对照 chunk_size、n_action_steps 和 temporal ensembling；不并行建立另一套主训练器。
SAC 数学参照 CleanRL：https://github.com/vwxyzjn/cleanrl 。
RLPD：https://github.com/ikostrikov/rlpd ；保留原有隔离基线及 Critic LayerNorm 先例。
Policy Decorator（ICLR 2025）：https://github.com/tongzhoumu/policy_decorator ，https://arxiv.org/abs/2412.13630 。基于冻结示范策略、残差在线 RL 和渐进探索；原论文是 BeT/Diffusion Policy，不可虚称它已在 ACT 上验证。其残差策略若部署仍调用冻结 ACT，只能作为不同范式的外部对照，不能取代本项目最终无 IL 路线。
PAC-ACT（2026 预印本）：https://arxiv.org/abs/2607.09590 。已研究从 ACT 向 chunk-level actor-critic 参数迁移及先验约束；本项目不可宣称“首次实现 ACT→RL/Actor-Critic”。核验全文、是否有官方代码及其真实训练算法（不默认当作 SAC）。
SEAR（2026 预印本）：https://arxiv.org/abs/2603.01891 ；DQC（ICLR 2026）：https://arxiv.org/abs/2512.10926 ，https://github.com/ColinQiyangLi/dqc 。研究 chunk Critic、回报及开环/闭环一致性；第一版不直接改成 chunk-SAC。
DAWN（ICML 2026）：https://github.com/Guozheng-Ma/DAWN ，Policy Decorator：研究残差 RL 的 Critic 归一化及预热，明确与既有 LayerNorm U1 的重合。
原始方案近邻 IBRL：https://github.com/hengyuan-hu/ibrl 、Q2RL：https://github.com/rai-opensource/q2rl 、QWM：https://arxiv.org/abs/2608.17163 、MRS.Q：https://github.com/facebookresearch/MRSQ 、TD-M(PC)^2：https://github.com/DarthUtopian/tdmpc_square_public ；用于双候选/价值错配/模型搜索查新，不同时改算法。
逐一登记完整 URL、锁定 SHA、许可证、实际读/复用的函数路径和源码差异；遇到错误 paper 链不得默默沿用。所有外部训练环境隔离。查不到源码时如实报告，不伪称已复现。
2. ACT 教师单独训练、评测（必须先过关）
使用本项目现成经过 GPU 真正动作回放审计、按完整原始 episode/source group 划分的数据，不重新用有 stale contact、跨批次污染或 CPU/GPU 不一致的历史数据；PickCube 以已核查 full-003 为起点，后续 Stack/Peg 也用各自有效来源组，Peg 的 126 成功轨迹不等于 126 独立来源。保持 obs_mode=state、pd_ee_delta_pos（Peg 使用其预登记控制模式）、reward_mode=sparse、physx_cuda、PickCube 200 步 episode 及已有 continuing-success 语义。ACT 只是专家动作监督，不把密集奖励或成功后特权标签加入训练。
先复用锁定 ManiSkill 官方 ACT 训练代码，改成项目兼容且轻量的 state-only teacher。训练目标是动作块，第一组 chunk K=8 或 16（依据实测调整，记录 K），预测头/Transformer 原始结构保持可追溯；先做 CPU shape/pad/seed 测试再做 GPU smoke。不得把分属不同 episode 的动作拼成一个 chunk；处理尾段 padding、动作尺度、状态/动作归一化统计只来自训练集。官方 ACT 推理使用 CVAE 先验（常用零潜变量），不得把训练时从真实动作编码得到的后验变量泄漏到测试时。原始 ACT 输出可能不是有界 tanh-Gaussian，请核实并记录 clip/尺度。
至少对比：旧单步 MLP-BC（原样冻结历史结果）与 ACT teacher 独立闭环，在同一有效演示 train IDs、同环境、同场景/验证种子下评价。ACT 测试两种明确不同的执行模式：A) 每步重新查询当前 chunk，只执行首动作，不做 temporal ensemble；B) 可选官方 temporal ensemble，一旦启用必须正确维护每个并行环境的历史、done/reset 缓存并记录结果。必须保证教师在真实部署条件下可运行，而非只看离线 L1/MSE。
初期可开发1 seed，小规模预先固定验证回合；若有希望，至少独立重训3 seed并按配对初始条件比较 success_once、success_at_end、阶段事件、动作误差、部署延时/吞吐、VRAM。正式最终测试 seed20000 不得在方案冻结前使用。若 ACT 仍失败，先检查数据对齐、ACT 的 chunk 尾部 padding、推理 latent/归一化、CPU/GPU 接触差异、失败的接近/抓取/放置阶段，有限度尝试一个有依据的改进；不要无证据增大到大模型。
3. ACT→SAC 迁移：先做可验证的蒸馏，再比较真正的参数移植
路线 A（优先、功能迁移）：ACT 教师 → 单步随机 SAC 学生
冻结独立 ACT 的全部推理参数、统计和可选时间缓存；保留 ACT 作训练期 IL 候选提供者。保持 ManiSkill 官方小型 Gaussian SAC Actor + Twin Q + entropy，使用专门的学生初始化数据（专家训练集状态 + 经实际 env.step 采集的 ACT 训练池状态；不得污染验证/测试数据；新增采样计真实环境预算）。从 ACT 在指定推理协议下得到当前执行动作 a_T，将学生 tanh-Gaussian 的确定性均值通过监督预热：
L_distill = E_s[ ||tanh(mu_SAC(s)) - clip(a_T(s), -1, 1)||^2 ]
可将 expert action MSE 作为独立可控的辅助实验，但不能无标注同时拼多个修改。优先训练 Actor 主干/mean 分支；log_std 采用显式有限初始值、保持非退化探索，不能把 ACT deterministic chunk 或 VAE 分布错误当 SAC 的 log_prob。学生必须实现有效 tanh Gaussian sample、log_prob/雅可比修正和独立的随机性，并在随后接受标准 SAC 的在线 Actor 更新。Critic/目标网络不从 ACT 假冒复制，按锁定主 SAC 初始化或用真实转移预热。
**重要：此路线是行为蒸馏/功能迁移，不是 ACT Transformer 的逐层权重复制。**检查初始 SAC 均值与冻结 ACT 在同一真实状态上的动作差和闭环成功率。对照：随机 SAC、旧 BC→SAC、ACT→蒸馏 SAC，无规划情况下相同在线预算的学习曲线。出现“教师强但学生弱”时，不允许绕过迁移验证直接进入 WM。
如果教师最强模式依赖 temporal ensembling，a_T 实际是 a_T(s,h)，仅观测 s 的单步学生可能无法表示该映射。先比较不做 ensemble 的教师与有 ensemble 教师；若只有带历史版本有效，增加最小必要的状态/动作历史作为明确可选的学生输入，SAC Q 和 WM 的观测接口相应保持一致，不能以随机的历史标签监督同一个 s 并假装能够精确拟合；历史输入及额外成本要独立对照。不要让历史预测缓存偷偷在终止后跨回合继续。
路线 B（真正参数复制；独立的第二组实验）
在路线 A 基础验证之后，取 ACT 推理用 state projection/Transformer decoder/第一个 action query 的特征，将兼容的预训练权重复制到独立的 SAC Actor。从首查询 h0(s) 接 mu(s)、log_sigma(s) 两个输出头，使其形成可计算 log_prob 的 tanh-Gaussian；用 ACT 当前首动作预热均值头，处理 ACT 原始 action_head、clip/tanh、动作归一化不一致。训练专用 CVAE 后验 encoder 不可要求在线推理依赖未来动作；冻结教师参数不能与学生 trainable 参数共享存储。Q 可以继续是原轻量 Twin Q，且无需强制输入整段 chunk。
必须测复制成功的参数 SHA/张量映射、ACT 本来首动作与新的 SAC 确定性首动作的一致程度、梯度/优化器状态、log_prob 有限且可微、Actor 参数量、整卡峰值、动作选择延迟和在线训练稳定性。路线 B 的新增参数和计算会影响可比性；其成绩与路线 A 分开报告。若超出本项目 <10M 可训练参数目标、20GiB 整卡显存目标或单卡墙钟预算，则明确停止 B，保留 A；不删除世界模型也不替换 SAC。不能无视差异直接调用 torch.load_state_dict(strict=False) 就称为成功迁移。
绝对不允许混淆的替代
“冻结 ACT + SAC residual 动作，最终动作仍需 ACT”是 Policy Decorator 范式，可作为独立方法对照，但不满足最终移除 IL，不得代替本项目主干。“SAC 一次输出 K 步，并只执行第一个动作，但将 K 步作为实际 replay 动作做 TD”存在执行/价值错配，暂不作为默认实现；要研究必须参考 SEAR/DQC，明确 K 步真实执行与折扣、终止、动作分布定义。
4. ACT 教师接入原世界模型 U0：严守两项评价语义
在原 scripts/planning.py、scripts/u0_modules.py 接口增加 ACT teacher adapter，不要重建第二套算法。ACT 提供 K 步 chunk，规划 H=2 时可使用其前两步作为教师候选（ACT chunk commitment），学生的八条候选用当前 SAC 在各自模型预测状态上递推采样；统一 predicted short reward + gamma^H * pure-task Q(s_H, mu_SAC(s_H)) 评分。每回合只实际执行被选序列首动作、下一步重新观测再规划；真正随机 SAC 探索执行的转移进入真实 replay。若尝试 ACT 在想象的第二步重新查询，必须作为单独的 candidate proposal 版本，并且不能假装等价于 ACT 原始 chunk。不得在想象 rollout 中修改 ACT 的时间集成历史状态。
原 SAC soft Q 与独立的 pure-task Q 不能混用；对后者继续使用真实 reward 和学生续接定义。保持冻结 ACT 的权重/统计不变、student SAC 可训练、同一 reward、两个候选来源、20% 实际 SAC 探索；记录 ACT/计划 SAC/随机 SAC 的实际执行次数而不只是候选数量。部署 no_il 模式必须在没有 ACT 模型构造或调用的情况下从头运行，并对照 checkpoint 的真实动作一致性。
执行公平消融（在同一预登记设置下，不同时加入更多模块）： A. 旧 BC teacher + 旧 BC-init SAC + 已验证 LayerNorm critic + 无规划 C-LN 续训； B. ACT teacher + 旧 BC-init SAC + 相同 LN/混合/WM/预算（仅改变教师候选来源）； C. ACT teacher + ACT-distilled SAC + 相同 LN/WM/预算（再改变初始化）； D. ACT-distilled SAC + 同等专家/在线数据、无 WM 规划的续训对照； E. ACT only 教师单独闭环；另记录 SAC only、IL+SAC+WM、no-IL SAC+WM。 不得把旧 U0（无 LN、发生崩溃）与新 ACT + LN 结果直接归因于 ACT；如果 B/C 较旧 U1 有差异，明确是 teacher、init、candidate 覆盖还是在线分布变化，必要时做固定候选/固定快照对照。早期只跑开发 seed，效应明确后再做至少三独立训练 seeds 及 Stack/Peg 多任务验证。
5. 既有漏洞不能因换 ACT 而跳过
F-BC：旧 BC 0/40；ACT 必须在同数据闭环证明自己更有用，否则不能宣称新教师更强。保存完整失败回合视频和接近/抓取/放置阶段统计。 F-Q：旧 U0 Critic 价值漂移、学生崩溃；保留原 U1 Critic LayerNorm 的稳定实现和无规划同父续训对照；核查 ACT 导致的经验分布偏移和纯任务价值不合理负值，不直接添加新 ensemble/dropout。参考 RLPD、DAWN、TD-M(PC)^2、MRS.Q。 F-WM：旧 WM 在专家与在线分布状态预测误差显著不同；增加 ACT 状态来源、学生来源、失败恢复和成功前阶段的验证分层；同等真实数据预算校验。较低 one-step loss 不证明高质量排序。 F-rank：旧 F1 四次分支全没有可排序对，不能将零 regret 写成完美判别；新 ACT 可能扩大候选质量差异。先在预登记的接近、抓取转换、扰动和失败恢复状态筛查候选真实续接差异，只有差距显著超过 Monte Carlo 误差/尾段容限，才测 WM 选错损失。统一共同 SAC 续接，分别报告首动作续接与执行完整 H 后续接；对固定候选比较直接 Q、WM H=1/H=2、随机选择及理想 oracle，诊断所有额外 env.step() 计入交互成本。不能再次无目标花数十万步得到全平局。 F-explore：逐步独立 20% SAC 接管不保证长序列探索；先统计有效连续控制长度和新成功状态覆盖。若确实不足，再固定总 SAC 执行预算比较按步随机 vs 连续自主片段，单独定义这项改变，不能同时改 teacher、Critic、评分。 F-credit：ACT chunk 高分可能来自后续 ACT 动作，学生实际仅接管首动作后的环境；不要直接用整段 ACT 分数监督当前 SAC 单步动作，先做共同学生续接评估。 F-remove：删除 ACT 后不能调用其 encoder、decoder、action cache 或教师权重；不仅要动作一致，也要完整闭环无 IL 成绩。
6. 实验统计与决定依据
保持 ManiSkill state、sparse、控制模式、回合长度、继续成功协议、源数据 ID 与 BC/旧 U1 尽可能一致。把 ACT 训练/额外教师轨迹、蒸馏、SAC 在线交互、WM 更新、真实排序分支、验证与最终测试的 GPU 小时、环境步、CPU RAM、整卡 VRAM 分账。旧正式最终测试 seed20000 在方法、配置和所有比较条件冻结之前不得触碰；开发/调参用旧协议 validation seed10000，排序使用独立诊断种子及状态池。多训练种子≠同一模型多测试回合。多任务先 PickCube 验证接口，再 StackCube 与 PegInsertionSide；Peg 成功轨迹和来源组不足的问题必须如实披露。
首轮 ACT 参数可从官方已验证 state ACT 降至轻量：例如 hidden 128/256、2 encoder/2 decoder、4 heads、chunk K 8/16、适配批次大小；这是建议的待测超参数，不能声称官方验证了这个具体配置。先实测可训练参数、预测速度和显存，再确认 <10M 本方法训练参数、整卡目标 <20GiB、单卡 RTX3090 24GiB。当前批次已经有总 GPU 小时上限与并行任务；不能重置已授权预算。若新阶段没有可用授权余额，先运行 CPU 审计/测试并提出最小 GPU 预算申请，不擅自购买云算力/API。
阶段停止标准：A 教师必须有已校验的闭环结果且与原 BC 同协议比较；B 蒸馏 SAC 必须在真实状态匹配 ACT 首动作且跑通独立闭环；C Gaussian log_prob、SAC 真实更新和完整 no-IL 路径均通过；D U0-ACT 的训练和无规划对照同预算；E 排序性能仅在真实可辨识候选上评估。若某项失败，先检验实现、数据和数学假设并保留负结果，不通过大模型或改任务奖励来“修好”数字。
7. GitHub、成果与开源归因
连续自主推进 一个预先登记并受预算约束的 ACT-迁移研究里程碑；每个模块修好才进入下一步，但不必每个小模块等待网页 ChatGPT。源码与研究章程的变更必须记录、原始已发布结果不可覆盖。新增 ACT_RL_TRANSFER.md（接口、公式、参数复制/蒸馏区别、数据与测试）、ACT_TEACHER_RESULTS.md、ACT_TO_SAC_RESULTS.md、ACT_WM_COMPARISON.md、LITERATURE_ACT_RL.md、研究日志和可复现的 configs/commands/logs/raw episodic metrics/plots/failed runs、dataset/upstream/checkpoint manifests。已有 U0/U1 的研究与失败归档保持原样。
每个里程碑完成时提交新 GitHub 分支，保留 commit 与 PR，不自动 merge main。不给出不存在的 ACT 训练结果，不称蒸馏为直接参数复制，不称 ACT、LayerNorm、PAC-ACT 近邻机制为本项目原创，不把 Policy Decorator 依赖 ACT 的 residual 部署冒称为 no-IL。报告明确回答：ACT 比基础 BC 强多少？什么模式真正使它强？ACT 能否在单卡预训练？功能蒸馏能否继承成功行为？参数移植相对蒸馏有没有净收益？ACT/新 SAC 是否形成互补候选？世界模型有无额外规划收益？规划是否仍伤害 SAC？真实探索有无改善？最终移除 IL 后是否仍有实证收益？
立即从检查现有训练、源码和数据开始。在未充分验证 ACT 教师/桥接前，不改原有 World Model 评分机制，不把本项目直接改为其他论文的完整方法。科研结果允许为负，必须保留并报告。