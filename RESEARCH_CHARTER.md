# il-rl-wm：研究章程

版本：2026-10-16 · 开工整合版
目标仓库：https://github.com/Adeline123456789/il-rl-wm

本文件规定研究边界、开源代码的角色及实验纪律。具体执行过程见同目录 `PROMPT.md`。
它是项目文件，不是 Skill，也不是自动调度程序。

## 1. 研究目的与原始主干

研究一种通用的轻量级策略改进机制。机器人操作是实验平台，不是算法必须依赖的任务特定设计。

1. 用固定的专家演示数据训练 IL。
2. 预训练结束后保留一份完整冻结、独立的 IL 推理函数，包括其编码器和归一化统计。
3. 将兼容的 IL 参数复制或通过明确的蒸馏初始化到随机 SAC Actor。不得用共享可变参数把“冻结 IL”偷偷更新。
4. 训练时，IL 与 SAC 从同一个当前环境观测出发，分别提供同一规划时域内的动作候选序列。
5. 动作条件世界模型预测短期状态/潜在状态与奖励，结合单独学习的终端价值，对候选统一评分。
6. 正常规划时执行所选序列的首个动作，再读取环境观测并重规划。
7. 保留 SAC 随机策略的实际探索执行机会，不把“产生很多想象候选”当成实际探索。
8. 用实际仿真 `env.step()` 产生的转移训练 SAC 和世界模型。
9. 最终部署不调用 IL，保留 SAC 多候选生成、世界模型、必要的价值评价与滚动规划。

目标是验证这个构想成立的条件、失败原因及有效的局部修补，而不是保证它一定成功。
“不能改变主干”不等于不能修正数学定义、数据接口和已知实现错误。

## 2. 可以改什么、不能改什么

可以在主干内研究候选排序、价值校准、短时预测、探索分配、数据覆盖、蒸馏、时域与采样等局部机制。

不得为了分数高而把 SAC 换成 PPO、把最终方案换成纯 BC、删除世界模型，或复制别人的整套算法后改名。
无世界模型、无 IL、无探索等版本可以作为明确标记的消融，不能冒充最终方法。

U0 是经过必要正确性修复的最小完整主干，而不是故意保留已知错误的程序。
U1、U2 等是独立、可回滚的研究修改；如果 U1 无效，U2 应从 U0 或已验证的父版本出发，不默认叠加失败模块。

## 3. 单卡资源边界

- 仅使用一张 RTX 3090 24GB；同一时间一个 GPU 训练作业。
- 初期小型 MLP、短时世界模型；视觉阶段优先小 CNN，不引入大型 VLA/VLM/视频生成基础模型。
- 项目初始自定上限：本方法全部可训练参数合计不超过 1000 万；整张 GPU 运行占用以 20 GiB 为目标，24GB 是物理限制，不是已验证用量。
- 达到目标上限时，先查回放、并行环境、候选缓存及 batch；不得自动扩展到多卡或云端。
- 原始图像回放优先位于 CPU 内存或磁盘，按 batch 送入 GPU；同时报告 CPU RAM 和磁盘占用。
- 必须先测量再声称能在 3090 上训练，不能根据参数量或上游 README 推断整个组合已满足要求。
- 不自动购买算力、调用计费 API 或扩大授权范围。

## 4. 多任务稀疏奖励实验

使用 ManiSkill3 任务与演示数据。按锁定代码版本核实每个任务的数据、稀疏奖励、成功判据和实际回放能力。

优先开发任务：PickCube-v1、PegInsertionSide-v1、StackCube-v1。
扩展候选：PushCube-v1、PullCubeTool-v1、StackPyramid-v1 等不同机制任务。
这些是待核实的实验选择，不代表每个版本都已验证全部可运行。

- 第一阶段是“同一算法在多个任务分别训练和评测”，不是默认训练一个大型联合多任务模型。
- 先 state 观测隔离机制问题，再验证视觉输入；不得把 state 结果写成视觉实验结果。
- 主实验使用环境稀疏奖励，不偷偷添加阶段奖励、距离奖励或特权标签。
- 不把所有任务的 sparse reward 写死成某种统一数值规则：核实实际实现，包括成功、失败、重复成功奖励和提前终止。
- 固定演示源、完整 episode 划分、奖励版本、动作模式、仿真后端、时限与评测初始化。
- 专家动作监督不等于完整 RL 转移；缺少奖励/下一状态必须按环境规则补齐并验证。
- 主训练数据、调参验证数据与最终测试隔离，不按测试成绩筛任务或选 checkpoint。
- 多任务均值不能掩盖个别任务下降，必须保留逐任务、逐训练种子结果。

## 5. Reference Implementations：明确网址与角色

### A. 唯一主训练底座：ManiSkill 官方仓库

Repository: https://github.com/mani-skill/ManiSkill
Docs: https://maniskill.readthedocs.io/en/latest/
Demonstration setup: https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html

BC entry:
https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/bc

SAC entry:
https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/sac

RLPD baseline instructions:
https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/rlpd

ManiSkill-side TD-MPC2 baseline entry:
https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/tdmpc2

用途：环境、演示处理、统一评测、BC/SAC 的初始可运行实现。
官方 SAC README 标明其代码改编自 CleanRL，故优先用这份已适配环境的 SAC，不先另造一套 CleanRL–ManiSkill 适配层。
官方 BC 的视觉支持与相机兼容性仍需实测，不默认所有视觉配置开箱即用。

### B. SAC 对照阅读：CleanRL

Repository: https://github.com/vwxyzjn/cleanrl
SAC source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
Docs: https://docs.cleanrl.dev/rl-algorithms/sac/

用途：核对标准随机 SAC、目标网络、熵项、动作缩放与回放处理。
默认是只读代码参考，不再并行维护第二套主 SAC 训练循环。

### C. 世界模型与规划来源：TD-MPC2

Repository: https://github.com/nicklashansen/tdmpc2
World model: https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py
Agent / planning: https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py
Paper: https://arxiv.org/abs/2310.16828

用途：阅读动作条件潜在动力学、奖励预测、多步训练、候选展开和终端价值拼接。
只有主项目 BC、SAC 与 BC→SAC 数据流经测试后，才引入相关模块。
复用局部机制而非整个 agent；不能因参考 TD-MPC2 就把其策略更新冒称为标准 SAC。
原仓库各任务域和依赖不保证直接兼容 ManiSkill3；环境接口以主项目为准，必要时核对 ManiSkill 内的移植入口。

### D. 示范辅助世界模型的相关工作/独立对照：DEMO³

Repository: https://github.com/adrialopezescoriza/demo3
Project: https://adrialopezescoriza.github.io/demo3/

用途：研究演示辅助训练、模型预热、奖励学习及相关失败机制；资源允许时作独立外部对照。
不是主工程底座，不作为 U0 实现的强制前置关卡，不直接替换 SAC。
该仓库含多阶段奖励学习与修改任务；不得把其 dense/semi-sparse 配置混入本项目 sparse 主实验。
复现它时，应区分任务、奖励、数据和监督权限；无法对齐就报告不可直接比较。
论文链接必须另行核对题名、作者和版本，不能仅凭 README 一个链接就确认论文身份。

### E. 稀疏奖励/离线在线混合的外部基线：RLPD

Original implementation: https://github.com/ikostrikov/rlpd
ManiSkill instructions: https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/rlpd
Implementation referenced by those instructions: https://github.com/StoneT2000/rfcl
Paper: https://arxiv.org/abs/2302.02948

用途：专家/既有数据与在线回放混合的强基线。
ManiSkill 当前说明指向 JAX 实现，运行时放在单独环境和入口，与主 PyTorch 环境隔离；通过数据协议和结果文件对接。
不得把“自己实现的 SAC＋50% 演示”直接命名为完整 RLPD 复现。

### F. 双策略竞争的相关工作：IBRL

Repository: https://github.com/hengyuan-hu/ibrl
Paper: https://arxiv.org/abs/2311.02198

用途：冻结 IL、候选竞争和引导探索的对照阅读及新颖性核验。
不能假定该代码的环境和策略更新可直接替代本项目；论文、具体算法与依赖都要核对。

### G. 不作为 ManiSkill3 主依赖：ManiSkill-Learn

Repository: https://github.com/haosulab/ManiSkill-Learn

其 README 明确这是 ManiSkill1 的学习框架。只可作为历史参考；不按先前清单将其安装进 ManiSkill3 主训练环境。

## 6. Integration Policy：防止混装，而不妨碍研究

1. 允许下载或 clone 多个仓库作只读阅读。禁止的是同时把多套依赖、训练循环、回放和更新规则混装进主系统，不是禁止阅读。
2. 唯一主程序是本项目的 PyTorch＋ManiSkill3 训练入口；首先依照 ManiSkill 官方 BC/SAC 建立它。
3. 上游参考 checkout 固定 commit 并保持不变；本地补丁另存，保留来源和许可。
4. 每次只接入一个主要模块：接口/单元测试 → 短实验 → 记录 → 再接下一模块。
5. 世界模型先旁路训练与诊断，再启用候选评分控制；诊断不应拖到全部多任务训练结束。
6. 外部基线独立环境、独立进程运行；不得为装一个旧基线随意降级主环境。
7. 核实真名、仓库所有者、项目页/论文作者关联和具体文件。URL 无法访问应报告，不默默换同名 fork。
8. 所有 `main/master` URL 仅是定位入口；首次取代码记录具体 commit，实验期间不自动追踪最新分支。
9. 维护 `UPSTREAMS.md`：repository、commit、paper、license、文件来源、原样复现/适配/新实现、用途及实际修改。
10. 条款不是正确性的证明；每次改动仍要用 diff、依赖清单和测试确认没有越界。

## 7. 必修正确性与待验证研究改进的界限

必须在 U0 中处理：完整 IL 冻结、独立 SAC 初始化、兼容观测/潜在状态、随机采样、动作单位、终止/截断、回放 next-observation、奖励标签、模型/价值预热和评分目标定义。

评分中的 soft Q、任务回报价值和实际规划器回报必须区分；不得把它们当同一量比较。任何目标对齐设计都要写出数学定义并记录。

奖励/动力学仅见专家数据，并不保证能外推 SAC 探索动作；不能称预热后“评分已可靠”。

候选整段价值高不等于首动作交给学生续接后的价值高。局部诊断必须固定续接协议。

校准、集成、悲观评分、持续探索片段、规划蒸馏等是待验证局部改进，不预先全部堆入 U0。

## 8. 自主研发和研究纪律

无需 ChatGPT 每一步审阅。Codex 可连续实施、实验、修 bug、查论文源码、提出局部假设并比较 U0/U1。

必须保留全部原始实验、失败尝试和已知限制。先定义目标指标、对照及预算，再运行比较。

一次主要机制变化；既有方法的局部适配也不自动构成新颖性。
不同上下文或不同模型审阅都不是科学独立性保证；证据可复算、对照公平和外部查新更重要。

没有可重复瓶颈时可以报告暂无证据，不得为完成“创新任务”制造失败。

## 9. 预算、结束与上传

先测吞吐，在本地记录有限的实验计划。没有用户额外预算时，首次自主批次以 24 GPU 小时、最多两个局部改进假设为默认上限；这是可调整工程预算，不是收敛保证。

达到预算、权限/额度阻塞或需改变核心时，保存完整进度并报告。不得把未完成的多任务评测写成已经完成。

本地小实验可以连续进行，无需逐项人工批准；完成一个完整批次后，统一向目标仓库的新工作分支上传代码、配置、可复算指标、日志和报告。不强推、不自动合并 main。

排除凭据、原始大数据集、回放和缓存。大 checkpoint 保存可靠位置、哈希和加载命令；只记录不可访问的本地路径不等于已经交付权重。

本文件的核心和资源边界不得由 Codex 为了取得更好结果自行改写。

## 2026-10-17 兼容性修订：用户明确授权 ACT→SAC 研究

本段依据用户于2026-10-17新增任务书及后续明确指令追加，原2026-10-16内容和Git历史完整保留。原始文件SHA256为 `b953dc82b8435bf1ada0174e5e049b56d4cf23ed346ca4651f82a3f01ef8dd90`。修订理由：旧MLP-BC闭环弱，需要实验验证更强ACT动作块教师能否有效初始化标准随机SAC，而非预设成功。

允许在独立ACT研究分支使用同一锁定ManiSkill的官方state ACT作为新IL教师；旧BC及U0/U1继续作为原样对照。按 `ACT_SAC_PLAN.md` 依次完成教师闭环、仅专家训练状态的行为蒸馏、真正推理网络参数复制、无规划在线SAC对照，再接原WM评分。在线状态扩展蒸馏及技能保持机制需独立有界对照，不混入首组。

本次结构升级由用户明确授权；保持原始冻结IL→随机SAC→双候选→WM短奖励/纯任务价值→真实探索→最终无IL主干。资源仍共享原24 GPU作业小时、单张3090、同一时间一个GPU作业、1000万可训练参数与20GiB目标；不取消正在执行训练，不自动新增24小时。后续局部假设仍逐项登记并受原有限改进规则约束。新比较冻结前保护最终测试种子，阶段失败如实保留。研究里程碑后台上传独立分支并创建PR，不自动合并main。

## 2026-10-19 新批算力授权（原批次边界不变）
用户明确回复“批准下一批 11,500 秒”。新批仅固定50%专家回放的初始化因果对照R0/R1/R3及短检/评测；原24小时实际85238.750349秒关闭保存，其余额不并入本批。独立cap11500秒，单3090串行，无第三机制、无WM/预热调参、无定时任务。授权及精确命令见configs/act_initialization_budget.json。
