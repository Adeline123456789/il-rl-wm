# 上游来源登记

## ACT 扩展来源（2026-10-17）

唯一新增训练底座仍是上述锁定 ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`。`vendor/maniskill/act/SOURCE_MANIFEST.json` 登记原样保存的 train.py、README、detr_vae.py、transformer.py、evaluate.py、make_env.py、utils.py 及许可证，blob逐一匹配；Apache-2.0/第三方声明随代码保留。ACT论文为 Zhao等，RSS2023，https://arxiv.org/abs/2304.13705 。

本地适配 `scripts/act_teacher.py` 复用 DETRVAE、Transformer、CVAE encoder，增加episode-tail有效mask并使用decoder最后一层；`train_act_teacher.py` 使用现有真实GPU回放、AdamW、L1+KL及StepLR，取消EMA、明确均匀有放回采样及首动作重查询；时间集成单列。Torch/numpy/h5py、已有IPython与ManiSkill环境沿用，不安装另一套学习框架。已完成CPU接口验证，尚未完成GPU教师闭环；必要偏差见ACT_FAILURES.md。

原始ACT https://github.com/tonyzhaozh/act ，commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`，MIT；LeRobot https://github.com/huggingface/lerobot ，commit `146d8474932c8b566717eb2738ba459337ab71c4`，Apache-2.0/第三方MIT。仅阅读推理latent、动作队列、temporal ensemble与normalization接口；六份文件及hash见literature/act_source_manifest.json，不安装其依赖。

Policy Decorator https://github.com/tongzhoumu/policy_decorator ，commit `92ba9ba442587ae5989c286355ace2200a0537fb`；论文 https://arxiv.org/abs/2412.13630 ，Yuan/Mu/Tao/Fang/Zhang/Su。只读README与online/pi_dec_bet_maniskill2.py；README声明MIT但该commit的LICENSE为空blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`，因此不复制其实现到发布代码。其ManiSkill2/BeT/残差SAC环境不接入。源码evaluate仍执行base+scaled residual，最终依赖教师；本项目只参考受控探索/遗忘诊断，不照搬残差主干。它将reward减1并采用不同bootstrap协议，也不直接与本项目数值等同。

PAC-ACT https://arxiv.org/html/2607.09590v1 为chunk-level PPO，当前未核实作者官方源码，故无代码复用。完整方法及实验正文已核对；其视觉骨干复制、Gaussian head、value critic和奖励级行为先验不是标准单步SAC结果。详见ACT_RELATED_WORK.md。

当前状态补充（2026-10-16）：以下各段保留接入时的历史状态。主ManiSkill BC、A/B/C、完整U0、C续训和两次F1诊断已经真实完成；RLPD隔离训练/评测smoke完成，完整运行排队；U1完整seed0规划段运行中。历史段落中的“尚未完成”不代表当前状态，逐实验结果以research_state.json及原始manifest为准。最新核查8份vendor文件SHA256不变，ManiSkill/rfcl缓存checkout仍为锁定commit且工作树干净。

## ManiSkill：唯一主环境与 BC/SAC 底座

- Repository: https://github.com/mani-skill/ManiSkill
- Locked commit: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`（2026-10-16 取回，禁止自动更新）。
- License: Apache-2.0，仓库另含 LICENSE-3RD-PARTY（PyTorch3D BSD）；机器人资产按各自目录许可。缓存 checkout 保持上游不变；后续复制代码时一并保存许可证。
- Paper identity: 已核验 [arXiv 2410.00425v2](https://arxiv.org/abs/2410.00425v2)，初版 2024-10-01、v2 2025-05-30；[RSS 2025 正式记录](https://roboticsproceedings.org/rss21/p021.html)，DOI `10.15607/RSS.2025.XXI.021`。两者题名不同：RSS 正式题名以 “Demonstrating” 开头，仓库 CITATION.cff 将预印本题名与 RSS 场合混合，项目 bibliography 采用正式记录并单列预印本。作者/来源见 literature/verified_sources.json；目前完成元数据核验，未宣称精读全文。CITATION 软件版本仍为 3.0.0，setup.py 为 3.0.1，实验以 commit 为准。
- `examples/baselines/bc/bc.py`、`behavior_cloning/{evaluate,make_env}.py`：官方 BC，已适配为 scripts/train_bc.py；其 README 声明源自 CORL，尚未引入该依赖。
- `examples/baselines/sac/sac.py`：官方 SAC，拟逐步适配；README 声明源自 CleanRL/ManiSkill Baselines，不另建第二个训练框架。
- `mani_skill/utils/download_demo.py`、`mani_skill/trajectory/replay_trajectory.py`、`scripts/data_generation/replay_for_il_baselines.sh`：下载器原样调用；回放在 scripts/replay_trajectory.py 保留官方循环并修复首帧记录/种子类型（patches/）；显式指定 sparse、record_rewards、state、控制模式；不使用逐步 use_env_states 作为真实转移。
- `mani_skill/vector/wrappers/gymnasium.py`、`mani_skill/envs/sapien_env.py` 和三个任务源文件：只读核查终止、reset、奖励、成功定义。
- 新实现：`scripts/run_experiment.py` 负责预算/证据；`scripts/smoke_env.py` 调用官方环境并验证转移接口。已完成 BC smoke，完整训练与 SAC 真实更新尚未完成。
- 官方 BC/SAC 及帮助模块已原样复制到 `vendor/maniskill/`；每个文件的 Git blob 与上游 commit 树一致，SHA256 及路径详见 `SOURCE_MANIFEST.json`；Apache 与第三方许可证已一并保留。

## 已核实的接口差异（不是研究改进）

1. SAC 默认 reward_mode 由环境选择，基类优先 normalized_dense；主协议必须显式 sparse。
2. SAC 默认三层 256 ReLU + 独立 mean/logstd + tanh/缩放；BC 默认两层 256 ReLU + 无 tanh 输出。初始化前需明确兼容架构或蒸馏，并验证均值动作一致。
3. SAC 默认 bootstrap_at_done=always；wrapper 默认忽略成功终止，直到时间限制 reset。必须把研究协议的 continuing/truncation 语义、原始任务 success 与 replay 的有效终止区分清楚。
4. 当前 replay CLI 使用 `--num-envs`；旧 BC examples.sh 仍写 `--num-procs`。使用锁定源码参数，不照搬过时脚本。Peg 的推荐控制为 pd_ee_delta_pose。

## 后续来源（尚未引入）

CleanRL 仅数学核对；TD-MPC2 仅在 BC→SAC 测试后局部参考；RLPD/rfcl 单独环境；DEMO³、IBRL 相关工作/独立对照。URL 与顺序以 RESEARCH_CHARTER.md 为准。各 commit、许可、论文核验将在实际首次使用时登记，未知项不伪填。

## 当前适配记录

- BC：3×256 ReLU+tanh 匹配 SAC 均值分支；保留 MSE/Adam/DataLoader，加入整回合拆分、固定末次检查点和原始指标。归一化关闭。scripts/policy_bridge.py 新实现独立冻结与均值复制。
- SAC：scripts/train_sac.py 从原始文件复制后适配 sparse、200 步 continuing-task 协议、原始终止标签、固定验证种子、完整 learner/replay/RNG 保存；actor/twin-Q/熵目标/Polyak 更新仍为官方实现。尚在工程验证。scripts/sac_evidence.py 是日志/快照辅助，不含另一套学习器。模拟器中途状态未保存，恢复必须标为新 segment。

## CleanRL：只读 SAC 数学核对

Repository https://github.com/vwxyzjn/cleanrl；commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`；MIT 与来源声明见缓存 LICENSE。只读取 cleanrl/sac_continuous_action.py、LICENSE 和 README，哈希记录 artifacts/bootstrap/cleanrl-source.json，没有安装或运行其框架。对比确认：tanh+尺度 Jacobian、min 双 target Q 减 alpha log π、actor 的 alpha log π−min Q、exp(log_alpha) 温度损失均与 ManiSkill 一致。差异：CleanRL 两层/ManiSkill 三层；延迟策略更新补偿方式不同（本项目 policy_frequency=1 避免该差异）；默认 gamma/UTD/环境封装不同。本项目固定 gamma=.99，保持官方 ManiSkill 的 batched collection/updates。仅以源码和官方文档核对公式，未宣称新增论文发现。

- 第三处 replay 必要补丁 F-E0-003：GPU 每批结束调用 flush_trajectory(save=False) 丢弃各环境已保存轨迹之后的 padding 残留，避免跨 batch 拼入下一 episode；真实两批及全量重放审计分别记录。scripts/audit_replay_alignment.py 新实现长度/动作/首状态来源检查，不能仅凭 HDF5 shape 自洽认定可训练。
- scripts/bc_evaluate.py 对官方 BC evaluate 仅增加显式 reset seed 与整向量回合数断言；避免调用方先 seeded reset 再被 helper 第二次 reset。原始 vendor helper 保留不变。

## TD-MPC2：世界模型阶段只读机制参考

Repository https://github.com/nicklashansen/tdmpc2；commit `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`；MIT (Nicklas Hansen, 2023)。源码哈希见 artifacts/bootstrap/tdmpc2-source.json。已读 world_model.py 的动作条件 next/reward 与 tdmpc2.py 的 _update、_estimate_value、_plan：从一个编码状态自由递推，逐步监督预测状态/奖励，短期折扣奖励接终端 Q；策略候选后续输入各自预测状态。其隐式潜空间、two-hot 价值、Q ensemble、MPPI 和策略更新不直接移入本项目。主项目将使用兼容 BC/SAC 的原 state 接口和独立 pure-task 价值定义；不把其 Q 与 SAC soft Q 混用。此时仅阅读来源，尚未引入 WM 模块或安装依赖。论文元数据已核验，未宣称精读全文。

## 后续独立对照：已锁定，只读准备

- rfcl `7bd660f05dbda1ec8881bcfb250a8d1881532dbb`，来源 https://github.com/StoneT2000/rfcl 。pyproject.toml 声明 MIT，但当前树没有 LICENSE 文件。当前main的训练入口是含反向/前向课程的 train.py；ManiSkill README 指定的 train_ms3.py 和 base_rlpd_ms3.yml 在该树中不存在。不能把默认 RFCL 当成 RLPD。正在查指定路径的历史，尚未安装或运行隔离基线。
- IBRL `3c457da3689cc90f881bcdf42a0a578f4a1f60aa`，来源 https://github.com/hengyuan-hu/ibrl 。已下载 rl/q_agent.py 只读参考；完整树无独立许可证文件，许可尚未核实，不复制源文件到主项目。论文身份为 arXiv2311.02198v6；尚未完成方法全文核对。
- DEMO³ `5a5424821498da07b6dcf6ee257e4171c35b4b83`，来源 https://github.com/adrialopezescoriza/demo3 。README 声明 MIT 并指向 LICENSE，但完整树没有该文件；尚不复制代码。README 的 Paper 链误指 TD-MPC2，独立查得正确论文为 arXiv2503.01837v2，题名见 literature/verified_sources.json。项目页明确使用多阶段奖励，不能与本项目无额外阶段监督的 sparse 协议直接比较。仅下载 world_model、demo3_trainer 和 maniskill_stages 用于核对。

上述初次只读文件树/逐文件哈希位于 artifacts/bootstrap/reference-trees 与 reference-sources。IBRL/DEMO³仍未安装、未运行；rfcl 后续隔离进展如下，不属于主训练框架。

## 当前最小U0适配（已通过真实短实验）

SAC核心actor/twin-Q/熵/Polyak公式仍来自ManiSkill官方。新增原state残差WM、独立pure-task TD评价头与批量双策略候选为本项目实现，局部参考TD-MPC2多步递推/短期奖励接终端价值思想；没有复制其整体agent或策略更新。E4旁路10000更新、E5真实候选检查、E6价值预热和U0真实4096交互均完成。新仿真段恢复明确保存/加载learner、replay、温度、模型、目标及优化器；采用新reset/seed而非声称精确恢复中途模拟器。完整训练与F1诊断仍未完成。

## RLPD 隔离适配：2026-10-16

原作者仓库 https://github.com/ikostrikov/rlpd 锁定 `c90fd4baf28c9c9ef40a81460a2e395092844f88`，MIT 许可文件名为 `LICENCE`。已读 configs/rlpd_config.py、td_config.py、sac_config.py、SACLearner、MLP、TanhNormal 与 train_finetuning.py，并核对论文§4.1–4.3及附录B超参数表；没有宣称精读全部实验。固定源码哈希见 external/rlpd/source_manifest.json。

ManiSkill 指向的 rfcl 已安装到独立 `.venv-rlpd`（JAX0.4.28），主 `.venv` 不改。external/rlpd/train.py 直接调用锁定 rfcl 的更新/网络/回放模块，不复制或修改上游源码。新增的收集/证据入口显式处理自动reset前的final_observation，因为当前 rfcl GPU GymLoop 对该分支仍为 pass；不调用其课程训练入口。原始RLPD的十critic、LayerNorm、50%混合、UTD20，以及稀疏Adroit起点的三层/无entropy backup均预先写入 protocol.json；与rfcl的初始化、方差参数化、向量收集和本项目任务差异明确列出。属于独立环境适配，不能叫原论文逐项复现，也不能把它的额外计算成本隐藏在等预算对比中。

隔离CPU实测通过：20critic/1actor/1温度更新、10critic输出、三层LayerNorm、训练ID/原始奖励一致、50%交替、完整优化器序列化，以及截断末观测与reset观测区分。实际GPU smoke排队，尚无基线分数。

## F-U0-001 的只读价值稳定性来源

完整U0已运行并观察末期失稳，因此增加DroQ/REDQ原论文和作者源码核对。DroQ来源https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning ，commit `4c7adddbaf7573ff9256b16091a3bbdbd536aad4`；REDQ来源https://github.com/watchernyu/REDQ ，commit `7b5d1bff39291a57325a2836bd397a55728960bb`。两者LICENSE均MIT。具体阅读文件及SHA256见literature/value_stability_source_manifest.json；方法/成本/边界见literature/F-U0-001_value_stability_sources.md。未安装、未运行、未复制到主训练器，尚未登记局部算法修改。

F-E0-005本地replay补丁在CPU/GPU保存时附加source/original episode ID，不改变环境步或动作；审计按显式ID核对，兼容既有唯一seed数据。Peg重复原始演示按seed组划分，证据与边界见diagnostics/F-E0-005.md。U1仅在本地主SAC critic隐藏层增加可选LayerNorm，默认U0输出逐位回归通过；来源与单一机制预登记见U1_CHANGE.md，无外部训练框架混入。
