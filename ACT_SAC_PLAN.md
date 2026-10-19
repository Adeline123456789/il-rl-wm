# ACT → SAC 迁移研究计划

2026-10-17。依据用户新任务建立 `research/act-sac-transfer`；原快照分支最新远端为 `bed60b00eeaac1be3dbbdade02844ce6a5f2bbc5`，只比已交付 `4e014a5` 新增任务书。任务书原始 blob `eb04c168de2f6436746636c46201ff0edb774e38` 已逐字节保存。后续用户消息优先：第一组蒸馏只用演示训练状态，真实在线状态扩展单列实验。

## 当前边界与保存

本地旧 BC/A/B/C、U0/U1、数据修复、所有失败和检查点保持原样。启动保存 271 个工作区变动文件与 155 条既有检查点哈希记录，见 `artifacts/act-startup-001`；运行中日志按文件捕获，不能当原子仿真快照。旧 GPU 作业及训练队列不抢占、不重新启动。

24 GPU 作业小时是共享总上限，ACT 不重置额度。启动核算：已结束 25225.872 秒，RLPD 当时已运行约 3140 秒；余量约 58034 秒还要覆盖旧注册训练，并非 ACT 可独占额度。保留原 7200 秒最终评测额度。旧最终测试 52 模型/88 条件注册内容不改，但新比较尚未冻结，增加显式测试放行文件 `configs/act_final_test_gate.json`，当前不允许访问 seed20000。门控等待不占 GPU；不能把等待计成实验完成。

## ACT-T0：先验证教师

唯一主来源为锁定 ManiSkill ACT state 实现，原文件、许可证和哈希保存在 `vendor/maniskill/act`。不用 LeRobot/ALOHA 训练框架，不升级主依赖。PickCube full-003 相同 717/180 整回合拆分、55092/13897 转移、state、pd_ee_delta_pos、sparse、200 步、physx_cuda、continuing success、reconfiguration_freq=1。

预登记有限 K 对照为 8、16、30；各 hidden128、前馈512、4 heads、2 encoder/2 decoder、latent32。K8 实测总训练参数 1348548；推理与后验专用参数在运行结果中分列。不是官方已验证的最佳小模型。均用 AdamW 1e-4、weight decay1e-4、有效动作 L1+10 KL、2/3 更新处学习率降十倍、固定末次模型。首次候选训练长度10000、batch256；先100更新GPU smoke实测，若1.5倍外推训练时间加300秒评测余量超过1800秒，停止并记录资源问题，不暗中缩短训练后声称充分训练。

与上游的明确适配：只读有效split、尾部action/posterior掩码、最后decoder层、每步重新查询执行首动作、固定验证种子、无EMA（不增加diffusers依赖）、均匀有放回转移采样替代epoch sampler、原始state/action不归一化。ACT线性action head输出在执行前clip到[-1,1]，不伪称tanh Gaussian。推理不传真实未来动作，latent=0，无后验泄漏。

每2500更新验证40完整回合，最终同时报告：A每步重查首动作；B系数.01的时间集成，按每个环境维护有效缓存并在done/reset清空。两者不混作一个教师。最终两模式的全部回合视频、状态/动作/奖励轨迹、首次抓取步、接近/抓取/抬升/放置事件、逐回合成功、延迟及资源保留。新仿真计入validation账，不进入蒸馏数据。

执行顺序为旧注册训练链完成 → ACT smoke≤300秒 → K8/K16/K30各≤1800秒，合计最多5700秒；每次重新检查共享余量且保留7200秒测试预算。不足时保留未运行条件，不能为了ACT取消旧种子。首次K8至少需要2100秒余额；后续配置不是无限搜索。失败不自动重试。

进入迁移的开发门槛：某个固定final无集成教师至少8/40成功且工程审计通过；这是预登记可用性门槛，不是显著性保证。有限三K比较按无集成success_once优先、success_at_end其次、相同则较小K选定开发教师；明确它仅是该网格的验证选择，不能称最优K。仅集成有效时先诊断历史依赖，不能把它的动作标签直接蒸馏到无历史学生。三个独立训练seed与Stack/Peg后续确认按真实余量登记，当前不宣称已完成。

## ACT-D1：优先行为蒸馏；之后再参数迁移

教师门槛未过前不启动迁移。A组只用expert train states，目标为 `MSE(tanh(mu_SAC(s)), clip(ACT(s)[0],-1,1))`。独立冻结教师；只训练学生backbone/mean，log_std显式初始化并单独核对真实采样/log_prob/熵更新。Q不从ACT复制。初始均值闭环和随机闭环分别评测，同validation初态记录teacher–student gap、关键阶段、动作误差和状态漂移。在线状态标签扩展是独立的后续假设，必须真实训练池采样、另记交互，不使用验证/测试状态。

随后B组复制ACT推理骨干到独立Transformer SAC Actor，保留全部query上下文以免改变首query自注意力；使用首query特征接Gaussian两头。参数名/shape/dtype/hash/存储独立性逐项核对，未来动作后验encoder不参与在线推理。ACT线性动作头与SAC pre-tanh头不等价，均值需要监督对齐；不得strict=False盲装后称迁移成功。单独报告模型/延迟/显存和真实SAC梯度，不默认复杂网络更优。

在线比较 T0旧BC、T1 ACT、T2随机SAC、T3旧BC-init、T4蒸馏SAC、T5参数迁移SAC、T6蒸馏+50%专家。先对齐critic结构、稀疏协议与在线预算；旧无LN结果是历史对照，不能与新LN方法混作初始化单因素。需要新增同LN对照时明确计费。离线/迁移/在线/验证成本分账。若迁移或在线技能保持失败，建Failure ID后只改一项，不借WM掩盖。

## ACT-W1：迁移成立后才接规划

目前不改 `planning.py` 或 `u0_modules.py`。未来ACT候选明确定义为当前chunk前H2；SAC八条候选各自经预测状态递推。共同短奖励+独立纯任务终端价值，20%真实随机SAC接管，最终无ACT九条SAC候选。保留无规划/直接Q/H2-WM以及教师来源、初始化两个因素的公平对照。新WM按expert/ACT/online及失败恢复、成功前分层验证；F1先做有界真实候选可辨识性筛查，不再盲花数十万分支步。

当前只登记教师阶段的GPU任务；D1、参数迁移、在线RL和W1预算待教师实测后在原剩余额度内具体化。阶段报告诚实列出未完成项；完整ACT迁移里程碑后后台上传新分支并创建PR，不自动merge main。

## 2026-10-19 ACT-T0 完成与 ACT-D1 有界登记

三K固定末次无集成成功为1/40、4/40、8/40，按原规则冻结K30，独立便携权重见releases/act-teacher-k30-001。K30集成30/40不能作为D1的教师目标或比较分母；D1实际教师是无集成8/40，末步4/40。仅通过开发门槛，不能称稳定高质量教师。

D1沿用3×256官方MLP Actor，原始state、单位动作范围、首动作MSE；只训练backbone/mean，Adam 3e-4、batch256、固定10000更新。log_std初始化为常数-2并冻结（约0.135的pre-tanh标准差），在线SAC时需显式重新启用更新。数据仅717演示训练episodes；180验证episodes只用于误差报告，不训练。GPU smoke100更新/每模式8回合≤300秒，通过后完整10000更新/每模式40回合≤1200秒，总新增硬上限1500秒；均保留7200秒最终测试额度，不依据验证成绩选中途checkpoint。保存逐回合阶段事件、教师真实轨迹同状态动作误差、学生状态分布摘要及完整最终trace，分别报告确定性与随机闭环。无RL/WM，teacher–student成功gap仅为开发描述。

按2026-10-19用户新要求，每个可验证小阶段完成后后台上传研究分支，代码/配置/原始证据一起提交；此规则取代上述等待整个迁移里程碑上传的频率。旧最终测试门控因24h等待超时退出，未访问测试；比较冻结后用新队列ID恢复，保留原失败记录。

## 2026-10-19 必要评测修复 F-ACT-EVAL-01

CPU审计发现D1与T0虽首次reset同seed，后续批次初态不配对。锁定ManiSkill `sapien_env.py:950` 无seed且非enhanced_determinism时直接初始化；PickCube使用全局torch.rand，向量wrapper自动reset不传seed，因而策略采样/训练消耗可改变后续初态。动作重算误差仅3.15e-6，不能把此问题归为动作执行错误。

新评测每8回合显式reset seed=10000+已完成回合数，保存实际seed并逐维核对40初态。固定既有K8/16/30教师的首动作/集成和D1均值/随机共8条件，各40回合；无重训、无checkpoint重选、不访问test，单次≤600秒，保持7200秒reserve。旧结果不改写，其8/40与1/40差值降为非配对描述。该项为正确性修复，不算新研究机制。旧BC/U0/U1的评测是否受同类问题影响另审查，不能把新旧协议混作严格配对。

## ACT-C1 参数移植登记（2026-10-19）

D1在修正配对下仍失败后，按原顺序独立研究参数迁移，不叠加在线state标签或新正则。复制冻结K30的state projection、Transformer encoder/decoder、全部30个query、zero-latent projection、位置embedding和action head；不复制CVAE训练后验encoder。教师与学生无共享storage，逐参数name/shape/dtype/hash核对。

均值明确取atanh(clip(ACT copied linear head(h0), -1+1e-6, 1-1e-6))，再用标准SAC tanh-Gaussian采样/修正log_prob；这是一种已授权的数值动作映射，非把ACT输出当pre-tanh或CVAE posterior当log_prob。新的log_std head初始化-2；后续SAC时可训练。为保持条件分布定义，推理骨干dropout在训练和评测均关闭，但参数允许梯度；所有query仍参与自注意力。饱和处clamp梯度为0是明确限制，统计比例，不预先宣称在线优化有效。

先CPU独立存储、全部decoder梯度、极值log_prob测试；真实GPU接口+闭环单作业≤300秒，先2次接口检查，再40回合均值与随机验证。用刚冻结的matched-reset种子和40初态核验，不重训教师、不用test。检查实际演示state上复制动作误差、饱和比例，隔离副本做一次SAC形式Actor/温度梯度探针（随机critic，仅工程检查），闭环只评估未更新的参数复制模型。无在线训练或WM，不能将初始动作对齐称为SAC学得更好。保留7200秒reserve。
