# ACT 变更记录

## 2026-10-17：用户明确授权 ACT 升级

完整读取新任务书、原核心文档与最新实际结果；远端新增任务书已核对，本地旧训练比快照更新。创建独立分支，保存未提交工作区与既有检查点记录。不改旧BC/U0/U1历史报告与算法。追加章程兼容性说明，预算不重置。

添加ManiSkill ACT原始vendor来源、state teacher、真实episode chunk数据接口、masked L1/KL、先验clip首动作、按环境清空时间集成及测试。首次CPU33测试通过，增加final decoder梯度检查后34测试通过，真实数据CPU10更新完成；后续记录003测试与002数据检查。

将教师顺序限定为旧训练完成后GPU smoke、K8/16/30有限对照；第一组蒸馏按用户最新指令仅expert train states，在线状态扩展单列。新比较冻结前增加最终测试gate，原模型/评测条件未改变，不启动或抢占GPU。上传遵循用户后台偏好，在完整迁移里程碑形成后统一提交新分支与PR。

同日补齐原始推理延迟/阶段汇总、真实数据接口断言及展开评测协议；最新004完整CPU测试35项、003真实数据CPU入口均成功。教师队列改为检查这两份最新结果。Policy Decorator源码锁定并记录许可缺口与教师依赖。登记本聊天有限24次小时续接，仍使用原预算及GPU锁。

GPU作业尚未启动时的预检修正：smoke batch从64改为与正式K8相同的256，且外推前断言模型和batch一致，避免用小batch低估正式训练成本。保持100次smoke更新和300秒上限。初始队列配置hash保留，修订hash见act-preflight-001。原样vendor有两处上游尾随空白，保留其blob一致性；本地新增源码diff检查通过。

## 2026-10-19

完成教师K8/16/30证据逐回合复算和K30固定选择，导出严格张量相等的便携权重。新增仅专家状态的MLP-SAC蒸馏入口及均值/方差契约测试，7项ACT相关测试通过。GPU smoke与完整D1按300+1200秒后台串行；用户要求每个小阶段后台上传已持久化。

D1真实闭环负结果后，追加36项CPU回归全通过。独立初态审计暴露F-ACT-EVAL-01，保留失败断言和002逐维诊断。最小修复每批显式reset seed，固定旧模型8条件重评，不训练、不从新成绩重选K。GitHub阶段1已完成并核对全部tree/blob；阶段2将携带负结果、修复和原始重评证据。

## 2026-10-19 最终评测入口同步显式reset（正确性修复）

审查发现独立主SAC和隔离RLPD评测入口未传matched_eval_resets，虽主helper已支持，最终多批次仍会回到旧自动reset语义。两个入口改为显式batch seed=base_seed+completed_episodes，并在result.protocol记录explicit_batch_seed；最终汇总拒绝缺失或错误的逐回合eval_seed。原88条件/权重选择/任务/预算不变，原历史验证不重写。9项CPU契约测试通过，包括不同策略随机数消耗、训练RNG恢复、错误reset seed拒绝；不声称这是新GPU评测，seed20000门控保持false。首个pytest启动因宿主ROS自动插件缺少lark失败，采用既有禁用插件自动加载和项目PYTHONPATH隔离后成功，未安装或修改训练依赖。

## 2026-10-19：独立 ACT 评测门控与在线权重导出

新增统一严格推理加载和完成作业的轻量actor导出；学生加载不构造教师，保留已学习方差。T2便携权重584114字节，全部actor张量与原final相等，CPU序列化动作逐位相等，见releases/act-t2-online-final-001与artifacts/act-t2-export-001。新增ACT独立评测入口，扩展既有闭环helper至10环境/100回合最终协议，逐批显式reset，最终gate在模型/CUDA/环境启动前检查；原默认8环境验证不变。16项CPU推理/门控/参数复制/reset/最终汇总测试通过（2.05秒），见artifacts/act-inference-cpu-tests-001；新10环境入口尚未真实GPU验证，未访问最终测试、未登记额外GPU预算。当前T3训练继续，未改在线算法。所有程序和上传按用户要求后台执行。

## 2026-10-19 固定训练状态的在线漂移诊断

新增CPU诊断入口scripts/diagnose_act_online_drift.py，只读取已成功完成的在线作业和哈希匹配检查点，不训练/采样/访问最终测试。固定前1024个expert-train转移，逐状态保存episode ID与时间索引，排除terminal obs和validation episode；报告与冻结K30首动作教师及初始学生的逐动作MSE、log_std、边界比例、逐参数L2和初始→当前pre-tanh Gaussian KL，关联既有40回合验证记录。它只检验固定专家分布上的变化，不能证明在线状态覆盖、Q校准或遗忘原因。

两项数据隔离/KL数学测试exit0；在已完成T5 smoke上核验，obs/初始与末次动作/log_std全部逐位复现原诊断（artifacts/act-drift-parity-001）。新增结果：gripper动作漂移MSE .942503、对教师MSE .942505、pre-tanh KL均值38.3999；初始8/40→末次0/40为原真实短检结果，未新增闭环仿真。完整T4/T5/T6均预先固定诊断step16384/65536/131072/262144，CPU后台队列2829356等待各作业成功后依次汇总、导出和诊断，失败停止、不重启GPU。启动回执写入时缺hashlib导入报错发生在Popen之后，已通过实际存活PID和status恢复回执，没有重复启动；原错误记录保留。

## 2026-10-19 T4 完整证据增量

提交T4完整原始配置、逐回合记录、资源、代码快照、便携actor及预登记固定输入漂移；汇总器校验17×40回合和同Critic协议。CPU后处理成功后等待T5/T6，不新增GPU任务。追加forgetting mitigation论文/作者NetHack源码身份与适用边界。训练算法未变，最终测试未开放。
