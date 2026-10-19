# 终态验收结论（2026-10-19）

整体科研目标未完成；本批已停止计算，发布收尾证据。正式104/106完成，U2均值time_budget失败、随机未开始；ACT-WM/ACT多seed/多任务仍缺。原始主干、历史88条件、16ACT条件、U2开发负结果、论文/源码角色与数据隔离证据保存。FINAL_BATCH_REPORT.md逐项回答原10问、ACT12问；ACT_TRANSFER_CAUSAL_GAPS保留新10问并追加正式结果。

终态审计核对214份作业manifest总账85131.546390秒、无区间重叠、46个真实执行commit在离线历史中存在，最终source tar/patch和权重检查完成。final墙钟超出cap的0.778426秒收尾开销明确披露；三次早期环境步计数不完整、旧plot字节缺失仍是证据限制。CPU复算16ACT条件100初态逐维一致。以上不证明未做的科学对照。

阶段22已后台交付104完整条件、U2失败原始数据及本终态报告，a00ff2c的3398远端文件逐一核验，回执github-act-stage-upload-022。后续元数据提交保存该回执和终态源码覆盖索引。main保持不变，PR2保持Draft；每小时自动化已删除。剩余GPU工作需新授权，不用机械审计通过代替研究完成。

---

以下为运行期验收快照，当前状态以上述终态结论和实际manifest为准。

# 首批交付验收矩阵（2026-10-19，执行中）

当前不能宣布整体目标完成。原106条件正式评测仍运行；终态后才以原始证据和实际预算决定本批交付状态。下表把已实现、科学证据不足、仍运行和待授权区分。文档存在不等于实验已完成。

## 原始要求及权威证据

|要求|权威证据入口|当前判定|剩余动作/限制|
|---|---|---|---|
|B：保存用户文件/原历史/适用入口|AGENTS、PROMPT、RESEARCH_CHARTER；act-startup-001；Git发布回执|原始历史保留；本地HEAD固定9f1536，临时index发布|禁止重置工作区、强推或覆盖main|
|B：持续状态和失败恢复|research_state、追加RESEARCH_LOG、各manifest/launch/status|持续维护；旧派生状态保存在act-next-stage-intake-001|最终完成后刷新终态，运行快照不能当完成|
|C：唯一主框架与隔离基线|UPSTREAMS；vendor/maniskill/act/SOURCE_MANIFEST；external/rlpd/source_manifest|ManiSkill主训练、JAX RLPD隔离；各自来源/许可已记录|不能将本地专家SAC称完整RLPD或复制参考框架|
|D/E0：真实环境与演示|各e0实验；full-003/dataset-audit；F-E0-005|三任务真实reset/step和回放已运行；分组隔离、哈希有记录|数据重建来源与巨大原始文件不混入Git|
|D2：稀疏奖励与监督公平|config、dataset audits；WM分组诊断；HISTORICAL_VALIDATION|主训练sparse，专家/在线额外监督分列|零奖励低loss不是Reward Head有效证据|
|E1：冻结IL与独立初始化|BC/ACT checkpoint、冻结/复制审计、C1和D1原记录|旧BC及新ACT冻结/迁移真实运行|教师时间集成与首动作目标不可混同|
|E2/E3：SAC A/B/C与强基线|旧注册52项；T2–T6 summary006；RLPD原manifest|全部已登记开发训练结束；T4/T6差异复算通过|同协议随机+专家R0、BC+专家R1仍缺；不能归因ACT独立贡献|
|E4：WM旁路与多步误差|旧E4/E6、F-U0-001及U1_seed0诊断|旧主干已实现运行，有专家/在线分组证据|新ACT行为分布下WM未验证|
|E5/E6：双候选/H2/纯任务终端值/真实探索|旧U0/U1执行配置、候选/价值测试、实际source统计|历史完整主干保留并运行，必要正确性补全已记录|ACT双候选尚未接入；不删目标、不用裸SAC冒充|
|E7：无IL部署及完整评测|历史deployment包/action-parity；旧四方法验证|已有独立无IL包及真实闭环开发评测|106条件正式评测待终态；ACT+WM无教师部署未完成|
|F1：真实反事实、目标分开、诊断计费|F1_batch_diagnosis；f1-all-probes-summary-001|4次真实诊断、496320交互、首动作/整H2目标分开|无可分对，排序质量不可识别，不能报100%|
|F2：失败链与竞争解释|F-U0-001、U1_seed0、ACT_FAILURES、CAUSAL_GAPS|失稳/状态支持/方差/漂移证据保存|不能把相关性、soft-Q大小或参数化限制当唯一根因|
|G：论文和作者源码|literature清单、UPSTREAMS、ACT_RELATED_WORK|来源版本/实际读取/许可限制分列；新增WSRL/DAWN已核验|PAC-ACT作者源码未核实处仍明确；无原创性证明|
|H：最多两项局部机制|U1_CHANGE、U2_CHANGE及预算状态|U1 LN和U2有限均值保持两项已用完|不追加第三机制；R系列/预热需新GPU授权|
|I：训练seed与评测episode分开|历史独立seed1–4原记录；ACT开发seed0；summary字段|旧5训练seed与ACT单开发seed分列|ACT多seed/多任务未完成，不能将40/100episode视作训练重复|
|I：验证/测试隔离及fixed-final|act-comparison-freeze、gate、最终注册表SHA|冻结早于正式访问，预审验证已完成条目|禁止根据final结果改教师/超参数/训练长度|
|I：单3090/24小时/7200秒final|全部GPU manifest、resources、同一runner锁|预审总账逐项相等，已结束作业无时间重叠|终态必须计入尾部/失败/关闭耗时；不足时未跑null|
|J：实际执行源码可恢复|基础+增量source-history；逐作业tar/patch/source_hashes|45历史commit离线恢复；新正式作业继承9f1536并核对完整tar|旧未提交plot字节缺项明确保留；不冒称所有历史文件完备|
|J：轻量权重可用/大数据边界|releases manifests、逐张量导出与便携推理检查|已发布ACT与旧关键推理权重，原完整大checkpoint留本地|最终新增包与原始结果需终态统一后台上传|
|J：远端新分支/PR/不合并|github-act-stage-upload-018；Draft PR2|阶段18已核验2491文件，main不变|本轮新增图/审计代码及最终数据尚需新回执|
|K：原10问和ACT12问/新10问|ACT_MILESTONE_REPORT；ACT_TRANSFER_CAUSAL_GAPS|开发证据问题已回答，限制明确|正式表、实际末次总账、最终远端回执未补齐|

## 新ACT验收

|要求|证据|判定|
|---|---|---|
|教师K8/16/30、块尾mask、episode隔离、真实闭环|ACT_TRANSFER_REPORT、matched-eval、数据audit|已执行；K选择固定，不用修复后分数重选|
|冻结首动作ACT→MLP-SAC|D1、F-ACT-01、matched-eval|完整负结果：均值1/40，不能认为继承成功|
|独立参数迁移和随机概率|C1、copy audit、gradients CPU002|均值8/40、随机1/40；真实clip梯度限制已测，未改结构|
|统一Critic/交互/更新的T2–T6|summary006、配置/初始化哈希、真实replay|开发完成；T6额外专家数据贡献与初始化分开|
|U2五阶段/A–H指标与分类|ACT_U2_FINAL_REVIEW、review CPU001|B类有限负结果；在线lift/首次抓取缺失明确null|
|下一批R0/R1/R3和预热|NEXT_RESEARCH_BATCH、JSON提案|配置/预算已准备；未授权、未执行、无虚构成绩|
|ACT-WM及最终无ACT净收益|原ACT计划、NEXT_RESEARCH_BATCH A–F|未完成，前置因果/稳定技能证据不足且无额外GPU授权|

## 机械审计的范围

`scripts/audit_act_final_delivery.py`从原始episode重新计算正式汇总、检查plan冻结哈希、gate时序、权重hash、逐文件源码tar/hash、离线Git commit、全部GPU manifest总账、资源和时间重叠。预审使用allow-running-snapshot标记，不能作终态完成证明。

CPU终态等待器`artifacts/act-final-delivery-audit-001/status.json`只在原003 launcher终态后运行，不创建GPU作业。它可能报告失败/未执行条件或预算收尾开销；机械审计通过也不能证明ACT初始化、WM排名或多任务泛化科学结论。最终交付要同时检查上述缺口，不用“全部测试绿”代替完整范围审查。
