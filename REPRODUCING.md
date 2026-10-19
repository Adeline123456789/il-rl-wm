# 复算与重跑

当前仍在首批实验中；`research_state.json`与各实验`manifest.json`记录实际完成状态。以下重跑命令使用新目录，不能在现有活跃实验目录重复执行。所有耗时程序通过后台启动器或独立后台进程运行，保留日志和退出码。

## 2026-10-19 当前状态与只读复算

以下历史重跑章节不表示应在本机重复启动。当前有效最终注册表是`configs/act_final_evaluation_plan.json`（70项/106条件），gate已在开发冻结后释放；003队列接续002完成的A/B，原整批7200秒上限扣除失败前已耗时间。`artifacts/act-final-evaluation-batch-003/status.json`和各manifest为权威状态。后文“seed20000未访问”是旧快照表述，现已访问，不能用于后续方法选择。

CPU新诊断：`scripts/review_act_u2_evidence.py --output <new-directory>`复算现有逐回合、指标、便携回放派生数组；`scripts/diagnose_act_mapping_gradients.py --output <new-directory>`需要本地原始固定快照和原审计数据，在CPU检查真实状态梯度。已执行结果及命令/退出码保存在`artifacts/act-u2-final-review-cpu-001/`、`artifacts/act-mapping-gradient-cpu-002/`；不要重复向已存在目录运行。CPU耗时程序仍应后台执行，CUDA_VISIBLE_DEVICES置空。

源代码恢复基包与至9f1536的增量包见`releases/source-history-act-20261019-001/README.md`，离线恢复45个实际实验commit已核验。新诊断代码和源码哈希随当前发布树交付；大型原始checkpoint/replay留本机，不混入普通Git。

## 环境

本批实测Linux、Python3.10.20、RTX3090 24GB、驱动575.51.03、Torch2.3.0+cu121、SAPIEN3.0.3。主环境与RLPD分别安装，不能合并锁文件。下面从公共包重新安装的整条路径尚未在第二台空机器验证；本机CUDA运行库最初由已安装同版本包按RECORD逐文件校验后复制，详情见bootstrap清单。锁文件保留全部版本，只有editable checkout位置改为相对仓库根目录。

在新clone的仓库根目录准备锁定上游，然后安装主环境：

```bash
git clone --filter=blob:none --no-checkout https://github.com/mani-skill/ManiSkill .cache/upstreams/ManiSkill
git -C .cache/upstreams/ManiSkill checkout --detach 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
uv venv --python 3.10.20 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip check --python .venv/bin/python
```

RLPD使用ManiSkill指定的rfcl独立环境及其原始更新，不使用课程入口：

```bash
git clone --filter=blob:none --no-checkout https://github.com/StoneT2000/rfcl .cache/upstreams/rfcl
git -C .cache/upstreams/rfcl checkout --detach 7bd660f05dbda1ec8881bcfb250a8d1881532dbb
uv venv --python 3.10.20 .venv-rlpd
uv pip install --python .venv-rlpd/bin/python -r external/rlpd/requirements.lock --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip check --python .venv-rlpd/bin/python
```

安装/CPU检查用`JAX_PLATFORMS=cpu`，实际RLPD GPU运行不能带这个变量。上游checkout保持只读；修复在`scripts/replay_trajectory.py`与`patches/`，不修改缓存上游。许可证及局部适配边界见`UPSTREAMS.md`。

## 不依赖大回放的复算

- U0崩溃前后固定真实输入、研究权重及误差复算命令：`releases/ANALYSIS_REPRODUCTION.md`。
- `releases/u0-final-001`只含SAC/WM/纯任务价值，明确排除IL；这是固定失败末次模型，不是最佳模型。
- `releases/u1-final-001`是U1 seed0固定末次无IL部署包，1.26MB；完整checkpoint与部署包64真实状态CPU动作逐位相同。另有只依赖已发布研究权重/真实输入的便携复算，见下方U1命令与`artifacts/u1-portable-no-il-check-001`。
- C续训对照的便携分析权重：`releases/c-continue-analysis-final-001`。64个真实状态CPU动作与原始checkpoint逐位一致。
- RLPD smoke可携带actor：`releases/rlpd-smoke-001`；实际GPU完整基线尚未由该文件表示。推理与完整learner的64状态CPU动作逐位一致。
- F1分支原始数组、奖励序列和分析在`artifacts/f1-u0-ranking-001`与`f1-u0-precollapse-ranking-001`；两个prefix40集合均没有尾段误差可分的候选排名，不应报告为排序正确率100%。

实际行为统计可仅由发布的逐转移派生数组复算：

```bash
.venv/bin/python scripts/diagnose_behavior.py --input-arrays artifacts/f-u0-behavior-u0-001/transitions.npz --bin-size 8192 --output /tmp/u0-behavior-recomputed
.venv/bin/python scripts/diagnose_behavior.py --input-arrays artifacts/f-u0-behavior-control-001/transitions.npz --bin-size 8192 --output /tmp/control-behavior-recomputed
```

每65536步的原始与便携复算已完全相等；8192步全时间段细分与原始learner日志的联合时间线位于`artifacts/f-u0-behavior-timeline-001`。这些是描述性关联，不是动作来源的因果收益。

## 数据与训练

官方数据仓库与revision在`artifacts/bootstrap/demo-source.json`；`scripts/download_demos.py`用该固定URL调用官方downloader。原始数据、全回放、完整训练checkpoint不放Git。`artifacts/datasets`和每次审计记录其哈希；不能把只给本地路径的完整checkpoint称为已交付。

逐实验`manifest.json`保存实际展开命令、代码commit、退出码和时限；`source_hashes.json`及`working-tree.patch`保存当时源码证据。复跑时换experiment ID、output和checkpoint目录，保留原命令的任务/seed/协议。完整SAC/U0单次阶段262144真实交互，U0加C预热共524288；独立RLPD共524288，但计算量更大。

`run_pickcube_repeat.py --task ...`是同一主训练器的有限任务/seed队列，不安装新框架；`run_u1_repeat.py`按预登记只加critic LayerNorm，先从同一BC重训C-LN，不能从已学习无LN checkpoint直接插入归一化。细节见`U1_CHANGE.md`。

Peg原始1000演示含312对完全相同动作/首状态的副本。回放保留明确episode来源ID，按整个seed组分离训练/验证；成功样本过滤使用GPU真实奖励。CPU转换成功率不能替代GPU重放成功率，详见F-E0-005。

## 固定最终测试

`configs/final_evaluation_plan.json`预登记当前队列的52个模型/阶段，共88个方法条件，每项100完整回合、seed20000、10环境，固定末次模型（RLPD262144另为预登记等交互中间点）。当前尚未访问最终测试。额外训练seed须在测试前追加登记，不能根据测试分数选模型。

训练和两个独立评测smoke完成、方法固定后，通过后台进程运行：

```bash
.venv/bin/python scripts/run_final_evaluation.py --plan configs/final_evaluation_plan.json --output artifacts/final-evaluation-001
```

该有限入口先检查所选训练/评测smoke成功，再核对保存checkpoint哈希、导出可携带分析权重、验证无IL部署动作一致性，最后在共享单卡锁内测试。总限7200秒仍受首批24小时账本限制。`--ids`只能明确指定已登记子集，未选项写为pending；不存在按成绩自动选择或失败自动重试。未完成项目必须在批次报告中列出。

2026-10-16补充：隔离RLPD独立加载器GPU smoke已完成4完整validation回合；主加载器smoke也已完成，最终测试仍未开始。分组统计与不确定性边界另见`configs/analysis_plan.md`。Peg full002最终可用126轨迹但只有88个不同来源组，训练/验证分别98/28轨迹和70/18组；完整失败保留，不把1000原始记录视作1000个独立成功演示。

五seed扩展后，最终登记为52模型/阶段、88方法条件；训练seed0仍标记开发，seed1–4单独报告独立重复。完成或部分完成评测后可用以下纯CPU命令复算逐条件表与跨训练seed汇总（输出目录须不存在）：

```bash
.venv/bin/python scripts/summarize_final_results.py --output artifacts/final-results-001
```

输入只有登记表及每个评测的manifest、result、interactions和逐回合JSONL；输出CSV/JSON附来源哈希。未执行、运行中或失败条件不填零分，软件合成测试不计为仿真实验。训练管线从已记录24h额度中预留2h用于最终评测；该预留不会增加总授权额度。

源码恢复预检见`artifacts/source-history-preflight-001`：54次已启动作业的原始代码commit均可从本地Git历史恢复，2998条记录源码哈希中两个仅在启动补丁中的版本已按SHA256逐字节恢复。另有一个未提交的历史绘图脚本版本不在提交/补丁中，涉及三次Stack数据作业的目录扫描，但不是这些作业的执行入口或导入依赖；保留缺口，不伪称完整恢复该目录。当前已发布绘图脚本及最终图形可以复算。

新启动作业会保存`source_snapshot.tar.gz`及其SHA256，内容与同次`source_hashes.json`来自同一次文件读取，包含未提交代码。它不包含环境依赖、数据或权重；这些仍按各自锁文件和清单恢复。已有作业不倒填启动快照。原始Git提交的离线bundle创建/恢复已在独立bare仓库验证；当前预检bundle仅在忽略缓存中，批次结束须生成并交付最终版本，不能将这次预检当作已上传。

## U1便携复算

U1的两个分析包`releases/u1-seed0-analysis-{step196608,final}-001`含冻结IL，供诊断/四方法评测，不能恢复训练；`releases/u1-final-001`只含最终无IL推理模块。比较报告为`diagnostics/U1_seed0.md`，目前只支持一条开发运行的稳定性改善，未证明跨seed或规划收益。以下命令均不需要完整训练checkpoint、大回放或新仿真：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/diagnose_value_drift.py --before releases/u1-seed0-analysis-step196608-001/weights.pt --after releases/u1-seed0-analysis-final-001/weights.pt --fixed-batches artifacts/f-u0-value-drift-portable-001/fixed_batches.npz --output /tmp/u1-common-drift-recomputed
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/diagnose_behavior.py --input-arrays artifacts/u1-seed0-behavior-001/transitions.npz --bin-size 8192 --output /tmp/u1-behavior-recomputed
CUDA_VISIBLE_DEVICES= .venv/bin/python artifacts/u1-portable-no-il-check-001/check.py
```

前两项输出目录必须不存在。最后一项会重写该检查目录的`actions.npz`与`result.json`，验证64个已发布真实输入上的无IL部署/分析加载器动作逐位一致，且禁止IL构造；只是CPU一致性检查，不是GPU独立最终评测。实际推理入口是`scripts.deployment.DeploymentPolicy`，输入原始state批量Tensor，动作通过9条SAC候选、H2世界模型与纯任务价值滚动规划生成。
