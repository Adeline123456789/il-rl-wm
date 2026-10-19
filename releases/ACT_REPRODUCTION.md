补评最新复算：`.venv/bin/python scripts/summarize_final_results.py --plan configs/act_final_evaluation_plan.json --attempt-map configs/act_final_completion_attempts.json --output <new-directory>`。映射保留旧失败；不带映射仍复算原104/106。详细账本见FINAL_BATCH_REPORT.md。

# ACT 阶段包复算

> 最终证据更新：原批次104/106条件完成、U2两项未完成；复算命令和范围见[FINAL_BATCH_REPORT](../FINAL_BATCH_REPORT.md)。原始summary与逐回合记录保留，未完成分数为null。

固定环境与原始演示重建先遵循根目录 `REPRODUCING.md`；ManiSkill commit仍为62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3。PickCube有效演示拆分为 `artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json`（717 train /180 validation）。不把validation/test状态加入训练。

## 便携权重

- `act-teacher-k30-001/weights.pt`：固定10000更新K30教师；包含完整模型张量（含仅训练时使用的后验参数以支持strict加载），不含优化器、回放、RNG。manifest记录完整checkpoint来源及两个哈希。
- `act-d1-negative-001/weights.pt`：D1失败学生的SAC Actor；不依赖教师推理，不含critic/优化器/回放，不能直接称完整SAC恢复点。构造 `scripts.train_sac.Actor` 时使用42维state和4维[-1,1]动作Box，再strict加载`actor`。评测时 `eval()`，均值`get_eval_action(s)`，随机`get_action(s)[0]`。在线训练前应明确启用log_std梯度并初始化critic/温度，完整在线T2–T6和U2现已运行，负结果与资源成本见根目录报告。

教师加载示例（repo根目录）：

```python
import torch
from scripts.act_teacher import ACTTeacher, ACTConfig
p = torch.load('releases/act-teacher-k30-001/weights.pt', map_location='cpu', weights_only=False)
teacher = ACTTeacher(p['state_dim'], p['action_dim'], ACTConfig(**p['model_config']))
teacher.load_state_dict(p['model'], strict=True)
teacher.eval().requires_grad_(False)
# teacher(s) = clip(chunk(s)[:,0], -1, 1)，原始state，无归一化，无时序集成。
```

## 原始指标及协议版本

T0/D1原始日志、逐回合数据、资源、交互账、全部最终状态/动作/奖励NPZ和source_snapshot已随阶段提交；视频、原始大演示、完整训练checkpoint/replay保留本机，不在普通Git中。上传范围在 `artifacts/act-teacher-summary-001/publication_scope.json`。负结果不删。

T0原评测和首个D1使用“仅首次seed”协议，F-ACT-EVAL-01审计发现后续reset与策略RNG耦合，不能当全部40回合严格配对。修正评测 `scripts/evaluate_act_matched.py` 固定已有模型、每8回合明确seed10000+batch起点，并保存全部初态配对误差；不使用seed20000，不重训或重新选checkpoint。

每次真实GPU作业的manifest中code_commit是本机实际提交，dirty部分由working-tree.patch及source_snapshot.tar.gz逐文件补全。GitHub连接器发布的是等内容snapshot commit，因此远端commit ID与本机ID可能不同，映射见 `artifacts/github-act-stage-upload-001/status.json`。复算历史算法应在独立目录解包对应source_snapshot，并校验source_hashes；当前HEAD可能已经包含评测修复，不应把新代码生成结果冒称旧实验复现。

当前修复评测需要原K8/K16完整checkpoint来复算全网格；它们未上传，来源配置、训练日志、哈希完整保存，可据固定配置重训但不承诺跨GPU逐位相同。便携K30和D1则可直接加载评测。所有新GPU运行仍须使用run_experiment.py账本、单卡锁与明确时限，不直接重复当前experiment ID。

## 2026-10-19 新增便携Actor与证据

`act-c1-init-001`独立Transformer-SAC初始化；`act-t2-online-final-001`至`act-t6-online-final-001`及`act-u2-online-final-001`为固定末次Actor。严格加载统一使用`scripts.act_inference.load_model`，kind为mlp或act；学生推理不构造教师，保存已学习log_std。每包manifest关联完整checkpoint hash并验证逐张量相等；不能把Actor包当含replay/优化器的完整恢复点。

T2–T6/U2完整曲线及成本：`artifacts/act-online-summary-006/`；固定训练状态漂移见各`act-*-online-drift-001/`；T5/U2原始梯度见`artifacts/act-mapping-gradient-cpu-002/analysis/`。原始训练和验证日志完整保留，所有最终测试条件仍遵守独立注册表及原预算。正式批次完成后由`summarize_final_results.py`生成总表，未执行值为null，不能作为0分。
