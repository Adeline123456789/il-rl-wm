# il-rl-wm

冻结 IL 初始化随机 SAC，训练时 IL/SAC 候选由世界模型规划选择，最终删除 IL、保留 SAC 与世界模型。固定约束在 `configs/charter.json`。

当前基准为 Meta-World v3。`bootstrap_probe` 实际检查 MT50 环境、采集三个 MT1 任务的官方专家轨迹、训练小型 BC 并闭环评测；它不是 MT50 多任务学习成绩，也不是 U0/U1 效果证明。

## 本机运行

所有命令在本仓库根目录执行。复制 `.venv` 后使用 `.venv/bin/python`，不依赖旧路径的激活脚本。

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python -m lab.cli make-plan --round R000 --experiment bootstrap_probe --steps 65536
.venv/bin/python -m lab.cli run plans/R000.json
.venv/bin/python -m lab.cli status
```

数据：`runtime/datasets/轮次/任务/trajectories.npz`。模型：`runtime/checkpoints/`。完整日志：`runs/轮次/`。上传的小型证据：`reports/轮次/`。

执行器在 `WAIT_REVIEW` 停下，不生成审批决定。独立网页 ChatGPT 的响应原样写入 GitHub `exchange/R000.json` 后，执行器读取固定提交并校验：

```sh
.venv/bin/python -m lab.cli pull-review --commit 固定提交SHA --path exchange/R000.json
```

`lab/handoff.py` 只消费外部审查，没有自行生成研究批准的代码。审查回执保存会话、响应标识和原文哈希；这提供可核对的来源记录，不是对模型身份的密码学证明。

`ref_sac` 使用 Stable-Baselines3 SAC。`lab/models.py` 的 SAC/世界模型集成仍是待验证的原型，尚未开放 U0 训练入口。`vendor/cleanrl_sac.py` 是未修改的参考源码，来源和许可证保留在 `vendor/`，不作为本项目的新算法贡献。
