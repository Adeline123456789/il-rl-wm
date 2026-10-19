# ACT 故障与未验证假设

## F-ACT-IMPL-01：尾段不是真实专家动作

锁定 ManiSkill `SmallDemoDataset_ACTPolicy.__getitem__` 在尾部合成静止臂/夹爪动作，`Agent.compute_loss` 对全部位置取mean，`DETRVAE.forward` 的posterior mask全false。新任务明确要求真实有效动作掩码。本地按真实episode结束位置标记pad；posterior attention和重建损失均排除pad。CPU测试验证不跨episode、任意改变pad值不改变posterior/loss，不把该接口修正称为性能改进。

## F-ACT-IMPL-02：第一层decoder输出使后续层无学习信号

锁定 `build_transformer` 设置 `return_intermediate_dec=True`，Transformer直接返回 `[layers,batch,query,hidden]`；DETRVAE取`[0]`后接action_head。因此后续decoder层虽计算却不影响损失。原作者锁定源码亦有该返回/索引组合。本地关闭intermediate返回，保留`[0]`但取得最后层；未修改vendor。两层梯度测试覆盖真实反向传播，未声称它提升闭环性能。无pad时本地训练forward与同一final-layer设置下的上游forward逐位一致；不把该检查误写为上游默认输出不变。

## F-ACT-IMPL-03：执行模式不能混称

官方无时间集成默认连续执行K个动作；本项目首模式必须每步重查首动作。时间集成单列、显式有效位而非用动作非零判断缓存存在，支持按环境清空。不能将它的历史依赖标签未经说明监督无历史学生。

迁移研究Failure ID F-ACT-01/02/03/04当前尚未成立：没有教师/学生真实结果，不捏造技能丢失、遗忘或参数迁移不稳定。后续按现象→竞争解释→论文/源码→单一变更→对照结果登记。

资料归档一次命令因参数列表超过操作系统上限被拒绝，未写文件或执行训练；改用逐文件保存并核对6份远端blob SHA。该传输问题与研究算法无关。

## 2026-10-19 F-ACT-T0-01：执行模式及保持能力缺口

固定K30首动作8/40、集成30/40，而末步仅4/40与5/40。有限网格开发结果，不是已证明的状态不可表达性。先以无历史首动作教师做D1，独立记录学生误差与闭环，不能把集成成绩套作教师基线。尚无迁移失败结论。
