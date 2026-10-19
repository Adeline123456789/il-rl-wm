import datetime as dt,json
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
a=read('artifacts/act-final-completion-postprocess-001/audit.json');assert a['completed_conditions']==106
review=read('artifacts/act-final-review-cpu-002/analysis/result.json');assert review['matched_completed_act_conditions']==18
u=[r for r in review['act_conditions'] if r['id'].startswith('act-u2-')];assert all(r['success_once']==0 and r['success_at_end']==0 and r['mean_return']==0 for r in u)
now=dt.datetime.now(dt.timezone.utc).isoformat();total=a['all_gpu_job_seconds'];added=a['completion_gpu_job_seconds'];wall=a['completion_wall_seconds'];remain=a['remaining_gpu_job_seconds']
p=Path('FINAL_BATCH_REPORT.md');s=p.read_text()
s=s.replace('2026-10-19。原有界批次已经停止计算，**104/106 条件完成，研究目标尚未全部完成**。历史 BC/U0/U1/RLPD 的 88 条件完成；ACT 的 18 条件完成 16 条，U2 均值因时间上限中断，U2 随机未启动。失败及未运行均为 null，不用开发分数替代。','2026-10-19。经用户明确批准300秒补评，**106/106正式条件全部完成；整个研究目标尚未全部完成**。历史88条件和ACT18条件均已完成。U2新attempt002的均值/随机各100完整回合，成功均0/100；原均值attempt001的预算中断和70回合仍完整保留，不把旧部分记录合入新成绩。')
s=s.replace('artifacts/act-final-summary-003/','artifacts/act-final-summary-004/').replace('artifacts/act-final-review-cpu-001/','artifacts/act-final-review-cpu-002/')
s=s.replace('16个完成条件的100个初始观测','18个完成条件的100个初始观测')
s=s.replace('|U2 有限保持：均值|未完成|未完成|未完成|','|U2 有限保持：均值|0|0|0.00|').replace('|U2 有限保持：随机|未启动|未启动|未启动|','|U2 有限保持：随机|0|0|0.00|')
s=s.replace('其完整独立测试缺失，不能将开发0/40写成正式0/100。','补评的两个独立测试模式均为0/100，接近/抓取/抬升/放置计数均0；与同初态T5末次一样，没有观察到持续闭环收益。')
s=s.replace('## 预算、失败和恢复边界','## 预算、失败和恢复边界\n\n原7200秒阶段的历史账目与失败保留如下。用户授权后两项补评GPU作业新增'+f'{added:.6f}秒，补评队列墙钟{wall:.6f}秒（低于300秒），共享总账{total:.6f}秒/{total/3600:.6f}小时，剩{remain:.6f}秒。作业总数216，新增40000个正式测试返回/尝试转移、0训练更新；正式测试累计返回2374180、尝试2374190，包含旧失败尾段。'+ '新审计见[completion audit](artifacts/act-final-completion-postprocess-001/audit.json)：其他104条件逐行不变、18个ACT条件100初态一致、源代码快照/权重/总账通过。')
s=s.replace('当前未授权、未排队。保留旧失败，新尝试使用独立ID，100回合完整重评，重复交互另记。','现已获用户明确授权并完成，授权原文、attempt映射及cap见configs/act_final_completion_authorization.json。保留旧失败，新尝试使用独立002 ID，100回合完整重评，重复交互另记。')
s=s.replace('但其正式结果缺失。','正式均值/随机均0/100，未观察到持续收益。')
s=s.replace('正式104条件完成。ACT-WM、ACT多seed/多任务及U2两项正式测试未完成。','正式106条件完成。ACT-WM、ACT多seed/多任务仍未完成。')
s=s.replace('CSV含全部106条件、失败及null；','CSV含全部106完整条件，旧失败及null汇总003仍保留；')
s=s.replace('计费23.648小时，采样峰值5770MiB，未新增GPU。先申请有限补评U2，再单独决定','计费'+f'{total/3600:.6f}'+'小时，历史采样峰值5770MiB，未新增GPU。U2已补齐，再单独决定')
s=s.replace('--plan configs/act_final_evaluation_plan.json --output','--plan configs/act_final_evaluation_plan.json --attempt-map configs/act_final_completion_attempts.json --output')
s=s.replace('scripts/review_act_final_evidence.py --output','scripts/review_act_final_evidence.py --summary artifacts/act-final-summary-004/summary.json --output').replace('核对16个完成ACT条件','核对18个完成ACT条件')
s=s.replace('收尾时GPU上另有其他环境进程，未干预或纳入本项目账本；任何获准的后续作业仍须重新检查GPU占用。','补评启动前重新核实GPU空闲，全程串行；任何后续作业仍须独立授权并检查占用。')
p.write_text(s)
notice='> 2026-10-19 补评完成：用户明确批准300秒后，106/106正式条件完成。U2均值/随机各0/100，旧预算中断记录保留；未增加训练或恢复自动化。最新表、补评账本和研究限制见[FINAL_BATCH_REPORT.md](FINAL_BATCH_REPORT.md)。下方旧104/106或未授权描述为此前历史。\n\n'
for name in ['ACT_MILESTONE_REPORT.md','ACT_TRANSFER_REPORT.md','ACT_U2_FINAL_REVIEW.md','ACT_TRANSFER_CAUSAL_GAPS.md','ACT_SAC_PLAN.md','PROGRESS_SNAPSHOT.md','README.md','NEXT_STEPS.md','NEXT_RESEARCH_BATCH.md','BATCH_REQUIREMENTS_AUDIT.md']:
 p=Path(name);p.write_text(notice+p.read_text())
for name in ['REPRODUCING.md','releases/ACT_REPRODUCTION.md']:
 p=Path(name);p.write_text('补评最新复算：`.venv/bin/python scripts/summarize_final_results.py --plan configs/act_final_evaluation_plan.json --attempt-map configs/act_final_completion_attempts.json --output <new-directory>`。映射保留旧失败；不带映射仍复算原104/106。详细账本见FINAL_BATCH_REPORT.md。\n\n'+p.read_text())
with open('RESEARCH_LOG.md','a') as f:f.write('\n## '+now+' U2补评完成\n两项新attempt002均exit0，各100回合、20000真实测试交互；均值/随机成功及末步成功均0/100。总106/106完成，18条件100初态逐元素一致，其他104行不变。补评GPU作业'+f'{added:.6f}秒，队列墙钟{wall:.6f}秒；总账{total:.6f}秒，剩{remain:.6f}秒。'+'旧失败70回合/14180返回转移保留并计费，0新增训练/机制。CPU等待器已结束；launch回执曾因缺json导入保存失败，从活动child状态恢复PID3180899，未重复启动。后台交付后停止，无新训练授权、无定时自动化。\n')
with open('ACT_CHANGE_LOG.md','a') as f:f.write('\n## '+now+' 显式补评attempt映射\n新增固定300秒U2补评入口；汇总可选attempt-map、轨迹审查可选summary。原默认汇总回归完全一致。两项完整新测试0/100，106条件闭合，旧失败未覆盖。\n')
with open('ACT_FAILURES.md','a') as f:f.write('\n## F-ACT-EVAL-03 补评关闭（'+now+'）\n用户批准300秒后两个002 attempt均完成，原001失败不覆盖；各100回合0成功。资源中断已补齐，科学负结果仍保留。补评audit见artifacts/act-final-completion-postprocess-001/audit.json。\n')
with open('AGENTS.md','a') as f:f.write('\n2026-10-19 补评终态：106/106已完成，U2两项002均0/100。原300秒仅消耗实测补评时长，无新训练授权；禁止重复队列。最新FINAL_BATCH_REPORT.md和act-final-summary-004。\n')
s=read('research_state.json');s['phase']='Authorized U2 completion finished; 106/106 final conditions complete; no further GPU work authorized';s['next_experiment']='None queued; R0/R1/R3 remains an unauthorized separate proposal';s['final_evaluation']['completion_result']=a;s['final_evaluation']['latest_summary']='artifacts/act-final-summary-004/summary.json';s['blockers']=['New training R0/R1/R3 and remaining ACT-WM/seeds/tasks require a separate explicit budget; 300-second evaluation authorization is consumed only by this completed scope.'];s['resume']='Read FINAL_BATCH_REPORT.md and completion audit; do not restart original or completion queues; hourly automation remains deleted.'
Path('research_state.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(completed_conditions=106,added_gpu_seconds=added,total_gpu_seconds=total,remaining=remain)))
