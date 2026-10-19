"""Plot fixed development evidence only; no generated observations or test selection."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--qa-tools',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.qa_tools));from audit_panel_alignment import require_matplotlib_panel_alignment
 a.output.mkdir(exist_ok=False,parents=True)
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
 curve_path=Path('artifacts/act-online-summary-006/curves.csv');curves=list(csv.DictReader(curve_path.open()))
 labels={'act-t2-online-seed0-002':'T2 Random SAC','act-t3-online-seed0-001':'T3 BC init','act-t4-online-seed0-001':'T4 ACT distilled','act-t5-online-seed0-001':'T5 ACT copied','act-t6-online-seed0-001':'T6 ACT distilled + expert replay','act-u2-retention-seed0-001':'U2 T5 + early mean retention'}
 colors=['#707070','#A5A5A5','#6D8FAC','#395B73','#369887','#C57A43'];markers=['+','x','s','o','D','^'];styles=[':',':','--','--','-','-']
 def canvas(title):
  fig,ax=plt.subplots(figsize=(7.0866141732,4.4094488189));fig.subplots_adjust(left=.12,right=.97,bottom=.20,top=.70)
  fig.text(.12,.96,title,fontsize=10,weight='bold',va='top')
  ax.set_xlim(-3000,266000);ax.set_xticks([0,65536,131072,196608,262144],['0','65.5','131.1','196.6','262.1']);ax.set_xlabel('Real training interactions (thousands)');ax.grid(axis='y',color='#EEEEEE',linewidth=.5)
  return fig,ax
 def save(fig,name):
  fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(a.output/(name+'.alignment.json')),strict=True)
  fig.savefig(a.output/f'{name}.pdf',facecolor='white')
  fig.savefig(a.output/f'{name}.svg',facecolor='white')
  fig.savefig(a.output/f'{name}.png',dpi=600,facecolor='white')
  plt.close(fig)
 fig,ax=canvas('Expert replay supports learning; initialization alone does not in this run')
 for (key,label),color,marker,style in zip(labels.items(),colors,markers,styles):
  data=[r for r in curves if r['id']==key];assert len(data)==17
  steps=[int(r['step']) for r in data];assert steps==sorted(set(steps))
  ax.plot(steps,[100*int(r['success_once'])/int(r['episodes']) for r in data],label=label,color=color,marker=marker,markersize=3,linewidth=1.2,linestyle=style)
 ax.set_ylim(-5,105);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Success at least once (%)')
 fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.11,.90),ncol=2,columnspacing=2,handlelength=2)
 fig.text(.12,.065,'One training seed; 40 fixed validation episodes per point; no seed-level error bars.',fontsize=8)
 fig.text(.12,.028,'T2–T5 overlap at zero after the initial evaluation. No final-test data used.',fontsize=8)
 save(fig,'online_success')
 sources={str(curve_path):hashlib.sha256(curve_path.read_bytes()).hexdigest()};drift_rows=[]
 fig,ax=canvas('Early mean retention reduces expert-state drift without lasting closed-loop skill')
 for label,path,color,marker in [('T5 ACT copied','artifacts/act-t5-online-drift-001/result.json',colors[3],'o'),('U2 Early mean retention','artifacts/act-u2-online-drift-001/result.json',colors[5],'^')]:
  obj=read(path);sources[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  data=obj['results'];assert [r['step'] for r in data]==[0,16384,65536,131072,262144]
  ax.plot([r['step'] for r in data],[r['teacher_action_mse_per_dim'][3] for r in data],label=label,color=color,marker=marker,markersize=4,linewidth=1.4)
  drift_rows.extend(dict(method=label,step=r['step'],gripper_teacher_mse=r['teacher_action_mse_per_dim'][3]) for r in data)
 ax.axvline(65536,color='#999999',linestyle=':',linewidth=.8,label='Retention ends (65,536)')
 ax.set_ylabel('Gripper action MSE vs frozen ACT');ax.set_ylim(-.04,1.4);ax.set_yticks([0,.4,.8,1.2])
 fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.11,.90),ncol=2,handlelength=2)
 fig.text(.12,.065,'Same 1,024 expert-training states; mean action only. U2 success is zero by 49,152 steps.',fontsize=8)
 fig.text(.12,.028,'This probe does not measure online-state coverage or identify the cause of forgetting.',fontsize=8)
 save(fig,'gripper_mean_drift')
 (a.output/'online_success_source.csv').write_text(curve_path.read_text())
 with (a.output/'gripper_mean_drift_source.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(drift_rows[0]));w.writeheader();w.writerows(drift_rows)
 (a.output/'sources.json').write_text(json.dumps(dict(source_hashes=sources,plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows_used=dict(success=len(curves),drift=len(drift_rows)),excluded_rows=0,interpolation='Straight segments connect recorded checkpoints; no smoothing or inferred observations.',training_repeats=1,uncertainty='No training-seed uncertainty available; no error bars invented.',final_test_used=False),indent=2)+'\n')
if __name__=='__main__':main()
