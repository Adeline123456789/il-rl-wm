"""Descriptive single-seed failure figure; no invented uncertainty or smoothing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path,records):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def build_figure(output):
    output.mkdir(parents=True,exist_ok=True)
    u0=Path('artifacts/e6-u0-full-001');control=Path('artifacts/e3-sac-c-continue-full-001')
    sources=[u0/'validation_episodes.jsonl',control/'validation_episodes.jsonl',
             u0/'metrics.jsonl',control/'metrics.jsonl',Path('artifacts/f-u0-value-drift-002/values.npz')]
    validation=[];critic=[]
    for directory,label in [(u0,'U0'),(control,'C')]:
        data=rows(directory/'validation_episodes.jsonl')
        for method in sorted({r.get('method','C') for r in data}):
            for step in sorted({r['step'] for r in data}):
                group=[r for r in data if r['step']==step and r.get('method','C')==method]
                assert len(group)==16
                validation.append(dict(run=label,method=method,online_interactions=262144+step,
                                       successes=sum(r['success_once'] for r in group),episodes=len(group)))
        for row in rows(directory/'metrics.jsonl'):
            if row['metric']=='losses/qf1_values':
                critic.append(dict(run=label,online_interactions=262144+row['step'],q1_batch_mean=row['value']))
    with np.load(sources[-1]) as data:
        distributions=[data[f'{time}_{source}_pure_q_mean_action'].copy()
                       for source in ['expert','online'] for time in ['before','after']]
    fixed=[dict(source=source,snapshot=time,sample=i,pure_task_value=float(value))
           for (source,time),values in zip([(s,t) for s in ['expert','online'] for t in ['before','after']],distributions)
           for i,value in enumerate(values)]
    write_csv(output/'validation.csv',validation);write_csv(output/'critic.csv',critic);write_csv(output/'fixed-input-values.csv',fixed)
    (output/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(7.2047244094,3.3464566929))  # 183 × 85 mm
    fig.subplots_adjust(left=.078,right=.985,top=.84,bottom=.30,wspace=.48)
    styles=[('C','C continuation','#353535','--','s'),('sac','U0: SAC','#0072B2','-','o'),
            ('sac_wm','U0: SAC + WM','#009E73','-','^'),('il_sac_wm','U0: IL + SAC + WM','#D55E00','-', 'D'),
            ('il','Frozen IL','#888888',':','x')]
    for method,label,color,style,marker in styles:
        values=[r for r in validation if r['method']==method]
        axes[0].plot([r['online_interactions']/1000 for r in values],
                     [r['successes']/r['episodes'] for r in values],label=label,color=color,
                     linestyle=style,marker=marker,markersize=3,linewidth=1.2)
    axes[0].set(ylabel='Validation success fraction',ylim=(-.04,1.04),yticks=[0,.5,1])
    for run,color,style in [('U0','#0072B2','-'),('C','#353535','--')]:
        values=[r for r in critic if r['run']==run]
        axes[1].plot([r['online_interactions']/1000 for r in values],[r['q1_batch_mean'] for r in values],
                     color=color,linestyle=style,linewidth=1.2)
    axes[1].set(ylabel='Soft Q1 (minibatch mean)')
    for ax in axes[:2]:ax.set(xlabel='Online interactions (thousands)',xticks=[262.144,393.216,524.288],xticklabels=['262','393','524'])
    boxes=axes[2].boxplot(distributions,patch_artist=True,whis=(5,95),widths=.55,
        medianprops=dict(color='#222222',linewidth=.8),boxprops=dict(linewidth=.7),
        whiskerprops=dict(linewidth=.7),capprops=dict(linewidth=.7),
        flierprops=dict(marker='.',markersize=1.5,alpha=.25,markeredgecolor='#555555'))
    for patch,color in zip(boxes['boxes'],['#A6CEE3','#F3BA98']*2):patch.set_facecolor(color)
    axes[2].set(ylabel='Pure-task value (fixed inputs)',xticks=[1,2,3,4],
                xticklabels=['Expert\nbefore','Expert\nafter','Online\nbefore','Online\nafter'])
    for ax,label,title in zip(axes,'abc',['Behavior','SAC critic','Task-value drift']):
        ax.text(0,1,label,transform=ax.transAxes+ScaledTranslation(-16/72,9/72,fig.dpi_scale_trans),fontweight='bold',fontsize=8)
        ax.set_title(title,loc='left',fontsize=8,pad=8)
        ax.tick_params(length=3,width=.7)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,fontsize=6.5,
               handlelength=2.3,columnspacing=1.8,labelspacing=.7)
    return fig


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('figures/seed0-diagnosis'))
    parser.add_argument('--qa-scripts',type=Path,required=True,help='nature-figure scripts directory for the required alignment audit')
    args=parser.parse_args();sys.path.insert(0,str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig=build_figure(args.output)
    require_matplotlib_panel_alignment(fig,json_out=args.output/'alignment.json',
        overlay_svg=args.output/'alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(args.output/'seed0-diagnosis.svg')
    fig.savefig(args.output/'seed0-diagnosis.pdf')
    fig.savefig(args.output/'seed0-diagnosis.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
