#!/usr/bin/env python3
"""Summarize a bounded train.py run and optionally plot measured losses."""
import argparse, json, math, pathlib, statistics
p=argparse.ArgumentParser(); p.add_argument('run_dir',type=pathlib.Path); p.add_argument('--window',type=int,default=20); a=p.parse_args()
rows=[json.loads(line) for line in (a.run_dir/'metrics.jsonl').read_text().splitlines() if line.strip()]
train=[r for r in rows if r['kind']=='train']; val=[r for r in rows if r['kind']=='validation']
report={'optimizer_steps':train[-1]['optimizer_step'] if train else 0,'completed':any(r['kind']=='complete' for r in rows),'train':{},'validation':val,'checkpoints':[str(x) for x in a.run_dir.glob('checkpoint-*/model.pt')]}
for key in ('total_loss','action_loss','tactile_loss','flare_loss'):
    values=[r[key] for r in train]
    if not values: continue
    assert all(math.isfinite(x) for x in values),key
    n=min(a.window,len(values)); first=statistics.mean(values[:n]); last=statistics.mean(values[-n:])
    report['train'][key]={'window':n,'first_mean':first,'last_mean':last,'relative_change_percent':100*(last-first)/first if first else None}
(a.run_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(0)
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
xs=[r['optimizer_step'] for r in train]
for key,label in [('total_loss','Total'),('action_loss','Action'),('tactile_loss','Tactile'),('flare_loss','FLARE')]:
    ys=[r[key] for r in train]
    rolling=[statistics.mean(ys[max(0,i-a.window+1):i+1]) for i in range(len(ys))]
    line=axes[0].plot(xs,rolling,label=label)[0]
    axes[0].plot(xs,ys,alpha=.12,color=line.get_color())
for key in ('val/action_loss','val/tactile_loss'):
    if val and key in val[0]: axes[1].plot([r['optimizer_step'] for r in val],[r[key] for r in val],'-o',label=key.removeprefix('val/'))
axes[0].set_title(f'Train loss ({a.window}-step rolling mean)'); axes[1].set_title('Held-out episodes, fixed samples and noise')
for ax in axes: ax.set_xlabel('Optimizer update'); ax.set_ylabel('Loss'); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(a.run_dir/'loss_curve.png',dpi=160)
