"""Read-only schema preflight for the release's canonical LeRobot loader."""
import argparse,json
import numpy as np
from pathlib import Path
from utils.lerobot_common import KEY_HEAD,KEY_WRIST_L,KEY_WRIST_R,KEY_TACF6,DEFORM_KEYS
p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
a=p.parse_args()
info=json.loads((a.root/'meta/info.json').read_text())
f=info['features']
issues=[]
for key in [KEY_HEAD,KEY_WRIST_L,KEY_WRIST_R,KEY_TACF6,*DEFORM_KEYS]:
    if key not in f: issues.append(f'Missing canonical feature: {key}')
for key,shape in [('observation.state',[62]),('action',[16,62]),(KEY_TACF6,[10,6])]:
    if key in f and f[key]['shape'] != shape: issues.append(f'{key}: got {f[key]["shape"]}, expected {shape}')
if not (a.root/'meta/trex_norm_stats.json').exists(): issues.append('Missing meta/trex_norm_stats.json (q01/q99 + mask)')
if (a.root/"CONVERSION_IN_PROGRESS").exists(): issues.append("Conversion is incomplete")
if (a.root/"meta/trex_norm_stats.json").exists():
    blocks=json.loads((a.root/"meta/trex_norm_stats.json").read_text())
    block=blocks[next(iter(blocks))]
    for key,shape in [("action",(16,62)),("state",(62,)),("tactile_f6",(60,))]:
        for stat in ["q01","q99"]:
            value=np.asarray(block.get(key,{}).get(stat,[]))
            if value.shape != shape or not np.isfinite(value).all():
                issues.append(f"Invalid norm stats: {key}.{stat}")
        if len(block.get(key,{}).get("mask",[])) != shape[-1]:
            issues.append(f"Invalid norm mask: {key}")
print(json.dumps({'root':str(a.root),'episodes':info['total_episodes'], 'frames':info['total_frames'],
    'compatible_with_release_loader':not issues,'issues':issues},indent=2))
raise SystemExit(1 if issues else 0)
