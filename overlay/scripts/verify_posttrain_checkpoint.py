#!/usr/bin/env python3
"""Validate exported policy tensors and prove selected weights were updated."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=pathlib.Path);p.add_argument('--checkpoint',required=True,type=pathlib.Path);a=p.parse_args()
torch.set_num_threads(4)
before=torch.load(a.release,map_location='cpu',weights_only=True,mmap=True)
after=torch.load(a.checkpoint/'model.pt',map_location='cpu',weights_only=True,mmap=True)
assert before.keys()==after.keys(),(set(before)-set(after),set(after)-set(before))
assert all(before[k].shape==v.shape for k,v in after.items())
checked=0;frozen=0;updates={}
for key,v in after.items():
    if v.is_floating_point():
        # Bound temporary allocations for the 311M-element embedding matrix.
        for piece in v.reshape(-1).split(4_000_000):assert torch.isfinite(piece).all().item(),key
    checked+=1
    if key.startswith(('visual.','deform_encoder.','tactile_vqvae.','tacf6_vqvae_')):
        assert torch.equal(before[key],v),f'Frozen tensor changed: {key}'
        frozen+=1
    if key in ('final_layer.weight','final_layer_tactile.weight','flare_queries') or key.endswith(('layers.0.self_attn.q_proj_action.weight','layers.0.self_attn.q_proj_tactile.weight','layers.0.self_attn.q_proj.weight')):
        delta=(v.float()-before[key].float()).abs()
        updates[key]={'changed_elements':int((delta>0).sum()),'total_elements':delta.numel(),'mean_abs_change':delta.mean().item(),'max_abs_change':delta.max().item()}
assert updates and all(x['changed_elements']>0 for x in updates.values()),updates
for name in ('config.json','training_args.json','stats_data.json'):assert (a.checkpoint/name).is_file(),name
assert (a.checkpoint/'processor').is_dir()
report={'matching_tensor_keys_and_shapes':checked,'all_tensors_finite':True,'unchanged_frozen_tensors':frozen,'updated_parameters':updates,'model_bytes':(a.checkpoint/'model.pt').stat().st_size}
(a.checkpoint.parent/'checkpoint_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
