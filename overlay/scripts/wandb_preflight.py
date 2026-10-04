"""Check W&B credentials/connectivity before training; optionally test metric upload."""
import argparse
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def check(smoke=False):
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "online":
        return {"status": "skipped", "mode": mode}
    from packaging.version import Version

    version = importlib.metadata.version("wandb")
    if Version(version) < Version("0.22.3"):
        raise RuntimeError("W&B SDK >=0.22.3 is required; install wandb==0.22.3 in the training environment.")
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        raise RuntimeError("Missing W&B credential; source scripts/wandb_env.sh first.")
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    if not base_url.startswith("https://"):
        raise RuntimeError("W&B credential checks require an HTTPS endpoint.")
    request = urllib.request.Request(
        base_url + "/graphql",
        data=json.dumps({"query": "query { viewer { username defaultEntity { name } } }"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Basic " + base64.b64encode(("api:" + key).encode()).decode()},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"W&B authentication failed (HTTP {exc.code}).") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("W&B endpoint is unreachable; check the training server network.") from None
    viewer = payload.get("data", {}).get("viewer")
    if payload.get("errors") or not viewer:
        raise RuntimeError("W&B did not return an authenticated account.")
    entity = os.environ.get("WANDB_ENTITY") or (viewer.get("defaultEntity") or {}).get("name")
    report = {"status": "PASS", "mode": mode, "sdk_version": version,
              "username": viewer["username"], "entity": entity,
              "project": os.environ.get("WANDB_PROJECT", "trex-posttrain")}
    if smoke:
        import wandb

        run = wandb.init(project=report["project"], entity=entity, mode="online",
                         name="handoff-monitoring-check", job_type="monitoring-check",
                         config={"purpose": "Verify monitoring only; no model training"},
                         settings=wandb.Settings(init_timeout=60))
        run.define_metric("optimizer_step")
        run.define_metric("*", step_metric="optimizer_step")
        run.log({"monitoring/upload_check": 1, "optimizer_step": 0})
        run.log({"monitoring/validation_check": 1, "optimizer_step": 0})
        report.update(smoke_run_url=run.url, smoke_run_path=f"{run.entity}/{run.project}/{run.id}")
        run.finish()
        uploaded = wandb.Api(timeout=20).run(report["smoke_run_path"])
        if any(uploaded.summary.get(name) != 1 for name in
               ("monitoring/upload_check", "monitoring/validation_check")):
            raise RuntimeError("W&B smoke metric was not confirmed by read-back.")
        report["metric_readback"] = "PASS"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.smoke)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
