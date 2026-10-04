"""Store a W&B token outside the repository with owner-only permissions."""
import getpass
import os
from pathlib import Path


if __name__ == "__main__":
    target = Path(os.environ.get("WANDB_API_KEY_FILE", "~/.config/trex/wandb_api_key")).expanduser()
    key = getpass.getpass("W&B API token (hidden): ").strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit("Enter a nonempty token without whitespace.")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(key + "\n")
    print(f"W&B credential saved with mode 0600: {target}")
