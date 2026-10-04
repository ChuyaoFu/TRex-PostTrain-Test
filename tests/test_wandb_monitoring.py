import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "overlay/scripts/wandb_env.sh"


class MonitoringTests(unittest.TestCase):
    def shell(self, env, trace=False):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
        clean.update(env)
        command = 'source "$1" || exit $?; set +x; test -n "${WANDB_API_KEY:-}" || test "$WANDB_MODE" != online'
        return subprocess.run(["bash", "-x" if trace else "-c", *(["-c"] if trace else []),
                               command, "test", str(HELPER)], env=clean, capture_output=True, text=True)

    def test_key_file_does_not_leak_with_shell_tracing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            token = "test-credential-never-print"
            path.write_text(token + "\n")
            result = self.shell({"WANDB_API_KEY_FILE": str(path)}, trace=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(token, result.stdout + result.stderr)

    def test_missing_key_stops_online_launch(self):
        result = self.shell({"WANDB_API_KEY_FILE": "/nonexistent/trex-test-key"})
        self.assertNotEqual(result.returncode, 0)

    def test_environment_key_takes_precedence(self):
        result = self.shell({"WANDB_API_KEY": "test-environment-key",
                             "WANDB_API_KEY_FILE": "/nonexistent/trex-test-key"})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_offline_requires_no_key(self):
        self.assertEqual(self.shell({"WANDB_MODE": "offline"}).returncode, 0)

    def test_metrics_keep_optimizer_step_instead_of_reusing_wandb_step(self):
        tree = ast.parse((ROOT / "overlay/scripts/train.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "wandb" and n.func.attr == "log"]
        self.assertEqual(len(calls), 2)
        class Recorder:
            def __init__(self): self.logs = []
            def log(self, payload): self.logs.append(payload)
        recorder = Recorder()
        namespace = dict(wandb=recorder, step=500, optimizer_steps=500,
                         result={"val/action_loss": 0.2}, log_dict={"action_loss": 0.3})
        for call in calls:
            module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[]))
            exec(compile(module, "<log>", "exec"), namespace)
        self.assertEqual([r["optimizer_step"] for r in recorder.logs], [500, 500])
        self.assertTrue(any("val/action_loss" in r for r in recorder.logs))
        self.assertTrue(any("action_loss" in r for r in recorder.logs))


if __name__ == "__main__":
    unittest.main()
