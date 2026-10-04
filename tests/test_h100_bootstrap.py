"""Exercise the actual bootstrap fragment without installing software or using GPUs."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_reused_setup_does_not_demand_fresh_disk_allocation(self):
        source = (ROOT / 'overlay/scripts/posttrain_h100.sh').read_text()
        start = source.index('        SETUP_FREE_GIB=200')
        end = source.index('\nPYSPACE', start) + len('\nPYSPACE')
        fragment = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / 'bin'
            tools.mkdir()
            python = tools / 'python3'
            python.write_text(f'#!{sys.executable}\nimport shutil, sys\n'
                              'from types import SimpleNamespace\n'
                              'shutil.disk_usage = lambda path: SimpleNamespace(free=100*2**30)\n'
                              'sys.argv = sys.argv[1:]\n'
                              'exec(sys.stdin.read())\n')
            python.chmod(0o755)
            env = os.environ.copy()
            env.pop('MIN_FREE_GIB', None)
            env.update(ROOT=str(root), TRAIN_VENV=str(root / 'train'),
                       DATA_VENV=str(root / 'data'), PATH=str(tools)+os.pathsep+env['PATH'])
            fresh = subprocess.run(['bash', '-euc', fragment], env=env, capture_output=True, text=True)
            self.assertNotEqual(fresh.returncode, 0)
            for name in ('train', 'data'):
                binary = root / name / 'bin/python'
                binary.parent.mkdir(parents=True)
                binary.symlink_to(sys.executable)
            reused = subprocess.run(['bash', '-euc', fragment], env=env, capture_output=True, text=True)
            self.assertEqual(reused.returncode, 0, reused.stderr)

    def test_setup_failure_survives_pipeline_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            source = ROOT / 'overlay/scripts/posttrain_h100.sh'
            launcher = root / 'scripts/posttrain_h100.sh'
            launcher.write_text(source.read_text())
            tools = root / 'bin'
            tools.mkdir()
            for name, status in [('nvidia-smi', 0), ('uv', 37)]:
                executable = tools / name
                executable.write_text(f'#!/bin/sh\nexit {status}\n')
                executable.chmod(0o755)
            env = os.environ.copy()
            for key in ('SKIP_SETUP', 'TRAIN_VENV', 'DATA_VENV', 'OUTPUT_DIR', 'LOG_DIR', 'LOG_FILE'):
                env.pop(key, None)
            env.update(MIN_FREE_GIB='0', RUN_NAME='failure-test', PATH=str(tools)+os.pathsep+env['PATH'])
            result = subprocess.run(['bash', str(launcher), 'prepare'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
            log = (root / 'logs/failure-test.pipeline.log').read_text()
            self.assertIn('Pipeline failed', log)
            self.assertNotIn('Launching: global batch', log)

    def test_existing_uv_creates_constraint_directory(self):
        source = (ROOT / "overlay/scripts/posttrain_h100.sh").read_text()
        start = source.index('    if [[ "${SKIP_SETUP:-0}" != 1 ]]')
        end = source.index('        TREX_DATA_VENV=', start)
        fragment = source[start:end] + '\nfi\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "bin"
            tools.mkdir()
            uv = tools / "uv"
            uv.write_text('#!/bin/sh\nexit 0\n')
            uv.chmod(0o755)
            env = os.environ.copy()
            env.update(ROOT=str(root), TRAIN_VENV=str(root / 'train'),
                       DATA_VENV=str(root / 'data'), MIN_FREE_GIB='0', SKIP_SETUP='0',
                       PATH=str(tools) + os.pathsep + env['PATH'])
            result = subprocess.run(['bash', '-euc', fragment], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            constraints = (root / '.tools/h100-torch-constraints.txt').read_text()
            self.assertEqual(constraints, 'torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\n')


if __name__ == '__main__':
    unittest.main()
