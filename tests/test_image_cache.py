"""Check real cache-validation helpers without video/model dependencies."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'overlay/utils/export_lerobot_to_json_images.py').read_text())
helpers = [n for n in source.body if isinstance(n, ast.FunctionDef)
           and n.name in ('suffix', 'validate_image_inventory', 'remaining_free_gib')]
ns = dict(os=os, json=json, RGB=['observation.images.head'], VIEWS=['observation.images.head'])
exec(compile(ast.Module(body=helpers, type_ignores=[]), '<cache helpers>', 'exec'), ns)


class CacheTests(unittest.TestCase):
    def test_missing_or_empty_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'images/episode_0'
            folder.mkdir(parents=True)
            first = folder/'image0_head.png'
            second = folder/'image1_head.png'
            first.write_bytes(b'nonempty')
            episodes = [{'episode_index': 0, 'length': 2}]
            with self.assertRaisesRegex(RuntimeError, 'Incomplete image cache'):
                ns['validate_image_inventory'](root, episodes)
            second.touch()
            with self.assertRaises(RuntimeError): ns['validate_image_inventory'](root, episodes)
            second.write_bytes(b'nonempty')
            ns['validate_image_inventory'](root, episodes)

    def test_retry_accounts_for_completed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(ns['remaining_free_gib'](root, 140), 140)
            (root/'jobs').mkdir()
            (root/'jobs/done.json').write_text(json.dumps({'png_bytes': 100 * 2**30}))
            self.assertEqual(ns['remaining_free_gib'](root, 140), 40)
            self.assertEqual(ns['remaining_free_gib'](root, 105), 8)
            self.assertEqual(ns['remaining_free_gib'](root, 0), 0)


if __name__ == '__main__': unittest.main()
