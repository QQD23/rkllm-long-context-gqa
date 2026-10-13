"""Host tests for runtime gating, environment isolation and frozen source integrity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_runtime import verify
from run import clean_environment


class HostTests(unittest.TestCase):
    def test_runtime_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.so'
            p.write_bytes(b'wrong build')
            with self.assertRaises(ValueError):
                verify(tmp, {'runtime_sha256': {'test.so': hashlib.sha256(b'expected').hexdigest()}})

    def test_runtime_missing_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                verify(tmp, {'runtime_sha256': {'missing.so': '0' * 64}})

    def test_baseline_removes_inherited_instrumentation(self):
        with patch.dict(os.environ, {'LD_PRELOAD': '/old/patch.so', 'LD_AUDIT': '/old/audit.so', 'RKDUANYADIAN_MODE': 'tile', 'WP_INFERENCE_GATE': 'old-gate'}):
            env = clean_environment(Path('/runtime'), 'baseline', Path('/patch.so'))
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('LD_AUDIT', env)
        self.assertNotIn('RKDUANYADIAN_MODE', env)
        self.assertNotIn('WP_INFERENCE_GATE', env)
        self.assertEqual(env['LD_LIBRARY_PATH'], str(Path('/runtime')))

    def test_tile_selects_only_explicit_patch(self):
        with patch.dict(os.environ, {'LD_PRELOAD': '/old/controller.so'}):
            env = clean_environment(Path('/runtime'), 'tile', Path('/selected.so'))
        self.assertEqual(env['LD_PRELOAD'], str(Path('/selected.so')))
        self.assertEqual(env['RKDUANYADIAN_MODE'], 'tile')

    def test_frozen_source_hashes(self):
        manifest = {'src/rkllm_pv_gqa.cpp': {'sha256': 'e7932c0c269405048091b6cafa7c02dbe09527650a090add08b150d83e8c45c3', 'origin': 'runtime_versions\\v23\\src\\rkllm_pv_gqa.cpp', 'copied_without_changes': True}, 'src/pv_gqa_engine.h': {'sha256': 'e481363eb6b324bbb74dfb1a2632228aeedb579fd6b5d27c8e0fb5f6d7a98dd6', 'origin': 'runtime_versions\\v23\\src\\pv_gqa_engine.h', 'copied_without_changes': True}, 'tests/pv_gqa_probe.cpp': {'sha256': 'fb51655dc0d5429fff7ad3c103b61ff324a7638b2986c2d5851984b18d15c9ac', 'origin': '03_rkllm_context_cliff_fix\\src\\pv_gqa_probe.cpp', 'copied_without_changes': True}, 'src/position_worker.cpp': {'sha256': 'de980d9939f62de52f5f8b1f9de62845056b599f6342dc34c883c71be4594fb7', 'origin': '03_rkllm_context_cliff_fix\\src\\position_worker.cpp', 'copied_without_changes': True}}
        for name, item in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), item['sha256'], name)



if __name__ == '__main__':
    unittest.main()
