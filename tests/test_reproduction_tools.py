from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.frame_io import import_cv2_np
from scripts.relocate_test_index import locate_input
from scripts.verify_exact_submission import verify, write_manifest
from wan_ticft.evaluation.ltx_detail_donor import transfer_detail


class ReproductionToolTests(unittest.TestCase):
    def test_locate_input_supports_nested_dataset_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "BDD" / "test" / "video3806" / "input"
            expected.mkdir(parents=True)
            self.assertEqual(locate_input(root, "video3806"), expected.resolve())

    def test_t55_detail_transfer_is_deterministic(self) -> None:
        try:
            cv2, _ = import_cv2_np()
        except SystemExit as exc:
            self.skipTest(str(exc))
        base = np.full((32, 48, 3), 100, dtype=np.uint8)
        donor = base.copy()
        donor[:, 24:] = 160
        first = transfer_detail(cv2, base, [donor], alpha=0.55, sigma=2.2)
        second = transfer_detail(cv2, base, [donor], alpha=0.55, sigma=2.2)
        self.assertEqual(first.shape, base.shape)
        self.assertTrue(np.array_equal(first, second))

    def test_sha256_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "submission"
            sample = root / "sample"
            sample.mkdir(parents=True)
            (sample / "0.png").write_bytes(b"frame-zero")
            (sample / "1.png").write_bytes(b"frame-one")
            manifest = Path(directory) / "sha256.txt"
            write_manifest(root, manifest)
            verify(root, manifest)


if __name__ == "__main__":
    unittest.main()
