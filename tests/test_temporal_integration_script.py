import unittest

import numpy as np

from tools.run_temporal_integration_test import (
    integrate_array,
    make_post_upchan_fixture,
)


class TemporalIntegrationScriptTest(unittest.TestCase):
    def test_post_upchan_fixture_reduces_time_by_32(self):
        raw = np.ones((64, 2, 3), dtype=np.float32)

        fixture = make_post_upchan_fixture(raw)

        self.assertEqual(fixture.shape, (2, 2, 3))
        np.testing.assert_array_equal(fixture, np.full((2, 2, 3), 32.0))

    def test_independent_paths_produce_48_windows(self):
        raw = np.ones((15360, 1, 1), dtype=np.float32)
        post_upchan = make_post_upchan_fixture(raw)

        direct = integrate_array(raw, 320)
        after_upchan = integrate_array(post_upchan, 10)

        self.assertEqual(post_upchan.shape, (480, 1, 1))
        self.assertEqual(direct.shape, (48, 1, 1))
        self.assertEqual(after_upchan.shape, (48, 1, 1))
        np.testing.assert_array_equal(direct, after_upchan)
        np.testing.assert_array_equal(direct, np.full((48, 1, 1), 320.0))

    def test_integration_rejects_incompatible_time_length(self):
        with self.assertRaises(ValueError):
            integrate_array(np.zeros((5, 1, 1), dtype=np.float32), 2)


if __name__ == "__main__":
    unittest.main()
