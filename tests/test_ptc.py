import unittest

import ptc


class MovingAverageTests(unittest.TestCase):
    def test_uses_available_values_until_window_is_full(self):
        self.assertEqual(ptc.moving_average([2.0, 4.0, 6.0], window=10), [2.0, 3.0, 4.0])

    def test_drops_values_outside_window(self):
        self.assertEqual(
            ptc.moving_average([1.0, 2.0, 3.0, 4.0], window=3),
            [1.0, 1.5, 2.0, 3.0],
        )

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ValueError):
            ptc.moving_average([1.0], window=0)


if __name__ == "__main__":
    unittest.main()
