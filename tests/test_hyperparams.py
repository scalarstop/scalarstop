"""Unit tests for the hyperparams module."""
import unittest

import scalarstop as sp


class TestHyperparams(unittest.TestCase):
    """Unit tests for the Hyperparams dataclass."""

    def test_hyperparams_are_not_frozen(self):
        """
        Test that hyperparams are not a frozen dataclass.

        Frozen dataclasses cannot mutate themselves in
        their ``__post_init__()`` method, and we want
        to allow such mutations.
        """

        @sp.dataclass
        class Hyperparams(sp.HyperparamsType):
            """Test hyperparams."""

            a: int
            b: str

            def __post_init__(self):
                self.b = self.b + str(self.a)

        hp = Hyperparams(a=1, b="hi")
        self.assertEqual(hp.a, 1)
        self.assertEqual(hp.b, "hi1")
