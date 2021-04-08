"""
Test the scalarstop.dataclasses module.
"""
import dataclasses
import unittest

import scalarstop as sp


def _serialize_dataclass():
    """Serialize a dataclass."""

    @sp.dataclasses.dataclass
    class blob:
        """Our example dataclass."""

        a: int
        b: int

    b = blob(1, 2)
    return sp.pickle.dumps(b)


def _load_serialized_fields(s):
    """Deserialize a dataclass."""
    return sp.pickle.loads(s)


class TestDataclass(unittest.TestCase):
    """Test our custom dataclass code."""

    def setUp(self):
        """Serialize and deserialize a dataclass."""
        self.dc = _load_serialized_fields(_serialize_dataclass())

    def test_dataclasses_library__is_broken(self):
        """Test that the dataclasses library doesn't work on cloudpickle'd dataclasses."""
        self.assertEqual(dataclasses.fields(self.dc), ())
        self.assertEqual(dataclasses.astuple(self.dc), ())
        self.assertEqual(dataclasses.asdict(self.dc), dict())

    def test_our_code_works(self):
        """Test that our versions of dataclasses code does work."""
        self.assertEqual(len(sp.dataclasses.fields(self.dc)), 2)
        self.assertEqual(sp.dataclasses.astuple(self.dc), (1, 2))
        self.assertEqual(sp.dataclasses.asdict(self.dc), dict(a=1, b=2))
