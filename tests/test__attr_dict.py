"""Unit tests for the scalarstop._attr_dict module."""
import doctest
import json
import unittest

from scalarstop import _attr_dict
from scalarstop._attr_dict import AttrDict


def load_tests(loader, tests, ignore):  # pylint: disable=unused-argument
    """Have the unittest loader also run doctests."""
    tests.addTests(doctest.DocTestSuite(_attr_dict))
    return tests


class TestAttrDict(unittest.TestCase):
    """Unit tests for scalarstop._attr_dict.AttrDict."""

    def test_simple(self):
        """Test that AttrDict works."""
        ad = AttrDict(a=1, b=2, clear=3, values=4)
        self.assertEqual(ad.a, 1)
        self.assertEqual(ad.b, 2)
        self.assertEqual(ad.clear, 3)
        self.assertEqual(ad.values, 4)

    def test_keys_with_special_chars(self):
        """
        Test that AttrDict can have dictionary keys with special characters.

        In this case, "special" characters refers to anything that
        cannot be a key in a ``dict()`` expression.
        """
        ad = AttrDict(**{"blob-a": 1, "omg.wtf": 2})
        self.assertEqual(ad["blob-a"], 1)
        self.assertEqual(ad["omg.wtf"], 2)

    def test_serialize_to_json(self):
        """Test that we can serialize an AttrDict to JSON."""
        ad = AttrDict(a=1, b=2, clear=3, values=4)
        loaded = json.loads(json.dumps(ad))
        self.assertEqual(ad, loaded)
