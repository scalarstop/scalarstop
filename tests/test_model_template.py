"""Unit tests for scalarstop.model_template."""
import doctest
import unittest

import tensorflow as tf

import scalarstop as sp


def load_tests(loader, tests, ignore):  # pylint: disable=unused-argument
    """Have the unittest loader also run doctests."""
    tests.addTests(doctest.DocTestSuite(sp.model_template))
    return tests


class MyModelTemplate(sp.ModelTemplate):
    """Our example model template for testing."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """HYperparams for MyModelTemplate."""

        a: int
        b: str = "hi"

    def set_model(self):
        """Setting a new model."""
        model = tf.keras.Sequential(
            layers=[tf.keras.layers.Dense(units=self.hyperparams.a)]
        )
        model.compile()
        return model


class MyModelTemplateForgotHyperparams(sp.ModelTemplate):
    """See what happens when we don't define hyperparams."""

    Hyperparams = None  # type: ignore


class TestModelTemplate(unittest.TestCase):
    """Tests for :py:class:`scalarstop.ModelTemplate`."""

    def test_name(self):
        """Test that names work."""
        model_template_1 = MyModelTemplate(
            hyperparams=dict(a=1),
        )
        model_template_2 = MyModelTemplate(hyperparams=dict(a=1, b="hi"))
        for i, model_template in enumerate((model_template_1, model_template_2)):
            with self.subTest(f"model_template_{i}"):
                self.assertEqual(
                    model_template.name, "MyModelTemplate-naro6iqyw9whazvkgp4w3qa2"
                )
                self.assertEqual(model_template.group_name, "MyModelTemplate")
                self.assertEqual(
                    sp.dataclasses.asdict(model_template.hyperparams), dict(a=1, b="hi")
                )

    def test_missing_hyperparams_class(self):
        """Test what happens when the hyperparams class itself is missing."""
        with self.assertRaises(sp.exceptions.YouForgotTheHyperparams):
            MyModelTemplateForgotHyperparams()
