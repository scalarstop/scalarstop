"""
Unit tests for ScalarStop.
"""
import os

# TensorFlow's C and C++ code emits a ton of
# annoying log messages that clutter up our tests.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
