"""
Test the scalarstop.tf_config module.
"""
import json
import unittest

import scalarstop as sp


class TestTFConfig(unittest.TestCase):
    """
    Unit tests for sp.TFConfig.
    """

    def test_env_var_is_missing_1(self):
        """
        Test our default values for when TF_CONFIG is missing.
        """
        cfg_str = "{}"
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.to_dict, {})
        self.assertEqual(cfg.num_nodes("all"), 0)
        self.assertEqual(cfg.num_nodes("worker"), 0)
        self.assertEqual(cfg.num_nodes("ps"), 0)
        self.assertEqual(cfg.task_type, None)
        self.assertEqual(cfg.task_index, 0)
        self.assertTrue(cfg.is_chief)

    def test_env_var_is_missing_2(self):
        """
        Test our default values for when TF_CONFIG is missing.
        """
        cfg_str = ""
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.to_dict, {})
        self.assertEqual(cfg.num_nodes("all"), 0)
        self.assertEqual(cfg.num_nodes("worker"), 0)
        self.assertEqual(cfg.num_nodes("ps"), 0)
        self.assertEqual(cfg.task_type, None)
        self.assertEqual(cfg.task_index, 0)
        self.assertTrue(cfg.is_chief)

    def test_parameter_server_1(self):
        """
        Test a parameter server TF_CONFIG where the current node
        is not the chief.
        """
        cfg_str = json.dumps(
            {
                "cluster": {
                    "worker": ["host1:port", "host2:port", "host3:port"],
                    "ps": ["host4:port", "host5:port"],
                },
                "task": {"type": "worker", "index": 1},
            }
        )
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.num_nodes("all"), 5)
        self.assertEqual(cfg.num_nodes("worker"), 3)
        self.assertEqual(cfg.num_nodes("ps"), 2)
        self.assertEqual(cfg.task_type, "worker")
        self.assertEqual(cfg.task_index, 1)
        self.assertFalse(cfg.is_chief)

    def test_parameter_server_2(self):
        """
        Test a parameter server TF_CONFIG where the current node
        is the chief.
        """
        cfg_str = json.dumps(
            {
                "cluster": {
                    "chief": ["host1:port"],
                    "ps": ["host2:port"],
                    "worker": [
                        "host3:port",
                        "host4:port",
                    ],
                },
                "environment": "cloud",
                "task": {
                    "cloud": "...",
                    "index": 0,
                    "trial": "1",
                    "type": "worker",
                },
            }
        )
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.num_nodes("all"), 4)
        self.assertEqual(cfg.num_nodes("chief"), 1)
        self.assertEqual(cfg.num_nodes("worker"), 2)
        self.assertEqual(cfg.num_nodes("ps"), 1)
        self.assertEqual(cfg.task_type, "worker")
        self.assertEqual(cfg.task_index, 0)
        self.assertTrue(cfg.is_chief)

    def test_worker_1(self):
        """
        Test a worker TF_CONFIG where the current node
        is the chief.
        """
        cfg_str = json.dumps(
            {
                "cluster": {"worker": ["host1:port", "host2:port"]},
                "task": {"type": "worker", "index": 0},
            }
        )
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.num_nodes("all"), 2)
        self.assertEqual(cfg.num_nodes("chief"), 0)
        self.assertEqual(cfg.num_nodes("worker"), 2)
        self.assertEqual(cfg.num_nodes("ps"), 0)
        self.assertEqual(cfg.task_type, "worker")
        self.assertEqual(cfg.task_index, 0)
        self.assertTrue(cfg.is_chief)

    def test_worker_2(self):
        """
        Test a worker TF_CONFIG where the current node is not
        the chief.
        """
        cfg_str = json.dumps(
            {
                "cluster": {"worker": ["host1:port", "host2:port"]},
                "task": {"type": "worker", "index": 1},
            }
        )
        cfg = sp.TFConfig(tf_config_str=cfg_str)
        self.assertEqual(cfg.num_nodes("all"), 2)
        self.assertEqual(cfg.num_nodes("chief"), 0)
        self.assertEqual(cfg.num_nodes("worker"), 2)
        self.assertEqual(cfg.num_nodes("ps"), 0)
        self.assertEqual(cfg.task_type, "worker")
        self.assertEqual(cfg.task_index, 1)
        self.assertFalse(cfg.is_chief)
