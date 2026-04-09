import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ConfigModuleTests(unittest.TestCase):
    def test_config_exports_all_constants(self):
        import config
        # Verify all expected constants are exported
        self.assertTrue(hasattr(config, "MAX_ITERATIONS"))
        self.assertTrue(hasattr(config, "MAX_HISTORY_MESSAGES"))
        self.assertTrue(hasattr(config, "TOOL_TIMEOUT_SEC"))
        self.assertTrue(hasattr(config, "MEMORY_AUTO_LOG"))
        self.assertTrue(hasattr(config, "MEMORY_PRIVATE_SESSION"))
        self.assertTrue(hasattr(config, "MAX_MEMORY_LOG_CHARS"))
        self.assertTrue(hasattr(config, "SESSION_STATE_ENABLED"))
        self.assertTrue(hasattr(config, "SESSION_STATE_FILE"))
        self.assertTrue(hasattr(config, "RUNTIME_EXECUTION_GUIDE"))
        self.assertTrue(hasattr(config, "MODEL"))
        self.assertTrue(hasattr(config, "MODEL_PROVIDER"))

    def test_config_constants_have_valid_ranges(self):
        import config
        self.assertGreaterEqual(config.MAX_ITERATIONS, 2)
        self.assertLessEqual(config.MAX_ITERATIONS, 40)
        self.assertGreaterEqual(config.MAX_HISTORY_MESSAGES, 20)
        self.assertGreater(config.TOOL_TIMEOUT_SEC, 0)
        self.assertGreaterEqual(config.MAX_MEMORY_LOG_CHARS, 600)
        self.assertIsInstance(config.SESSION_STATE_FILE, Path)

    def test_config_model_provider_is_known(self):
        import config
        self.assertIn(config.MODEL_PROVIDER, {"openai", "azure"})

    def test_config_runtime_execution_guide_is_nonempty(self):
        import config
        self.assertIsInstance(config.RUNTIME_EXECUTION_GUIDE, str)
        self.assertGreater(len(config.RUNTIME_EXECUTION_GUIDE), 50)

    def test_create_client_and_model_returns_tuple(self):
        import config
        result = config.create_client_and_model()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        # (client_or_none, model_name, provider, error_string)
        _client, model_name, provider, error = result
        self.assertIsInstance(model_name, str)
        self.assertIn(provider, {"openai", "azure"})
        self.assertIsInstance(error, str)
