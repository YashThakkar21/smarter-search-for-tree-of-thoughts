import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import test_evaluator

class DummyTask:
    def __init__(self, prompts):
        self.prompts = prompts

    def get_ensemble_prompts(self, x, y):
        return self.prompts

    def extract_numerical_score(self, output):
        data = json.loads(output)
        return float(data["score"])


class EvaluatorUnitTests(unittest.TestCase):
    def test_hardcoded_terminal_prompts_skip_llm_and_log_scores(self):
        prompts = [
            '{"reasoning": "Already solved", "score": 10}',
            '{"reasoning": "Already solved", "score": 10}',
        ]
        task = DummyTask(prompts)
        x = "4 4 6 8"
        y = "12 * 2 = 24 (left: 24)\n"
        state_str = y.strip()
        expected_hash = hashlib.md5(state_str.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                with patch("test_evaluator.bfs.gpt") as mock_gpt:
                    score = test_evaluator._get_ensemble_value(task, x, y, to_print=False)
                mock_gpt.assert_not_called()

                self.assertEqual(score, 1.0)

                log_path = os.path.join(tmp_dir, "eval_logs", f"{expected_hash}.json")
                self.assertTrue(os.path.exists(log_path))

                with open(log_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                self.assertEqual(payload["state"], state_str)
                self.assertEqual(payload["average_score"], 10.0)
                self.assertEqual(payload["final_normalized_score"], 1.0)
                self.assertEqual(len(payload["evaluations"]), 2)
                for item in payload["evaluations"]:
                    self.assertEqual(item["note"], "Hardcoded terminal state")
                    self.assertEqual(item["extracted_score"], 10.0)
            finally:
                os.chdir(cwd)

    def test_llm_prompts_use_model_and_normalize_average(self):
        prompts = ["prompt-a", "prompt-b"]
        task = DummyTask(prompts)
        x = "4 4 6 8"
        y = ""
        state_str = x
        expected_hash = hashlib.md5(state_str.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                with patch(
                    "test_evaluator.bfs.gpt",
                    side_effect=[["{\"reasoning\":\"ok\",\"score\":8}"], ["{\"reasoning\":\"ok\",\"score\":6}"]],
                ) as mock_gpt:
                    score = test_evaluator._get_ensemble_value(task, x, y, to_print=False)

                self.assertEqual(mock_gpt.call_count, 2)
                self.assertAlmostEqual(score, 0.7)

                log_path = os.path.join(tmp_dir, "eval_logs", f"{expected_hash}.json")
                self.assertTrue(os.path.exists(log_path))

                with open(log_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                self.assertEqual(payload["average_score"], 7.0)
                self.assertEqual(payload["final_normalized_score"], 0.7)
                self.assertEqual(len(payload["evaluations"]), 2)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()