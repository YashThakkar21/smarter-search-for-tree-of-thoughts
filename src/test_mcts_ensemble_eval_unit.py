import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

from tot.methods.mcts import Node, _evaluate


class DummyTaskWithEnsemble:
    def get_ensemble_prompts(self, x, y):
        return ["p1", "p2", "p3"]

    def extract_numerical_score(self, output):
        return float(output)


class DummyTaskNoEnsemble:
    pass


class MCTSEnsembleEvalUnitTests(unittest.TestCase):
    def test_average_aggregation(self):
        args = SimpleNamespace(mcts_use_ensemble_evaluator=True, mcts_ensemble_aggregation="average")
        task = DummyTaskWithEnsemble()
        node = Node("4 + 8 = 12 (left: 4 6 12)\n")

        with patch("tot.methods.mcts.bfs.gpt", side_effect=[["8"], ["6"], ["7"]]):
            reward = _evaluate(node, args, task, x="4 4 6 8", idx=0)

        self.assertAlmostEqual(reward, 0.7)

    def test_majority_aggregation(self):
        args = SimpleNamespace(mcts_use_ensemble_evaluator=True, mcts_ensemble_aggregation="majority")
        task = DummyTaskWithEnsemble()
        node = Node("4 + 8 = 12 (left: 4 6 12)\n")

        with patch("tot.methods.mcts.bfs.gpt", side_effect=[["6"], ["8"], ["8"]]):
            reward = _evaluate(node, args, task, x="4 4 6 8", idx=0)

        self.assertAlmostEqual(reward, 0.8)

    def test_fallback_to_legacy_value_when_ensemble_unavailable(self):
        args = SimpleNamespace(mcts_use_ensemble_evaluator=True, mcts_ensemble_aggregation="average", n_mcts_evaluate_sample=1)
        task = DummyTaskNoEnsemble()
        node = Node("4 + 8 = 12 (left: 4 6 12)\n")

        with patch("tot.methods.mcts.bfs.get_value", return_value=1.0):
            reward = _evaluate(node, args, task, x="4 4 6 8", idx=0)

        self.assertAlmostEqual(reward, 0.5)


if __name__ == "__main__":
    unittest.main()