import argparse
from dotenv import load_dotenv
from tot.methods.mcts import solve
from tot.tasks.game24 import Game24Task

load_dotenv()

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.7,
    task="game24",
    naive_run=False,
    prompt_sample=None,
    method_generate="propose",
    method_evaluate="value",

    # BFS-only knobs; harmless if left in, but not used by MCTS
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5,

    # your index range
    task_start_index=900,
    task_end_index=901,

    # MCTS knobs
    n_mcts_simulations=100,
    mcts_exploration=1.4,
)

task = Game24Task()
ys, info = solve(args, task, 900, to_print=True)
# solution = ys[0] if ys else ""
print(ys)