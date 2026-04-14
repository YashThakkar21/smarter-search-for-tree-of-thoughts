import argparse
from dotenv import load_dotenv
<<<<<<< HEAD
from debug_utils import (
    explain_evaluation,
    print_plain_solution,
    print_selected_solution,
)
from tot.methods.bfs import solve as bfs_solve
from tot.methods.mcts import solve as mcts_solve
=======
from tot.methods.mcts import solve
>>>>>>> c9a4fb8 (Modified the game24.py file and the minimal_run file, added the tinker to requirements)
from tot.tasks.game24 import Game24Task

load_dotenv()

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.7,
    task="game24",
<<<<<<< HEAD

=======
>>>>>>> c9a4fb8 (Modified the game24.py file and the minimal_run file, added the tinker to requirements)
    naive_run=False,
    prompt_sample=None,
    method_generate="propose",
    method_evaluate="value",

<<<<<<< HEAD
    # ranges from 0 - 1362
    task_start_index=40,
    task_end_index=41,

    # Modify the Search Algorithm
    search_method="mcts",

    # BFS knobs
=======
    # BFS-only knobs; harmless if left in, but not used by MCTS
>>>>>>> c9a4fb8 (Modified the game24.py file and the minimal_run file, added the tinker to requirements)
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5,

<<<<<<< HEAD
    # MCTS knobs
    n_mcts_simulations=500,
    n_mcts_evaluate_sample=1,
    mcts_exploration=0.5,
)

task = Game24Task()
solve = bfs_solve if args.search_method == "bfs" else mcts_solve

# Debuging
debug = False

for i in range(args.task_start_index, args.task_end_index):
    results, info = solve(args, task, i, to_print=debug)

    if debug:
        solution = print_selected_solution(results)
        explain_evaluation(task, i, solution)
        print()
    else:
        print_plain_solution(results)
=======
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
>>>>>>> c9a4fb8 (Modified the game24.py file and the minimal_run file, added the tinker to requirements)
