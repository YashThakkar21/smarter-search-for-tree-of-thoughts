import argparse
from dotenv import load_dotenv
from tot.methods.bfs import solve as bfs_solve
from tot.methods.mcts import solve as mcts_solve
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
    task_start_index=0,
    task_end_index=1362,
    
    # Modify the Search Algorithm
    search_method="bfs",

    # BFS-only knobs; harmless if left in, but not used by MCTS
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5,

    # MCTS knobs
    n_mcts_simulations=500,
    mcts_exploration=1.4,
)

task = Game24Task()
solve = bfs_solve if args.search_method == "bfs" else mcts_solve

# For debugging purposes (set to_print to True print all of ys): 
debug = False
for i in range(args.task_start_index, args.task_end_index):
    res, info = solve(args, task, i, to_print=debug)
    if debug:
        print(res)
        print('\n')
    else:
        solution = res[0] if res else ""
        print(solution)
        print('\n')