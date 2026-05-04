import argparse
from dotenv import load_dotenv
from debug_utils import (
    print_plain_solution,
    print_selected_solution,
)
from tot.methods.bfs import solve as bfs_solve
from tot.methods.mcts import solve as mcts_solve
from tot.tasks.cryptic import CrypticTask

load_dotenv()

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.3,
    task="bfs",

    naive_run=False,
    prompt_sample=None,
    method_generate="propose",
    method_evaluate="value",

    # ranges from 0 - 1362 for 24
    task_start_index=22,
    task_end_index=42,

    # Modify the Search Algorithm
    search_method="mcts",

    # BFS knobs
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=1,
    n_select_sample=2,

    # MCTS knobs
    n_mcts_simulations=100,
    n_mcts_evaluate_sample=1,
    mcts_exploration=0.5,
)

task = CrypticTask()
solve = bfs_solve if args.search_method == "bfs" else mcts_solve

# Debugging
debug = True
sum = 0

for i in range(args.task_start_index, args.task_end_index):
    results, info = solve(args, task, i, to_print=debug)

    if debug:
        # Get the selected solution string
        solution = print_selected_solution(results)
        if solution is None and results:
            solution = results[0]
        elif not results:
            solution = ""

        print(f"Puzzle {i}: {task.get_input(i)}")

        # --- THE FIX: Use CrypticTask's built-in evaluation ---
        eval_result = task.test_output(i, solution)

        print(f"Gold Answer: {eval_result['gold']}")
        print(f"Extracted Answer: {eval_result['extracted']}")

        if eval_result['r'] == 1:
            print("Evaluation: CORRECT!")
            sum += 1
        else:
            print("Evaluation: INCORRECT")

        if eval_result.get('parse_failed'):
            print("(Parse failed - could not extract an answer from the model's output)")

        print()
    else:
        print_plain_solution(results)