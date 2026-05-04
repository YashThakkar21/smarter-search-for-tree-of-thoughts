import argparse
import contextlib
import io
from dotenv import load_dotenv
from debug_utils import (
    _select_solution,
    explain_evaluation,
    print_plain_solution,
    print_selected_solution,
)
from tot.methods.bfs import solve as bfs_solve, naive_solve
from tot.methods.dfs import solve as dfs_solve
from tot.methods.mcts import solve as mcts_solve
from tot.tasks.game24 import Game24Task
from tot.tasks.crosswords import MiniCrosswordsTask

load_dotenv()

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.7,
    task="crosswords", #crosswords, game24

    naive_run=False,
    prompt_sample="propose",
    method_generate="propose",
    method_evaluate="value",

    # ranges from 0 - 1362
    task_start_index=0,
    task_end_index=1,

    # Modify the Search Algorithm
    search_method="bfs",

    # BFS knobs
    method_select="greedy",
    n_generate_sample=1, # paper crossword DFS sampled 8 proposal responses per state
    n_evaluate_sample=1,
    n_select_sample=3, # og was 5

    # MCTS knobs
    n_mcts_simulations=500,
    n_mcts_evaluate_sample=1,
    mcts_exploration=0.5,

    # Crossword DFS knobs, matching scripts/crosswords/search_crosswords-dfs.ipynb
    dfs_time_limit=100,
    dfs_max_per_state=3,
    dfs_prune=True,
    dfs_finalize_with_model=False,
)

# ---------------------------------------------------------------------------
# Paper-style mini-crossword evaluation.
#
# The paper's crossword notebook evaluates 20 puzzles with:
#   range(0, 100, 5) == [0, 5, 10, ..., 95]
#
# To return to the original one-puzzle minimal run, set this flag to False
# or comment out this whole block and the branch at the bottom of the file.
# ---------------------------------------------------------------------------
PAPER_CROSSWORDS_EVAL = False
PAPER_CROSSWORDS_INDEXES = list(range(0, 100, 5))
PAPER_CROSSWORDS_QUIET_SOLVER = False
PAPER_CROSSWORDS_PRINT_SOLUTIONS = False

task = MiniCrosswordsTask() if args.task == "crosswords" else Game24Task()
if args.naive_run:
    solve = naive_solve
elif args.search_method == "bfs":
    solve = bfs_solve
elif args.search_method == "dfs":
    if args.task != "crosswords":
        raise ValueError('search_method="dfs" only supports crosswords')
    solve = dfs_solve
else:
    solve = mcts_solve

# Debuging
debug = True

def _evaluate_selected_crossword(task, idx: int, results: list) -> tuple:
    solution = _select_solution(results)
    if not solution.strip():
        return solution, {'r_letter': 0.0, 'r_word': 0.0, 'r_game': False}
    return solution, task.test_output(idx, solution)

def _run_paper_crosswords_eval():
    summaries = []
    total = len(PAPER_CROSSWORDS_INDEXES)

    print("Paper-style crossword evaluation")
    print(f"indices: {PAPER_CROSSWORDS_INDEXES}")
    print()

    for run_num, idx in enumerate(PAPER_CROSSWORDS_INDEXES, start=1):
        print(f"[{run_num:02d}/{total}] puzzle index {idx}")

        if PAPER_CROSSWORDS_QUIET_SOLVER:
            with contextlib.redirect_stdout(io.StringIO()):
                results, _ = solve(args, task, idx, to_print=False)
        else:
            results, _ = solve(args, task, idx, to_print=debug)

        solution, metrics = _evaluate_selected_crossword(task, idx, results)
        r_letter = float(metrics.get('r_letter', 0.0))
        r_word = float(metrics.get('r_word', 0.0))
        r_game = bool(metrics.get('r_game', False))
        summaries.append({
            'idx': idx,
            'solution': solution,
            'r_letter': r_letter,
            'r_word': r_word,
            'r_game': r_game,
        })

        print(f"r_letter={r_letter:.2f}  r_word={r_word:.2f}  r_game={r_game}")
        if PAPER_CROSSWORDS_PRINT_SOLUTIONS:
            print(solution)
        print()

    n = len(summaries)
    avg_letter = sum(row['r_letter'] for row in summaries) / n if n else 0.0
    avg_word = sum(row['r_word'] for row in summaries) / n if n else 0.0
    solved = sum(1 for row in summaries if row['r_game'])

    print("Paper-style crossword summary")
    print(f"puzzles: {n}")
    print(f"avg r_letter: {avg_letter:.3f}")
    print(f"avg r_word:   {avg_word:.3f}")
    print(f"r_game:       {solved}/{n} ({(solved / n if n else 0):.1%})")

if PAPER_CROSSWORDS_EVAL and args.task == "crosswords":
    _run_paper_crosswords_eval()
else:
    for i in range(args.task_start_index, args.task_end_index):
        results, info = solve(args, task, i, to_print=debug)

        if debug:
            solution = print_selected_solution(results)
            explain_evaluation(task, i, solution)
            print()
        else:
            print_plain_solution(results)
