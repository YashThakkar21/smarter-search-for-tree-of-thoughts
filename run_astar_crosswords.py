import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from tot.methods.astar import solve as astar_solve
from tot.tasks.crosswords import MiniCrosswordsTask

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.7,
    n_generate_sample=3,
    n_astar_expansions=20,
    n_astar_evaluate_sample=1,
    prompt_sample="propose",
)

task = MiniCrosswordsTask()
idx = 0
print(f"=== A* on Crosswords | puzzle {idx} ===")
print("Clues:")
print(task.get_input(idx))
results, info = astar_solve(args, task, idx, to_print=True)

# Direct evaluation: replay the raw y-string into the board and score it.
# This avoids finalize_output's LLM call and fallback issues with partial solutions.
best_y = info.get('best_y', '')
task.env.reset(idx)
task._apply_history(idx, best_y)
board = task.env.board
r_letter = sum(a == b for a, b in zip(board, task.env.board_gt)) / 25
ans = task.env.get_ans(board)
ans_gt = task.env.get_ans(task.env.board_gt)
r_word = sum(a == b for a, b in zip(ans, ans_gt)) / 10

print("\n--- Direct Evaluation (from A* best partial solution) ---")
print(f"Words placed:\n{best_y.strip() if best_y.strip() else '(none)'}")
print(f"r_letter={r_letter:.2f}  r_word={r_word:.2f}  r_game={board == task.env.board_gt}")
print(f"expansions used: {len(info['expansions'])}")
