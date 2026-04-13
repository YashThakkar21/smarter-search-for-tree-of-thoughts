import argparse
from dotenv import load_dotenv
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
from tot.models import _steps_to_answer

load_dotenv()

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.2,
    task="game24",
    naive_run=False,
    prompt_sample=None,
    method_generate="propose",
    method_evaluate="value",
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=1,
    n_select_sample=1,
    task_start_index=900,
    task_end_index=901,
)

task = Game24Task()
# ys, info = solve(args, task, 900, to_print=True)
# print(ys)

ys, info = solve(args, task, 900, to_print=False)
solution = ys[0] if ys else ""
print(solution)
# answer = _steps_to_answer("4 5 6 10", solution)
# print(answer)