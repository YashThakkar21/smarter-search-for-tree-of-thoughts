import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from tot.methods.astar import solve as astar_solve
from tot.tasks.text import TextTask

args = argparse.Namespace(
    backend="openai/gpt-oss-120b",
    temperature=0.7,
    n_generate_sample=2,
    n_astar_expansions=20,
    n_astar_evaluate_sample=1,
    prompt_sample="cot",
)

task = TextTask()
idx = 0
print(f"=== A* on Text | puzzle {idx} ===")
print("Input:", task.get_input(idx).strip())
print()
results, info = astar_solve(args, task, idx, to_print=True)

print("\n--- Output ---")
output = results[0] if results[0] else "(empty)"
# strip the internal reasoning channel prefix if present
if "Passage:\n" in output:
    output = "Passage:\n" + output.split("Passage:\n", 1)[-1]
print(output[:800])
print(f"\nexpansions used: {len(info['expansions'])}")
