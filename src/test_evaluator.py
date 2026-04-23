from functools import partial
from tot.models import gpt as base_gpt
import tot.methods.bfs as bfs
from tot.tasks.game24 import Game24Task

def _get_ensemble_value(task, x, y, to_print=True) -> float:
    prompts = task.get_ensemble_prompts(x, y)
    total_score = 0.0

    if to_print:
        print(f"\n--- Evaluating State ---")
        print(f"Path so far:\n{y.strip() if y.strip() else '[Start] ' + x}")

    for i, prompt in enumerate(prompts):
        if prompt.startswith("Score:"):
            # Caught by the hardcoded logic for '24' or 'impossible'
            score = task.extract_numerical_score(prompt)
            if to_print:
                print(f"Prompt {i+1}: Hardcoded Score -> {score}")
        else:
            # Query the LLM
            outputs = bfs.gpt(prompt, n=1, stop=None, max_tokens=300)
            raw_output = outputs[0]
            score = task.extract_numerical_score(raw_output)
            if to_print:
                print(f"Prompt {i+1} LLM Output: {raw_output.strip()} -> Extracted: {score}")

        total_score += score

    avg_score = total_score / len(prompts)
    normalized = avg_score / 10.0

    if to_print:
        print(f"=> Average Score: {avg_score:.2f} | UCB Normalized: {normalized:.3f}")

    return normalized

def run_tests():
    # 1. Setup the LLM just like your main run
    # Replace 'openai/gpt-oss-120b' with whatever model string you are actually using
    args_backend = 'openai/gpt-oss-120b'
    bfs.gpt = partial(base_gpt, model=args_backend, temperature=0.7)

    task = Game24Task()

    # 2. Define our test cases
    test_cases = [
        {
            "name": "Test 1: Start State (Decent numbers)",
            "x": "4 4 6 8",
            "y": ""
        },
        {
            "name": "Test 2: Intermediate State (Looking good)",
            "x": "4 4 6 8",
            "y": "4 + 8 = 12 (left: 4 6 12)\n"
        },
        {
            "name": "Test 3: Dead End (Impossible numbers)",
            "x": "1 1 1 1",
            "y": "1 + 1 = 2 (left: 1 1 2)\n"
        },
        {
            "name": "Test 4: Already Solved",
            "x": "4 4 6 8",
            "y": "4 + 8 = 12 (left: 4 6 12)\n12 * 2 = 24 (left: 24)\n"
        }
    ]

    # 3. Run the evaluation on each case
    for case in test_cases:
        print(f"\n\n{'='*40}")
        print(f"RUNNING: {case['name']}")
        print(f"{'='*40}")
        _get_ensemble_value(task, case["x"], case["y"], to_print=True)

if __name__ == "__main__":
    run_tests()