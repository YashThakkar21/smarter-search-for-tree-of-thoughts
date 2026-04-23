from functools import partial
from tot.models import gpt as base_gpt
import tot.methods.bfs as bfs
from tot.tasks.game24 import Game24Task

import os
import json
import hashlib

def _get_ensemble_value(task, x, y, to_print=True) -> float:
    prompts = task.get_ensemble_prompts(x, y)

    # 1. Setup Logging Directory
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Create a unique filename based on the current state
    state_str = y.strip() if y.strip() else x
    state_hash = hashlib.md5(state_str.encode()).hexdigest()
    log_filepath = os.path.join(log_dir, f"{state_hash}.json")

    total_score = 0.0
    log_data = {"state": state_str, "evaluations": []}

    if to_print:
        print(f"\n--- Evaluating State ---")
        print(f"Path so far:\n{state_str}")

    # 3. Evaluate the ensemble
    for i, prompt in enumerate(prompts):
        if prompt.startswith('{"reasoning"'):
                score = task.extract_numerical_score(prompt)
                if to_print:
                    print(f"Prompt {i+1}: Hardcoded Score -> {score}")

                log_data["evaluations"].append({
                    "prompt_version": i + 1,
                    "raw_output": prompt,
                    "extracted_score": score,
                    "note": "Hardcoded terminal state"
                })
        else:
            # Query the LLM with max_tokens=800 so it can finish its JSON thought
            outputs = bfs.gpt(prompt, n=1, stop=None, max_tokens=800)
            raw_output = outputs[0]
            score = task.extract_numerical_score(raw_output)

            if to_print:
                print(f"Prompt {i+1} LLM Output: {raw_output.strip()} \n-> Extracted: {score}")

            log_data["evaluations"].append({
                "prompt_version": i + 1,
                "raw_output": raw_output,
                "extracted_score": score
            })

        total_score += score

    # 4. Calculate Final Scores
    avg_score = total_score / len(prompts)
    normalized = avg_score / 10.0

    log_data["average_score"] = avg_score
    log_data["final_normalized_score"] = normalized

    if to_print:
        print(f"=> Avg: {avg_score:.2f} | Normalized: {normalized:.3f} "
              f"| Individual: {scores} ({len(scores)}/{len(prompts)})")

    with open(log_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    # 5. Save the log to the directory
    with open(log_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    # 5. Save the log to the directory
    with open(log_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    return normalized

# def _get_ensemble_value(task, x, y, to_print=True) -> float:
#     prompts = task.get_ensemble_prompts(x, y)
#     total_score = 0.0

#     if to_print:
#         print(f"\n--- Evaluating State ---")
#         print(f"Path so far:\n{y.strip() if y.strip() else '[Start] ' + x}")

#     for i, prompt in enumerate(prompts):
#         if prompt.startswith("Score:"):
#             # Caught by the hardcoded logic for '24' or 'impossible'
#             score = task.extract_numerical_score(prompt)
#             if to_print:
#                 print(f"Prompt {i+1}: Hardcoded Score -> {score}")
#         else:
#             # Query the LLM
#             outputs = bfs.gpt(prompt, n=1, stop=None, max_tokens=300)
#             raw_output = outputs[0]
#             score = task.extract_numerical_score(raw_output)
#             if to_print:
#                 print(f"Prompt {i+1} LLM Output: {raw_output.strip()} -> Extracted: {score}")

#         total_score += score

#     avg_score = total_score / len(prompts)
#     normalized = avg_score / 10.0

#     if to_print:
#         print(f"=> Average Score: {avg_score:.2f} | UCB Normalized: {normalized:.3f}")

#     return normalized

def run_tests():
    args_backend = 'openai/gpt-oss-120b'
    bfs.gpt = partial(base_gpt, model=args_backend, temperature=0.3)

    task = Game24Task()

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

    for case in test_cases:
        print(f"\n\n{'='*40}")
        print(f"RUNNING: {case['name']}")
        print(f"  Group:    {case['group']}")
        print(f"  Expected: {case['expected'][0]}-{case['expected'][1]}  ({case['note']})")
        print(f"{'='*60}")
        normalized = _get_ensemble_value(task, case["x"], case["y"], to_print=True)
        avg = normalized * 10
        in_range = case["expected"][0] <= avg <= case["expected"][1]
        results.append({
            "group": case["group"],
            "name": case["name"],
            "avg": avg,
            "expected": case["expected"],
            "in_range": in_range,
        })
        marker = "  IN RANGE" if in_range else "  OUT OF RANGE"
        print(f"=> Result: avg={avg:.2f}, expected {case['expected'][0]}-{case['expected'][1]} {marker}")

    # === Final summary ===
    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Group':<14} {'Name':<45} {'Avg':>5}  {'Exp':<7} {'OK?'}")
    print("-" * 80)
    for r in results:
        exp_str = f"{r['expected'][0]}-{r['expected'][1]}"
        ok = "yes" if r["in_range"] else "NO"
        print(f"{r['group']:<14} {r['name']:<45} {r['avg']:>5.2f}  {exp_str:<7} {ok}")

    n_ok = sum(1 for r in results if r["in_range"])
    print(f"\nIn-range: {n_ok}/{len(results)}")

    # Calibration check: are we using the full 1-10 range?
    distinct_avgs = sorted(set(round(r["avg"]) for r in results))
    print(f"Distinct rounded avgs across all cases: {distinct_avgs}")
    if len(distinct_avgs) <= 3:
        print("WARNING: Evaluator is collapsing to {} values - not using full 1-10 range."
              .format(len(distinct_avgs)))


if __name__ == "__main__":
    # run_tests()
    # Or run a single group:
    run_tests(only_groups=["D-FRACTION"])
    run_tests(only_groups=["C-PAPER", "E-MIDGAME"])