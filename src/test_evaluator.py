from functools import partial
from tot.models import gpt as base_gpt
import tot.methods.bfs as bfs
from tot.tasks.game24 import Game24Task

import os
import json
import hashlib


def _get_ensemble_value(task, x, y, to_print=True) -> float:
    prompts = task.get_ensemble_prompts(x, y)

    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)

    state_str = y.strip() if y.strip() else x
    state_hash = hashlib.md5(state_str.encode()).hexdigest()
    log_filepath = os.path.join(log_dir, f"{state_hash}.json")

    log_data = {"state": state_str, "evaluations": []}

    if to_print:
        print(f"\n--- Evaluating State ---\nPath so far:\n{state_str}")

    scores = []
    for i, prompt in enumerate(prompts):
        if prompt.startswith('{"reasoning"'):
            score = task.extract_numerical_score(prompt)
            log_data["evaluations"].append({
                "prompt_version": i + 1,
                "raw_output": prompt,
                "extracted_score": score,
                "note": "Hardcoded terminal state",
            })
            if to_print:
                print(f"Prompt {i+1}: Hardcoded Score -> {score}")
        else:
            outputs = bfs.gpt(prompt, n=1, stop=None, max_tokens=800)
            raw = outputs[0]
            score = task.extract_numerical_score(raw)
            attempts = [{"raw_output": raw, "extracted_score": score}]
            if score is None:
                outputs = bfs.gpt(prompt, n=1, stop=None, max_tokens=1500)
                raw = outputs[0]
                score = task.extract_numerical_score(raw)
                attempts.append({"raw_output": raw, "extracted_score": score})
            log_data["evaluations"].append({
                "prompt_version": i + 1,
                "attempts": attempts,
                "final_score": score,
                "parse_failed": score is None,
            })
            if to_print:
                # Show the TAIL on success (where the JSON lives) and the
                # HEAD on failure (where the wreckage is). Previously we
                # always showed the head, which made successful parses
                # look like they came out of nowhere.
                if score is None:
                    preview = "[head] " + raw.strip()[:250]
                    tag = "PARSE_FAIL"
                else:
                    preview = "...[tail] " + raw.strip()[-200:]
                    tag = f"-> Extracted: {score}"
                print(f"Prompt {i+1}: {preview}\n  {tag}")
        if score is not None:
            scores.append(score)

    n_succeeded = len(scores)
    n_total = len(prompts)

    if n_succeeded == 0:
        avg_score = 5.0
        log_data["all_parses_failed"] = True
        if to_print:
            print("  WARNING: all prompts failed to parse - falling back to midpoint (5.0)")
    else:
        avg_score = sum(scores) / n_succeeded

    normalized = avg_score / 10.0
    log_data["average_score"] = avg_score
    log_data["final_normalized_score"] = normalized
    log_data["successful_evaluations"] = n_succeeded
    log_data["total_evaluations"] = n_total
    log_data["confidence"] = n_succeeded / n_total

    if to_print:
        print(f"=> Avg: {avg_score:.2f} | Normalized: {normalized:.3f} "
              f"| Individual: {scores} ({len(scores)}/{len(prompts)})")

    with open(log_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    return normalized


# Each case has:
#   group:    A/B/C... so the summary at the end can group results
#   name:     short identifier
#   x, y:     puzzle and path so far (mirrors the real solver's call sites)
#   expected: rough score range we'd expect from a well-calibrated evaluator
#   note:     why this case is interesting
#
# Expected ranges are NOT asserted - the point is to see whether the
# evaluator naturally lands near them, and where it doesn't.

test_cases = [
    # === GROUP A: Clearly solvable start states (expect 9-10) ===
    {
        "group": "A-SOLVABLE",
        "name": "A1: Classic 4 4 6 8",
        "x": "4 4 6 8", "y": "",
        "expected": (9, 10),
        "note": "(4+8)*(6-4)=24",
    },
    {
        "group": "A-SOLVABLE",
        "name": "A2: Paper-sure 4 9 10 13",
        "x": "4 9 10 13", "y": "",
        "expected": (9, 10),
        "note": "(13-9)*(10-4)=24",
    },
    {
        "group": "A-SOLVABLE",
        "name": "A3: Paper-sure 2 9 10 12",
        "x": "2 9 10 12", "y": "",
        "expected": (9, 10),
        "note": "2*12*(10-9)=24",
    },

    # === GROUP B: Clearly impossible start states (expect 1-2) ===
    {
        "group": "B-IMPOSSIBLE",
        "name": "B1: All ones",
        "x": "1 1 1 1", "y": "",
        "expected": (1, 2),
        "note": "max value 4",
    },
    {
        "group": "B-IMPOSSIBLE",
        "name": "B2: All thirteens",
        "x": "13 13 13 13", "y": "",
        "expected": (1, 2),
        "note": "all too big",
    },
    {
        "group": "B-IMPOSSIBLE",
        "name": "B3: All twos",
        "x": "2 2 2 2", "y": "",
        "expected": (1, 3),
        "note": "max via products is 16",
    },

    # === GROUP C: Paper's labeled mid-range states ===
    # These come straight from the value_prompt few-shot in game24.py.
    # They test whether the model can replicate the paper's intuition.
    {
        "group": "C-PAPER",
        "name": "C1: Paper-likely 5 7 8 (actually unsolvable!)",
        "x": "5 7 8", "y": "",
        "expected": (3, 7),
        "note": "Paper labels 'likely'; in fact no solution exists. "
                "Tests false-positive rate.",
    },
    {
        "group": "C-PAPER",
        "name": "C2: Paper-likely 5 6 6 (actually solvable)",
        "x": "5 6 6", "y": "",
        "expected": (5, 9),
        "note": "5*6-6=24. Paper says 'likely' because the path isn't obvious.",
    },
    {
        "group": "C-PAPER",
        "name": "C3: Paper-impossible 10 10 11",
        "x": "10 10 11", "y": "",
        "expected": (1, 3),
        "note": "Numbers too large per paper",
    },
    {
        "group": "C-PAPER",
        "name": "C4: Paper-impossible 1 3 3",
        "x": "1 3 3", "y": "",
        "expected": (1, 3),
        "note": "Numbers too small per paper",
    },
    {
        "group": "C-PAPER",
        "name": "C5: Paper-sure 4 4 10",
        "x": "4 4 10", "y": "",
        "expected": (9, 10),
        "note": "(10-4)*4=24",
    },

    # === GROUP D: Tricky-solvable (require non-integer intermediates) ===
    # These test whether the model is biased toward integer-only solutions.
    # Models often score these LOW because they can't find the path.
    {
        "group": "D-FRACTION",
        "name": "D1: Fraction trick 1 5 5 5",
        "x": "1 5 5 5", "y": "",
        "expected": (5, 10),  # wide; we want to see what model actually does
        "note": "(5 - 1/5) * 5 = 24. Models often miss the fraction.",
    },
    {
        "group": "D-FRACTION",
        "name": "D2: Fraction trick 3 3 7 7",
        "x": "3 3 7 7", "y": "",
        "expected": (5, 10),
        "note": "(3 + 3/7) * 7 = 24. Famous hard puzzle.",
    },
    {
        "group": "D-FRACTION",
        "name": "D3: Fraction trick 3 3 8 8",
        "x": "3 3 8 8", "y": "",
        "expected": (5, 10),
        "note": "8 / (3 - 8/3) = 24",
    },

    # === GROUP E: Mid-game states (the actual MCTS workload) ===
    {
        "group": "E-MIDGAME",
        "name": "E1: Good move from 4 4 6 8",
        "x": "4 4 6 8",
        "y": "4 + 8 = 12 (left: 4 6 12)\n",
        "expected": (9, 10),
        "note": "(12-6)*4=24, easy from here",
    },
    {
        "group": "E-MIDGAME",
        "name": "E2: Bad move from 4 4 6 8 (dead end)",
        "x": "4 4 6 8",
        "y": "4 * 4 = 16 (left: 6 8 16)\n",
        "expected": (1, 4),
        "note": "6 8 16 cannot reach 24. MCTS should NOT explore further.",
    },
    {
        "group": "E-MIDGAME",
        "name": "E3: Dead end from 1 1 1 1",
        "x": "1 1 1 1",
        "y": "1 + 1 = 2 (left: 1 1 2)\n",
        "expected": (1, 2),
        "note": "max 4",
    },
    {
        "group": "E-MIDGAME",
        "name": "E4: Mid-game 2 9 10 12",
        "x": "2 9 10 12",
        "y": "10 - 9 = 1 (left: 1 2 12)\n",
        "expected": (8, 10),
        "note": "1*2*12=24",
    },

    # === GROUP F: Terminal states (hardcoded - no LLM call) ===
    {
        "group": "F-TERMINAL",
        "name": "F1: Already solved",
        "x": "4 4 6 8",
        "y": "4 + 8 = 12 (left: 4 6 12)\n12 - 6 = 6 (left: 4 6)\n6 * 4 = 24 (left: 24)\n",
        "expected": (10, 10),
        "note": "hardcoded path",
    },

    # === GROUP G: Two-number end states ===
    {
        "group": "G-TWONUM",
        "name": "G1: Trivial solvable 12 12",
        "x": "12 12", "y": "",
        "expected": (9, 10),
        "note": "12+12=24",
    },
    {
        "group": "G-TWONUM",
        "name": "G2: Trivial impossible 13 13",
        "x": "13 13", "y": "",
        "expected": (1, 2),
        "note": "13+13=26",
    },
    {
        "group": "G-TWONUM",
        "name": "G3: Solvable two-num 7 17",
        "x": "7 17", "y": "",
        "expected": (9, 10),
        "note": "7+17=24",
    },
    {
        "group": "G-TWONUM",
        "name": "G4: Off-by-one 7 18",
        "x": "7 18", "y": "",
        "expected": (1, 2),
        "note": "7+18=25, no path",
    },
]


def run_tests(only_groups=None):
    """
    Run all test cases or only specified groups (e.g. only_groups=['D-FRACTION']).
    """
    args_backend = 'openai/gpt-oss-120b'
    bfs.gpt = partial(base_gpt, model=args_backend, temperature=0.3)

    task = Game24Task()

    cases = test_cases
    if only_groups is not None:
        cases = [c for c in cases if c["group"] in only_groups]

    results = []
    for case in cases:
        print(f"\n\n{'='*60}")
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