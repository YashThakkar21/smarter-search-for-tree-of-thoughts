import re
import sympy

def explain_evaluation(task, idx: int, solution: str) -> None:
    puzzle = task.get_input(idx)
    print(f"Puzzle {idx}: {puzzle}")

    if not solution.strip():
        print("Evaluation: incorrect (empty solution)")
        return

    final_line = solution.strip().split("\n")[-1]
    expression = final_line.lower().replace("answer: ", "").split("=")[0].strip()
    answer_numbers = sorted(re.findall(r"\d+", expression))
    puzzle_numbers = sorted(re.findall(r"\d+", puzzle))

    print(f"Final line: {final_line}")
    print(f"Parsed expression: {expression}")
    print(f"Numbers in expression: {answer_numbers}")
    print(f"Numbers in puzzle:     {puzzle_numbers}")

    if answer_numbers != puzzle_numbers:
        print("Evaluation: incorrect (number mismatch; not all puzzle numbers were used exactly once)")
        return

    try:
        simplified = sympy.simplify(expression)
        print(f"Simplified value: {simplified}")
        if simplified == 24:
            print("Evaluation: correct")
        else:
            print("Evaluation: incorrect (expression does not evaluate to 24)")
    except Exception as exc:
        print(f"Evaluation: incorrect (invalid expression: {exc})")

def print_selected_solution(results) -> str:
    print("\nSelected solution:")
    print(results)
    solution = results[0] if results else ""
    print()
    return solution

def print_plain_solution(results) -> None:
    solution = results[0] if results else ""
    print(solution)
    print()