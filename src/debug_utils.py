import re

def _normalize_crossword_grid(solution: str) -> str:
    rows = []
    for line in solution.strip().splitlines():
        line = line.strip()
        compact = re.fullmatch(r'[A-Za-z]{5}', line)
        spaced = re.fullmatch(r'[A-Za-z](?:\s+[A-Za-z]){4}', line)
        if compact:
            letters = compact.group(0).upper()
        elif spaced:
            letters = re.sub(r'\s+', '', spaced.group(0)).upper()
        else:
            continue
        rows.append(' '.join(letters))
        if len(rows) == 5:
            break
    return '\n'.join(rows) if len(rows) == 5 else ''

def _select_solution(results) -> str:
    candidates = [result for result in results if result and result.strip()]
    if not candidates:
        return ''

    ranked_grids = []
    for index, candidate in enumerate(candidates):
        grid = _normalize_crossword_grid(candidate)
        if not grid:
            continue
        letters = re.sub(r'[^A-Z]', '', grid)
        placeholder_count = letters.count('X')
        ranked_grids.append(((placeholder_count, index), grid))

    if ranked_grids:
        return min(ranked_grids, key=lambda item: item[0])[1]

    return candidates[0]

def explain_evaluation(task, idx: int, solution: str) -> None:
    from tot.tasks.crosswords import MiniCrosswordsTask
    if isinstance(task, MiniCrosswordsTask):
        _explain_crosswords(task, idx, solution)
    else:
        _explain_game24(task, idx, solution)

def _explain_crosswords(task, idx: int, solution: str) -> None:
    puzzle = task.get_input(idx)
    print(f"Puzzle {idx}: {puzzle}")

    if not solution.strip():
        print("Evaluation: incorrect (empty solution)")
        return

    # Detect format:
    #   A) JSON: model reasoning + {"h1": "WORD", ...} anywhere in output
    #   B) finalize_output: 5 compact rows like "MOTOR\nABCDE\n..."
    #   C) model grid: spaced uppercase "A B C D E" or "Output:" header
    #   D) BFS history: render_ans() lines like "h1. clue: ANSWER"
    json_rows = task._parse_json_board(solution)
    compact_rows = re.findall(r'^[A-Za-z]{5}$', solution.strip(), re.MULTILINE)
    spaced_rows  = re.findall(r'^[A-Z] [A-Z] [A-Z] [A-Z] [A-Z]$', solution.strip(), re.MULTILINE)

    if len(json_rows) == 5:
        info = task.test_output(idx, '\n'.join(json_rows))
        r_letter = info.get('r_letter', 0)
        r_word   = info.get('r_word', 0)
        r_game   = info.get('r_game', False)
    elif len(compact_rows) >= 5 or len(spaced_rows) >= 5 or 'Output:' in solution:
        info = task.test_output(idx, solution)
        r_letter = info.get('r_letter', 0)
        r_word   = info.get('r_word', 0)
        r_game   = info.get('r_game', False)
    else:
        task.env.reset(idx)
        pattern = re.compile(r'^([hv][1-5])\.\s+.+?:\s+([A-Za-z]{5})\s*$', re.MULTILINE)
        for m in pattern.finditer(solution):
            task.env.step(f'{m.group(1)}. {m.group(2).lower()}')
        r_letter = sum(a == b for a, b in zip(task.env.board, task.env.board_gt)) / 25
        r_word   = sum(a == b for a, b in zip(task.env.ans,   task.env.ans_gt))   / 10
        r_game   = task.env.board == task.env.board_gt

    print(f"r_letter: {r_letter:.2f}  r_word: {r_word:.2f}  r_game: {r_game}")
    if r_game:
        print("Evaluation: correct (board solved)")
    else:
        print("Evaluation: incorrect")

def _explain_game24(task, idx: int, solution: str) -> None:
    import sympy

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
    solution = _select_solution(results)
    print("\nSelected solution:")
    print(solution)
    print()
    return solution

def print_plain_solution(results) -> None:
    solution = _select_solution(results)
    print(solution)
    print()
