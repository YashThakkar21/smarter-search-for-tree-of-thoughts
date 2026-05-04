import re
from functools import partial
from tot.models import gpt as _gpt


confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}


def parse_line(input_str):
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5}) \((certain|high|medium|low)\).*$'
    match = re.match(pattern, input_str.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    pos, word, confidence = match.groups()
    return pos.lower(), word.lower(), confidence.lower()


def parse_response(response):
    parsed_lines = []
    for line in response.split('\n'):
        parsed_line = parse_line(line)
        if parsed_line is None:
            continue
        pos, word, confidence = parsed_line
        parsed_lines.append((f'{pos}. {word}', confidence_to_value.get(confidence, 0)))
    return parsed_lines if parsed_lines else None


def get_candidates_to_scores(task, x, actions, n_generate_sample):
    y = ''.join(f'{action}\n' for action in actions)
    task._apply_history(task.env.idx, y)
    obs = task.env.render()
    cache_key = (obs, n_generate_sample)
    if cache_key in task.env.cache:
        return task.env.cache[cache_key]

    prompt = task.propose_prompt_wrap(x, y)
    responses = gpt(prompt, n=n_generate_sample)
    candidates_to_scores = {}
    for response in responses:
        parsed_response = parse_response(response)
        if not parsed_response:
            continue
        for candidate, score in parsed_response:
            if hasattr(task, '_proposal_is_usable') and not task._proposal_is_usable(candidate):
                continue
            candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score

    task.env.cache[cache_key] = candidates_to_scores
    return candidates_to_scores


def _format_history(actions):
    return ''.join(f'{action}\n' for action in actions)


def _format_current_board(task):
    if hasattr(task, '_format_board_rows'):
        return '\n'.join(task._format_board_rows(task.env.board))
    return ''


def _better_info(candidate, best):
    if best is None:
        return True
    candidate_info = candidate['info']
    best_info = best['info']
    candidate_key = (
        candidate_info.get('r_word', 0),
        candidate_info.get('r_letter', 0),
        candidate['env_step'],
    )
    best_key = (
        best_info.get('r_word', 0),
        best_info.get('r_letter', 0),
        best['env_step'],
    )
    return candidate_key > best_key


def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(_gpt, model=args.backend, temperature=args.temperature)
    if hasattr(task, 'set_gpt_fn'):
        task.set_gpt_fn(gpt)
    if to_print:
        print(gpt)

    x = task.get_input(idx)
    actions = []
    infos = []
    best = None
    time_limit = getattr(args, 'dfs_time_limit', 100)
    max_per_state = getattr(args, 'dfs_max_per_state', 3)
    prune = getattr(args, 'dfs_prune', True)
    n_generate_sample = getattr(args, 'n_generate_sample', 8)

    def dfs():
        nonlocal best
        candidates_to_scores = get_candidates_to_scores(task, x, actions, n_generate_sample)
        if not candidates_to_scores:
            return

        sorted_candidates = sorted(
            candidates_to_scores,
            key=candidates_to_scores.get,
            reverse=True,
        )
        if to_print:
            print(sorted(candidates_to_scores.items(), key=lambda item: item[1], reverse=True))

        board = task.env.board.copy()
        status = task.env.status.copy()
        steps = task.env.steps

        cnt_per_state = 0
        for action in sorted_candidates:
            obs, _, _, step_info = task.env.step(action)
            if (
                len(infos) < time_limit
                and task.env.steps < 10
                and not any(status_value == 2 for status_value in task.env.status)
            ):
                cnt_per_state += 1
                if cnt_per_state > max_per_state:
                    task.env.reset(task.env.idx, board=board.copy(), status=status.copy(), steps=steps)
                    break

                count = task._paper_status_count()
                actions.append(action)

                info = {
                    'total_step': len(infos),
                    'env_step': task.env.steps,
                    'actions': actions.copy(),
                    'info': step_info,
                    'count': count,
                }
                infos.append(info)
                if _better_info(info, best):
                    best = info

                if to_print:
                    print(len(infos) - 1)
                    print(actions)
                    print(task.env.render_board())
                    print(step_info)
                    print(count)
                    if best:
                        print('best', best)
                    print('--------------')
                    print()

                if step_info.get('r_game') or (not prune or count['impossible'] < 1):
                    dfs()
                actions.pop()

            task.env.reset(task.env.idx, board=board.copy(), status=status.copy(), steps=steps)

    dfs()

    best_actions = best['actions'] if best else []
    best_y = _format_history(best_actions)
    task._apply_history(idx, best_y)
    if getattr(args, 'dfs_finalize_with_model', False) and hasattr(task, 'finalize_output'):
        output = task.finalize_output(x, best_y)
    else:
        output = _format_current_board(task) or best_y

    if to_print:
        print([output])

    return [output], {'steps': infos, 'best': best}
