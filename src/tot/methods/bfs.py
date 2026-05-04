import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    # max_tokens=4000 gives gpt-oss harmony format room for its analysis
    # channel before the final-channel JSON. Hard cryptic clues can spend
    # 2000+ tokens in analysis, so we need genuine headroom rather than
    # the 800/1500 we used originally. models.py enforces a floor of 4000
    # for cryptic mode, so this number aligns the call sites.
    raw = gpt(propose_prompt, n=1, stop=None, max_tokens=4000)[0]
    # If the task knows how to filter raw output (e.g. extract JSON
    # proposals from harmony-format leakage), let it. Otherwise fall back
    # to the legacy "every non-empty line is a proposal" behaviour so this
    # patch is backward-compatible with game24 / text / crosswords.
    if hasattr(task, 'propose_outputs_unwrap'):
        proposals = task.propose_outputs_unwrap(x, y, raw)
    else:
        proposals = [p.strip() for p in raw.split('\n') if p.strip()]
    return [y + p + '\n' for p in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def _is_game24_complete(y: str) -> bool:
    lines = [line.strip() for line in y.strip().split('\n') if line.strip()]
    if not lines:
        return False
    return lines[-1].endswith('(left: 24)') or lines[-1].lower().startswith('answer:')

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)

    x = task.get_input(idx)
    ys = ['']
    infos = []

    for step in range(task.steps):
        if args.method_generate == 'sample':
            new_ys = [
                get_samples(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                )
                for y in ys
            ]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        else:
            raise ValueError(f'Unknown generation method: {args.method_generate}')

        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        if not new_ys:
            break

        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        else:
            raise ValueError(f'Unknown evaluation method: {args.method_evaluate}')

        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:args.n_select_sample]
        else:
            raise ValueError(f'Unknown selection method: {args.method_select}')

        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if to_print and new_ys:
            sorted_pairs = sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
            sorted_new_ys, sorted_values = zip(*sorted_pairs)
            print(
                f'-- new_ys --: {sorted_new_ys}\n'
                f'-- sol values --: {sorted_values}\n'
                f'-- choices --: {select_new_ys}\n'
            )

        infos.append({
            'step': step,
            'x': x,
            'ys': ys,
            'new_ys': new_ys,
            'values': values,
            'select_new_ys': select_new_ys,
        })
        ys = select_new_ys

        # Early stop: if every surviving candidate already reaches 24, finalize now.
        if ys and all(_is_game24_complete(y) for y in ys):
            break

    if hasattr(task, "finalize_output"):
        ys = [task.finalize_output(x, y) for y in ys]

    if to_print:
        print(ys)

    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    if to_print:
        print(gpt)
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}