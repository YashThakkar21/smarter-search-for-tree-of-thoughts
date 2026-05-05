import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from tot.methods.astar import solve as astar_solve
from tot.tasks.game24 import Game24Task
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.tasks.text import TextTask


def _eval_crosswords_direct(task, idx, best_y):
    """Score by replaying the raw y-string into the board.
    Bypasses finalize_output's LLM call, which is unreliable for partial solutions."""
    task.env.reset(idx)
    task._apply_history(idx, best_y)
    board = task.env.board
    r_letter = sum(a == b for a, b in zip(board, task.env.board_gt)) / 25
    ans = task.env.get_ans(board)
    ans_gt = task.env.get_ans(task.env.board_gt)
    r_word = sum(a == b for a, b in zip(ans, ans_gt)) / 10
    return {'r_letter': r_letter, 'r_word': r_word, 'r_game': board == task.env.board_gt, 'r': r_word}


def run(args):
    if args.task == 'crosswords':
        task = MiniCrosswordsTask()
    elif args.task == 'text':
        task = TextTask()
    else:
        task = Game24Task()

    os.makedirs(f'logs/{args.task}', exist_ok=True)
    backend_slug = args.backend.replace('/', '-')
    step = args.task_step
    log_file = (
        f'logs/{args.task}/astar_{backend_slug}_{args.temperature}'
        f'_gen{args.n_generate_sample}_exp{args.n_astar_expansions}'
        f'_start{args.task_start_index}_end{args.task_end_index}_step{step}.json'
    )

    indices = list(range(args.task_start_index, args.task_end_index, step))
    total = len(indices)
    logs = []

    for run_num, i in enumerate(indices, 1):
        print(f'[{run_num:03d}/{total}] idx={i}', file=sys.stderr, flush=True)

        results, info = astar_solve(args, task, i, to_print=False)

        best_y = info.get('best_y', '')
        solution = results[0] if results else ''

        if args.task == 'crosswords':
            metrics = _eval_crosswords_direct(task, i, best_y)
        elif args.task == 'text':
            # Text scoring calls gpt-4 (5 calls/example) — skip by default; save passage only.
            metrics = {'r': 0}
        else:
            metrics = task.test_output(i, solution) if solution else {'r': 0}

        r = metrics.get('r', 0)
        extra = ''
        if args.task == 'crosswords':
            extra = f'  r_letter={metrics.get("r_letter",0):.2f}  r_word={metrics.get("r_word",0):.2f}  r_game={metrics.get("r_game",False)}'
        print(f'       done  expansions={len(info.get("expansions", []))}  r={r:.3f}{extra}', file=sys.stderr, flush=True)
        print(f'       ----', file=sys.stderr, flush=True)

        # Stdout: one solution line per example (matches bfs/mcts results format).
        print(solution)
        sys.stdout.flush()

        logs.append({
            'idx': i,
            'solution': solution,
            'best_y': best_y,
            'metrics': metrics,
            'expansions_used': len(info.get('expansions', [])),
        })
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    # Summary to stderr
    n = len(logs)
    if n == 0:
        return
    if args.task == 'game24':
        solved = sum(1 for l in logs if l['metrics'].get('r', 0) == 1)
        print(f'\n[A*] game24  solved={solved}/{n} ({solved/n:.1%})', file=sys.stderr)
    elif args.task == 'crosswords':
        avg_letter = sum(l['metrics'].get('r_letter', 0) for l in logs) / n
        avg_word = sum(l['metrics'].get('r_word', 0) for l in logs) / n
        solved = sum(1 for l in logs if l['metrics'].get('r_game', False))
        print(
            f'\n[A*] crosswords  avg_letter={avg_letter:.3f}  avg_word={avg_word:.3f}'
            f'  r_game={solved}/{n} ({solved/n:.1%})',
            file=sys.stderr,
        )
    elif args.task == 'text':
        print(f'\n[A*] text  {n} passages saved (scoring skipped)', file=sys.stderr)

    print(f'Logs: {log_file}', file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', type=str, default='openai/gpt-oss-120b')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    p.add_argument('--task_start_index', type=int, default=0)
    p.add_argument('--task_end_index', type=int, default=1)
    p.add_argument('--task_step', type=int, default=1)
    p.add_argument('--n_generate_sample', type=int, default=1)
    p.add_argument('--n_astar_expansions', type=int, default=50)
    p.add_argument('--n_astar_evaluate_sample', type=int, default=1)
    p.add_argument('--prompt_sample', type=str, default='cot', choices=['standard', 'cot'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
