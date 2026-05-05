from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
import tot.methods.bfs as bfs
from tot.models import gpt as base_gpt


@dataclass
class AStarNode:
    y: str
    g: float = 0.0       # number of generation steps taken (not line count)
    h: float = 0.0       # heuristic (estimated cost to goal)
    parent: Optional["AStarNode"] = field(default=None, compare=False, repr=False)
    is_solved: bool = False
    is_dead_end: bool = False

    @property
    def f(self) -> float:
        return self.g + self.h

    @property
    def step(self) -> int:
        """Generation step index (same as g for integer budgets)."""
        return int(self.g)

    @property
    def depth(self) -> int:
        """Line-count depth — used only for display; do NOT use for task logic."""
        if not self.y.strip():
            return 0
        return len([line for line in self.y.strip().split("\n") if line.strip()])


def _normalize(raw: float) -> float:
    """Map raw value score to [0, 1]. Mirrors MCTS _normalize_reward."""
    if raw <= 0:
        return 0.0
    if raw <= 1.0:
        return raw
    return raw / (raw + 1.0)


def _heuristic(node: AStarNode, task, args, x: str) -> float:
    """
    True heuristic requiring an LLM evaluation call.
    Called once per node when it is first popped and not yet closed.

    h(n) = remaining_steps * (1 - normalized_value(n))
    High value → low h → expanded sooner.
    Falls back to 0.5 * remaining for tasks without a value function (TextTask).
    """
    remaining = max(0, task.steps - node.step)
    if remaining == 0:
        return 0.0
    n_eval = getattr(args, 'n_astar_evaluate_sample', 1)
    try:
        raw = bfs.get_value(task, x, node.y, n_eval)
        v = _normalize(raw)
    except AttributeError:
        v = 0.5  # no value fn (TextTask) — neutral estimate
    return remaining * (1.0 - v)


def _cheap_h(parent: AStarNode, child_step: int, task) -> float:
    """
    Cheap insertion-time h estimate — no LLM call.
    Propagates the parent's value to the child.

    Special case: root (step=0) scores 1.0 on an empty board, which propagates
    cheap_h=0 to every depth-1 child — guaranteeing a re-queue for all of them.
    Use a neutral 0.5 estimate for root's children instead.
    """
    remaining = max(0, task.steps - child_step)
    if remaining == 0:
        return 0.0
    if parent.step == 0:
        # Root's empty-board value is uninformative; use neutral estimate.
        return remaining * 0.5
    parent_remaining = max(1, task.steps - parent.step)
    parent_v = max(0.0, min(1.0, 1.0 - parent.h / parent_remaining))
    return remaining * (1.0 - parent_v)


def _generate_children(node: AStarNode, task, args, x: str) -> List[str]:
    """
    Expand a node into child y-strings.
    Uses propose_prompt_wrap (game24, crosswords) when available;
    falls back to get_samples for tasks without it (TextTask).
    """
    n_gen = getattr(args, 'n_generate_sample', 1)
    if hasattr(task, 'propose_prompt_wrap'):
        return bfs.get_proposals(task, x, node.y, n_gen)
    prompt_sample = getattr(args, 'prompt_sample', 'cot')
    stop = None
    if hasattr(task, 'stops') and node.step < len(task.stops):
        stop = task.stops[node.step]
    return bfs.get_samples(task, x, node.y, n_gen, prompt_sample, stop)


def _finalize(task, x: str, y: str) -> str:
    if hasattr(task, 'finalize_output'):
        return task.finalize_output(x, y)
    return y


def _is_solved(task, idx: int, x: str, y: str) -> bool:
    try:
        return task.test_output(idx, _finalize(task, x, y)).get('r', 0) == 1
    except Exception:
        return False


def _tail(y: str) -> str:
    lines = [l.strip() for l in y.strip().split('\n') if l.strip()]
    return lines[-1] if lines else '<root>'


def solve(args, task, idx: int, to_print: bool = True):
    bfs.gpt = partial(base_gpt, model=args.backend, temperature=args.temperature)
    if hasattr(task, 'set_gpt_fn'):
        task.set_gpt_fn(bfs.gpt)
    if to_print:
        print(bfs.gpt)

    x = task.get_input(idx)

    root = AStarNode(y='', g=0.0, h=float(task.steps))  # worst-case initial h
    counter = 0
    heap: list = [(root.f, counter, root)]
    closed: set = set()
    leaves: List[AStarNode] = []
    best_closed: Optional[AStarNode] = None  # deepest/best node actually expanded
    infos = []
    n_expansions = getattr(args, 'n_astar_expansions', 200)

    # FIX: count only real expansions (proposal calls) against the budget.
    # Heuristic re-queues are free — they correct priority estimates, not explore new states.
    real_exp = 0
    # Safety cap: prevents infinite loops if heap never empties (e.g. heuristic keeps rising).
    max_iters = n_expansions * 20

    for _ in range(max_iters):
        if real_exp >= n_expansions:
            break
        if not heap:
            if to_print:
                print(f'[A*] open set empty after {real_exp} expansions')
            break

        _, _, node = heapq.heappop(heap)
        canon = node.y.strip()
        if canon in closed:
            continue

        # ---- Lazy heuristic refinement (does NOT count as an expansion) --------
        # Compute the true h when we first pop this node.
        # If the cheap estimate was too optimistic (true_h > node.h), re-queue
        # with corrected priority. This does NOT consume expansion budget.
        if node.step < task.steps and not node.is_solved:
            true_h = _heuristic(node, task, args, x)
            if to_print:
                print(f'  [heuristic] h_stored={node.h:.3f} → h_true={true_h:.3f}')
            if true_h > node.h:
                node.h = true_h
                counter += 1
                heapq.heappush(heap, (node.f, counter, node))
                continue  # re-queue; NOT an expansion
            node.h = true_h

        closed.add(canon)
        real_exp += 1

        # Track the best node actually expanded: prefer deeper, then lower h.
        if best_closed is None or node.step > best_closed.step or \
                (node.step == best_closed.step and node.h < best_closed.h):
            best_closed = node

        infos.append({
            'expansion': real_exp,
            'f': node.f, 'g': node.g, 'h': node.h,
            'step': node.step, 'depth': node.depth,
            'tail': _tail(node.y), 'is_solved': node.is_solved,
        })
        if to_print:
            print(
                f'[A* {real_exp}/{n_expansions}] '
                f'f={node.f:.3f} g={node.g:.1f} h={node.h:.3f} '
                f'step={node.step}/{task.steps} | {_tail(node.y)}'
            )

        if node.is_solved:
            leaves.append(node)
            if to_print:
                print(f'[A*] solved node reached at expansion {real_exp}')
            break

        # Use node.step (= g) for terminal check, NOT line-count depth.
        if node.step >= task.steps:
            if _is_solved(task, idx, x, node.y):
                node.is_solved = True
                leaves.append(node)
                if to_print:
                    print(f'[A*] solved at expansion {real_exp}')
                break
            leaves.append(node)
            continue

        # ---- Expand: generate children ------------------------------------------
        try:
            child_ys = _generate_children(node, task, args, x)
        except Exception as e:
            if to_print:
                print(f'[A*] expand error at exp {real_exp}: {e}')
            node.is_dead_end = True
            continue

        if not child_ys:
            if _is_solved(task, idx, x, node.y):
                node.is_solved = True
                leaves.append(node)
                if to_print:
                    print(f'[A*] solved (early terminal) at expansion {real_exp}')
                break
            node.is_dead_end = True
            continue

        if to_print:
            print(f'  [expand] {len(child_ys)} children generated')

        for child_y in child_ys:
            if child_y.strip() in closed:
                continue
            child_step = node.step + 1
            child = AStarNode(y=child_y, g=float(child_step), parent=node)
            child.h = _cheap_h(node, child_step, task)
            counter += 1
            heapq.heappush(heap, (child.f, counter, child))

    # ---- Select best output -----------------------------------------------------
    solved_leaves = [n for n in leaves if n.is_solved]
    if solved_leaves:
        best = solved_leaves[0]
    elif leaves:
        best = min(leaves, key=lambda n: n.h)
    elif best_closed is not None:
        # Use the deepest node actually expanded — better than a heap node
        # that was never evaluated with its true heuristic.
        best = best_closed
    elif heap:
        best = min(heap, key=lambda item: item[0])[2]
    else:
        best = root

    best_output = _finalize(task, x, best.y)

    if to_print:
        print(
            f'\n[A*] result: step={best.step}/{task.steps} '
            f'f={best.f:.3f} solved={best.is_solved}'
        )
        print([best_output])

    return [best_output], {'expansions': infos, 'best_y': best.y}
