from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
import tot.methods.bfs as bfs
from tot.models import gpt as base_gpt

@dataclass
class Node:
    y: str
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    untried: Optional[List[str]] = None
    visits: int = 0
    value_sum: float = 0.0
    is_solved: bool = False
    is_dead_end: bool = False

    @property
    def depth(self) -> int:
        if not self.y.strip():
            return 0
        return len([line for line in self.y.strip().split("\n") if line.strip()])

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

def _tail_line(y: str) -> str:
    lines = [line.strip() for line in y.strip().split("\n") if line.strip()]
    return lines[-1] if lines else "<root>"

def _finalize_candidate(task, x: str, y: str) -> str:
    if hasattr(task, "finalize_output"):
        return task.finalize_output(x, y)
    return y

def _is_valid_solution(task, idx: int, x: str, y: str) -> bool:
    try:
        candidate = _finalize_candidate(task, x, y)
        result = task.test_output(idx, candidate)
        return result.get("r", 0) == 1
    except Exception:
        return False

def _is_terminal(node: Node, max_steps: int) -> bool:
    return node.is_solved or node.is_dead_end or node.depth >= max_steps

def _normalize_reward(raw_value: float) -> float:
    if raw_value <= 0:
        return 0.0
    return raw_value / (raw_value + 1.0)


def _clip_score_1_to_10(score: float) -> int:
    return max(1, min(10, int(round(score))))


def _evaluate_with_ensemble(task, x: str, y: str, aggregation: str) -> Optional[float]:
    if not hasattr(task, "get_ensemble_prompts") or not hasattr(task, "extract_numerical_score"):
        return None

    prompts = task.get_ensemble_prompts(x, y)
    if not prompts:
        return None

    scores: List[int] = []
    for prompt in prompts:
        if isinstance(prompt, str) and prompt.startswith('{"reasoning"'):
            raw_output = prompt
        else:
            raw_output = bfs.gpt(prompt, n=1, stop=None, max_tokens=800)[0]

        score = task.extract_numerical_score(raw_output)
        scores.append(_clip_score_1_to_10(score))

    mode = (aggregation or "average").lower()
    if mode == "majority":
        counts = Counter(scores)
        # Break ties deterministically toward higher confidence.
        best_score = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        return best_score / 10.0

    avg_score = sum(scores) / len(scores)
    return avg_score / 10.0

def _ucb_score(parent_visits: int, child: Node, c: float) -> float:
    if child.visits == 0:
        return float("inf")
    exploit = child.mean_value
    explore = c * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)
    return exploit + explore

def _select(node: Node, c: float, max_steps: int) -> Node:
    cur = node
    while not _is_terminal(cur, max_steps):
        if cur.untried is None:
            return cur
        if cur.untried:
            return cur
        if not cur.children:
            return cur
        cur = max(cur.children, key=lambda child: _ucb_score(cur.visits, child, c))
    return cur

def _expand(node: Node, task, idx: int, x: str) -> Node:
    if node.is_solved:
        return node

    if node.depth >= task.steps:
        if _is_valid_solution(task, idx, x, node.y):
            node.is_solved = True
        else:
            node.is_dead_end = True
        return node

    if node.untried is None:
        proposals = bfs.get_proposals(task, x, node.y)
        deduped = list(dict.fromkeys(proposals))
        existing = {child.y for child in node.children}
        node.untried = [proposal for proposal in deduped if proposal not in existing]

    if not node.untried:
        node.is_dead_end = True
        return node

    child_y = node.untried.pop(0)
    child = Node(child_y, parent=node)

    if _is_valid_solution(task, idx, x, child_y):
        child.is_solved = True
    elif child.depth >= task.steps:
        child.is_dead_end = True

    node.children.append(child)
    return child


def _evaluate(node: Node, args, task, x: str, idx: int) -> float:
    if node.is_solved:
        return 1.0
    if node.is_dead_end:
        return 0.0

    agg_mode = getattr(args, "mcts_ensemble_aggregation", "average")
    use_ensemble = getattr(args, "mcts_use_ensemble_evaluator", True)
    if use_ensemble:
        ensemble_reward = _evaluate_with_ensemble(task, x, node.y, agg_mode)
        if ensemble_reward is not None:
            return ensemble_reward

    n_eval = getattr(args, "n_mcts_evaluate_sample", 1)
    raw_value = bfs.get_value(task, x, node.y, n_eval)
    return _normalize_reward(raw_value)


def _backpropagate(node: Node, reward: float) -> None:
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += reward
        cur = cur.parent


def _collect_non_root_nodes(root: Node) -> List[Node]:
    nodes: List[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node is not root:
            nodes.append(node)
        stack.extend(node.children)
    return nodes

def _rank_candidates(root: Node) -> List[Node]:
    candidates = _collect_non_root_nodes(root)
    return sorted(
        candidates,
        key=lambda n: (
            n.is_solved,
            not n.is_dead_end,
            n.mean_value,
            n.depth,
            n.visits,
        ),
        reverse=True,
    )

def _best_node(root: Node) -> Node:
    ranked = _rank_candidates(root)
    if ranked:
        return ranked[0]
    return root

def solve(args, task, idx: int, to_print: bool = True):
    bfs.gpt = partial(base_gpt, model=args.backend, temperature=args.temperature)

    if to_print:
        print(bfs.gpt)

    x = task.get_input(idx)
    root = Node("")
    infos = []

    n_simulations = getattr(args, "n_mcts_simulations", 100)
    c = getattr(args, "mcts_exploration", 1.4)

    for sim in range(n_simulations):
        leaf = _select(root, c, task.steps)
        child = _expand(leaf, task, idx, x)
        reward = _evaluate(child, args, task, x, idx)
        _backpropagate(child, reward)

        if to_print:
            infos.append(
                {
                    "simulation": sim,
                    "selected_y": leaf.y,
                    "expanded_y": child.y,
                    "reward": reward,
                    "is_solved": child.is_solved,
                    "is_dead_end": child.is_dead_end,
                }
            )
            print(
                f"[sim {sim + 1}/{n_simulations}] "
                f"select={_tail_line(leaf.y)} | "
                f"expand={_tail_line(child.y)} | "
                f"reward={reward:.3f} | "
                f"visits={child.visits} mean={child.mean_value:.3f} | "
                f"solved={child.is_solved} dead_end={child.is_dead_end}"
            )

        if child.is_solved:
            if to_print:
                print(f"[early-stop] solved at simulation {sim + 1}/{n_simulations}")
            break

    ranked = _rank_candidates(root)

    best_node = None
    best_output = ""

    for node in ranked:
        candidate = _finalize_candidate(task, x, node.y)
        if task.test_output(idx, candidate).get("r", 0) == 1:
            best_node = node
            best_output = candidate
            break

    if best_node is None:
        best_node = _best_node(root)
        best_output = _finalize_candidate(task, x, best_node.y)

    ys = [best_output]

    if to_print:
        top_k = ranked[: min(10, len(ranked))]
        print("-- mcts top candidates --")
        for rank, node in enumerate(top_k, start=1):
            print(
                f"{rank:02d}. step={node.depth} visits={node.visits} "
                f"mean={node.mean_value:.3f} solved={node.is_solved} "
                f"dead_end={node.is_dead_end} | {_tail_line(node.y)}"
            )
        print("-- mcts selected best --")
        print(
            f"step={best_node.depth} visits={best_node.visits} "
            f"mean={best_node.mean_value:.3f} solved={best_node.is_solved} "
            f"| {_tail_line(best_node.y)}"
        )
        print(ys)

    return ys, {"simulations": infos}