import math
from functools import partial

import tot.methods.bfs as bfs
from tot.models import gpt as base_gpt


class Node:
    def __init__(self, y, parent=None):
        self.y = y
        self.parent = parent
        self.children = []
        self.untried = None
        self.visits = 0
        self.value_sum = 0.0
        self.is_solved = False
        self.is_dead_end = False

    @property
    def depth(self):
        if not self.y.strip():
            return 0
        return len([line for line in self.y.strip().split("\n") if line.strip()])

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _is_valid_solution(task, idx, y) -> bool:
    try:
        result = task.test_output(idx, y)
        return result.get("r", 0) == 1
    except Exception:
        return False


def _is_terminal(node: Node, max_steps: int) -> bool:
    return node.is_solved or node.is_dead_end or node.depth >= max_steps


def _normalize_reward(raw_value: float) -> float:
    if raw_value <= 0:
        return 0.0
    return raw_value / (raw_value + 1.0)


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


def _expand(node: Node, task, idx, x) -> Node:
    if node.is_solved:
        return node

    if node.depth >= task.steps:
        node.is_dead_end = not _is_valid_solution(task, idx, node.y)
        return node

    if node.untried is None:
        proposals = bfs.get_proposals(task, x, node.y)
        deduped = list(dict.fromkeys(proposals))
        existing = {child.y for child in node.children}
        node.untried = [p for p in deduped if p not in existing]

    if not node.untried:
        node.is_dead_end = True
        return node

    child_y = node.untried.pop(0)
    child = Node(child_y, parent=node)

    if _is_valid_solution(task, idx, child_y):
        child.is_solved = True
    elif child.depth >= task.steps:
        child.is_dead_end = True

    node.children.append(child)
    return child


def _evaluate(node: Node, args, task, x, idx) -> float:
    if node.is_solved:
        return 1.0

    if node.is_dead_end and node.depth >= task.steps:
        return 0.0

    raw_value = bfs.get_value(task, x, node.y, args.n_evaluate_sample)
    return _normalize_reward(raw_value)


def _backpropagate(node: Node, reward: float):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += reward
        cur = cur.parent


def _best_node(root: Node) -> Node:
    solved_nodes = []
    other_nodes = []

    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_solved:
            solved_nodes.append(node)
        elif node is not root:
            other_nodes.append(node)
        stack.extend(node.children)

    if solved_nodes:
        return max(solved_nodes, key=lambda n: (n.mean_value, n.visits))

    if other_nodes:
        return max(other_nodes, key=lambda n: (n.mean_value, n.visits))

    return root


def solve(args, task, idx, to_print=True):
    # Important: bfs.get_proposals/get_value use bfs.gpt internally.
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
            infos.append({
                "simulation": sim,
                "selected_y": leaf.y,
                "expanded_y": child.y,
                "reward": reward,
                "is_solved": child.is_solved,
                "is_dead_end": child.is_dead_end,
            })

    best = _best_node(root)
    ys = [best.y]

    if hasattr(task, "finalize_output"):
        ys = [task.finalize_output(x, y) for y in ys]

    if to_print:
        print(ys)

    return ys, {"simulations": infos}