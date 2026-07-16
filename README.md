# Smarter Search for Tree of Thoughts · [Paper](https://drive.google.com/file/d/1LqR2tY1I4ndmFvawkI0gjoOPjsv9bUo7/view?usp=sharing)

Arnold Jiang, Emma Hsieh, Yash Thakkar — Princeton University

## Overview

Tree of Thoughts (ToT) lets LLMs solve problems requiring planning and
backtracking by exploring a tree of intermediate reasoning steps — but the
original framework uses simple BFS/DFS. This work asks whether *smarter
search* improves LLM reasoning, replacing ToT's search layer with classical
algorithms: **A\*** (with a lazily evaluated LLM-based heuristic) and
**Monte Carlo Tree Search (MCTS)**.

## Approach

- **A\* for ToT** — each node's cost combines depth (g) with an LLM-scored
  estimate of remaining work (h). Heuristic evaluation is lazy: cheap
  estimates on insertion, true LLM evaluation only when a node is popped,
  so search budget is spent on promising branches.
- **MCTS for ToT** — balances repeated testing of promising branches with
  exploration of less-visited alternatives, offering more flexibility than
  BFS when the search space is semantically ambiguous.

## Evaluation

Benchmarked against the original ToT BFS/DFS baselines on **Game of 24**,
**Mini Crosswords**, and **Cryptic Crosswords**, controlling for LLM-call
budget.

## Key findings

- Stronger search helps *when the evaluator is reliable*: MCTS improves
  full-solve accuracy over BFS on Cryptic Crosswords, where partial states
  give useful guidance.
- Search alone is not sufficient: on Mini Crosswords, MCTS underperforms
  DFS despite a larger LLM-call budget — noisy or poorly calibrated state
  evaluations send even sophisticated search down unproductive branches.
- The central bottleneck in ToT-style reasoning is **evaluator quality**,
  as much as the choice of search algorithm — motivating future work on
  heuristic design, constraint checking, and budget-aware search.


## Acknowledgments

This project builds on the original
[Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm)
repository by Yao et al. (Princeton NLP), forked and extended with A* and
MCTS search implementations, the lazy LLM heuristic, and the Cryptic
Crosswords task.
