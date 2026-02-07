# gp_engine.py
import random
from typing import List, Dict, Any

from moudle1_gp_config import Config
from moudle3_gp_operators import OperatorSet
from moudle4_gp_tree import GPTree, Node
from moudle5_evaluator import FactorEvaluator


class GPEngine:
    def __init__(self, cfg: Config, ops: OperatorSet, evaluator: FactorEvaluator):
        self.cfg = cfg
        self.ops = ops
        self.evaluator = evaluator

        # 可用特征（只用实际存在于panel的列）
        self.features = list(evaluator._X.keys())
        if len(self.features) < 3:
            raise RuntimeError(f"too few features: {self.features}")

    def _init_population(self) -> List[GPTree]:
        pop = []
        for _ in range(self.cfg.pop_size):
            t = GPTree.random_tree(self.features, self.ops, self.cfg.max_tree_depth)
            pop.append(t)
        return pop

    def _tournament(self, pop: List[GPTree], scores: List[float]) -> GPTree:
        k = self.cfg.tournament_k
        idxs = random.sample(range(len(pop)), k)
        best = max(idxs, key=lambda i: scores[i])
        return pop[best]

    def _crossover(self, a: GPTree, b: GPTree) -> GPTree:
        ca = a.clone()
        cb = b.clone()

        na, pa, wa = ca.random_subtree()
        nb, pb, wb = cb.random_subtree()

        # swap
        ca.replace_subtree(pa, wa, nb.clone())
        # 控制规模
        if ca.size() > self.cfg.max_tree_nodes or ca.depth() > self.cfg.max_tree_depth:
            return a.clone()
        return ca

    def _mutate(self, t: GPTree) -> GPTree:
        ct = t.clone()
        n, p, w = ct.random_subtree()

        # 方式1：用随机新子树替换
        if random.random() < 0.6:
            new_sub = GPTree.random_tree(self.features, self.ops, max_depth=3).root
            ct.replace_subtree(p, w, new_sub)
        else:
            # 方式2：节点内变异：feature换一个 / op换一个同arity
            if n.kind == "feature":
                n.value = random.choice(self.features)
            else:
                op = n.value
                n.value = self.ops.sample_op(arity=op.arity)

        if ct.size() > self.cfg.max_tree_nodes or ct.depth() > self.cfg.max_tree_depth:
            return t.clone()
        return ct

    def evolve(self) -> List[Dict[str, Any]]:
        pop = self._init_population()

        best_archive: List[Dict[str, Any]] = []

        for gen in range(self.cfg.n_generations):
            scores = [self.evaluator.fitness(t) for t in pop]

            # 排序
            ranked = sorted(zip(pop, scores), key=lambda x: x[1], reverse=True)
            elites = ranked[: self.cfg.elitism]

            # 记录
            best_tree, best_score = elites[0]
            best_archive.append({
                "gen": gen,
                "score": float(best_score),
                "size": best_tree.size(),
                "expr": best_tree.to_string(),
                "tree": best_tree.clone(),
            })

            print(f"[Gen {gen:02d}] best={best_score:.6f}  expr={best_tree.to_string()}")

            # 新一代
            new_pop = [t.clone() for t, _ in elites]  # 精英保留

            while len(new_pop) < self.cfg.pop_size:
                r = random.random()
                if r < self.cfg.p_crossover:
                    p1 = self._tournament(pop, scores)
                    p2 = self._tournament(pop, scores)
                    child = self._crossover(p1, p2)
                    new_pop.append(child)
                elif r < self.cfg.p_crossover + self.cfg.p_mutation:
                    p = self._tournament(pop, scores)
                    child = self._mutate(p)
                    new_pop.append(child)
                else:
                    p = self._tournament(pop, scores)
                    new_pop.append(p.clone())

            pop = new_pop

        # 全部archive再按score排序输出
        best_archive = sorted(best_archive, key=lambda x: x["score"], reverse=True)
        return best_archive
