# gp_tree.py
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import torch

from moudle3_gp_operators import Op, OperatorSet


@dataclass
class Node:
    kind: str  # "feature" / "op"
    value: Any
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def size(self) -> int:
        if self.kind == "feature":
            return 1
        s = 1
        if self.left:
            s += self.left.size()
        if self.right:
            s += self.right.size()
        return s

    def depth(self) -> int:
        if self.kind == "feature":
            return 1
        dl = self.left.depth() if self.left else 0
        dr = self.right.depth() if self.right else 0
        return 1 + max(dl, dr)

    def clone(self) -> "Node":
        return Node(
            kind=self.kind,
            value=self.value,
            left=self.left.clone() if self.left else None,
            right=self.right.clone() if self.right else None
        )


class GPTree:
    def __init__(self, root: Node):
        self.root = root

    def size(self) -> int:
        return self.root.size()

    def depth(self) -> int:
        return self.root.depth()

    def clone(self) -> "GPTree":
        return GPTree(self.root.clone())

    def to_string(self) -> str:
        def rec(n: Node) -> str:
            if n.kind == "feature":
                return str(n.value)
            op: Op = n.value
            if op.arity == 1:
                return f"{op.name}({rec(n.left)})"
            return f"{op.name}({rec(n.left)},{rec(n.right)})"
        return rec(self.root)

    @staticmethod
    def random_tree(features: List[str], ops: OperatorSet, max_depth: int) -> "GPTree":
        def grow(depth: int) -> Node:
            # depth==1 -> 必须feature
            if depth <= 1 or (depth < max_depth and random.random() < 0.35):
                return Node("feature", random.choice(features))
            op = ops.sample_op()
            if op.arity == 1:
                return Node("op", op, left=grow(depth - 1))
            return Node("op", op, left=grow(depth - 1), right=grow(depth - 1))

        depth = random.randint(2, max_depth)
        return GPTree(grow(depth))

    def eval(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        X: feature_name -> tensor [T,N]
        return factor: [T,N]
        """
        def rec(n: Node) -> torch.Tensor:
            if n.kind == "feature":
                return X[n.value]
            op: Op = n.value
            if op.arity == 1:
                return op.fn(rec(n.left))
            return op.fn(rec(n.left), rec(n.right))

        out = rec(self.root)
        # 防止爆炸
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # --------- 用于交叉/变异：节点采样与替换 ----------
    def _collect_nodes(self) -> List[Tuple[Node, Optional[Node], str]]:
        """
        返回 (node, parent, which_child) 方便替换
        which_child: "root" / "left" / "right"
        """
        out = []

        def rec(n: Node, parent: Optional[Node], which: str):
            out.append((n, parent, which))
            if n.left:
                rec(n.left, n, "left")
            if n.right:
                rec(n.right, n, "right")

        rec(self.root, None, "root")
        return out

    def random_subtree(self) -> Tuple[Node, Optional[Node], str]:
        nodes = self._collect_nodes()
        return random.choice(nodes)

    def replace_subtree(self, parent: Optional[Node], which: str, new_sub: Node):
        if which == "root":
            self.root = new_sub
        elif which == "left":
            parent.left = new_sub
        elif which == "right":
            parent.right = new_sub
        else:
            raise ValueError(which)
