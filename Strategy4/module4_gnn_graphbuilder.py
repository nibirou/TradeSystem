# module_gnn_graphbuilder.py（相关性图 TopK）
import numpy as np

def build_corr_graph(returns_mat: np.ndarray, topk: int) -> np.ndarray:
    """
    returns_mat: [N, L]  N stocks, L lookback returns
    输出：A_hat [N,N]（GCN 归一化）
    """
    R = np.nan_to_num(returns_mat, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(R)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    N = corr.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k = min(topk, N - 1)
        idx = np.argpartition(-corr[i], kth=k)[:k+1]
        A[i, idx] = corr[i, idx]
    A = np.maximum(A, A.T)

    A = A + np.eye(N, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    A_hat = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
    return A_hat.astype(np.float32)
