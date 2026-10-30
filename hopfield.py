"""Hopfield associative-memory capacity: Hebbian vs. pseudoinverse (projection) learning.

Stores P random +-1 patterns in N=200 binary neurons, cues with a corrupted pattern (10% flipped),
runs asynchronous updates to convergence and measures retrieval overlap m = (1/N) x_final . x_target.
Sweeps load alpha = P/N. Theory: Hebbian capacity ~0.138 N (statistical mechanics, Amit-Gutfreund-Sompolinsky).
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 200

def train(P, rule, rng):
    X = rng.choice([-1, 1], (P, N))
    if rule == "hebbian":
        W = X.T @ X / N
    else:                                               # pseudoinverse / projection rule
        W = X.T @ np.linalg.pinv(X @ X.T) @ X
    np.fill_diagonal(W, 0)
    return X, W

def recall(W, cue, rng, sweeps=30):
    x = cue.copy()
    for _ in range(sweeps):
        changed = False
        for i in rng.permutation(N):
            s = 1 if W[i] @ x >= 0 else -1
            if s != x[i]: x[i] = s; changed = True
        if not changed: break
    return x

def overlap_curve(rule, alphas, trials=5, flip=0.10, seed=0):
    rng = np.random.default_rng(seed); out = []
    for a in alphas:
        P = max(1, int(round(a * N))); ms = []
        for _ in range(trials):
            X, W = train(P, rule, rng)
            for mu in rng.choice(P, min(P, 6), replace=False):
                cue = X[mu].copy(); idx = rng.choice(N, int(flip * N), replace=False); cue[idx] *= -1
                ms.append(abs(recall(W, cue, rng) @ X[mu]) / N)
        out.append(float(np.mean(ms)))
    return out

if __name__ == "__main__":
    alphas = np.round(np.arange(0.02, 0.62, 0.04), 2)
    res = {r: overlap_curve(r, alphas) for r in ("hebbian", "pseudoinverse")}
    cap = {r: next((float(a) for a, m in zip(alphas, v) if m < 0.9), None) for r, v in res.items()}
    out = dict(N=N, alphas=alphas.tolist(), mean_overlap=res, first_alpha_below_0p9_overlap=cap)
    json.dump(out, open("results.json", "w"), indent=2); print(json.dumps(out["first_alpha_below_0p9_overlap"]))
    for r in res: print(r, np.round(res[r], 2))
    plt.figure(figsize=(5.5, 3.5))
    for r, v in res.items(): plt.plot(alphas, v, "o-", ms=3, label=r)
    plt.axvline(0.138, color="k", ls="--", lw=0.8, label="theory 0.138 (Hebbian)")
    plt.xlabel("load P/N"); plt.ylabel("mean retrieval overlap"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("capacity.png", dpi=150)
