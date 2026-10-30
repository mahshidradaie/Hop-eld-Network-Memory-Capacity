# Hopfield Associative-Memory Capacity: Hebbian vs. Pseudoinverse Learning

N = 200 binary neurons store P random ±1 patterns; recall starts from a cue with 10 % of bits flipped and runs
asynchronous updates to convergence. Retrieval overlap is measured against the target pattern while the load
α = P/N is swept (5 networks × 6 cues per load).

```
pip install numpy matplotlib
python hopfield.py     # writes results.json and capacity.png
```

## Results (`results.json`)
| Rule | Overlap at α=0.10 | Overlap at α=0.18 | First α with mean overlap < 0.9 |
|---|---|---|---|
| Hebbian | 0.91 | 0.73 | 0.18 |
| Pseudoinverse | 1.00 | 1.00 | 0.54 |

Hebbian retrieval degrades around α ≈ 0.14, consistent with the Amit–Gutfreund–Sompolinsky capacity of 0.138 N.
The pseudoinverse (projection) rule stays near-perfect to α ≈ 0.4 and degrades gradually after.

## Caveats
Small N (finite-size effects blur the sharp theoretical transition), random uncorrelated patterns, few trials; the
0.9-overlap cut-off is a practical criterion, not the exact theoretical capacity.
