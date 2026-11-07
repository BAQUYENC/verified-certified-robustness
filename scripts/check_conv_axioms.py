"""Numerical checks of the axioms that the CNN extension adds to
operator_norms.dfy for convolutions (ConvGramIdentity, ConvCrop and ConvSchur),
and of the bound that Norm2Toep computes, on 2,592 random layers. It uses the
same index conventions as convolution.dfy (ToeplitzEntry) and GramStep, builds
each layer's matrix explicitly and compares with its spectral norm (by SVD, in
double precision), so every difference should be at the level of rounding.

Usage: python3 check_conv_axioms.py"""
import numpy as np

rng = np.random.default_rng(1)


def toeplitz(K, cIn, n, cOut, k, s, p):
    m = (n + 2 * p - k) // s + 1
    T = np.zeros((cOut * m * m, cIn * n * n))
    for o in range(cOut):
        for y in range(m):
            for x in range(m):
                for c in range(cIn):
                    for i in range(n):
                        for j in range(n):
                            a, b = i + p - y * s, j + p - x * s
                            if 0 <= a < k and 0 <= b < k:
                                T[o * m * m + y * m + x, c * n * n + i * n + j] = K[o, c, a, b]
    return T


def gram_step(F):
    cOut, cIn, k, _ = F.shape
    s = 2 * k - 1
    G = np.zeros((cIn, cIn, s, s))
    for i1 in range(cIn):
        for i2 in range(cIn):
            for u in range(s):
                for v in range(s):
                    du, dv = u - k + 1, v - k + 1
                    t = 0.0
                    for j in range(cOut):
                        for a in range(k):
                            for b in range(k):
                                a2, b2 = a + du, b + dv
                                if 0 <= a2 < k and 0 <= b2 < k:
                                    t += F[j, i1, a, b] * F[j, i2, a2, b2]
                    G[i1, i2, u, v] = t
    return G


def schur(F):
    col = max(np.abs(F[:, i]).sum() for i in range(F.shape[1]))
    row = max(np.abs(F[j]).sum() for j in range(F.shape[0]))
    return max(col, row)


def norm2toep(K, n, N):
    F, ex = K, []
    for _ in range(N):
        G = gram_step(F)
        c = np.sqrt((G ** 2).sum()) or 1.0
        H = G / c
        R = np.where(H >= 0, np.ceil(H * 1e16), np.floor(H * 1e16)) / 1e16
        e = schur(H - R)
        ex = [(c, e)] + ex
        F = R
    v = schur(F)
    for c, e in ex:
        v = np.sqrt((v + e) * c)
    return v


sigma = lambda T: np.linalg.norm(T, 2)
worst = {"gram": 0.0, "crop": -np.inf, "schur": -np.inf, "n2t": -np.inf}
cases = 0
for _ in range(300):
    cOut, cIn, k, n = rng.integers(1, 4), rng.integers(1, 4), rng.integers(1, 4), rng.integers(2, 6)
    K = rng.standard_normal((cOut, cIn, k, k))
    full = toeplitz(K, cIn, n, cOut, k, 1, k - 1)
    same_gram = toeplitz(gram_step(K), cIn, n, cIn, 2 * k - 1, 1, k - 1)
    worst["gram"] = max(worst["gram"], np.abs(full.T @ full - same_gram).max())
    for s in (1, 2, 3):
        for p in range(0, k + 1):
            if k > n + 2 * p:
                continue
            T = toeplitz(K, cIn, n, cOut, k, s, p)
            cases += 1
            worst["crop"] = max(worst["crop"], sigma(T) - sigma(full))
            worst["schur"] = max(worst["schur"], sigma(T) - schur(K))
            for N in (1, 2, 3):
                worst["n2t"] = max(worst["n2t"], sigma(T) - norm2toep(K, n, N))
print(f"{cases} layers (random filters, strides 1-3, paddings 0..k):")
print(f"  ConvGramIdentity: max |T^T T - SameConv(GramStep)| = {worst['gram']:.2e}")
print(f"  ConvCrop:  max sigma(T) - sigma(T_full)  = {worst['crop']:.2e}  (must be <= 0)")
print(f"  ConvSchur: max sigma(T) - SchurBound(K)   = {worst['schur']:.2e}  (must be <= 0)")
print(f"  Norm2Toep, N = 1..3: max sigma(T) - bound = {worst['n2t']:.2e}  (must be <= 0)")
