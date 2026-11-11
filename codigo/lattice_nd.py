# -*- coding: utf-8 -*-
"""
lattice_nd.py
=============
Dinâmica confinada (E < E_sep) de osciladores de rede em 2 a 8 dimensões, comparando
o potencial cossenoidal COMPLETO com o seu truncamento HARMÔNICO + QUÁRTICO.

Potenciais (S = um representante de cada par ±k de vetores de norma mínima do dual Λ*):
    completo:  U(r) = V0 Σ_{k∈S} [1 − cos(k·r)]
    quártico:  U(r) = V0 Σ_{k∈S} [(k·r)²/2 − (k·r)⁴/24]
Os k são reescalados para que Σ k kᵀ = I, isto é, todos os modos normais têm ω = 1.
Isso só muda as unidades de r e de t: E_sep, a anisotropia e os expoentes por unidade
de ω₀ são comparáveis entre reticulados.

Reticulados (n = 2…8): ℤⁿ, Aₙ, Aₙ*, Dₙ, Dₙ* (n ≥ 3), E₆, E₆*, E₇, E₇*, E₈.
Potenciais isométricos são detectados automaticamente e calculados uma só vez:
    A₂* ≡ A₂,  D₃ ≡ A₃ (FCC),  D₃* ≡ A₃* (BCC),  D₄* ≡ D₄,  Dₙ ≡ ℤⁿ para n ≥ 5.
(E₈* = E₈ não aparece; F₄ não é listado porque seu reticulado de raízes é D₄.)

Limiar de confinamento
----------------------
completo: E_sep = menor valor de U nos pontos críticos que não são mínimos. A busca usa
          sementes nos meios-períodos (pontos críticos exatos, pois U é par e periódico),
          nos máximos de U ao longo dos períodos curtos e sementes aleatórias na célula.
          Cota superior de controle: UB = min_p max_t U(t p) sobre períodos curtos p.
quártico: o potencial é ilimitado inferiormente; E_sep4 = menor sela (U > 0), com cota
          UB = min_u max_t U(t u) sobre direções u. Só energias E < 0,97·E_sep4 são rodadas.

Diagnósticos (uma integração em lote por energia; Velocity-Verlet + mapa tangente exato)
---------------------------------------------------------------------------------------
  • Expoentes de Lyapunov (N−1 maiores, Benettin + QR) e h_KS;
  • λ₁ em janelas, medido só após o início do regime exponencial t* de cada órbita;
  • SALI, GALI₂ e fração de órbitas caóticas (SALI < 1e-8);
  • momento angular: |L|² = r²P² − (r·P)² (invariante de rotação; conservado se e só se
    o potencial é central) — δL² = (max − min)/(max + min) ao longo da órbita;
  • frequência dominante de r_j − iP_j, difusão de frequência (Laskar), espectro;
  • amplitude ⟨|r|⟩ e max|r|;
  • seção de Poincaré r_N = 0, P_N > 0 projetada em (r₁, P₁) nas energias escolhidas.

Cada tarefa (reticulado, modelo, energia) é gravada em um .npz próprio: a rodada pode
ser interrompida e retomada, e só refaz o que falta ou o que tem outra configuração.

Uso: veja run_nd.py (Spyder: F5) ou  python lattice_nd.py --help
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import root

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

VERSION = "nd-1.0"
SALI_CHAOS = 1e-8
MODELS = ("full", "quartic")
MODEL_PT = {"full": "completo", "quartic": "quártico"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Sistemas de vetores
# ═══════════════════════════════════════════════════════════════════════════

def _e(n, i):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def _reps(V, tol=1e-9):
    """Um representante por par ±v, sem repetições."""
    out = []
    for v in V:
        if not any(np.allclose(v, w, atol=tol) or np.allclose(v, -w, atol=tol) for w in out):
            out.append(v)
    return np.array(out)


def _roots_A(n):
    return np.array([_e(n + 1, i) - _e(n + 1, j)
                     for i in range(n + 1) for j in range(i + 1, n + 1)])


def _mins_Astar(n):
    return np.array([_e(n + 1, i) - 1.0 / (n + 1) for i in range(n + 1)])


def _roots_D(n):
    return np.array([v for i in range(n) for j in range(i + 1, n)
                     for v in (_e(n, i) + _e(n, j), _e(n, i) - _e(n, j))])


def _mins_Dstar(n):
    half = np.array([0.5 * np.array(s) for s in itertools.product([1, -1], repeat=n)
                     if s[0] == 1])
    if n > 4:
        return np.eye(n)
    if n == 4:
        return np.vstack([np.eye(4), half])
    return half


def _roots_E8():
    R = list(_roots_D(8)) + [0.5 * np.array(s, float)
                            for s in itertools.product([1, -1], repeat=8)
                            if s[0] == 1 and np.prod(s) == 1]
    return np.array(R)


def _E_systems():
    F = np.vstack([_roots_E8(), -_roots_E8()])
    a = _e(8, 0) + _e(8, 1)
    E7r = _reps(F[np.abs(F @ a) < 1e-9])
    E7m = _reps(F[np.abs(F @ a - 1) < 1e-9] @ (np.eye(8) - np.outer(a, a) / 2))
    a1, a2 = _e(8, 0) - _e(8, 1), _e(8, 1) - _e(8, 2)
    A = np.vstack([a1, a2])
    E6r = _reps(F[(np.abs(F @ a1) < 1e-9) & (np.abs(F @ a2) < 1e-9)])
    Pr = F @ (np.eye(8) - A.T @ np.linalg.inv(A @ A.T) @ A)
    E6m = _reps(Pr[np.abs((Pr ** 2).sum(1) - 4 / 3) < 1e-9])
    assert (len(E7r), len(E7m), len(E6r), len(E6m)) == (63, 28, 36, 27)
    return {"E8": _roots_E8(), "E7": E7m, "E7*": E7r, "E6": E6m, "E6*": E6r}


def _raw(name):
    """Vetores de norma mínima do DUAL do reticulado `name` (um por par ±)."""
    if name[0] == "E":
        return _E_systems()[name]
    dual = name.endswith("*")
    fam, n = name[0], int(name.rstrip("*")[1:])
    if fam == "Z":
        return np.eye(n)
    if fam == "A":
        return _roots_A(n) if dual else _mins_Astar(n)
    if fam == "D":
        return _roots_D(n) if dual else _mins_Dstar(n)
    raise ValueError(name)


def _to_coords(V):
    """Coordenadas ortonormais do subespaço gerado por V."""
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    N = int((s > 1e-9 * s[0]).sum())
    return V @ Vt[:N].T


ORDER = []
for _n in range(2, 9):
    ORDER += [f"Z{_n}", f"A{_n}", f"A{_n}*"]
    if _n >= 3:
        ORDER += [f"D{_n}", f"D{_n}*"]
    if _n == 6:
        ORDER += ["E6", "E6*"]
    if _n == 7:
        ORDER += ["E7", "E7*"]
    if _n == 8:
        ORDER += ["E8"]


def safe(name):
    return name.replace("*", "s")


@lru_cache(maxsize=None)
def lattice_info(name):
    """S normalizado (Σ k kᵀ = I) e propriedades geométricas."""
    S = _to_coords(_raw(name))
    M, N = S.shape
    C = S.T @ S
    c = np.trace(C) / N
    iso_err = float(np.linalg.norm(C - c * np.eye(N)) / c)
    S = S / np.sqrt(c)
    # ordem da primeira anisotropia: menor n par com Σ (k·u)ⁿ não constante na esfera
    u = np.random.default_rng(7).normal(size=(20000, N))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = u @ S.T
    order = None
    for n in range(2, 14, 2):
        f = (p ** n).sum(1)
        if (f.max() - f.min()) / f.mean() > 1e-7:
            order = n
            break
    q4 = (p ** 4).sum(1)
    G = S @ S.T
    separable = bool(np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-9))
    inv = np.sort(np.round(np.sort(np.abs(G), axis=1), 6), axis=0)
    return dict(S=S, N=N, M=M, iso_err=iso_err, aniso_order=order,
                t=(order - 1) if order else 13, quartic_aniso=float(q4.max() / q4.min()),
                separable=separable, invariant=(N, M, inv.tobytes()))


@lru_cache(maxsize=None)
def aliases():
    """{nome: nome canônico} para potenciais isométricos (mesmo invariante)."""
    seen, out = {}, {}
    for n in ORDER:
        key = lattice_info(n)["invariant"]
        out[n] = seen.setdefault(key, n)
    return out


def canonical_names(names=None):
    al = aliases()
    names = ORDER if names is None else names
    out = []
    for n in names:
        c = al[n]
        if c not in out:
            out.append(c)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Modelo dinâmico
# ═══════════════════════════════════════════════════════════════════════════

class Lattice:
    def __init__(self, name, model="full"):
        info = lattice_info(name)
        self.name, self.model = name, model
        self.K = info["S"]
        self.KT = self.K.T.copy()
        self.M, self.N = self.K.shape
        self.info = info

    # f0, f1 = f0', f2 = f0''  (por termo, em função de s = k·r)
    def f0(self, s):
        return 2 * np.sin(s / 2) ** 2 if self.model == "full" else s * s / 2 - s ** 4 / 24

    def f1(self, s):
        return np.sin(s) if self.model == "full" else s - s ** 3 / 6

    def f2(self, s):
        return np.cos(s) if self.model == "full" else 1 - s * s / 2

    def U(self, r):
        return self.f0(r @ self.KT).sum(-1)

    def grad(self, r):
        return self.f1(r @ self.KT) @ self.K

    def hess(self, r):
        return (self.KT * self.f2(r @ self.KT)) @ self.K

    def hess_apply(self, c2, D):
        """H·D para D (B, N, q), dado c2 = f2(K r) de forma (B, M)."""
        return self.KT @ (c2[:, :, None] * (self.K @ D))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Períodos e pontos críticos
# ═══════════════════════════════════════════════════════════════════════════

def _int_row_basis(Z):
    A = [list(map(int, row)) for row in Z]
    m, n, r = len(A), len(A[0]), 0
    for c in range(n):
        if r >= m:
            break
        while True:
            nz = [i for i in range(r, m) if A[i][c] != 0]
            if not nz:
                break
            i0 = min(nz, key=lambda i: abs(A[i][c]))
            A[r], A[i0] = A[i0], A[r]
            clean = True
            for i in range(r + 1, m):
                if A[i][c]:
                    q = A[i][c] // A[r][c]
                    A[i] = [a - q * b for a, b in zip(A[i], A[r])]
                    clean = clean and A[i][c] == 0
            if clean:
                break
        if A[r][c] != 0:
            r += 1
    return np.array(A[:r], float)


def _lll(B, delta=0.75):
    """LLL nas colunas de B."""
    B = B.copy()
    n = B.shape[1]

    def gs(B):
        Q, mu = np.zeros_like(B), np.zeros((n, n))
        for i in range(n):
            v = B[:, i].copy()
            for j in range(i):
                mu[i, j] = B[:, i] @ Q[:, j] / (Q[:, j] @ Q[:, j])
                v -= mu[i, j] * Q[:, j]
            Q[:, i] = v
        return Q, mu
    Q, mu, k = *gs(B), 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                B[:, k] -= q * B[:, j]
                Q, mu = gs(B)
        if Q[:, k] @ Q[:, k] >= (delta - mu[k, k - 1] ** 2) * (Q[:, k - 1] @ Q[:, k - 1]):
            k += 1
        else:
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            Q, mu = gs(B)
            k = max(k - 1, 1)
    return B


@lru_cache(maxsize=None)
def periods(name):
    """Base reduzida (colunas) do reticulado de períodos de U e os períodos mais curtos."""
    S = lattice_info(name)["S"]
    M, N = S.shape
    idx = []
    for i in range(M):
        if np.linalg.matrix_rank(S[idx + [i]], tol=1e-8) > len(idx):
            idx.append(i)
        if len(idx) == N:
            break
    B0 = S[idx]
    C = S @ np.linalg.inv(B0)
    d = next(d for d in range(1, 721) if np.abs(d * C - np.round(d * C)).max() < 1e-7)
    Bspan = _int_row_basis(np.round(d * C).astype(np.int64))[:N] / d @ B0
    P = _lll(2 * np.pi * np.linalg.inv(Bspan))
    chk = S @ P / (2 * np.pi)
    assert np.abs(chk - np.round(chk)).max() < 1e-7, "períodos inválidos"
    co = np.array(list(itertools.product([-1, 0, 1], repeat=N)), float)
    V = co @ P.T
    nv = np.linalg.norm(V, axis=1)
    V, nv = V[nv > 1e-9], nv[nv > 1e-9]
    o = np.argsort(nv)
    short = []
    for i in o:
        v = V[i]
        if not any(np.allclose(v, w) or np.allclose(v, -w) for w in short):
            short.append(v)
        if len(short) >= 40:
            break
    return P, np.array(short)


def _classify(lat, x):
    u = float(lat.U(x))
    ev = np.linalg.eigvalsh(lat.hess(x))
    tol = 1e-6 * max(1.0, np.abs(ev).max())
    n_neg, n_zero = int((ev < -tol).sum()), int((np.abs(ev) <= tol).sum())
    return u, n_neg, n_zero


def critical_points(name, model, seed=0):
    lat = Lattice(name, model)
    N, rng = lat.N, np.random.default_rng(seed)
    g, J = lat.grad, lat.hess
    seeds = []
    if model == "full":
        P, short = periods(name)
        Pinv = np.linalg.inv(P)
        ts = np.linspace(0, 1, 401)[:, None]
        UB = np.inf
        for p in short:
            seeds.append(p / 2)
            Ul = lat.U(ts * p)
            UB = min(UB, Ul.max())
            seeds.append(ts[np.argmax(Ul), 0] * p)
        seeds += list((P @ rng.uniform(-0.5, 0.5, size=(N, 1500 + 300 * N))).T)
        fold = lambda x: P @ (Pinv @ x - np.round(Pinv @ x))
        pmin = float(np.linalg.norm(short[0]))
    else:
        kn = np.sqrt(N / lat.M)
        R0 = np.sqrt(6) / kn
        dirs = rng.normal(size=(3000, N))
        dirs = np.vstack([dirs / np.linalg.norm(dirs, axis=1, keepdims=True),
                          lat.K / kn, -lat.K / kn])
        ts = np.linspace(0, 4 * R0, 601)
        umax, targ = np.empty(len(dirs)), np.empty(len(dirs))
        for j0 in range(0, len(dirs), 200):            # em blocos: memória limitada
            Ur = lat.U(ts[None, :, None] * dirs[j0:j0 + 200, None, :])
            umax[j0:j0 + 200] = Ur.max(1)
            targ[j0:j0 + 200] = ts[np.argmax(Ur, 1)]
        UB = float(umax.min())
        for j in np.argsort(umax)[:300]:
            seeds.append(targ[j] * dirs[j])
        seeds += list(dirs[:1500] * rng.uniform(0.3, 1.6, size=(1500, 1)) * R0)
        fold = lambda x: x
        pmin = 4 * R0
    found = {}
    for x0 in seeds:
        try:
            sol = root(g, x0, jac=J, method="hybr")
        except Exception:
            continue
        if not sol.success or np.linalg.norm(g(sol.x)) > 1e-9:
            continue
        x = fold(sol.x)
        if model == "quartic" and np.linalg.norm(x) > 6 * pmin:
            continue
        u, nn, nz = _classify(lat, x)
        key = (round(u, 6), nn, nz)
        dist = float(np.linalg.norm(x))
        if key not in found or dist < found[key]["dist"]:
            found[key] = dict(U=u, n_neg=nn, n_zero=nz, dist=dist, x=x.tolist())
    pts = sorted(found.values(), key=lambda p: (p["U"], p["n_neg"]))
    nonmin = [p for p in pts if (p["n_neg"] or p["n_zero"]) and p["U"] > 1e-8]
    E_found = min((p["U"] for p in nonmin), default=np.inf)
    E_sep = min(E_found, UB)
    sad = [p for p in nonmin if abs(p["U"] - E_found) < 1e-6]
    r_sep = min((p["dist"] for p in sad), default=np.nan)
    return dict(E_sep=float(E_sep), E_found=float(E_found), UB=float(UB), r_sep=r_sep,
                p_min=pmin, missed=bool(E_found > UB * (1 + 1e-3) + 1e-6),
                secondary_minima=sorted({round(p["U"], 6) for p in pts
                                         if not (p["n_neg"] or p["n_zero"]) and p["U"] > 1e-8}),
                classes=[(round(p["U"], 5), p["n_neg"], p["n_zero"]) for p in pts][:40])


def _crit_job(name, model):
    return critical_points(name, model)


def load_critical(names, path, workers):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cache = json.load(fh)
    todo = [(n, m) for n in names for m in MODELS if f"{n}|{m}" not in cache]
    if todo:
        print(f"Procurando pontos críticos ({len(todo)} casos) ...", flush=True)
        for (n, m), c in run_tasks(_crit_job, [((n, m), (n, m)) for n, m in todo], workers):
            cache[f"{n}|{m}"] = c
            print(f"  {n:5s} {MODEL_PT[m]:9s} E_sep = {c['E_sep']:.4f}  (UB = {c['UB']:.4f})"
                  f"{'  ATENÇÃO: sela não encontrada abaixo da cota' if c['missed'] else ''}",
                  flush=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=1)
    return {k: v for k, v in cache.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integração em lote
# ═══════════════════════════════════════════════════════════════════════════

def section_ics(lat, E, n, rng, rmax):
    N = lat.N
    t = np.linspace(0, rmax, 3000)
    r0, P0 = np.zeros((n, N)), np.zeros((n, N))
    for i in range(n):
        u = np.zeros(N)
        u[:N - 1] = rng.normal(size=N - 1)
        u /= np.linalg.norm(u)
        Ut = lat.U(t[:, None] * u)
        bad = np.nonzero(Ut >= E)[0]
        amax = t[bad[0] - 1] if len(bad) else rmax
        r = ((i + rng.uniform()) / n) * amax * u
        v = rng.normal(size=N)
        v[-1] = abs(v[-1]) + 1e-3
        v /= np.linalg.norm(v)
        r0[i], P0[i] = r, np.sqrt(2 * max(E - lat.U(r), 0.0)) * v
    return r0, P0


def _dominant_freqs(z, dt):
    n = z.shape[0]
    F = np.abs(np.fft.fft((z - z.mean(0)) * np.hanning(n)[:, None, None], axis=0))
    F[0] = 0
    k = F.argmax(0)
    take = lambda idx: np.log(np.take_along_axis(F, (idx % n)[None], 0)[0] + 1e-300)
    a, b, c = take(k - 1), take(k), take(k + 1)
    den = a - 2 * b + c
    kk = k + np.where(np.abs(den) > 1e-14, 0.5 * (a - c) / den, 0.0)
    kk = np.where(kk > n / 2, kk - n, kk)
    return np.abs(2 * np.pi * kk / (n * dt))


def run_batch(name, model, eps, E_sep, n_orbits, h, t_max, seed, rmax, keep_section=False,
              max_samples=8192):
    lat = Lattice(name, model)
    rng = np.random.default_rng(seed)
    E = eps * E_sep
    N, B = lat.N, n_orbits
    D2, kL = 2 * lat.N, max(1, lat.N - 1)
    q = kL + 2
    n_steps = int(round(t_max / h))
    reortho = max(1, int(round(1.0 / h)))
    renorm = max(1, int(round(0.1 / h)))
    sample_every = max(int(round(0.1 / h)), -(-n_steps // max_samples))
    r, P = section_ics(lat, E, B, rng, rmax)
    H0 = 0.5 * (P ** 2).sum(1) + lat.U(r)

    Wr, WP = np.zeros((B, N, q)), np.zeros((B, N, q))
    Q0 = np.linalg.qr(rng.normal(size=(B, D2, kL)))[0]
    w0 = rng.normal(size=(B, D2, 2))
    w0 /= np.linalg.norm(w0, axis=1, keepdims=True)
    Wr[:, :, :kL], WP[:, :, :kL] = Q0[:, :N], Q0[:, N:]
    Wr[:, :, kL:], WP[:, :, kL:] = w0[:, :N], w0[:, N:]

    s = r @ lat.KT
    c2, g = lat.f2(s), lat.f1(s) @ lat.K
    h2 = 0.5 * h
    lsum = np.zeros((B, kL))
    n_qr = n_steps // reortho
    ck_every = max(1, n_qr // 3000)
    n_ck = n_qr // ck_every
    lhist, thist = np.zeros((n_ck, B, kL)), np.zeros(n_ck)
    n_samp = n_steps // sample_every
    zs = np.zeros((n_samp, B, N), complex)
    rsum, rmx = np.zeros(B), np.zeros(B)
    L2min, L2max = np.full(B, np.inf), np.zeros(B)
    esc = np.zeros(B, bool)
    sec_xy, sec_id = [], []
    rN_prev = r[:, -1].copy()
    ick = isamp = iqr = 0

    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(1, n_steps + 1):
            Ph = P - h2 * g
            WP -= h2 * lat.hess_apply(c2, Wr)
            r1o, P1o = r[:, 0].copy(), P[:, 0].copy()
            r = r + h * Ph
            Wr += h * WP
            s = r @ lat.KT
            c2, g = lat.f2(s), lat.f1(s) @ lat.K
            P = Ph - h2 * g
            WP -= h2 * lat.hess_apply(c2, Wr)

            if keep_section:
                rN = r[:, -1]
                for b in np.nonzero((rN_prev < 0) & (rN >= 0) & (P[:, -1] > 0))[0]:
                    if len(sec_id) < 6000 * B:
                        a = -rN_prev[b] / (rN[b] - rN_prev[b])
                        sec_xy.append((r1o[b] + a * (r[b, 0] - r1o[b]),
                                       P1o[b] + a * (P[b, 0] - P1o[b])))
                        sec_id.append(b)
                rN_prev = rN.copy()

            if step % renorm == 0:
                ws = np.concatenate([Wr[:, :, kL:], WP[:, :, kL:]], 1)
                ws /= np.linalg.norm(ws, axis=1, keepdims=True)
                Wr[:, :, kL:], WP[:, :, kL:] = ws[:, :N], ws[:, N:]

            if step % reortho == 0:
                X = np.concatenate([Wr[:, :, :kL], WP[:, :, :kL]], 1)
                Qm, Rm = np.linalg.qr(X)
                lsum += np.log(np.abs(np.diagonal(Rm, axis1=1, axis2=2)) + 1e-300)
                Wr[:, :, :kL], WP[:, :, :kL] = Qm[:, :N], Qm[:, N:]
                iqr += 1
                if iqr % ck_every == 0 and ick < n_ck:
                    thist[ick] = step * h
                    lhist[ick] = lsum / (step * h)
                    ick += 1

            if step % sample_every == 0 and isamp < n_samp:
                rn = np.linalg.norm(r, axis=1)
                bad = ~np.isfinite(rn) | (rn > rmax)
                if model == "quartic":
                    bad |= lat.U(r) < -0.5
                if bad.any():                  # órbita escapou: congela na origem
                    esc |= bad
                    r[bad], P[bad] = 0.0, 0.0
                    s = r @ lat.KT
                    c2, g = lat.f2(s), lat.f1(s) @ lat.K
                    rn = np.linalg.norm(r, axis=1)
                zs[isamp] = r - 1j * P
                isamp += 1
                rsum += rn
                rmx = np.maximum(rmx, rn)
                L2 = (r ** 2).sum(1) * (P ** 2).sum(1) - ((r * P).sum(1)) ** 2
                L2min, L2max = np.minimum(L2min, L2), np.maximum(L2max, L2)

    H1 = 0.5 * (P ** 2).sum(1) + lat.U(r)
    lhist, thist = lhist[:ick], thist[:ick]
    # expoentes extrapolados: λ(t) = λ∞ + a/t + b ln t / t nos últimos 60 %
    i0 = int(0.4 * ick)
    tt = thist[i0:]
    A = np.stack([np.ones_like(tt), 1 / tt, np.log(tt) / tt], 1)
    coef = np.linalg.lstsq(A, lhist[i0:].reshape(len(tt), -1), rcond=None)[0]
    lam = -np.sort(-coef[0].reshape(B, kL), axis=1)
    lam_raw = -np.sort(-lsum / (n_steps * h), axis=1)
    hks = np.clip(lam, 0, None).sum(1)

    ws = np.concatenate([Wr[:, :, kL:], WP[:, :, kL:]], 1)
    ws /= np.linalg.norm(ws, axis=1, keepdims=True)
    w1, w2 = ws[:, :, 0], ws[:, :, 1]
    sali = np.maximum(np.minimum(np.linalg.norm(w1 + w2, axis=1),
                                 np.linalg.norm(w1 - w2, axis=1)), 1e-16)
    gali2 = np.maximum(np.linalg.norm(w1 - (w1 * w2).sum(1)[:, None] * w2, axis=1), 1e-16)

    # λ₁ em janelas, só após o início do regime exponencial
    S_EXP, NW, NF = 10.0, 20, 200
    S1 = lhist[:, :, 0] * thist[:, None]
    lam_b = np.maximum(S1[-1] / thist[-1], 1e-12)
    tstar = np.full(B, np.inf)
    for b in range(B):
        kk = np.nonzero(S1[:, b] - np.log1p(thist) > S_EXP)[0]
        if len(kk):
            tstar[b] = max(thist[kk[0]], 10 / lam_b[b])

    def windows(n):
        edges = np.linspace(0, thist[-1], n + 1)
        Se = np.array([np.interp(edges, np.r_[0, thist], np.r_[0, S1[:, b]])
                       for b in range(B)]).T
        lw = np.diff(Se, axis=0) / np.diff(edges)[:, None]
        return edges, np.where(edges[:-1][:, None] >= tstar[None, :], lw, np.nan)
    we, wl = windows(NW)
    _, fl = windows(NF)
    tc = 0.5 * (we[:-1] + we[1:])
    wslope = np.full(B, np.nan)
    for b in range(B):
        ok = np.isfinite(wl[:, b]) & (wl[:, b] > 0)
        if ok.sum() >= 4:
            wslope[b] = np.polyfit(np.log(tc[ok]), np.log(wl[ok, b]), 1)[0]

    dt = h * sample_every
    zs = zs[:isamp]
    nh = isamp // 2
    nu1, nu2 = _dominant_freqs(zs[:nh], dt), _dominant_freqs(zs[nh:2 * nh], dt)
    nu = _dominant_freqs(zs, dt)
    dnu = np.maximum(np.max(np.abs(nu2 - nu1) / np.maximum(nu1, 1e-12), axis=1), 1e-12)
    fr = 2 * np.pi * np.fft.rfftfreq(isamp, dt)
    keep = fr <= 4.0
    spec = lambda b: np.abs(np.fft.rfft((zs[:, b, 0].real - zs[:, b, 0].real.mean())
                                        * np.hanning(isamp))) ** 2
    ok = ~esc
    b_reg = int(np.argmax(np.where(ok, sali, -1)))
    b_cha = int(np.argmin(np.where(ok, sali, 9)))

    out = dict(
        E=E, eps=eps, lam=lam, lam_raw=lam_raw, hks=hks, sali=sali, gali2=gali2,
        nu=nu, dnu=dnu, r_mean=rsum / max(isamp, 1), r_max=rmx,
        dH=np.abs(H1 - H0) / np.maximum(np.abs(H0), 1e-300),
        dL2=(L2max - L2min) / np.maximum(L2max + L2min, 1e-300), esc=esc,
        tstar=tstar, win_t=tc, win_lam=wl, fine_lam=fl, win_slope=wslope,
        spec_f=fr[keep], spec_reg=spec(b_reg)[keep], spec_cha=spec(b_cha)[keep],
        sec_xy=np.array(sec_xy).reshape(-1, 2), sec_id=np.array(sec_id, int),
    )
    for k in ("lam", "lam_raw", "hks", "sali", "gali2", "nu", "dnu", "r_mean", "r_max",
              "dH", "dL2", "tstar", "win_slope"):
        a = out[k].astype(float)
        a[esc] = np.nan
        out[k] = a
    out["win_lam"][:, esc] = np.nan
    out["fine_lam"][:, esc] = np.nan
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. Tarefas, arquivos e execução
# ═══════════════════════════════════════════════════════════════════════════

def run_tasks(func, jobs, workers):
    if workers <= 1:
        for key, a in jobs:
            yield key, func(*a)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(func, *a): key for key, a in jobs}
        for f in as_completed(futs):
            yield futs[f], f.result()


def task_path(out, name, model, eps):
    return os.path.join(out, "tasks", model, safe(name), f"E_{eps:.4f}.npz")


def task_meta(args, keep_section):
    return dict(version=VERSION, t_max=float(args.t_max), h=float(args.h),
                n_orbits=int(args.n_orbits), seed=int(args.seed), section=bool(keep_section))


def task_done(path, meta):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            old = json.loads(str(z["meta"]))
        return all(old.get(k) == v for k, v in meta.items() if k != "section") and \
            (old.get("section") or not meta["section"])
    except Exception:
        return False


def _task(name, model, eps, E_sep, rmax, args_d, keep_section, path, meta):
    a = argparse.Namespace(**args_d)
    t0 = time.time()
    res = run_batch(name, model, eps, E_sep, a.n_orbits, a.h, a.t_max,
                    a.seed + int(1e4 * eps) + 7 * ORDER.index(name) + (0 if model == "full" else 3),
                    rmax, keep_section)
    res["elapsed"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **res)
    os.replace(tmp, path)
    return dict(lam_med=float(np.nanmedian(res["lam"][:, 0])),
                f_chaos=float(np.nanmean(np.where(np.isnan(res["sali"]), np.nan,
                                                  res["sali"] < SALI_CHAOS))),
                f_esc=float(res["esc"].mean()), elapsed=res["elapsed"])


def simulate(args):
    os.makedirs(args.out, exist_ok=True)
    names = canonical_names(args.lattices)
    al = aliases()
    skipped = [n for n in (args.lattices or ORDER) if al[n] != n]
    if skipped:
        print("Potenciais equivalentes (calculados uma vez): " +
              ", ".join(f"{n} ≡ {al[n]}" for n in skipped))
    crit = load_critical(names, os.path.join(args.out, "critical_points_nd.json"), args.workers)
    jobs, todo_info = [], []
    for name in names:
        for model in args.models:
            c = crit[f"{name}|{model}"]
            cf = crit[f"{name}|full"]
            for eps in args.eps:
                E = eps * cf["E_sep"]
                if E >= 0.97 * c["E_sep"]:
                    continue                    # acima da barreira deste modelo
                keep = any(abs(eps - e) < 1e-9 for e in args.eps_plot)
                meta = task_meta(args, keep)
                path = task_path(args.out, name, model, eps)
                if args.skip_done and task_done(path, meta):
                    continue
                rmax = c["p_min"] if model == "full" else 3 * max(c["r_sep"], 1.0)
                cost = lattice_info(name)["M"] * lattice_info(name)["N"] ** 2
                # energia absoluta E = eps·E_sep(completo) nos dois modelos
                jobs.append(((name, model, eps), (name, model, eps, cf["E_sep"], rmax,
                                                  vars(args), keep, path, meta), cost))
    jobs.sort(key=lambda j: -j[2])
    jobs = [(k, a) for k, a, _ in jobs]
    print(f"Tarefas a rodar: {len(jobs)}  (T = {args.t_max}, h = {args.h}, "
          f"{args.n_orbits} órbitas, {args.workers} processos)", flush=True)
    t0 = time.time()
    for i, ((n, m, e), r) in enumerate(run_tasks(_task, jobs, args.workers), 1):
        print(f"  [{i:4d}/{len(jobs)}] {n:5s} {MODEL_PT[m]:9s} E/E_sep = {e:6.3f}  "
              f"λ_med = {r['lam_med']:.2e}  f_caos = {r['f_chaos']:.2f}"
              f"{'  escapes = %.2f' % r['f_esc'] if r['f_esc'] > 0 else ''}  "
              f"({r['elapsed']:.0f} s; total {(time.time() - t0) / 60:.1f} min)", flush=True)
    return crit


# ═══════════════════════════════════════════════════════════════════════════
# 6. Análise
# ═══════════════════════════════════════════════════════════════════════════

def _nanmed(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def summarize_task(z):
    ok = ~z["esc"].astype(bool)
    lam1 = z["lam"][:, 0]
    sali = z["sali"]
    tstar = z["tstar"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wmean = np.nanmean(z["win_lam"], axis=0)
        fl = z["fine_lam"][np.isfinite(z["fine_lam"])]
    return dict(
        E=float(z["E"]), n_ok=int(ok.sum()),
        lam_med=_nanmed(lam1), lam_mean=float(np.nanmean(lam1)) if ok.any() else np.nan,
        lam_q25=float(np.nanpercentile(lam1, 25)) if ok.any() else np.nan,
        lam_q75=float(np.nanpercentile(lam1, 75)) if ok.any() else np.nan,
        hks=float(np.nanmean(z["hks"])) if ok.any() else np.nan,
        f_chaos=float(np.mean(sali[ok] < SALI_CHAOS)) if ok.any() else np.nan,
        f_exp=float(np.mean(np.isfinite(tstar[ok]))) if ok.any() else np.nan,
        lam_win=_nanmed(wmean), win_slope=_nanmed(z["win_slope"]),
        sticky=float(np.mean(fl < 0.2 * np.median(fl))) if len(fl) else np.nan,
        dL2_med=_nanmed(z["dL2"]),
        dL2_q75=float(np.nanpercentile(z["dL2"], 75)) if ok.any() else np.nan,
        dnu_med=_nanmed(z["dnu"]), nu_mean=float(np.nanmean(z["nu"])) if ok.any() else np.nan,
        nu_std=float(np.nanstd(z["nu"])) if ok.any() else np.nan,
        r_mean=float(np.nanmean(z["r_mean"])) if ok.any() else np.nan,
        r_max=float(np.nanmax(z["r_max"])) if ok.any() else np.nan,
        dH_max=float(np.nanmax(z["dH"])) if ok.any() else np.nan,
        f_esc=float(z["esc"].mean()),
    )


def load_results(out, names):
    R = {}
    for name in names:
        for model in MODELS:
            d = os.path.join(out, "tasks", model, safe(name))
            if not os.path.isdir(d):
                continue
            rows = {}
            for f in sorted(os.listdir(d)):
                if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                    with np.load(os.path.join(d, f)) as z:
                        rows[float(z["eps"])] = summarize_task(z)
            if rows:
                eps = np.array(sorted(rows))
                R[(name, model)] = {"eps": eps, **{k: np.array([rows[e][k] for e in eps])
                                                   for k in next(iter(rows.values()))}}
    return R


def lattice_class(name):
    info = lattice_info(name)
    if info["separable"]:
        return "separável"
    if info["N"] == 2:
        return "N = 2"
    return f"t = {info['t']}"


CLS_COLOR = {"separável": "0.55", "N = 2": "C2", "t = 3": "C0", "t = 5": "C3", "t = 7": "C4"}


def _floor(R, N, model):
    """Piso de ruído: λ_med do ℤᴺ (separável, integrável) no mesmo modelo."""
    z = R.get((f"Z{N}", model))
    return float(np.nanmedian(z["lam_med"])) if z is not None else np.nan


def _logy(ax):
    """Escala log em y só se houver algum dado positivo (evita erro com eixos vazios)."""
    for ln in ax.get_lines():
        y = np.asarray(ln.get_ydata(), float)
        if (np.isfinite(y) & (y > 0)).any():
            ax.set_yscale("log")
            return
    ax.text(0.5, 0.5, "sem dados\n(nenhuma órbita no regime)", transform=ax.transAxes,
            ha="center", va="center", color="0.5", fontsize=9)


def plot_lattice(name, R, crit, out):
    info = lattice_info(name)
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.6))
    axs = axs.ravel()
    sty = {"full": dict(ls="-", marker="o"), "quartic": dict(ls="--", marker="s")}
    panels = [("lam_med", "λ₁ extrapolado (mediana)", "log"),
              ("lam_win", "λ₁ em janelas (órbitas exponenciais)", "log"),
              ("hks", "$h_{KS}$ (média)", "log"),
              ("f_chaos", "fração caótica (SALI)", "lin"),
              ("dL2_med", "δ|L|² (mediana)", "log"),
              ("r_mean", "⟨|r|⟩", "lin"),
              ("nu_mean", "ν dominante (média; barra: desvio)", "lin"),
              ("dnu_med", "|Δν|/ν (mediana)", "log")]
    for ax, (k, lab, sc) in zip(axs, panels):
        for model in MODELS:
            z = R.get((name, model))
            if z is None:
                continue
            y = z[k]
            if sc == "log":
                y = np.where(y > 0, y, np.nan)
            if k == "nu_mean":
                ax.errorbar(z["eps"], y, z["nu_std"], capsize=2, label=MODEL_PT[model],
                            **sty[model])
            else:
                ax.plot(z["eps"], y, label=MODEL_PT[model], **sty[model])
        if sc == "log":
            _logy(ax)
        if k == "f_chaos":
            ax.set_ylim(-0.03, 1.03)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("$E/E_{sep}$ (do potencial completo)")
        ax.set_title(lab, fontsize=11)
    fl = _floor(R, info["N"], "full")
    if np.isfinite(fl):
        axs[0].axhline(fl, color="0.5", ls=":", lw=1, label=f"piso (ℤ{info['N']})")
    axs[0].legend(fontsize=9)
    cf, cq = crit[f"{name}|full"], crit[f"{name}|quartic"]
    fig.suptitle(f"{name}   (N = {info['N']}, {info['M']} termos, {lattice_class(name)}, "
                 f"anisotropia quártica = {info['quartic_aniso']:.3f})   "
                 f"E_sep: completo = {cf['E_sep']:.4g}, quártico = {cq['E_sep']:.4g}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figuras", f"{safe(name)}_dinamica.png"), dpi=130)
    plt.close(fig)


def plot_poincare(name, out, eps_plot):
    rows = []
    for model in MODELS:
        for e in eps_plot:
            p = task_path(out, name, model, e)
            if os.path.exists(p):
                with np.load(p) as z:
                    if len(z["sec_id"]):
                        rows.append((model, e, z["sec_xy"].copy(), z["sec_id"].copy(),
                                     z["sali"].copy()))
    if not rows:
        return
    models = [m for m in MODELS if any(r[0] == m for r in rows)]
    ne = len(eps_plot)
    fig, axs = plt.subplots(len(models), ne, figsize=(4.3 * ne, 4.1 * len(models)),
                            squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)
    for model, e, xy, idx, sali in rows:
        i, j = models.index(model), [abs(e - x) < 1e-9 for x in eps_plot].index(True)
        ax = axs[i, j]
        ax.set_visible(True)
        c = np.nan_to_num(sali[idx], nan=1.0)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.6, lw=0, cmap="coolwarm_r",
                        norm=LogNorm(1e-12, 1))
        ax.set_title(f"{MODEL_PT[model]}, E/E_sep = {e:g}", fontsize=10)
        ax.set_xlabel("$r_1$")
        ax.set_ylabel("$P_1$")
    fig.colorbar(sc, ax=axs, fraction=0.02, label="SALI")
    fig.suptitle(f"{name}: seção $r_N = 0$, $P_N > 0$ projetada em $(r_1, P_1)$")
    fig.savefig(os.path.join(out, "figuras", f"{safe(name)}_poincare.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(R, names, out):
    dims = sorted({lattice_info(n)["N"] for n in names})
    for model in MODELS:
        for key, lab, fname in (("lam_med", "λ₁ (mediana)", "lambda"),
                                ("dL2_med", "δ|L|² (mediana)", "momento_angular"),
                                ("f_chaos", "fração caótica", "fracao_caotica")):
            fig, axs = plt.subplots(1, len(dims), figsize=(3.6 * len(dims), 3.9),
                                    squeeze=False, sharey=True)
            for ax, N in zip(axs[0], dims):
                for n in names:
                    if lattice_info(n)["N"] != N or (n, model) not in R:
                        continue
                    z = R[(n, model)]
                    y = z[key] if key == "f_chaos" else np.where(z[key] > 0, z[key], np.nan)
                    ax.plot(z["eps"], y, "o-", ms=3.5, lw=1.4,
                            color=CLS_COLOR.get(lattice_class(n), "k"), label=n)
                    ok = np.isfinite(y)
                    if ok.any():
                        ax.annotate(n, (z["eps"][ok][-1], y[ok][-1]), fontsize=8,
                                    xytext=(2, 0), textcoords="offset points")
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if key != "f_chaos":
                    _logy(ax)
                ax.set_title(f"N = {N}")
                ax.set_xlabel("$E/E_{sep}$")
            axs[0, 0].set_ylabel(lab)
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([], [], color=c, lw=2, label=k)
                                for k, c in CLS_COLOR.items()],
                       loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
            fig.suptitle(f"{lab} — potencial {MODEL_PT[model]}  "
                         "(cor: classe — separável, N = 2, força do design t)")
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(os.path.join(out, "figuras", f"comparacao_{fname}_{model}.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

    # completo × quártico, mesma energia absoluta
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, lab in ((axs[0], "lam_med", "λ₁ (mediana)"), (axs[1], "dL2_med", "δ|L|²")):
        allv = []
        for n in names:
            zf, zq = R.get((n, "full")), R.get((n, "quartic"))
            if zf is None or zq is None:
                continue
            common = np.intersect1d(np.round(zf["eps"], 6), np.round(zq["eps"], 6))
            for e in common:
                a = zf[key][np.isclose(zf["eps"], e)][0]
                b = zq[key][np.isclose(zq["eps"], e)][0]
                if a > 0 and b > 0:
                    ax.scatter(a, b, s=18 + 120 * e, color=CLS_COLOR.get(lattice_class(n), "k"),
                               alpha=0.75, lw=0)
                    allv += [a, b]
        if allv:
            lo, hi = min(allv) / 2, max(allv) * 2
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"{lab}, completo")
        ax.set_ylabel(f"{lab}, quártico")
        ax.set_title("mesma energia (tamanho do ponto ∝ E/E_sep)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=k)
                        for k, c in CLS_COLOR.items()], loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "figuras", "completo_vs_quartico.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # mapa de calor: fração caótica e λ
    epsall = sorted({round(float(e), 6) for z in R.values() for e in z["eps"]})
    for key, lab, norm in (("f_chaos", "fração caótica", None),
                           ("lam_med", "log₁₀ λ₁", "log")):
        fig, axs = plt.subplots(1, 2, figsize=(14, 0.32 * len(names) + 2), sharey=True)
        for ax, model in zip(axs, MODELS):
            Mx = np.full((len(names), len(epsall)), np.nan)
            for i, n in enumerate(names):
                z = R.get((n, model))
                if z is None:
                    continue
                for e, v in zip(z["eps"], z[key]):
                    Mx[i, epsall.index(round(float(e), 6))] = np.log10(v) if norm and v > 0 \
                        else (v if not norm else np.nan)
            im = ax.imshow(Mx, aspect="auto", cmap="magma" if norm else "viridis",
                           vmin=None if norm else 0, vmax=None if norm else 1)
            ax.set_xticks(range(len(epsall)))
            ax.set_xticklabels([f"{e:g}" for e in epsall], rotation=60, fontsize=8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([f"{n} ({lattice_class(n)})" for n in names], fontsize=8)
            ax.set_xlabel("$E/E_{sep}$")
            ax.set_title(MODEL_PT[model])
            fig.colorbar(im, ax=ax, label=lab)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figuras", f"mapa_{key}.png"), dpi=130)
        plt.close(fig)


def write_report(R, names, crit, out, args):
    al = aliases()
    L = ["DINÂMICA CONFINADA EM 2–8 DIMENSÕES — POTENCIAL COMPLETO × QUÁRTICO\n",
         f"T = {args.t_max}, h = {args.h}, {args.n_orbits} órbitas/energia, "
         f"caótica: SALI < {SALI_CHAOS:g}; normalização Σkkᵀ = I (ω = 1)\n\n",
         "1. RETICULADOS\n"]
    L.append(f"{'nome':6s} {'N':>2s} {'termos':>6s} {'classe':>10s} {'t':>3s} "
             f"{'aniso4':>7s} {'E_sep':>8s} {'UB':>8s} {'E_sep4':>8s} {'r_sep':>6s}  equivalentes\n")
    for n in names:
        i, cf, cq = lattice_info(n), crit[f"{n}|full"], crit[f"{n}|quartic"]
        eq = ", ".join(k for k, v in al.items() if v == n and k != n) or "—"
        L.append(f"{n:6s} {i['N']:2d} {i['M']:6d} {lattice_class(n):>10s} {i['t']:3d} "
                 f"{i['quartic_aniso']:7.3f} {cf['E_sep']:8.4f} {cf['UB']:8.4f} "
                 f"{cq['E_sep']:8.4f} {cf['r_sep']:6.3f}  {eq}"
                 f"{'  [sela não encontrada abaixo de UB]' if cf['missed'] else ''}\n")
    L.append("\n   Previsão (forma normal): regular a baixa energia se separável, N = 2 ou t ≥ 5\n"
             "   (termo quártico isotrópico); caótico desde E → 0 se t = 3 e N ≥ 3.\n"
             "   No modelo quártico com t ≥ 5 o potencial é central: |L|² é conservado exatamente.\n")
    L.append("\n2. RESUMO A BAIXA ENERGIA (E/E_sep ≤ 0,1)\n")
    L.append(f"{'nome':6s} {'classe':>10s} {'previsão':>9s} | {'λ_med comp':>10s} "
             f"{'λ_med quárt':>11s} {'f_caos c':>8s} {'f_caos q':>8s} {'δL² comp':>9s} "
             f"{'δL² quárt':>9s}\n")
    for n in names:
        pred = "regular" if (lattice_info(n)["separable"] or lattice_info(n)["N"] == 2
                             or lattice_info(n)["t"] >= 5) else "caótico"
        vals = []
        for model in MODELS:
            z = R.get((n, model))
            if z is None:
                vals.append((np.nan, np.nan, np.nan))
                continue
            m = z["eps"] <= 0.1 + 1e-9
            vals.append((np.nanmedian(z["lam_med"][m]) if m.any() else np.nan,
                         np.nanmax(z["f_chaos"][m]) if m.any() else np.nan,
                         np.nanmedian(z["dL2_med"][m]) if m.any() else np.nan))
        (lf, ff, df), (lq, fq, dq) = vals
        L.append(f"{n:6s} {lattice_class(n):>10s} {pred:>9s} | {lf:10.2e} {lq:11.2e} "
                 f"{ff:8.2f} {fq:8.2f} {df:9.2e} {dq:9.2e}\n")
    L.append("\n3. TABELAS POR RETICULADO\n")
    keys = ("E", "lam_med", "lam_win", "hks", "f_chaos", "f_exp", "dL2_med", "dnu_med",
            "nu_mean", "r_mean", "r_max", "dH_max", "f_esc")
    for n in names:
        for model in MODELS:
            z = R.get((n, model))
            if z is None:
                continue
            L.append(f"\n   {n} — {MODEL_PT[model]}\n   {'E/Esep':>7s} " +
                     " ".join(f"{k:>9s}" for k in keys) + "\n")
            for i, e in enumerate(z["eps"]):
                L.append(f"   {e:7.3f} " + " ".join(f"{z[k][i]:9.3g}" for k in keys) + "\n")
    txt = "".join(L)
    with open(os.path.join(out, "resultados_nd.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(out, "resultados_nd.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reticulado", "N", "termos", "classe", "t", "aniso4", "modelo", "E/E_sep"]
                   + list(keys))
        for n in names:
            i = lattice_info(n)
            for model in MODELS:
                z = R.get((n, model))
                if z is None:
                    continue
                for j, e in enumerate(z["eps"]):
                    w.writerow([n, i["N"], i["M"], lattice_class(n), i["t"],
                                f"{i['quartic_aniso']:.4f}", MODEL_PT[model], f"{e:g}"]
                               + [f"{z[k][j]:.6g}" for k in keys])
    return txt


def analyze(args, crit=None):
    names = canonical_names(args.lattices)
    if crit is None:
        with open(os.path.join(args.out, "critical_points_nd.json"), encoding="utf-8") as fh:
            crit = json.load(fh)
    os.makedirs(os.path.join(args.out, "figuras"), exist_ok=True)
    R = load_results(args.out, names)
    names = [n for n in names if any((n, m) in R for m in MODELS)]
    print(f"Analisando {len(names)} reticulados ...", flush=True)
    for n in names:
        plot_lattice(n, R, crit, args.out)
        plot_poincare(n, args.out, args.eps_plot)
    plot_comparisons(R, names, args.out)
    txt = write_report(R, names, crit, args.out, args)
    print(txt[:txt.index("3. TABELAS")])
    print(f"Relatório completo, CSV e figuras em '{args.out}'.")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Linha de comando
# ═══════════════════════════════════════════════════════════════════════════

def make_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattices", nargs="+", default=None, help=f"subconjunto de {ORDER}")
    ap.add_argument("--dims", nargs="+", type=int, default=None, help="só estas dimensões")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--eps-plot", nargs="+", type=float, default=[0.1, 0.5, 0.95])
    ap.add_argument("--t-max", type=float, default=5000.0)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--n-orbits", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="resultados_nd")
    ap.add_argument("--no-skip", dest="skip_done", action="store_false")
    ap.add_argument("--only-analysis", action="store_true")
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    args.workers = max(1, min(args.workers, 61))
    if args.lattices is None:
        args.lattices = list(ORDER)
    bad = [n for n in args.lattices if n not in ORDER]
    if bad:
        raise SystemExit(f"Reticulados desconhecidos: {bad}. Disponíveis: {ORDER}")
    if args.dims:
        args.lattices = [n for n in args.lattices if lattice_info(n)["N"] in args.dims]
    for e in args.eps_plot:
        if not any(abs(e - x) < 1e-9 for x in args.eps):
            args.eps.append(e)
    args.eps = sorted(args.eps)
    assert all(0 < e < 1 for e in args.eps), "use 0 < E/E_sep < 1"
    crit = None if args.only_analysis else simulate(args)
    analyze(args, crit)


if __name__ == "__main__":
    main()
