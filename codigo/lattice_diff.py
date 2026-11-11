# -*- coding: utf-8 -*-
"""
lattice_diff.py
===============
Regime de DIFUSÃO (E > E_sep): passeio aleatório determinístico nos potenciais de rede
U(r) = Σ_{k∈S} [1 − cos(k·r)] (mesma normalização de lattice_nd.py: Σkkᵀ = I).

Acima da sela a partícula salta de poço em poço; a dinâmica hamiltoniana gera, sem ruído,
um passeio aleatório no reticulado de períodos. O programa mede esse passeio e o compara
com dois passeios markovianos de referência.

Medidas na dinâmica (uma integração em lote por energia, condições iniciais
microcanônicas sobre toda a camada de energia):
  • deslocamento quadrático médio ⟨|r(t) − r(0)|²⟩ (média temporal e sobre órbitas),
    expoente α (MSD ∝ t^α) e coeficiente de difusão D = MSD/(2N t);
  • k_cross: taxa de cruzamento da fronteira da célula de Voronoi do poço atual;
  • k_c: taxa de saltos COMPROMETIDOS (núcleo → núcleo de poços diferentes; o núcleo é
    a bola de raio |v_min|/4 em torno do mínimo), que descarta recruzamentos;
  • distribuição dos tempos de permanência e dos voos (sequências de saltos
    comprometidos na mesma direção) — caudas longas indicam voos de Lévy;
  • f_trap: fração de órbitas que nunca deixam a célula inicial.

Referências markovianas:
  • TST microcanônica (independente da dinâmica), com a célula de Voronoi como superfície
    divisória:
        k_TST = c_N Σ_faces ∫_face (2K)^{(N−1)/2} dS  /  ∫_célula (2K)^{(N−2)/2} dV,
        K = E − U,   c_N = Γ(N/2) / (2√π Γ((N+1)/2)),
    calculada por Monte Carlo (amostragem por importância em cada face).
    O passeio no grafo das selas tem D_TST = c_N Σ_v F_v |v|² / (2N Ω).
    Se a dinâmica fosse ergódica, k_cross = k_TST exatamente; k_cross/k_TST < 1 mede
    a fração do espaço de fase presa em órbitas regulares.
  • passeio markoviano efetivo, com a taxa e os saltos comprometidos medidos:
        D_M = k_c ⟨|Δ|²⟩ / (2N).
    D/D_M > 1: saltos correlacionados (voos); D/D_M < 1: retroespalhamento.
  • κ = k_c / k_TST: coeficiente de transmissão.

Uso: veja run_diff.py (Spyder: F5) ou  python lattice_diff.py --help
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import math
import sys
import time
import warnings
from functools import lru_cache

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lattice_nd as L

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

VERSION = "diff-1.0"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Geometria do reticulado de períodos
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def geometry(name):
    """Base P (colunas), vetores relevantes de Voronoi, |v_min| e raio de cobertura μ."""
    P, _ = L.periods(name)
    N = P.shape[0]
    Pinv = np.linalg.inv(P)
    co1 = np.array(list(itertools.product([-1, 0, 1], repeat=N)), float) @ P.T
    n1 = (co1 ** 2).sum(1)
    o = np.argsort(n1)
    S1 = co1[o][1:min(len(o), 400)]                      # vetores curtos (sem o zero)
    vmin = float(np.sqrt(n1[o][1]))
    # raio de cobertura (estimado por amostragem; usado só como limite generoso)
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(20000, N)) @ P.T
    mu = 0.0
    for i in range(0, len(X), 400):
        x = X[i:i + 400]
        qb = np.round(x @ Pinv.T) @ P.T
        d = np.linalg.norm(x[:, None, :] - qb[:, None, :] - co1[None], axis=2).min(1)
        mu = max(mu, float(d.max()))
    # candidatos: combinações {-2..2} (N ≤ 6) ou {-1,0,1} + somas de pares curtos (N > 6)
    if N <= 6:
        C = np.array(list(itertools.product(range(-2, 3), repeat=N)), float) @ P.T
    else:
        C = np.vstack([co1, (S1[:, None, :] + S1[None, :, :]).reshape(-1, N),
                       (S1[:, None, :] - S1[None, :, :]).reshape(-1, N)])
    nC = (C ** 2).sum(1)
    cut = (2 * mu * 1.15) ** 2
    C, nC = C[(nC > 1e-9) & (nC <= cut)], nC[(nC > 1e-9) & (nC <= cut)]
    C = np.unique(np.round(C, 8), axis=0)
    nC = (C ** 2).sum(1)
    o = np.argsort(nC)
    C, nC = C[o], nC[o]
    rel = []
    for i, v in enumerate(C):
        W = C[nC <= nC[i] + 1e-9]
        W = W[np.linalg.norm(W - v, axis=1) > 1e-9]
        if np.all((W * (W - v)).sum(1) > 1e-9 * nC[i]):
            rel.append(v)
    rel = np.array(rel)
    return dict(P=P, Pinv=Pinv, N=N, vmin=vmin, mu=mu, rel=rel, rel_n2=(rel ** 2).sum(1),
                co1=co1, vol=float(abs(np.linalg.det(P))))


def nearest_point(X, g):
    """Ponto do reticulado mais próximo (Babai + vizinhança {-1,0,1})."""
    qb = np.round(X @ g["Pinv"].T) @ g["P"].T
    out = np.empty_like(X)
    for i in range(0, len(X), 256):
        cand = qb[i:i + 256, None, :] + g["co1"][None]
        d = ((X[i:i + 256, None, :] - cand) ** 2).sum(2)
        out[i:i + 256] = cand[np.arange(len(cand)), d.argmin(1)]
    return out


@lru_cache(maxsize=None)
def U_max(name):
    lat = L.Lattice(name)
    g = geometry(name)
    N = g["N"]
    seeds = [g["P"] @ (np.array(c) / 2) for c in itertools.product([0, 1], repeat=N)]
    seeds += list(np.random.default_rng(1).uniform(size=(200, N)) @ g["P"].T)
    best = 0.0
    for x0 in seeds:
        r = minimize(lambda x: -lat.U(x), x0, jac=lambda x: -lat.grad(x), method="BFGS")
        best = max(best, -r.fun)
    return float(best)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Teoria do estado de transição microcanônica (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

def _c_N(N):
    return math.exp(math.lgamma(N / 2) - math.lgamma((N + 1) / 2)) / (2 * math.sqrt(math.pi))


def _ball_vol(d, R):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1) * R ** d


def _in_face(X, g, v, tol=1e-9):
    """X no plano bissetor de 0 e v: pertence à face se w·x ≤ |w|²/2 para todo relevante w."""
    return np.all(X @ g["rel"].T <= g["rel_n2"] / 2 + tol * g["vmin"] ** 2, axis=1)


def _face_minima(lat, g, v, Bp, Rf, rng):
    """Pontos de menor U na face (busca com múltiplos inícios e penalidade fora da face)."""
    def f(y):
        x = v / 2 + Bp @ y
        viol = np.clip(g["rel"] @ x - g["rel_n2"] / 2, 0, None)
        return lat.U(x) + 1e3 * (viol ** 2).sum() / g["vmin"] ** 2
    d = Bp.shape[1]
    starts = [np.zeros(d)] + list(rng.normal(size=(24, d)) * Rf / 3)
    found = []
    for y0 in starts:
        r = minimize(f, y0, method="L-BFGS-B")
        x = v / 2 + Bp @ r.x
        if _in_face(x[None], g, v, 1e-6)[0]:
            found.append((float(lat.U(x)), r.x))
    found.sort(key=lambda t: t[0])
    pts = []
    for u, y in found:
        if u <= found[0][0] + 1e-3 * max(1.0, abs(found[0][0])) and \
                not any(np.linalg.norm(y - p) < 1e-3 * Rf for _, p in pts):
            pts.append((u, y))
    return pts


def tst(name, energies, n_vol=300000, n_face=60000, seed=0):
    """k_TST, D_TST, fluxo por classe de face e erros estatísticos, para cada energia."""
    lat = L.Lattice(name)
    g = geometry(name)
    N, rng = g["N"], np.random.default_rng(seed)
    cN = _c_N(N)
    # volume de fase do poço (por célula): ∫ (2K)^{(N−2)/2} dV
    X = rng.uniform(size=(n_vol, N)) @ g["P"].T
    Uv = np.concatenate([lat.U(X[i:i + 20000]) for i in range(0, n_vol, 20000)])
    # classes de faces equivalentes
    keys = {}
    for v in g["rel"]:
        k = (round(float(v @ v), 5), round(float(lat.U(v / 2)), 5))
        keys.setdefault(k, []).append(v)
    classes = []
    for k, vs in keys.items():
        v = vs[0]
        Bp = np.linalg.svd(np.eye(N) - np.outer(v, v) / (v @ v))[0][:, :N - 1]
        Rf = math.sqrt(max((1.3 * g["mu"]) ** 2 - (v @ v) / 4, 1e-6))
        mins = _face_minima(lat, g, v, Bp, Rf, rng)
        classes.append(dict(v=v, count=len(vs), Bp=Bp, Rf=Rf, mins=mins,
                            n2=float(v @ v), Umin=mins[0][0] if mins else np.inf))
    out = []
    for E in energies:
        K = E - Uv
        w = np.where(K > 0, (2 * np.clip(K, 0, None)) ** ((N - 2) / 2), 0.0)
        Om, Om_err = g["vol"] * w.mean(), g["vol"] * w.std() / math.sqrt(n_vol)
        flux, flux_err, Dsum, Dsum_err = 0.0, 0.0, 0.0, 0.0
        per_class = []
        for c in classes:
            d = N - 1
            if E <= c["Umin"] or not c["mins"]:
                per_class.append(0.0)
                continue
            # proposta: mistura de uniforme no disco e gaussianas nos mínimos da face
            sig = np.clip(math.sqrt(2 * (E - c["Umin"])), 0.02 * c["Rf"], c["Rf"] / 2)
            n_u = n_face // 2
            dirs = rng.normal(size=(n_u, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            Yu = dirs * (c["Rf"] * rng.uniform(size=(n_u, 1)) ** (1 / d))
            m = len(c["mins"])
            comp = rng.integers(0, m, size=n_face - n_u)
            Yg = np.array([c["mins"][j][1] for j in comp]) + sig * rng.normal(size=(n_face - n_u, d))
            Y = np.vstack([Yu, Yg])
            Xf = c["v"] / 2 + Y @ c["Bp"].T
            qu = (np.linalg.norm(Y, axis=1) <= c["Rf"]) / _ball_vol(d, c["Rf"])
            qg = np.zeros(len(Y))
            for _, ym in c["mins"]:
                qg += np.exp(-((Y - ym) ** 2).sum(1) / (2 * sig ** 2)) / (2 * math.pi * sig ** 2) ** (d / 2)
            q = 0.5 * qu + 0.5 * qg / m
            Kf = E - lat.U(Xf)
            fw = np.where((Kf > 0) & _in_face(Xf, g, c["v"]),
                          (2 * np.clip(Kf, 0, None)) ** ((N - 1) / 2), 0.0) / q
            F, Ferr = fw.mean(), fw.std() / math.sqrt(len(fw))
            per_class.append(F)
            flux += c["count"] * F
            flux_err += (c["count"] * Ferr) ** 2
            Dsum += c["count"] * F * c["n2"]
            Dsum_err += (c["count"] * Ferr * c["n2"]) ** 2
        k = cN * flux / Om if Om > 0 else np.nan
        k_err = k * math.sqrt((math.sqrt(flux_err) / flux) ** 2 + (Om_err / Om) ** 2) if flux > 0 else np.nan
        D = cN * Dsum / (2 * N * Om) if Om > 0 else np.nan
        D_err = D * math.sqrt((math.sqrt(Dsum_err) / Dsum) ** 2 + (Om_err / Om) ** 2) if Dsum > 0 else np.nan
        out.append(dict(E=float(E), k_TST=float(k), k_TST_err=float(k_err), D_TST=float(D),
                        D_TST_err=float(D_err), face_flux=[float(x) for x in per_class]))
    face_info = [dict(n2=c["n2"], count=c["count"], Umin=c["Umin"], n_mins=len(c["mins"]))
                 for c in classes]
    kl = list(keys)
    rel_class = [kl.index((round(float(v @ v), 5), round(float(lat.U(v / 2)), 5)))
                 for v in g["rel"]]
    return out, dict(faces=face_info, rel_class=rel_class)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dinâmica
# ═══════════════════════════════════════════════════════════════════════════

def microcanonical_ics(lat, g, E, n, rng):
    N = g["N"]
    r, P = [], []
    while len(r) < n:
        X = rng.uniform(size=(4000, N)) @ g["P"].T
        K = E - lat.U(X)
        acc = (K > 0) & (rng.uniform(size=len(X)) < (np.clip(K, 0, None) / E) ** ((N - 2) / 2))
        for x, k in zip(X[acc], K[acc]):
            u = rng.normal(size=N)
            u /= np.linalg.norm(u)
            r.append(x)
            P.append(np.sqrt(2 * k) * u)
            if len(r) == n:
                break
    return np.array(r), np.array(P)


def run_diffusion(name, eps, E_sep, n_orbits, h, t_max, seed, dt_sample=0.1, max_store=8192):
    lat = L.Lattice(name)
    g = geometry(name)
    N, B = g["N"], n_orbits
    rng = np.random.default_rng(seed)
    E = eps * E_sep
    r, P = microcanonical_ics(lat, g, E, B, rng)
    H0 = 0.5 * (P ** 2).sum(1) + lat.U(r)
    n_steps = int(round(t_max / h))
    every = max(1, int(round(dt_sample / h)))
    n_samp = n_steps // every
    store_every = max(1, -(-n_samp // max_store))
    traj = np.zeros((n_samp // store_every + 1, B, N))
    traj[0] = r
    V0 = np.vstack([np.zeros(N), g["rel"]])                 # 0 e vizinhos de Voronoi
    q_cur = nearest_point(r, g)
    core = g["vmin"] / 4
    in_core = np.linalg.norm(r - q_cur, axis=1) < core
    q_core = q_cur.copy()                                   # último núcleo visitado
    t_core = np.where(in_core, 0.0, np.nan)
    n_cross = np.zeros(B, int)
    hops = [[] for _ in range(B)]                           # (t, Δ) de saltos comprometidos
    left = np.zeros(B, bool)
    g_cur = lat.grad(r)
    i_store = 1
    n_show, b_show = 6000, min(B, 4)
    show = np.zeros((n_show, b_show, N))                    # trecho inicial (Δt = 0,2)
    i_show = 0
    for step in range(1, n_steps + 1):
        P -= 0.5 * h * g_cur
        r += h * P
        g_cur = lat.grad(r)
        P -= 0.5 * h * g_cur
        if step % every:
            continue
        t = step * h
        # cruzamentos da célula de Voronoi
        d = ((r[:, None, :] - q_cur[:, None, :] - V0[None]) ** 2).sum(2)
        j = d.argmin(1)
        moved = j > 0
        if moved.any():
            far = moved & (np.sqrt(d[np.arange(B), j]) > 1.05 * g["mu"])
            q_new = q_cur + V0[j]
            if far.any():                                   # saltou mais de uma célula
                q_new[far] = nearest_point(r[far], g)
            n_cross += moved
            q_cur = np.where(moved[:, None], q_new, q_cur)
        # saltos comprometidos (entrada no núcleo de outro poço)
        incore = np.linalg.norm(r - q_cur, axis=1) < core
        newc = incore & (np.linalg.norm(q_cur - q_core, axis=1) > 1e-6 * g["vmin"])
        for b in np.nonzero(newc)[0]:
            hops[b].append((t, *(q_cur[b] - q_core[b])))
            q_core[b] = q_cur[b]
            left[b] = True
        if i_show < n_show and (step // every) % 2 == 0:
            show[i_show] = r[:b_show]
            i_show += 1
        if (step // every) % store_every == 0 and i_store < len(traj):
            traj[i_store] = r
            i_store += 1
    traj = traj[:i_store]
    H1 = 0.5 * (P ** 2).sum(1) + lat.U(r)

    # MSD por média temporal (lags logarítmicos)
    dt_st = every * store_every * h
    n_st = len(traj)
    lags = np.unique(np.geomspace(1, max(2, n_st // 4), 40).astype(int))
    msd = np.array([((traj[l:] - traj[:-l]) ** 2).sum(2).mean() for l in lags])
    tl = lags * dt_st
    # α e D na última década de lags
    m = tl >= tl[-1] / 10
    alpha = np.polyfit(np.log(tl[m]), np.log(msd[m]), 1)[0] if m.sum() >= 3 else np.nan
    D = np.polyfit(tl[m], msd[m], 1)[0] / (2 * N) if m.sum() >= 3 else np.nan

    # estatísticas de saltos comprometidos
    T_tot = B * t_max
    allh = [np.array(hh) for hh in hops if len(hh)]
    n_hops = sum(len(a) for a in allh)
    k_c = n_hops / T_tot
    if n_hops:
        Dv = np.vstack([a[:, 1:] for a in allh])
        d2 = float((Dv ** 2).sum(1).mean())
        taus = np.concatenate([np.diff(a[:, 0]) for a in allh if len(a) > 1]) \
            if any(len(a) > 1 for a in allh) else np.array([])
        flights = []
        for a in allh:
            run = 1
            for i in range(1, len(a)):
                if np.allclose(a[i, 1:], a[i - 1, 1:], atol=1e-6 * g["vmin"]):
                    run += 1
                else:
                    flights.append(run)
                    run = 1
            flights.append(run)
        flights = np.array(flights)
        jump_n = np.sqrt((Dv ** 2).sum(1)) / g["vmin"]
    else:
        d2, taus, flights, jump_n = np.nan, np.array([]), np.array([], int), np.array([])
    D_M = k_c * d2 / (2 * N) if n_hops else np.nan
    # sequência de saltos comprometidos (em coordenadas inteiras do reticulado)
    hop_b = np.concatenate([np.full(len(hh), b) for b, hh in enumerate(hops) if len(hh)]) \
        if n_hops else np.zeros(0, int)
    hop_c = np.round(np.vstack([np.array(hh)[:, 1:] for hh in hops if len(hh)]) @ g["Pinv"].T) \
        .astype(np.int64) if n_hops else np.zeros((0, N), np.int64)
    show = show[:i_show]
    return dict(E=E, eps=eps, msd_t=tl, msd=msd, alpha=alpha, D=D,
                hop_b=hop_b, hop_c=hop_c, traj_show=show,
                k_cross=float(n_cross.sum() / T_tot), k_c=float(k_c), D_M=float(D_M),
                d2=float(d2) if n_hops else np.nan, n_hops=int(n_hops),
                f_trap=float(np.mean(n_cross == 0)), f_noleave=float(np.mean(~left)),
                taus=taus, flights=flights, jump_n=jump_n,
                dH=float(np.max(np.abs(H1 - H0) / np.abs(H0))))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Execução
# ═══════════════════════════════════════════════════════════════════════════

def _task(name, eps, E_sep, args_d, path, meta):
    a = argparse.Namespace(**args_d)
    t0 = time.time()
    res = run_diffusion(name, eps, E_sep, a.n_orbits, a.h, a.t_max,
                        a.seed + int(1000 * eps) + 7 * L.ORDER.index(name))
    res["elapsed"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **res)
    os.replace(tmp, path)
    return dict(alpha=res["alpha"], D=res["D"], k_c=res["k_c"], f_trap=res["f_trap"],
                n_hops=res["n_hops"], elapsed=res["elapsed"], dH=res["dH"])


def _tst_job(name, energies):
    return tst(name, energies)


def simulate(args):
    os.makedirs(args.out, exist_ok=True)
    names = L.canonical_names(args.lattices)
    crit = L.load_critical(names, os.path.join(args.out, "critical_points_nd.json"), args.workers)
    # TST (uma tarefa por reticulado; reaproveitada entre rodadas)
    tst_path = os.path.join(args.out, "tst.json")
    TS = {}
    if os.path.exists(tst_path):
        with open(tst_path, encoding="utf-8") as fh:
            TS = json.load(fh)
    todo = [n for n in names if n not in TS or
            sorted(TS[n]["eps"]) != sorted(args.eps)]
    if todo:
        print(f"Teoria do estado de transição (Monte Carlo) para {len(todo)} reticulados ...",
              flush=True)
        jobs = [(n, (n, [e * crit[f"{n}|full"]["E_sep"] for e in args.eps])) for n in todo]
        for n, (res, fi) in L.run_tasks(_tst_job, jobs, args.workers):
            TS[n] = dict(eps=list(args.eps), res=res, faces=fi["faces"],
                         rel_class=fi["rel_class"], U_max=U_max(n))
            print(f"  {n:4s} U_max/E_sep = {TS[n]['U_max'] / crit[f'{n}|full']['E_sep']:.3f}  "
                  f"classes de faces = {len(fi['faces'])}  k_TST(1,1·E_sep) = "
                  f"{next((r['k_TST'] for e, r in zip(args.eps, res) if abs(e - 1.1) < 1e-9), np.nan):.3g}",
                  flush=True)
            with open(tst_path, "w", encoding="utf-8") as fh:
                json.dump(TS, fh, indent=1)
    # dinâmica
    meta = dict(version=VERSION, t_max=float(args.t_max), h=float(args.h),
                n_orbits=int(args.n_orbits), seed=int(args.seed))
    jobs = []
    for n in names:
        Es = crit[f"{n}|full"]["E_sep"]
        for e in args.eps:
            path = os.path.join(args.out, "tasks", L.safe(n), f"E_{e:.4f}.npz")
            if args.skip_done and L.task_done(path, dict(meta, section=False)):
                continue
            cost = L.lattice_info(n)["M"] * L.lattice_info(n)["N"] + len(geometry(n)["rel"])
            jobs.append(((n, e), (n, e, Es, vars(args), path, meta), cost))
    jobs.sort(key=lambda j: -j[2])
    jobs = [(k, a) for k, a, _ in jobs]
    print(f"Tarefas de dinâmica: {len(jobs)}  (T = {args.t_max}, h = {args.h}, "
          f"{args.n_orbits} órbitas, {args.workers} processos)", flush=True)
    t0 = time.time()
    for i, ((n, e), r) in enumerate(L.run_tasks(_task, jobs, args.workers), 1):
        print(f"  [{i:4d}/{len(jobs)}] {n:4s} E/E_sep = {e:5.2f}  α = {r['alpha']:.2f}  "
              f"D = {r['D']:.3g}  saltos = {r['n_hops']}  presas = {r['f_trap']:.2f}  "
              f"ΔH/H = {r['dH']:.1e}  ({r['elapsed']:.0f} s; total "
              f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    return crit, TS


# ═══════════════════════════════════════════════════════════════════════════
# 5. Análise
# ═══════════════════════════════════════════════════════════════════════════

FIELDS = ("E", "alpha", "D", "D_M", "k_cross", "k_c", "d2", "n_hops", "f_trap", "f_noleave", "dH")


def load(out, names, TS):
    R = {}
    for n in names:
        d = os.path.join(out, "tasks", L.safe(n))
        if not os.path.isdir(d):
            continue
        rows = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                with np.load(os.path.join(d, f)) as z:
                    rows.append({k: (z[k].copy() if z[k].ndim else float(z[k]))
                                 for k in z.files if k != "meta"})
        if not rows:
            continue
        rows.sort(key=lambda r: r["eps"])
        z = {"eps": np.array([r["eps"] for r in rows]), "raw": rows}
        for k in FIELDS:
            z[k] = np.array([r[k] for r in rows], float)
        ts = {round(e, 6): r for e, r in zip(TS[n]["eps"], TS[n]["res"])}
        for k in ("k_TST", "k_TST_err", "D_TST", "D_TST_err"):
            z[k] = np.array([ts.get(round(e, 6), {}).get(k, np.nan) for e in z["eps"]])
        R[n] = z
    return R


def _style(ax, logx=True, logy=True):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if logy:
        L._logy(ax)


def plot_lattice(n, z, crit, TS, out):
    info, Es = L.lattice_info(n), crit[f"{n}|full"]["E_sep"]
    um = TS[n]["U_max"] / Es
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.6))
    cm = plt.get_cmap("viridis")
    ax = axs[0, 0]
    for i, r in enumerate(z["raw"]):
        ax.loglog(r["msd_t"], r["msd"], color=cm(i / max(len(z["raw"]) - 1, 1)),
                  label=f"{r['eps']:g}")
    tt = z["raw"][-1]["msd_t"]
    y0 = z["raw"][-1]["msd"][0]
    ax.loglog(tt, y0 * (tt / tt[0]), "k:", lw=0.8)
    ax.loglog(tt, y0 * (tt / tt[0]) ** 2, "k--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle |r(t)-r(0)|^2 \rangle$")
    ax.set_title("(a) MSD  (pontilhado ∝ t, tracejado ∝ t²)")
    ax.legend(title="E/E_sep", fontsize=7, ncol=2)
    ax = axs[0, 1]
    ax.plot(z["eps"], z["alpha"], "o-")
    ax.axhline(1, color="0.5", ls=":")
    ax.axhline(2, color="0.5", ls="--")
    ax.set_ylim(0, 2.2)
    ax.set_title("(b) expoente α (MSD ∝ t^α)")
    _style(ax, logy=False)
    ax = axs[0, 2]
    ax.plot(z["eps"], z["D"], "o-", label="D (dinâmica)")
    ax.errorbar(z["eps"], z["D_TST"], z["D_TST_err"], fmt="s--", label="D_TST (grafo, TST)")
    ax.plot(z["eps"], z["D_M"], "^:", label="D_M (Markov, saltos medidos)")
    ax.set_title("(c) coeficiente de difusão")
    ax.legend(fontsize=8)
    _style(ax)
    ax = axs[1, 0]
    ax.plot(z["eps"], z["k_cross"] / z["k_TST"], "o-", label="k_cross / k_TST")
    ax.plot(z["eps"], z["k_c"] / z["k_TST"], "s-", label="κ = k_c / k_TST")
    ax.plot(z["eps"], z["f_trap"], "^-", label="órbitas presas")
    ax.axhline(1, color="0.5", ls=":")
    ax.set_title("(d) taxas relativas à TST e aprisionamento")
    ax.legend(fontsize=8)
    _style(ax, logy=False)
    ax = axs[1, 1]
    for i, r in enumerate(z["raw"]):
        tau = r["taus"]
        if len(tau) > 20:
            b = np.geomspace(max(tau.min(), 1e-2), tau.max(), 30)
            hh, e = np.histogram(tau, b, density=True)
            ax.loglog(np.sqrt(e[1:] * e[:-1]), np.where(hh > 0, hh, np.nan),
                      color=cm(i / max(len(z["raw"]) - 1, 1)))
    ax.set_xlabel("tempo de permanência τ")
    ax.set_ylabel("densidade")
    ax.set_title("(e) tempos entre saltos comprometidos")
    ax = axs[1, 2]
    for i, r in enumerate(z["raw"]):
        fl = r["flights"]
        if len(fl) > 20:
            ks = np.arange(1, fl.max() + 1)
            pk = np.array([(fl >= k).mean() for k in ks])
            ax.loglog(ks, pk, "o-", ms=3, color=cm(i / max(len(z["raw"]) - 1, 1)))
    ax.set_xlabel("comprimento do voo ℓ (saltos na mesma direção)")
    ax.set_ylabel("P(voo ≥ ℓ)")
    ax.set_title("(f) voos: cauda longa ⇒ Lévy")
    for ax in axs[0, 1:].tolist() + [axs[1, 0]]:
        ax.set_xlabel("$E/E_{sep}$")
        if um < z["eps"].max() * 1.05:
            ax.axvline(um, color="C3", ls="-.", lw=1)
    fig.suptitle(f"{n} — difusão (N = {info['N']}, {L.lattice_class(n)}); "
                 f"E_sep = {Es:.4g}, U_max = {um:.3f}·E_sep (linha vermelha)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figuras", f"{L.safe(n)}_difusao.png"), dpi=130)
    plt.close(fig)


def plot_comparisons(R, out):
    names = list(R)
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.8))
    for n in names:
        z = R[n]
        c = L.CLS_COLOR.get(L.lattice_class(n), "k")
        kw = dict(color=c, ms=4, lw=1.3)
        axs[0].plot(z["eps"], z["alpha"], "o-", **kw)
        axs[1].plot(z["eps"], z["D"] / z["D_TST"], "o-", **kw)
        axs[2].plot(z["eps"], z["k_cross"] / z["k_TST"], "o-", **kw)
        axs[3].plot(z["eps"], z["D"] / z["D_M"], "o-", **kw)
        for ax, y in zip(axs, (z["alpha"], z["D"] / z["D_TST"], z["k_cross"] / z["k_TST"],
                               z["D"] / z["D_M"])):
            ok = np.isfinite(y)
            if ok.any():
                ax.annotate(n, (z["eps"][ok][-1], y[ok][-1]), fontsize=7,
                            xytext=(2, 0), textcoords="offset points")
    tit = ("expoente α", "D / D_TST", "k_cross / k_TST (ergodicidade)", "D / D_M (correlação)")
    for ax, t, ly in zip(axs, tit, (False, True, False, True)):
        ax.set_title(t)
        ax.set_xlabel("$E/E_{sep}$")
        ax.axhline(1, color="0.5", ls=":")
        _style(ax, logy=ly)
    axs[0].axhline(2, color="0.5", ls="--")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=k) for k, c in L.CLS_COLOR.items()],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "figuras", "comparacao_difusao.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ── Pólya: retorno à origem e sítios distintos na sequência de saltos ────────
# Constantes de Pólya (probabilidade de retorno do passeio simples aos primeiros vizinhos).
POLYA = {2: 1.0, 3: 0.3405373, 4: 0.1932017, 5: 0.1351786, 6: 0.1047155,
         7: 0.0858449, 8: 0.0729126}
POLYA_FCC = 0.2563182          # A₃: o grafo de saltos é o FCC (12 vizinhos)
N_LIST = np.unique(np.r_[np.geomspace(1, 2000, 28).astype(int), 10, 100, 1000])


def _walk_stats(seqs, nlist=N_LIST, min_walkers=10):
    """R(n) = P(voltou à origem até o passo n); S(n)/(n+1) = fração de sítios distintos
    entre as n+1 posições (→ 1 − F, F = probabilidade de retorno de Pólya)."""
    if not seqs:
        return np.full(len(nlist), np.nan), np.full(len(nlist), np.nan), np.zeros(len(nlist), int)
    N = seqs[0].shape[1]
    hv = np.random.default_rng(99).integers(1, 2 ** 61, size=N, dtype=np.int64) | 1
    R, S, W = [], [], []
    for n in nlist:
        sel = [q[:n + 1] for q in seqs if len(q) >= n + 1]
        if len(sel) < min_walkers:
            R.append(np.nan); S.append(np.nan); W.append(len(sel))
            continue
        K = np.stack(sel) @ hv                                      # (w, n+1) chaves
        R.append(float(np.mean((K[:, 1:] == K[:, :1]).any(1))))
        Ks = np.sort(K, axis=1)
        S.append(float(np.mean(1 + (np.diff(Ks, axis=1) != 0).sum(1)) / (n + 1)))
        W.append(len(sel))
    return np.array(R), np.array(S), np.array(W)


def _markov_walks(jumps, weights, n_walk=2000, n_max=int(N_LIST[-1]), seed=5):
    rng = np.random.default_rng(seed)
    p = np.asarray(weights, float)
    p = p / p.sum()
    idx = rng.choice(len(jumps), size=(n_walk, n_max), p=p)
    pos = np.concatenate([np.zeros((n_walk, 1, jumps.shape[1]), np.int64),
                          np.cumsum(jumps[idx], axis=1)], axis=1)
    return list(pos)


def polya_analysis(n, z, TS):
    g = geometry(n)
    relc = np.round(g["rel"] @ g["Pinv"].T).astype(np.int64)
    ts_by_e = {round(e, 6): r for e, r in zip(TS[n]["eps"], TS[n]["res"])}
    out = []
    for r in z["raw"]:
        seqs = []
        hb, hc = r["hop_b"].astype(int), r["hop_c"].astype(np.int64)
        for b in np.unique(hb):
            c = hc[hb == b]
            seqs.append(np.vstack([np.zeros((1, c.shape[1]), np.int64), np.cumsum(c, axis=0)]))
        det = _walk_stats(seqs)
        iid = _walk_stats(_markov_walks(hc, np.ones(len(hc)))) if len(hc) > 20 else None
        tsr = ts_by_e.get(round(float(r["eps"]), 6))
        tstw = None
        if tsr is not None and sum(tsr["face_flux"]) > 0:
            w = np.array([tsr["face_flux"][k] for k in TS[n]["rel_class"]])
            tstw = _walk_stats(_markov_walks(relc, w))
        out.append(dict(eps=float(r["eps"]), det=det, iid=iid, tst=tstw))
    return out


def plot_polya(n, P, out):
    info = L.lattice_info(n)
    F = POLYA_FCC if n == "A3" else (POLYA.get(info["N"]) if n.startswith("Z") or info["N"] == 2
                                      else None)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    cm = plt.get_cmap("viridis")
    for i, p in enumerate(P):
        c = cm(i / max(len(P) - 1, 1))
        for k, (ax, lab) in enumerate(zip(axs, ("R(n)", "S(n)/n"))):
            ax.plot(N_LIST, p["det"][k], "-", color=c, lw=1.6, label=f"{p['eps']:g}" if k == 0 else None)
            if p["tst"] is not None:
                ax.plot(N_LIST, p["tst"][k], "--", color=c, lw=0.9)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("número de saltos n")
    if F is not None:
        axs[0].axhline(F, color="k", ls=":", lw=1.2, label=f"Pólya: {F:.4f}")
        if F < 1:
            axs[1].axhline(1 - F, color="k", ls=":", lw=1.2)
    axs[0].set_ylabel("P(retorno ao poço inicial até n)")
    axs[1].set_ylabel("sítios distintos / (n+1)   (→ 1 − F)")
    axs[0].set_title("(a) probabilidade de retorno")
    axs[1].set_title("(b) sítios distintos visitados")
    axs[0].legend(title="E/E_sep (— dinâmica, -- Markov TST)", fontsize=7, ncol=2,
                  loc="center right")
    fig.suptitle(f"{n} — comparação com o passeio de Pólya (N = {info['N']})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figuras", f"{L.safe(n)}_polya.png"), dpi=130)
    plt.close(fig)


def plot_traj_2d(R, crit, out):
    """Trajetórias 2D: trecho inicial de cada órbita, até sair de uma vizinhança de ~5 células."""
    names = [n for n in R if L.lattice_info(n)["N"] == 2]
    if not names:
        return
    sel_e = [1.02, 1.1, 1.5, 3.0]
    fig, axs = plt.subplots(len(names), len(sel_e), figsize=(4.4 * len(sel_e), 4.6 * len(names)),
                            squeeze=False)
    for i, n in enumerate(names):
        lat, g = L.Lattice(n), geometry(n)
        Rw = 5 * g["vmin"]
        for j, e in enumerate(sel_e):
            ax = axs[i, j]
            r = min(R[n]["raw"], key=lambda q: abs(q["eps"] - e))
            T = r["traj_show"]
            c0 = nearest_point(T[0], g)                      # centra no poço inicial
            xs = np.linspace(-Rw, Rw, 300)
            X, Y = np.meshgrid(xs, xs)
            ax.contourf(X, Y, lat.U(np.stack([X, Y], -1)), 30, cmap="Greys", alpha=0.45)
            ax.contour(X, Y, lat.U(np.stack([X, Y], -1)), [crit[f"{n}|full"]["E_sep"]],
                       colors="k", linewidths=0.5)
            for b in range(min(T.shape[1], 3)):
                tr = T[:, b] - c0[b]
                fora = np.nonzero(np.abs(tr).max(1) > Rw)[0]
                tr = tr[:fora[0]] if len(fora) else tr
                ax.plot(tr[:, 0], tr[:, 1], lw=0.7, alpha=0.9)
                ax.plot(*tr[0], "o", ms=4, color="k")
            ax.set_xlim(-Rw, Rw); ax.set_ylim(-Rw, Rw)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{n}, E/E_sep = {r['eps']:g}  (α = {r['alpha']:.2f})", fontsize=11)
    fig.suptitle("Trajetórias em 2D — início de 3 órbitas (ponto: posição inicial; "
                 "contorno preto: U = E_sep)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figuras", "trajetorias_2D.png"), dpi=130)
    plt.close(fig)


def write_report(R, crit, TS, out, args, PO=None):
    Lr = ["REGIME DE DIFUSÃO (E > E_sep) — PASSEIO ALEATÓRIO DETERMINÍSTICO\n",
          f"T = {args.t_max}, h = {args.h}, {args.n_orbits} órbitas/energia\n\n",
          "1. RETICULADOS\n",
          f"{'nome':5s} {'N':>2s} {'classe':>10s} {'E_sep':>8s} {'U_max/E_sep':>11s} "
          f"{'|v_min|':>8s} {'vizinhos':>8s}  faces (|v|², nº, U mínimo na face)\n"]
    for n in R:
        g = geometry(n)
        Lr.append(f"{n:5s} {g['N']:2d} {L.lattice_class(n):>10s} {crit[f'{n}|full']['E_sep']:8.4f} "
                  f"{TS[n]['U_max'] / crit[f'{n}|full']['E_sep']:11.3f} {g['vmin']:8.3f} "
                  f"{len(g['rel']):8d}  " + "; ".join(
                      f"({f['n2']:.1f}, {f['count']}, {f['Umin']:.3f})" for f in TS[n]["faces"]) + "\n")
    Lr.append("\n2. RESULTADOS POR RETICULADO\n"
              "   α: MSD ∝ t^α; D: difusão (dinâmica); D_TST: passeio no grafo com taxas TST;\n"
              "   D_M: Markov com saltos comprometidos medidos; k_x = k_cross; κ = k_c/k_TST\n")
    keys = ("alpha", "D", "D_TST", "D_M", "k_TST", "k_cross", "k_c", "f_trap", "n_hops", "dH")
    for n, z in R.items():
        Lr.append(f"\n   {n}\n   {'E/Esep':>7s} " + " ".join(f"{k:>9s}" for k in keys) +
                  f" {'D/D_TST':>9s} {'k_x/k_TST':>9s} {'κ':>9s} {'D/D_M':>9s}\n")
        for i, e in enumerate(z["eps"]):
            Lr.append(f"   {e:7.3f} " + " ".join(f"{z[k][i]:9.3g}" for k in keys) +
                      f" {z['D'][i] / z['D_TST'][i]:9.3g} {z['k_cross'][i] / z['k_TST'][i]:9.3g}"
                      f" {z['k_c'][i] / z['k_TST'][i]:9.3g} {z['D'][i] / z['D_M'][i]:9.3g}\n")
    if PO:
        n_ref = [10, 100, 1000]
        Lr.append("\n3. PÓLYA: retorno R(n) e sítios distintos S(n)/n na sequência de saltos\n"
                  "   din = dinâmica determinística; iid = Markov com os saltos medidos embaralhados;\n"
                  "   TST = passeio no grafo de vizinhos com pesos do fluxo TST\n")
        for n, P in PO.items():
            F = POLYA_FCC if n == "A3" else POLYA.get(L.lattice_info(n)["N"]) \
                if (n.startswith("Z") or L.lattice_info(n)["N"] == 2) else None
            Lr.append(f"\n   {n}" + (f"  (constante de Pólya do grafo simples: {F:.4f})" if F else "") +
                      "\n   " + f"{'E/Esep':>7s}" + "".join(
                          f" {w + '@' + str(m):>11s}" for m in n_ref
                          for w in ("R_din", "R_iid", "R_TST", "S/n_din", "S/n_TST")) + "\n")
            for p in P:
                vals = []
                for m in n_ref:
                    i = int(np.nonzero(N_LIST == m)[0][0]) if m in N_LIST else -1
                    g_ = lambda t, k: (t[k][i] if t is not None else np.nan)
                    vals += [g_(p["det"], 0), g_(p["iid"], 0), g_(p["tst"], 0),
                             g_(p["det"], 1), g_(p["tst"], 1)]
                Lr.append(f"   {p['eps']:7.3f}" + "".join(f" {v:11.3g}" for v in vals) + "\n")
    txt = "".join(Lr)
    with open(os.path.join(out, "resultados_difusao.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(out, "resultados_difusao.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reticulado", "N", "classe", "E/E_sep"] + list(keys) +
                   ["D_TST_err", "k_TST_err", "U_max/E_sep"])
        for n, z in R.items():
            for i, e in enumerate(z["eps"]):
                w.writerow([n, L.lattice_info(n)["N"], L.lattice_class(n), f"{e:g}"] +
                           [f"{z[k][i]:.6g}" for k in keys] +
                           [f"{z['D_TST_err'][i]:.3g}", f"{z['k_TST_err'][i]:.3g}",
                            f"{TS[n]['U_max'] / crit[f'{n}|full']['E_sep']:.4f}"])
    return txt


def analyze(args, crit=None, TS=None):
    names = L.canonical_names(args.lattices)
    if crit is None:
        with open(os.path.join(args.out, "critical_points_nd.json"), encoding="utf-8") as fh:
            crit = json.load(fh)
    if TS is None:
        with open(os.path.join(args.out, "tst.json"), encoding="utf-8") as fh:
            TS = json.load(fh)
    os.makedirs(os.path.join(args.out, "figuras"), exist_ok=True)
    R = load(args.out, [n for n in names if n in TS], TS)
    print(f"Analisando {len(R)} reticulados ...", flush=True)
    PO = {}
    for n, z in R.items():
        plot_lattice(n, z, crit, TS, args.out)
        PO[n] = polya_analysis(n, z, TS)
        plot_polya(n, PO[n], args.out)
    plot_comparisons(R, args.out)
    plot_traj_2d(R, crit, args.out)
    txt = write_report(R, crit, TS, args.out, args, PO)
    print(txt[:txt.index("2. RESULTADOS")])
    print(f"Relatório, CSV e figuras em '{args.out}'.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattices", nargs="+",
                    default=["Z2", "A2", "Z3", "A3", "Z4", "A4", "D4", "Z6", "A6", "E6",
                             "Z8", "A8", "E8"])
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0])
    ap.add_argument("--t-max", type=float, default=20000.0)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--n-orbits", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="resultados_difusao")
    ap.add_argument("--no-skip", dest="skip_done", action="store_false")
    ap.add_argument("--only-analysis", action="store_true")
    args = ap.parse_args(argv)
    args.workers = max(1, min(args.workers, 61))
    bad = [n for n in args.lattices if n not in L.ORDER]
    if bad:
        raise SystemExit(f"Reticulados desconhecidos: {bad}")
    assert all(e > 1 for e in args.eps), "o regime de difusão exige E/E_sep > 1"
    args.eps = sorted(args.eps)
    crit, TS = (None, None) if args.only_analysis else simulate(args)
    analyze(args, crit, TS)


if __name__ == "__main__":
    main()
