import argparse
import os
import warnings
import re

import numpy as np
import pandas as pd

from bench.data import clone_pairs, load_branches, load_larry

RATIOS = {"acc": ("ok", "n"), "miss_depth": ("depth", "miss"), "severe": ("severe", "n")}


def load_pairs(split=None, part=None):
    meta, clones = load_larry()
    day = meta["Time point"].to_numpy()
    branch = load_branches(len(meta))
    i, j, c = clone_pairs(day, clones)
    keep = pd.notna(branch[i]) & (branch[i] == branch[j])
    if split is not None:
        s = pd.read_csv(split)
        keep &= np.isin(c, s.loc[s["split"] == part, "clone"].to_numpy())
    i, j, c = i[keep], j[keep], c[keep]
    pairs = pd.DataFrame({"i": i, "j": j, "clone": c, "branch": branch[i].astype(str),
                          "daypair": [f"{int(x)}->{int(y)}" for x, y in zip(day[i], day[j])]})
    return pairs, branch


def branch_rank(x, branch):
    r = np.full(len(x), np.nan)
    ok = pd.notna(branch) & ~np.isnan(x)
    for b in pd.unique(branch[ok]):
        m = ok & (branch == b)
        r[m] = pd.Series(x[m]).rank(pct=True).to_numpy()
    return r


def pair_stats(x, pairs, branch, severe):
    rk = branch_rank(x, branch)
    early = rk[pairs["i"].to_numpy()]
    d = rk[pairs["j"].to_numpy()] - early
    valid = ~np.isnan(d)
    miss = valid & (d < 0)
    return early, dict(n=valid.astype(float),
                       ok=np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)),
                       miss=miss.astype(float),
                       depth=np.where(miss, -d, 0.0),
                       severe=(valid & (d < -severe)).astype(float))


def read_pt(path, n):
    pt = pd.read_csv(path, index_col="cell_id")
    ids = pt.index.to_numpy().astype(int)
    out = {}
    for r in [col for col in pt.columns if col != "label"]:
        out[r] = np.full(n, np.nan)
        out[r][ids] = pt[r].to_numpy()
    return out


def method_name(path):
    m = re.search(r"pt/(?P<fe>[^/]+)/(?P<be>[^/]+)/(?:(?P<rest>.*)/)?[^/]+_pseudotime\.csv$", path)
    if m is None:
        return os.path.basename(path), "-", "-"
    return m["fe"], m["be"], m["rest"] or "-"


def ratio(sums, key, w):
    num, den = RATIOS[key]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean((w @ sums[num]) / (w @ sums[den]), 0)


def summarize(sums, w):
    out = {}
    for key in RATIOS:
        r = ratio(sums, key, w)
        out[key] = np.column_stack([r, np.nanmean(r, 1)])
    return out


def ci(v):
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def evaluate(methods, pairs, branch, ref=None, severe=0.1, n_bins=10, n_boot=2000, seed=0):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ucl, ci_ = np.unique(pairs["clone"].to_numpy(), return_inverse=True)
    ubr, bi = np.unique(pairs["branch"].to_numpy(), return_inverse=True)
    cell = ci_ * len(ubr) + bi
    labels = list(ubr) + ["mean"]
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(ucl), len(ucl)), minlength=len(ucl))
                  for _ in range(n_boot)]).astype(float)
    ones = np.ones((1, len(ucl)))

    point, boot, root_acc, slices = {}, {}, {}, []
    for name, roots in methods.items():
        sums = {k: [] for k in ["n", "ok", "miss", "depth", "severe"]}
        for r, x in roots.items():
            early, st = pair_stats(x, pairs, branch, severe)
            for k, v in st.items():
                sums[k].append(np.bincount(cell, weights=v, minlength=len(ucl) * len(ubr)).reshape(len(ucl), len(ubr)))
            df = pairs[["branch", "daypair"]].assign(pos=np.minimum((early * n_bins) // 1, n_bins - 1), **st)
            for by in ["pos", "daypair"]:
                g = df.groupby(["branch", by])[list(st)].sum()
                g = pd.DataFrame({key: g[RATIOS[key][0]] / g[RATIOS[key][1]] for key in RATIOS}).assign(n=g["n"])
                slices.append(g.reset_index().rename(columns={by: "value"}).assign(
                    slice=by, frontend=name[0], backend=name[1], setting=name[2], root=r))
        sums = {k: np.stack(v) for k, v in sums.items()}
        point[name] = summarize(sums, ones)
        boot[name] = summarize(sums, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            root_acc[name] = np.nanmean(sums["ok"].sum(1) / sums["n"].sum(1), -1)
        point[name]["n_roots"], point[name]["n_pairs"] = len(roots), np.append(sums["n"][0].sum(0), sums["n"][0].sum())

    rows = []
    for name in methods:
        rn = (ref, name[1], name[2])
        for b, lab in enumerate(labels):
            row = dict(frontend=name[0], backend=name[1], setting=name[2], branch=lab,
                       n_roots=point[name]["n_roots"], n_pairs=int(point[name]["n_pairs"][b]),
                       root_sd=np.std(root_acc[name], ddof=1) if lab == "mean" and len(root_acc[name]) > 1 else np.nan)
            for key in RATIOS:
                row[key] = point[name][key][0, b]
                row[key + "_lo"], row[key + "_hi"] = ci(boot[name][key][:, b])
            if ref is not None and rn in methods and name != rn:
                for key in RATIOS:
                    d = boot[name][key][:, b] - boot[rn][key][:, b]
                    row["d_" + key] = point[name][key][0, b] - point[rn][key][0, b]
                    row["d_" + key + "_lo"], row["d_" + key + "_hi"] = ci(d)
                    row["p_" + key] = min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
            rows.append(row)
    scores = pd.DataFrame(rows)
    keys = ["slice", "frontend", "backend", "setting", "branch", "value"]
    sl = pd.concat(slices).groupby(keys)[list(RATIOS) + ["n"]].mean().reset_index()
    return scores, sl


def score(a):
    pairs, branch = load_pairs(a.split, a.part)
    methods = {method_name(path): read_pt(path, len(branch)) for path in a.pts}
    scores, slices = evaluate(methods, pairs, branch, ref=a.ref, severe=a.severe, n_bins=a.n_bins,
                              n_boot=a.n_boot, seed=a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    scores.to_csv(os.path.join(a.outdir, "scores.csv"), index=False)
    slices.to_csv(os.path.join(a.outdir, "slices.csv"), index=False)
    cols = ["frontend", "backend", "setting", "acc", "acc_lo", "acc_hi", "miss_depth", "severe"]
    cols += [c for c in ["d_acc", "d_acc_lo", "d_acc_hi", "p_acc"] if c in scores]
    print(f"{a.part}: {pairs['clone'].nunique()} clones, {len(pairs)} pairs, severe = d < -{a.severe}")
    print(scores.loc[scores["branch"] == "mean", cols].round(3).to_string(index=False))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pt = pd.read_csv(a.pt, index_col="cell_id")
    meta = pd.read_csv(a.meta, sep="\t", usecols=["SPRING-x", "SPRING-y", "Cell type annotation"])
    m = meta.iloc[pt.index.to_numpy()]
    roots = [c for c in pt.columns if c != "label"]
    rng = np.random.default_rng(a.seed)
    show = np.sort(rng.choice(len(pt), min(a.max_cells, len(pt)), replace=False))
    x, y = m["SPRING-x"].to_numpy()[show], m["SPRING-y"].to_numpy()[show]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = m["Cell type annotation"].to_numpy()[show]
    for lab in sorted(set(labels)):
        k = labels == lab
        axes[0].scatter(x[k], y[k], s=1, alpha=0.5, label=lab, rasterized=True)
    axes[0].legend(markerscale=6, fontsize=7, frameon=False)
    axes[0].set_title("labels")
    r = pd.Series(pt[roots[0]].to_numpy()).rank(pct=True).to_numpy()[show]
    o = np.argsort(np.nan_to_num(r, nan=-1))
    sc_ = axes[1].scatter(x[o], y[o], c=r[o], s=1, cmap="plasma", rasterized=True)
    fig.colorbar(sc_, ax=axes[1], label="rank")
    for rt in [r for r in roots if str(r).isdigit()]:
        axes[1].scatter(meta["SPRING-x"].iloc[int(rt)], meta["SPRING-y"].iloc[int(rt)], s=60, c="cyan",
                        edgecolors="black", marker="*")
    axes[1].set_title(f"pseudotime rank, root {roots[0]} (stars: all roots)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(a.pt)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("score")
    s.add_argument("--pts", nargs="+", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--part", choices=["dev", "test"], required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--ref")
    s.add_argument("--severe", type=float, default=0.1)
    s.add_argument("--n-bins", type=int, default=10)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=score)

    s = sub.add_parser("plot")
    s.add_argument("--pt", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-cells", type=int, default=30000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=plot)

    a = p.parse_args()
    a.func(a)
