import argparse

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

DATA = "rt_data/"


def load_larry():
    meta = pd.read_csv(DATA + "stateFate_inVitro_metadata.txt.gz", sep="\t")
    clones = sio.mmread(DATA + "stateFate_inVitro_clone_matrix.mtx.gz").tocsr()
    return meta, clones


def clone_pairs(day, clones):
    clones = clones.tocsc()
    ii, jj, cc = [], [], []
    for c in range(clones.shape[1]):
        cells = clones.indices[clones.indptr[c]:clones.indptr[c + 1]]
        if len(cells) < 2:
            continue
        d = day[cells]
        a, b = np.triu_indices(len(cells), 1)
        keep = d[a] != d[b]
        a, b = cells[a[keep]], cells[b[keep]]
        swap = day[a] > day[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        ii.append(a)
        jj.append(b)
        cc.append(np.full(len(a), c))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(cc)


def load_branches(n):
    neu = pd.read_csv(DATA + "stateFate_inVitro_neutrophil_pseudotime.txt.gz", sep="\t")["Cell index"].to_numpy()
    nm = pd.read_csv(DATA + "stateFate_inVitro_neutrophil_monocyte_trajectory.txt.gz", sep="\t")["Cell index"].to_numpy()
    branch = np.full(n, None, dtype=object)
    branch[nm] = "Monocyte"
    branch[neu] = "Neutrophil"
    return branch


def cache(a):
    X = sio.mmread(a.normed).tocsr().astype(np.float32)
    sp.save_npz(a.out, X)


def raw_counts(X):
    rmin = np.minimum.reduceat(X.data, X.indptr[:-1])
    cnt = X.copy()
    cnt.data = np.round(cnt.data / np.repeat(rmin, np.diff(X.indptr)))
    return cnt


def thin(a):
    X = sp.load_npz(a.normed).tocsr()
    target = float(np.median(np.asarray(X.sum(1)).ravel())) * a.p
    cnt = raw_counts(X)
    del X
    raw = np.asarray(cnt.sum(1)).ravel().astype(np.float64)
    n = cnt.data.astype(np.int64)
    cnt.data = np.random.default_rng(a.seed).binomial(n, a.p).astype(np.float32)
    low = np.flatnonzero(raw * a.p < a.min_counts)
    rng = np.random.default_rng([a.seed, 1])
    for i in low:
        s, e = cnt.indptr[i], cnt.indptr[i + 1]
        cnt.data[s:e] = rng.binomial(n[s:e], min(1.0, a.min_counts / raw[i]))
    print("cells raised to floor", len(low))
    cnt.eliminate_zeros()
    tot = np.asarray(cnt.sum(1)).ravel()
    if (tot == 0).any():
        raise SystemExit(f"{int((tot == 0).sum())} cells lost all counts")
    out = sp.diags((target / tot).astype(np.float32)) @ cnt
    sp.save_npz(a.out, out.tocsr().astype(np.float32))
    print("p", a.p, "target", target, "mean raw total", tot.mean(), "nnz", out.nnz)


def split(a):
    labels = pd.read_csv(a.meta, sep="\t")["Cell type annotation"].to_numpy()
    cl = sio.mmread(a.clones).tocsc()
    rows = []
    for c in range(cl.shape[1]):
        lab = labels[cl.indices[cl.indptr[c]:cl.indptr[c + 1]]]
        mature = lab[lab != "Undifferentiated"]
        fate = pd.Series(mature).mode()[0] if len(mature) else "Undifferentiated"
        rows.append((c, len(lab), fate))
    df = pd.DataFrame(rows, columns=["clone", "size", "fate"])
    df["size_bin"] = pd.cut(df["size"], [0, 2, 4, 8, 16, np.inf], labels=["1-2", "3-4", "5-8", "9-16", "17+"])
    rng = np.random.default_rng(a.seed)
    df["split"] = "test"
    for _, g in df.groupby(["fate", "size_bin"], observed=True):
        dev = rng.choice(g.index.to_numpy(), size=int(round(a.dev_frac * len(g))), replace=False)
        df.loc[dev, "split"] = "dev"
    df.to_csv(a.out, index=False)


def counts(a):
    import anndata as ad
    import scanpy as sc
    X = sp.load_npz(a.normed).tocsr()
    genes = pd.read_csv(a.genes, header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(a.meta, sep="\t")
    rmin = np.minimum.reduceat(X.data, X.indptr[:-1])
    cnt = X.copy()
    cnt.data = cnt.data / np.repeat(rmin, np.diff(X.indptr))
    dev = np.abs(cnt.data - np.round(cnt.data)).max()
    if dev > 1e-3:
        raise SystemExit(f"Counts not integer, max deviation {dev}")
    cnt.data = np.round(cnt.data).astype(np.float32)
    norm = sc.AnnData(X)
    sc.pp.log1p(norm)
    sc.pp.highly_variable_genes(norm, n_top_genes=a.n_hvg)
    hvg = norm.var["highly_variable"].to_numpy()
    obs = pd.DataFrame({"label": meta["Cell type annotation"].to_numpy(),
                        "time": meta["Time point"].to_numpy().astype(float),
                        "n_genes_by_counts": np.diff(cnt.indptr),
                        "total_counts": np.asarray(cnt.sum(1)).ravel()},
                       index=pd.Index(np.arange(len(meta)).astype(str), name="cell_id"))
    out = ad.AnnData(X=cnt[:, hvg].tocsr(), obs=obs, var=pd.DataFrame(index=genes[hvg]))
    out.uns["max_integer_deviation"] = float(dev)
    out.write_h5ad(a.out)
    print(out, "max deviation", dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("cache")
    s.add_argument("--normed", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cache)

    s = sub.add_parser("thin")
    s.add_argument("--normed", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--p", type=float, required=True)
    s.add_argument("--min-counts", type=float, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=thin)

    s = sub.add_parser("split")
    s.add_argument("--meta", required=True)
    s.add_argument("--clones", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dev-frac", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=split)

    s = sub.add_parser("counts")
    s.add_argument("--normed", required=True)
    s.add_argument("--genes", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-hvg", type=int, default=2000)
    s.set_defaults(func=counts)

    a = p.parse_args()
    a.func(a)
