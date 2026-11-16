import argparse
import os
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd


def load(a):
    data = ad.read_h5ad(a.adata)
    if a.latent_key not in data.obsm:
        raise SystemExit(f"'{a.latent_key}' not in obsm")
    missing = [r for r in getattr(a, "roots", []) if r not in data.obs_names]
    if missing:
        raise SystemExit(f"Roots not in obs_names: {', '.join(missing)}")
    return data


def frame(data, a):
    out = pd.DataFrame(index=pd.Index(data.obs_names, name="cell_id"))
    if a.label_col in data.obs:
        out["label"] = data.obs[a.label_col].to_numpy()
    return out


def write(out, a):
    os.makedirs(a.outdir, exist_ok=True)
    out.to_csv(os.path.join(a.outdir, f"{a.prefix}_pseudotime.csv"))


def pick_roots(latent, pool, n_roots, pool_size, seed):
    center = latent[pool].mean(0)
    dist = ((latent[pool] - center) ** 2).sum(1)
    near = pool[np.argsort(dist)[:pool_size]]
    return np.random.default_rng(seed).choice(near, size=n_roots, replace=False)


def roots(a):
    data = load(a)
    latent = np.asarray(data.obsm[a.latent_key], dtype=np.float32)
    pool = np.flatnonzero((data.obs[a.label_col].to_numpy() == a.start) & (data.obs[a.time_col].to_numpy() == a.start_time))
    if len(pool) < a.pool_size or a.pool_size < a.n_roots:
        raise SystemExit("Pool too small")
    chosen = pick_roots(latent, pool, a.n_roots, a.pool_size, a.seed)
    with open(a.out, "w") as f:
        f.write("\n".join(data.obs_names[chosen]) + "\n")


def euclid(a):
    data = load(a)
    x = np.asarray(data.obsm[a.latent_key], dtype=np.float64)
    out = frame(data, a)
    for r in a.roots:
        out[r] = np.linalg.norm(x - x[data.obs_names.get_loc(r)], axis=1)
    write(out, a)


def dpt(a):
    import scanpy as sc
    data = load(a)
    key = a.neighbors_key
    if key is None:
        key = "neighbors"
        sc.pp.neighbors(data, n_neighbors=a.n_neighbors, use_rep=a.latent_key, transformer="sklearn")
    elif key not in data.uns:
        raise SystemExit(f"Graph '{key}' not in uns")
    sc.tl.diffmap(data, n_comps=a.n_dcs + 5, neighbors_key=key)
    out = frame(data, a)
    for r in a.roots:
        data.uns["iroot"] = int(data.obs_names.get_loc(r))
        sc.tl.dpt(data, n_dcs=a.n_dcs, neighbors_key=key)
        pt = data.obs["dpt_pseudotime"].to_numpy().copy()
        pt[~np.isfinite(pt)] = np.nan
        out[r] = pt
    write(out, a)


def palantir(a):
    import palantir as pl
    data = load(a)
    data.obsm[a.latent_key] = np.asarray(data.obsm[a.latent_key], dtype=np.float64)
    pl.utils.run_diffusion_maps(data, n_components=a.n_components, knn=a.n_neighbors, pca_key=a.latent_key, seed=a.seed)
    pl.utils.determine_multiscale_space(data)
    out = frame(data, a)
    os.makedirs(a.outdir, exist_ok=True)
    for r in a.roots:
        pl.core.run_palantir(data, r, knn=a.n_neighbors, num_waypoints=a.num_waypoints, use_early_cell_as_start=True,
                             n_jobs=a.n_jobs, seed=a.seed, save_as_df=True)
        out[r] = data.obs["palantir_pseudotime"].to_numpy()
        fate = pd.DataFrame(data.obsm["palantir_fate_probabilities"], index=out.index)
        fate.to_csv(os.path.join(a.outdir, f"{a.prefix}_fate_{r}.csv"))
    write(out, a)


def slingshot(a):
    import scanpy as sc
    data = load(a)
    if a.cluster_by == "label":
        cl = data.obs[a.label_col].astype(str).to_numpy()
    else:
        sc.pp.neighbors(data, n_neighbors=a.n_neighbors, use_rep=a.latent_key, transformer="sklearn")
        sc.tl.leiden(data, resolution=a.resolution, random_state=a.seed, flavor="igraph", n_iterations=2, directed=False)
        cl = data.obs["leiden"].astype(str).to_numpy()
        if a.cluster_by == "leiden_start":
            m = (data.obs[a.label_col].to_numpy() == a.start) & (data.obs[a.time_col].to_numpy() == a.start_time)
            cl = np.where(m, "start", cl)
    starts = {r: cl[data.obs_names.get_loc(r)] for r in a.roots}
    uniq = sorted(set(starts.values()))
    x = np.ascontiguousarray(data.obsm[a.latent_key], dtype=np.float64)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slingshot.R")
    os.makedirs(a.outdir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.outdir) as tmp:
        x.tofile(os.path.join(tmp, "x.bin"))
        with open(os.path.join(tmp, "cl.txt"), "w") as f:
            f.write("\n".join(cl) + "\n")
        cmd = ["Rscript", script, "--x", os.path.join(tmp, "x.bin"), "--n", str(x.shape[0]),
               "--d", str(x.shape[1]), "--clusters", os.path.join(tmp, "cl.txt"),
               "--starts", ",".join(uniq), "--outdir", tmp, "--seed", str(a.seed), "--prefix", a.prefix]
        if a.plot:
            meta = pd.read_csv(a.meta, sep="\t", usecols=["SPRING-x", "SPRING-y"])
            xy = np.ascontiguousarray(meta.to_numpy()[data.obs_names.astype(int)], dtype=np.float64)
            xy.tofile(os.path.join(tmp, "xy.bin"))
            with open(os.path.join(tmp, "lab.txt"), "w") as f:
                f.write("\n".join(data.obs[a.label_col].astype(str)) + "\n")
            cmd += ["--plot-xy", os.path.join(tmp, "xy.bin"), "--plot-labels", os.path.join(tmp, "lab.txt"),
                    "--plot-dir", a.outdir]
        subprocess.run(cmd, check=True)
        pts = {s: np.fromfile(os.path.join(tmp, f"pt_{s}.bin")) for s in uniq}
        lins = {s: open(os.path.join(tmp, f"lineages_{s}.txt")).read().strip().split("\n") for s in uniq}
    out = frame(data, a)
    for r in a.roots:
        out[r] = pts[starts[r]]
    write(out, a)
    pd.DataFrame({"root": a.roots, "start_cluster": [starts[r] for r in a.roots],
                  "n_lineages": [len(lins[starts[r]]) for r in a.roots],
                  "lineages": [" | ".join(lins[starts[r]]) for r in a.roots],
                  "n_clusters": len(set(cl))}).to_csv(os.path.join(a.outdir, f"{a.prefix}_info.csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)

    def add(name, func, backend=True):
        s = sub.add_parser(name)
        s.add_argument("--adata", required=True)
        s.add_argument("--latent-key", default="X_latent")
        s.add_argument("--label-col", default="label")
        s.add_argument("--time-col", default="time")
        s.add_argument("--start", default="Undifferentiated")
        s.add_argument("--start-time", type=float, default=2.0)
        s.add_argument("--seed", type=int, default=0)
        if backend:
            s.add_argument("--roots", nargs="+", required=True)
            s.add_argument("--outdir", required=True)
            s.add_argument("--prefix", default=name)
        s.set_defaults(func=func)
        return s

    s = add("roots", roots, backend=False)
    s.add_argument("--out", required=True)
    s.add_argument("--n-roots", type=int, default=5)
    s.add_argument("--pool-size", type=int, default=100)

    add("euclid", euclid)

    s = add("dpt", dpt)
    s.add_argument("--neighbors-key")
    s.add_argument("--n-dcs", type=int, default=10)
    s.add_argument("--n-neighbors", type=int, default=15)

    s = add("palantir", palantir)
    s.add_argument("--n-components", type=int, default=20)
    s.add_argument("--n-neighbors", type=int, default=15)
    s.add_argument("--num-waypoints", type=int, default=1200)
    s.add_argument("--n-jobs", type=int, default=-1)

    s = add("slingshot", slingshot)
    s.add_argument("--n-neighbors", type=int, default=15)
    s.add_argument("--resolution", type=float, default=1.0)
    s.add_argument("--cluster-by", choices=["leiden", "leiden_start", "label"], default="leiden")
    s.add_argument("--plot", action="store_true")
    s.add_argument("--meta", default="rt_data/stateFate_inVitro_metadata.txt.gz")

    a = p.parse_args()
    a.func(a)
