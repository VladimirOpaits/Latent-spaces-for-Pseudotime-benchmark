import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from bench.data import DATA, clone_pairs
from bench.latents import add_graph, get_graph, make_frontend, pca, save_frontend, smooth
from bench.metric import evaluate, load_pairs


PAIRS = pd.DataFrame({"i": [0, 2, 4, 6], "j": [1, 3, 5, 7], "clone": [0, 1, 2, 3],
                      "branch": "A", "daypair": ["2->4", "2->4", "4->6", "2->6"]})
BRANCH = np.array(["A"] * 8, dtype=object)


def score(x, ref=None, severe=0.1, **more):
    methods = {("m", "b", "-"): {"r": np.asarray(x, dtype=float)}}
    methods.update({(k, "b", "-"): {"r": np.asarray(v, dtype=float)} for k, v in more.items()})
    scores, slices = evaluate(methods, PAIRS, BRANCH, ref=ref, severe=severe, n_boot=50)
    return scores.set_index(["frontend", "branch"]), slices


def test_later_sister_must_have_larger_pseudotime():
    assert score(np.arange(8))[0].loc[("m", "A"), "acc"] == 1.0
    assert score(-np.arange(8))[0].loc[("m", "A"), "acc"] == 0.0
    assert score(np.zeros(8))[0].loc[("m", "A"), "acc"] == 0.5


def test_miss_depth_and_severe_use_branch_rank():
    x = np.array([1, 0, 2, 3, 4, 5, 7, 6], dtype=float)
    s = score(x)[0].loc[("m", "A")]
    assert s["acc"] == 0.5
    assert np.isclose(s["miss_depth"], 1 / 8)
    assert s["severe"] == 0.5
    assert score(x, severe=0.2)[0].loc[("m", "A"), "severe"] == 0.0


def test_scale_does_not_matter():
    x = np.array([3, 1, 2, 9, 0.5, 4, 8, 7])
    a, b = score(x)[0], score(np.exp(x) * 100)[0]
    for k in ["acc", "miss_depth", "severe"]:
        assert a.loc[("m", "A"), k] == b.loc[("m", "A"), k]


def test_difference_to_reference():
    s = score(np.arange(8), ref="pca", pca=-np.arange(8))[0]
    assert s.loc[("m", "A"), "d_acc"] == 1.0
    assert np.isnan(s.loc[("pca", "A"), "d_acc"])


def test_nan_pairs_are_dropped():
    x = np.arange(8, dtype=float)
    x[[0, 1]] = np.nan
    s = score(x)[0].loc[("m", "A")]
    assert s["n_pairs"] == 3
    assert s["acc"] == 1.0


def test_slices_by_daypair():
    _, sl = score(-np.arange(8))
    d = sl[sl["slice"] == "daypair"].set_index("value")
    assert d.loc["2->4", "n"] == 2
    assert (d["acc"] == 0).all()


def test_clone_pairs_are_cross_day_and_early_first():
    day = np.array([2, 4, 4, 6, 2])
    rows, cols = [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 3]
    clones = csr_matrix((np.ones(6), (cols, rows)), shape=(5, 2))
    i, j, _ = clone_pairs(day, clones)
    assert len(i) == 5
    assert np.all(day[i] < day[j])


@pytest.mark.skipif(not os.path.exists(DATA + "stateFate_inVitro_metadata.txt.gz"), reason="no LARRY data")
def test_larry_reference_values():
    pairs, branch = load_pairs()
    ps = pd.read_csv(DATA + "stateFate_inVitro_neutrophil_pseudotime.txt.gz", sep="\t")
    pt = np.full(len(branch), np.nan)
    pt[ps["Cell index"].to_numpy()] = ps["pseudotime"].to_numpy()
    rnd = np.random.default_rng(0).random(len(branch))
    methods = {("authors", "-", "-"): {"r": pt}, ("random", "-", "-"): {"r": rnd}}
    s = evaluate(methods, pairs, branch, n_boot=50)[0].set_index(["frontend", "branch"])
    authors, random = s.loc[("authors", "Neutrophil")], s.loc[("random", "Neutrophil")]
    assert 0.92 <= authors["acc"] <= 0.96
    assert authors["miss_depth"] < 0.2
    assert 0.47 <= random["acc"] <= 0.53
    assert 0.3 <= random["miss_depth"] <= 0.36
    assert 0.38 <= random["severe"] <= 0.43


def frontend(n=200, d=10):
    x = np.random.default_rng(0).normal(size=(n, d))
    obs = pd.DataFrame({"label": "A", "time": 2.0}, index=[f"c{i}" for i in range(n)])
    return make_frontend(x, obs)


def test_pca_is_deterministic():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(rng.gamma(0.5, 2.0, 500), (300, 500)).astype(np.float32))
    a = pca(X.copy(), n_hvg=200, n_pcs=10)
    b = pca(X.copy(), n_hvg=200, n_pcs=10)
    assert np.array_equal(a, b)


def test_graph_is_deterministic():
    _, c1 = get_graph(frontend(), 15)
    _, c2 = get_graph(frontend(), 15)
    assert (c1 != c2).nnz == 0


def test_smooth_averages_neighbours():
    data = frontend()
    x = data.obsm["X_latent"]
    _, conn = get_graph(data, 10)
    assert np.allclose(smooth(np.ones_like(x), conn, t=3), 1)
    assert smooth(x, conn, t=3).var() < x.var()


def run_dpt(path, outdir, roots, neighbors_key=None, **kw):
    cmd = [sys.executable, "-m", "bench.backends", "dpt", "--adata", path, "--outdir", outdir, "--roots", *roots]
    if neighbors_key is not None:
        cmd += ["--neighbors-key", neighbors_key]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(os.path.join(outdir, "dpt_pseudotime.csv"), index_col="cell_id", dtype={"cell_id": str})[roots]


@pytest.fixture
def curve(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    t = np.sort(rng.uniform(0, 1, n))
    x = np.column_stack([t, np.sin(6 * t), rng.normal(0, 0.01, (n, 3))]) + rng.normal(0, 0.02, (n, 5))
    obs = pd.DataFrame({"label": np.where(t < 0.2, "Start", "Other"), "time": np.where(t < 0.2, 2.0, 4.0)},
                       index=[f"c{i:04d}" for i in range(n)])
    data = make_frontend(x, obs)
    add_graph(data, 15)
    path = str(tmp_path / "curve.h5ad")
    save_frontend(data, path)
    return path, data, t


def test_pseudotime_aligned_ordered_and_rooted(curve, tmp_path):
    path, data, t = curve
    root = data.obs_names[0]
    pt = run_dpt(path, str(tmp_path / "a"), [root], neighbors_key="k15", n_dcs=2)
    assert list(pt.index) == list(data.obs_names)
    assert pt.loc[root, root] == 0
    late = t > 0.2
    assert spearmanr(pt[root].to_numpy()[late], t[late])[0] > 0.95


def test_stored_graph_equals_graph_built_in_script(curve, tmp_path):
    path, data, _ = curve
    roots = list(data.obs_names[[0, 5]])
    pt1 = run_dpt(path, str(tmp_path / "a"), roots, neighbors_key="k15")
    pt2 = run_dpt(path, str(tmp_path / "b"), roots, n_neighbors=15)
    assert np.allclose(pt1.to_numpy(), pt2.to_numpy())


def test_many_roots_equal_single_runs(curve, tmp_path):
    path, data, _ = curve
    roots = list(data.obs_names[[0, 7, 30]])
    many = run_dpt(path, str(tmp_path / "many"), roots, neighbors_key="k15")
    for r in roots:
        one = run_dpt(path, str(tmp_path / r), [r], neighbors_key="k15")
        assert np.allclose(many[r].to_numpy(), one[r].to_numpy())


def test_gauss_weights_match_scanpy():
    import scanpy as sc
    from bench.backends import gauss_weights
    rng = np.random.default_rng(0)
    data = make_frontend(rng.normal(size=(300, 5)), pd.DataFrame(index=[f"c{i}" for i in range(300)]))
    sc.pp.neighbors(data, n_neighbors=15, use_rep="X_latent", method="gauss", transformer="sklearn")
    w = gauss_weights(data.obsp["distances"])
    assert abs(w - data.obsp["connectivities"]).max() < 1e-6
    assert gauss_weights(data.obsp["distances"], 2.0).sum() > w.sum()
