import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from bench.data import DATA

LATENT = "X_latent"
ROOT = "runs/frontends"


def load_counts(cells=None):
    X = sp.load_npz(DATA + "cache/normed_counts_f32.npz").tocsr()
    return X if cells is None else X[cells]


def larry_obs(meta, cells=None):
    idx = np.arange(len(meta)) if cells is None else np.asarray(cells)
    m = meta.iloc[idx]
    return pd.DataFrame(
        {"label": m["Cell type annotation"].to_numpy(), "time": m["Time point"].to_numpy().astype(float)},
        index=pd.Index(idx.astype(str), name="cell_id"))


def pca(X, n_hvg=2000, n_pcs=50):
    data = sc.AnnData(X)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=n_hvg)
    data = data[:, data.var["highly_variable"]].copy()
    sc.pp.scale(data, max_value=10)
    sc.pp.pca(data, n_comps=n_pcs, svd_solver="covariance_eigh")
    return data.obsm["X_pca"].astype(np.float32)


def add_graph(data, k):
    key = f"k{k}"
    if key not in data.uns:
        sc.pp.neighbors(data, n_neighbors=k, use_rep=LATENT, key_added=key, transformer="sklearn")
    return key


def get_graph(data, k):
    key = add_graph(data, k)
    return data.obsp[f"{key}_distances"], data.obsp[f"{key}_connectivities"]


def diffusion_operator(conn, self_loops=True):
    w = conn + sp.identity(conn.shape[0], format="csr") if self_loops else conn
    deg = np.asarray(w.sum(1)).ravel()
    return (sp.diags(1 / deg) @ w).tocsr()


def smooth(latent, conn, t=3):
    p = diffusion_operator(conn)
    x = np.asarray(latent, dtype=np.float32)
    for _ in range(t):
        x = p @ x
    return x.astype(np.float32)


def make_frontend(latent, obs, **params): 
    data = ad.AnnData(obs=obs.copy())
    data.obsm[LATENT] = np.asarray(latent, dtype=np.float32)
    data.uns["frontend"] = params   #  все, что не привязанно ни к клеткам ни к генам,
    return data                     #  в нашем случае настройки латента


def frontend_path(name, root=ROOT):
    return os.path.join(root, f"{name}.h5ad")


def save_frontend(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data.write_h5ad(path)


def load_frontend(path):
    return ad.read_h5ad(path)


def dump(out, suffix, obj):
    with open(os.path.splitext(out)[0] + suffix, "w") as f:
        json.dump(obj, f)


def run_pca(a):
    from bench.data import load_larry
    meta, _ = load_larry()
    X = sp.load_npz(a.input).tocsr()
    data = make_frontend(pca(X, n_hvg=a.n_hvg, n_pcs=a.n_pcs), larry_obs(meta), method="pca", n_hvg=a.n_hvg,
                         n_pcs=a.n_pcs, svd_solver="covariance_eigh")
    add_graph(data, a.k)
    save_frontend(data, a.out)

def pca_slice(a):
    data = load_frontend(a.input)
    x = np.asarray(data.obsm[LATENT])[:, :a.n]
    save_frontend(make_frontend(x, data.obs, method="pca_slice", n_pcs=a.n), a.out)


def pca_smooth(a):
    data = load_frontend(a.input)
    _, conn = get_graph(data, a.k)
    x = smooth(np.asarray(data.obsm[LATENT]), conn, t=a.t)
    save_frontend(make_frontend(x, data.obs, method="pca_smooth", t=a.t, graph_k=a.k), a.out)


def run_umap(a):
    data = load_frontend(a.input)
    key = add_graph(data, a.k)
    sc.tl.umap(data, neighbors_key=key, min_dist=a.min_dist, n_components=a.n_dim, random_state=a.seed)
    save_frontend(make_frontend(data.obsm["X_umap"], data.obs, method="umap", graph_k=a.k, min_dist=a.min_dist,
                                n_dim=a.n_dim, seed=a.seed), a.out)


def scvi(a):
    import anndata as ad
    import scvi
    scvi.settings.seed = a.seed
    data = ad.read_h5ad(a.input)
    scvi.model.SCVI.setup_anndata(data)
    model = scvi.model.SCVI(data, n_latent=a.n_latent)
    plan = {} if a.kl_weight is None else {"max_kl_weight": a.kl_weight, "n_epochs_kl_warmup": a.kl_warmup}
    model.train(plan_kwargs=plan)
    z = model.get_latent_representation()
    hist = {k: v.iloc[:, 0].astype(float).tolist() for k, v in model.history.items()
            if k in ["elbo_train", "elbo_validation", "reconstruction_loss_train", "kl_local_train", "kl_weight"]}
    active = int((z.var(0) > 0.01).sum())
    fe = make_frontend(z, data.obs[["label", "time"]], method="scvi", seed=a.seed, n_latent=a.n_latent,
                       kl_weight=a.kl_weight if a.kl_weight is not None else "default",
                       kl_warmup=a.kl_warmup if a.kl_weight is not None else 400,
                       n_epochs=len(hist.get("elbo_train", [])), active_dims=active)
    save_frontend(fe, a.out)
    dump(a.out, "_history.json", hist)
    print("latent", z.shape, "active dims", active, "var per dim", np.round(z.var(0), 3))


def sctour(a):
    import anndata as ad
    import pandas as pd
    import sctour as sct
    import torch
    data = ad.read_h5ad(a.input)
    data.X = data.X.toarray().astype(np.float32)
    tnode = sct.train.Trainer(data, random_state=a.seed)
    tnode.train()
    t = tnode.get_time()
    default_reversed = bool(tnode.time_reverse)
    start = (data.obs["label"].to_numpy() == a.start) & (data.obs["time"].to_numpy() == a.start_time)
    flip = t[start].mean() > t[~start].mean()
    if flip:
        t = 1 - t
    torch.manual_seed(a.seed)
    mix, _, _ = tnode.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
    info = dict(seed=a.seed, percent=tnode.percent, nepoch=int(tnode.nepoch), default_reversed=default_reversed,
                flipped_by_start_rule=bool(flip),
                start_mean_t=float(t[start].mean()), other_mean_t=float(t[~start].mean()))
    save_frontend(make_frontend(mix, data.obs[["label", "time"]], method="sctour", **info), a.out)
    os.makedirs(os.path.dirname(a.pt_out), exist_ok=True)
    pd.DataFrame({"label": data.obs["label"].to_numpy(), "sctour": t},
                 index=pd.Index(data.obs_names, name="cell_id")).to_csv(a.pt_out)
    dump(a.out, "_info.json", info)
    print(info, "latent", mix.shape)


def gae(a):
    import torch
    from torch_geometric.nn import GAE, VGAE, GCNConv
    from torch_geometric.utils import negative_sampling

    class Encoder(torch.nn.Module):
        def __init__(self, d_in, d_hid, d_out, variational):
            super().__init__()
            self.conv1 = GCNConv(d_in, d_hid)
            self.conv_mu = GCNConv(d_hid, d_out)
            self.conv_logstd = GCNConv(d_hid, d_out) if variational else None

        def forward(self, x, edge_index):
            h = torch.relu(self.conv1(x, edge_index))
            if self.conv_logstd is None:
                return self.conv_mu(h, edge_index)
            return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_frontend(a.input)
    x = np.asarray(data.obsm[LATENT], dtype=np.float32)
    x = (x - x.mean(0)) / x.std(0)
    conn = data.obsp[f"k{a.k}_connectivities"].tocoo()
    edge_index = torch.tensor(np.vstack([conn.row, conn.col]), dtype=torch.long, device=dev)
    xt = torch.tensor(x, device=dev)
    variational = a.kind == "vgae"
    enc = Encoder(x.shape[1], a.hidden, a.n_latent, variational)
    model = (VGAE(enc) if variational else GAE(enc)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = []
    for _ in range(a.epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(xt, edge_index)
        neg = negative_sampling(edge_index, num_nodes=xt.shape[0], num_neg_samples=edge_index.shape[1])
        loss = model.recon_loss(z, edge_index, neg)
        if variational:
            loss = loss + model.kl_loss() / xt.shape[0]
        loss.backward()
        opt.step()
        hist.append(float(loss))
    model.eval()
    with torch.no_grad():
        z = model.encoder(xt, edge_index)
        z = (z[0] if variational else z).cpu().numpy()
    active = int((z.var(0) > 0.01).sum())
    fe = make_frontend(z, data.obs, method=a.kind, n_latent=a.n_latent, hidden=a.hidden, epochs=a.epochs,
                       lr=a.lr, seed=a.seed, graph_k=a.k, input="pca50_standardized", active_dims=active)
    save_frontend(fe, a.out)
    dump(a.out, "_history.json", {"loss": hist})
    print(a.kind, "latent", z.shape, "active", active, "loss first/last", hist[:1], hist[-1:])


def run_magic(a):
    import magic
    import scprep
    from bench.data import load_larry
    meta, _ = load_larry()
    x = sp.load_npz(a.input).tocsr()
    x = scprep.filter.filter_rare_genes(x, min_cells = a.min_cells)  # вместо hvg
    #x = scprep.normalize.library_size_normalize(x)       # данные уже нормализированны
    x = scprep.transform.sqrt(x)
    op = magic.MAGIC(knn = a.knn, decay = a.decay, t = a.t, n_pca = a.n_pca, random_state = a.seed, verbose = 0, n_jobs = a.n_jobs)
    z = np.asarray(op.fit_transform(x, genes = 'pca_only'))[:, :a.n_pcs] # выбираем из 100 компонент сколько
    # нам надо, сообщаем что как аутпут нас интересует сразу PCA.
    out = make_frontend(z, larry_obs(meta), method = 'magic', min_cells = a.min_cells, transform = 'sqrt',
        knn = a.knn, decay = a.decay, t = a.t, n_pca = a.n_pca, # внутри MAGIC для построенния графа и диффузий
        n_pcs = a.n_pcs, n_genes = x.shape[1])
    add_graph(out, a.k)
    save_frontend(out, a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)

    def add(name, func):
        s = sub.add_parser(name)
        s.add_argument("--input", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--seed", type=int, default=0)
        s.set_defaults(func=func)
        return s

    s = add("pca", run_pca)
    s.add_argument("--n-hvg", type=int, default=2000)
    s.add_argument("--n-pcs", type=int, default=50)
    s.add_argument("--k", type=int, default=15)

    s = add("slice", pca_slice)
    s.add_argument("--n", type=int, required=True)

    s = add("smooth", pca_smooth)
    s.add_argument("--k", type=int, required=True)
    s.add_argument("--t", type=int, required=True)

    s = add("umap", run_umap)
    s.add_argument("--k", type=int, default=15)
    s.add_argument("--min-dist", type=float, default=0.5)
    s.add_argument("--n-dim", type=int, default=2)

    s = add('magic', run_magic)
    s.add_argument('--min-cells', type = int, default = 10)
    s.add_argument('--knn', type = int, default = 5)
    s.add_argument('--decay', type = float, default = 1)
    s.add_argument('--t', type = int, default = 3)
    s.add_argument('--n-pca', type = int, default = 100)
    s.add_argument('--n-pcs', type = int, default = 50)
    s.add_argument('--n-jobs', type = int, default = 8)
    s.add_argument('--k', type = int, default = 15)

    s = add("scvi", scvi)
    s.add_argument("--n-latent", type=int, default=10)
    s.add_argument("--kl-weight", type=float)
    s.add_argument("--kl-warmup", type=int, default=20)

    s = add("sctour", sctour)
    s.add_argument("--pt-out", required=True)
    s.add_argument("--start", default="Undifferentiated")
    s.add_argument("--start-time", type=float, default=2.0)

    s = add("gae", gae)
    s.add_argument("--kind", choices=["gae", "vgae"], required=True)
    s.add_argument("--k", type=int, default=15)
    s.add_argument("--n-latent", type=int, default=10)
    s.add_argument("--hidden", type=int, default=128)
    s.add_argument("--epochs", type=int, default=200)
    s.add_argument("--lr", type=float, default=0.01)

    a = p.parse_args()
    a.func(a)
