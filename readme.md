bench/ : library + cli, one file per pipeline stage (python -m bench.<file> <command>)
    $ data.py :  loads data, clones, branches; cache | split | counts
    $ latents.py :  latent format, knn graph, smoothing; pca | slice | smooth | scvi | sctour | gae
    $ backends.py :  pseudotime from latent; roots | euclid | dpt | palantir | slingshot
    $ slingshot.R :  slingshot curves, called from backends.py
    $ metric.py :  sister order metric, bootstrap over clones, slices; score | plot
workflow/ : snakemake
    $ Snakefile :  all rules; targets pre | calib | kcross
    $ config.yaml :  split, roots, backend settings, evaluate, frontend list
tests/ : pytest
    $ test_bench.py :  metric, latents, dpt
runs/ : outputs, not in git
    $ fixed/ :  clone split, roots, hvg counts
    $ frontends/ :  latents, <name>.h5ad
    $ <run>/pt/ :  pseudotime, <latent>/<backend>/[k15/]<backend>_pseudotime.csv
    $ <run>/eval/ :  scores.csv + slices.csv
    $ <run>/plots/ :  pseudotime on spring

run: snakemake pre -c 8 --resources mem_gb=8 gpu=1 --rerun-triggers mtime
