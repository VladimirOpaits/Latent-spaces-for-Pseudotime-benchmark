suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
eval_dir <- if (length(args) >= 1) args[1] else "runs/calib_v1/eval/dev/pre"
out_dir <- if (length(args) >= 2) args[2] else "pics"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ink <- "#0b0b0b"
ink2 <- "#52514e"
muted <- "#898781"
grid <- "#e8e7e3"
surface <- "#fcfcfb"
family_col <- c("PCA" = "#2a78d6", "smoothed PCA" = "#eb6834", "scVI" = "#1baf7a",
                "scTour" = "#eda100", "graph AE" = "#e87ba4")
blue_ramp <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
orange_ramp <- c("#fde6da", "#f9c3a6", "#f39a6f", "#eb6834", "#c4501f", "#933a14", "#64270c")

latent_order <- c(
  "pca" = "PCA-50", "pca20" = "PCA-20", "pca10" = "PCA-10", "pca5" = "PCA-5",
  "pca_smooth_t1_k15" = "smoothed t=1", "pca_smooth_t3_k15" = "smoothed t=3", "pca_smooth_t10_k15" = "smoothed t=10",
  "scvi_s0" = "scVI-10 default KL, s0", "scvi_s1" = "scVI-10 default KL, s1",
  "scvi_l10_kl0.01_s0" = "scVI-10 KL 0.01", "scvi_l10_kl1.0_s0" = "scVI-10 KL 1.0",
  "scvi_l30_kl0.01_s0" = "scVI-30 KL 0.01", "scvi_l30_kl1.0_s0" = "scVI-30 KL 1.0",
  "sctour_s0" = "scTour s0", "sctour_s1" = "scTour s1",
  "gae_l50_e0_s0" = "GAE-50 untrained", "gae_l10_e200_s0" = "GAE-10", "gae_l50_e200_s0" = "GAE-50",
  "vgae_l10_e200_s0" = "VGAE-10", "vgae_l50_e200_s0" = "VGAE-50")
family_of <- function(fe) case_when(
  grepl("^pca_smooth", fe) ~ "smoothed PCA", grepl("^pca", fe) ~ "PCA", grepl("^scvi", fe) ~ "scVI",
  grepl("^sctour", fe) ~ "scTour", TRUE ~ "graph AE")
backend_lab <- c(euclid = "Euclidean", dpt = "DPT", palantir = "Palantir", sctour = "scTour own")
metric_lab <- c(acc = "Accuracy  ↑", miss_depth = "Miss depth  ↓", severe = "Severe misses  ↓")

tidy_names <- function(d) {
  d |>
    filter(frontend %in% names(latent_order)) |>
    mutate(family = factor(family_of(frontend), levels = names(family_col)),
           latent = factor(latent_order[frontend], levels = rev(latent_order)),
           backend = factor(backend_lab[backend], levels = backend_lab))
}

theme_bench <- function(base = 9) {
  theme_minimal(base_size = base) +
    theme(text = element_text(colour = ink2),
          plot.title = element_text(colour = ink, face = "bold", size = base + 3),
          plot.subtitle = element_text(colour = ink2, size = base),
          plot.caption = element_text(colour = muted, size = base - 1.5, hjust = 0),
          strip.text = element_text(colour = ink, face = "bold", size = base),
          axis.text = element_text(colour = ink2),
          axis.title = element_text(colour = ink2),
          panel.grid.major.y = element_blank(),
          panel.grid.minor = element_blank(),
          panel.grid.major.x = element_line(colour = grid, linewidth = 0.3),
          panel.spacing = unit(9, "pt"),
          legend.position = "bottom",
          legend.title = element_text(colour = ink, face = "bold"),
          plot.background = element_rect(fill = surface, colour = NA),
          panel.background = element_rect(fill = surface, colour = NA))
}

main <- c("pca", "pca10", "pca_smooth_t3_k15", "scvi_s0", "scvi_l10_kl0.01_s0", "sctour_s0",
          "gae_l50_e200_s0", "gae_l50_e0_s0")
dashed <- c("pca10", "scvi_l10_kl0.01_s0", "gae_l50_e0_s0")
diverging <- c("#933a14", "#eb6834", "#f9c3a6", "#e8e7e3", "#9ec5f4", "#3987e5", "#184f95")

save_both <- function(p, name, w, h) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = w, height = h, dpi = 200, device = ragg::agg_png, bg = surface)
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = w, height = h, device = cairo_pdf, bg = surface)
}

scores <- read_csv(file.path(eval_dir, "scores.csv"), show_col_types = FALSE) |> tidy_names()

long <- bind_rows(lapply(names(metric_lab), function(m) {
  scores |>
    transmute(latent, family, backend, branch, frontend, metric = m,
              value = .data[[m]], lo = .data[[paste0(m, "_lo")]], hi = .data[[paste0(m, "_hi")]],
              p = if (paste0("p_", m) %in% names(scores)) .data[[paste0("p_", m)]] else NA_real_)
})) |>
  mutate(metric = factor(metric_lab[metric], levels = metric_lab),
         filled = frontend == "pca" | (!is.na(p) & p < 0.05))

plot_scores <- function(keep, title) {
  d <- filter(long, frontend %in% keep) |> mutate(latent = droplevels(latent))
  m <- filter(d, branch == "mean")
  ref <- m |> filter(frontend == "pca") |> select(backend, metric, ref = value)
  ggplot(m, aes(y = latent)) +
    geom_vline(data = ref, aes(xintercept = ref), colour = muted, linewidth = 0.4, linetype = "22") +
    geom_point(data = filter(d, branch != "mean"), aes(x = value, shape = branch), colour = muted, size = 1.3, stroke = 0.4) +
    geom_linerange(aes(xmin = lo, xmax = hi, colour = family), linewidth = 0.7, alpha = 0.55) +
    geom_point(data = filter(m, !filled), aes(x = value, colour = family), shape = 21, fill = surface, size = 2.4, stroke = 0.9) +
    geom_point(data = filter(m, filled), aes(x = value, colour = family, fill = family), shape = 21, size = 2.4, stroke = 0.9) +
    facet_grid(backend ~ metric, scales = "free", space = "free_y", switch = "y") +
    scale_colour_manual(values = family_col, name = "Latent family", drop = FALSE) +
    scale_fill_manual(values = family_col, guide = "none", drop = FALSE) +
    scale_shape_manual(values = c(Monocyte = 2, Neutrophil = 0), name = "Single branch") +
    scale_x_continuous(labels = function(x) sprintf("%.2f", x), n.breaks = 4) +
    labs(x = NULL, y = NULL, title = title,
         subtitle = paste("LARRY in vitro, dev clones (767 clones, 26 106 sister pairs), mean over 5 roots.",
                          "Dot = mean of the two branches, line = 95% clone bootstrap CI, grey marks = single branches.",
                          "Filled = paired difference to PCA-50 on the same backend has p < 0.05; hollow = not (or no PCA-50 counterpart).",
                          "Dashed line = PCA-50.", sep = "\n"),
         caption = "Miss depth: mean rank gap of wrongly ordered pairs (fraction of branch). Severe: share of pairs with rank gap < -0.1.") +
    guides(colour = guide_legend(order = 1, override.aes = list(fill = family_col)), shape = guide_legend(order = 2)) +
    theme_bench() +
    theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1))
}

save_both(plot_scores(c(main, "sctour_s0"), "Which latent space orders sisters correctly?"), "fig1_scores", 10.5, 8.5)
save_both(plot_scores(names(latent_order), "Which latent space orders sisters correctly? All latents"), "figS_scores_all", 11, 12.5)

slices <- read_csv(file.path(eval_dir, "slices.csv"), show_col_types = FALSE, col_types = cols(value = "c")) |>
  tidy_names() |>
  filter(backend != "scTour own")
n_bins <- slices |> filter(slice == "pos") |> pull(value) |> as.numeric() |> max() + 1
bin_lab <- sprintf("%d–%d%%", (0:(n_bins - 1)) * 100 / n_bins, (1:n_bins) * 100 / n_bins)
diffs <- slices |>
  left_join(slices |> filter(frontend == "pca") |> select(slice, backend, branch, value, ref = acc),
            by = c("slice", "backend", "branch", "value")) |>
  mutate(d = acc - ref)

where <- diffs |>
  filter(slice == "pos", frontend %in% setdiff(main, "pca")) |>
  mutate(x = factor(bin_lab[as.integer(as.numeric(value)) + 1], levels = bin_lab),
         lt = ifelse(frontend %in% dashed, "dashed", "solid"))
npairs <- slices |> filter(slice == "pos", frontend == "pca", backend == "Palantir") |>
  mutate(x = factor(bin_lab[as.integer(as.numeric(value)) + 1], levels = bin_lab))

fig2 <- ggplot(where, aes(x = x, y = d, group = latent)) +
  geom_hline(yintercept = 0, colour = muted, linewidth = 0.4, linetype = "22") +
  geom_line(aes(colour = family, linetype = lt), linewidth = 0.8) +
  geom_point(aes(colour = family), size = 1.6) +
  geom_text(data = npairs, aes(x = x, y = -Inf, label = paste0("n=", n)), inherit.aes = FALSE,
            vjust = -0.6, size = 2.4, colour = muted) +
  facet_grid(backend ~ branch, scales = "free_y", switch = "y") +
  scale_colour_manual(values = family_col, name = "Latent family", drop = FALSE) +
  scale_linetype_identity() +
  scale_y_continuous(labels = function(x) sprintf("%+.2f", x)) +
  labs(x = "Position of the earlier sister along the branch (consensus rank of DPT and Palantir over all latents)",
       y = "Accuracy minus PCA-50 on the same backend",
       title = "Where along the branch do latents gain or lose against PCA-50?",
       subtitle = paste("Dev clones, mean over 5 roots. Lines: PCA-10, smoothed t=3, scVI default, scVI KL 0.01, scTour, GAE-50, GAE-50 untrained.",
                        "Dashed = second member of a family (PCA-10, scVI KL 0.01, GAE-50 untrained). n = sister pairs per segment.", sep = "\n")) +
  theme_bench() +
  theme(strip.placement = "outside", panel.grid.major.y = element_line(colour = grid, linewidth = 0.3))
save_both(fig2, "fig2_where", 10, 8)

heat <- diffs |>
  filter(frontend != "pca") |>
  mutate(x = ifelse(slice == "pos", bin_lab[as.integer(suppressWarnings(as.numeric(value))) + 1], value),
         x = factor(x, levels = c(bin_lab, "2->4", "4->6", "2->6")),
         slice = factor(ifelse(slice == "pos", "Position (consensus rank)", "Sampling days"),
                        levels = c("Position (consensus rank)", "Sampling days")))
figS <- ggplot(heat, aes(x = x, y = latent, fill = d)) +
  geom_tile(colour = surface, linewidth = 0.6) +
  facet_grid(backend ~ branch + slice, scales = "free", space = "free", switch = "y") +
  scale_fill_gradientn(colours = diverging, limits = c(-0.1, 0.1), oob = scales::squish,
                       labels = function(x) sprintf("%+.2f", x), name = "Accuracy minus PCA-50",
                       guide = guide_colourbar(barwidth = unit(160, "pt"), barheight = unit(7, "pt"))) +
  labs(x = NULL, y = NULL, title = "Accuracy difference to PCA-50 by position and sampling days, all latents",
       subtitle = "Dev clones, mean over 5 roots. Colour saturates at ±0.10.") +
  theme_bench() +
  theme(panel.grid = element_blank(), strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 1), axis.text.x = element_text(angle = 45, hjust = 1))
save_both(figS, "figS_slices_all", 14, 12)
depth_file <- file.path(dirname(eval_dir), "depth", "depth_bins.csv")
if (file.exists(depth_file)) {
  p_col <- c("1" = "#0d366b", "0.7" = "#184f95", "0.5" = "#256abf", "0.3" = "#3987e5", "0.2" = "#6da7ec", "0.1" = "#9ec5f4")
  depth_lat <- c(pca = "PCA-50", pca_smooth_t3_k15 = "smoothed PCA t=3", scvi_s0 = "scVI", magic_t3 = "MAGIC t=3")
  dep <- read_csv(depth_file, show_col_types = FALSE) |>
    filter(draw == 0, frontend %in% names(depth_lat)) |>
    mutate(latent = factor(depth_lat[frontend], levels = depth_lat),
           backend = factor(backend_lab[backend], levels = backend_lab),
           p = factor(as.character(p), levels = names(p_col)))
  log_x <- scale_x_log10(breaks = c(100, 300, 1000, 3000), labels = scales::label_comma())
  figD1 <- ggplot(dep, aes(depth, acc, colour = p, fill = p, group = p)) +
    geom_ribbon(aes(ymin = acc_lo, ymax = acc_hi), colour = NA, alpha = 0.15) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 1.2) +
    facet_grid(latent ~ backend) +
    log_x +
    scale_colour_manual(values = p_col, name = "Kept fraction of molecules p") +
    scale_fill_manual(values = p_col, guide = "none") +
    labs(x = "Molecules in the shallower cell of the pair (log scale)", y = "Accuracy",
         title = "Accuracy of sister pairs by the depth of the pair",
         subtitle = "Dev pairs split into 10 equal bins by depth at each p; thinning draw 0, mean over 5 roots, 95% clone bootstrap band.",
         caption = "Within one p, the deepest bin is enriched in pairs where both cells are undifferentiated (31% vs 17-25%).") +
    theme_bench() +
    theme(panel.grid.major.y = element_line(colour = grid, linewidth = 0.3))
  save_both(figD1, "figS_depth_acc_all", 10, 8)
  full_depth <- dep |> filter(p == "1", frontend == "pca", backend == "DPT") |> arrange(bin) |> pull(depth)
  main_dep <- dep |>
    filter(backend == "DPT", frontend %in% c("pca", "scvi_s0"), p %in% c("1", "0.3", "0.1")) |>
    mutate(group = bin + 1, p = droplevels(p))
  figD0 <- ggplot(main_dep, aes(group, acc, colour = p, fill = p, group = p)) +
    geom_ribbon(aes(ymin = acc_lo, ymax = acc_hi), colour = NA, alpha = 0.12) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.6) +
    facet_wrap(~latent, nrow = 1) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0.4, 1)) +
    scale_colour_manual(values = c("1" = "#0d366b", "0.3" = "#3987e5", "0.1" = "#9ec5f4"),
                        name = "Kept fraction of molecules p") +
    scale_fill_manual(values = c("1" = "#0d366b", "0.3" = "#3987e5", "0.1" = "#9ec5f4"), guide = "none") +
    labs(x = "Group of sister pairs by molecules in the shallower cell (1 = fewest, 10 = most)", y = "Accuracy (DPT)",
         title = "Removing molecules lowers accuracy for all pairs, not only for shallow ones",
         subtitle = "Same dev pairs in each group at every p; thinning draw 0, mean over 5 roots, 95% clone bootstrap band.",
         caption = sprintf("At full depth the median of the groups ranges from %s to %s molecules. Group 10 is enriched in pairs of two undifferentiated cells.",
                           format(round(min(full_depth)), big.mark = ","), format(round(max(full_depth)), big.mark = ","))) +
    theme_bench() +
    theme(panel.grid.major.y = element_line(colour = grid, linewidth = 0.3), panel.grid.major.x = element_blank())
  save_both(figD0, "fig_depth_acc", 9, 4.5)
  figD2 <- ggplot(filter(dep, frontend != "pca"), aes(depth, d, colour = p, fill = p, group = p)) +
    geom_hline(yintercept = 0, colour = muted, linewidth = 0.4, linetype = "22") +
    geom_ribbon(aes(ymin = d_lo, ymax = d_hi), colour = NA, alpha = 0.15) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 1.2) +
    facet_grid(latent ~ backend) +
    log_x +
    scale_y_continuous(labels = function(x) sprintf("%+.2f", x)) +
    scale_colour_manual(values = p_col, name = "Kept fraction of molecules p") +
    scale_fill_manual(values = p_col, guide = "none") +
    labs(x = "Molecules in the shallower cell of the pair (log scale)", y = "Accuracy minus PCA-50",
         title = "Gain over PCA-50 by the depth of the pair",
         subtitle = "Same pairs and bins as the accuracy figure; paired clone bootstrap band.") +
    theme_bench() +
    theme(panel.grid.major.y = element_line(colour = grid, linewidth = 0.3))
  save_both(figD2, "fig_depth_diff", 10, 6)
}

cat("wrote", out_dir, "\n")
