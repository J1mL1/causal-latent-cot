#!/usr/bin/env Rscript

# R port of scripts/plot/python/rq1/analyze_ablation_jsonl.py
# Produces the same heatmaps with the ggplot style used in plot_latent_causal_graph.R.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
script_dir <- if (length(script_path) > 0 && nzchar(script_path)) {
  dirname(normalizePath(script_path))
} else {
  normalizePath(".")
}
source(file.path(script_dir, "ablation_parse_utils.R"))

MODE_DESCRIPTIONS <- c(
  "zero" = "Replace latent with zeros",
  "mean" = "Replace latent with global mean",
  "gaussian_h" = "Add Gaussian noise to latent h_t",
  "gaussian_mu" = "Add Gaussian noise around global mean",
  "mean_step" = "Replace latent with step-specific mean",
  "gaussian_mu_step" = "Add Gaussian noise around step-specific mean"
)

to_step_int <- function(x) suppressWarnings(as.integer(as.character(x)))

plot_heatmap <- function(mat, title, fill_label, out_png, diverging = FALSE) {
  if (!nrow(mat)) return(invisible(NULL))
  vals <- mat$value
  if (diverging && any(is.finite(vals))) {
    max_abs <- max(abs(vals[is.finite(vals)]))
    limits <- c(-max_abs, max_abs)
    fill_scale <- scale_fill_distiller(palette = "RdBu", direction = -1, limits = limits, na.value = "grey95")
  } else {
    fill_scale <- scale_fill_viridis_c(option = "C", na.value = "grey95")
  }

  p <- ggplot(mat, aes(x = step, y = mode, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = label), size = 3) +
    fill_scale +
    coord_equal() +
    labs(x = "step", y = "mode", fill = fill_label, title = title) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank())

  ggsave(out_png, p, width = 7, height = 6, dpi = 300)
}

parse_args <- function(args) {
  opts <- list(path = NULL, out_dir = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--path") {
      opts$path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$path)) {
    stop("Missing required --path")
  }
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)
df <- read_jsonl_df(opts$path)

if (!all(c("sample_id", "mode", "step") %in% names(df))) {
  stop("Missing required columns: sample_id, mode, step")
}

df <- df %>%
  mutate(
    step_i = to_step_int(step),
    batch_idx = ifelse("batch_idx" %in% names(df), as.integer(batch_idx), 0L),
    gold_is_choice = grepl("^[A-E]$", as.character(gold_answer)),
    gold = sapply(gold_answer, parse_gold),
    baseline_text = {
      b_list <- to_list_col(baseline, nrow(df))
      vapply(seq_len(nrow(df)), function(i) get_text_field(b_list[[i]], batch_idx[[i]]), character(1))
    },
    ablated_text = {
      a_list <- to_list_col(ablated, nrow(df))
      vapply(seq_len(nrow(df)), function(i) get_text_field(a_list[[i]], batch_idx[[i]]), character(1))
    },
    gold_choice = ifelse(gold_is_choice, as.character(gold_answer), NA_character_),
    gold_num = ifelse(!gold_is_choice, suppressWarnings(as.numeric(gold)), NA_real_),
    baseline_choice = sapply(baseline_text, extract_choice),
    ablated_choice = sapply(ablated_text, extract_choice),
    baseline_pred = sapply(baseline_text, extract_number),
    ablated_pred = sapply(ablated_text, extract_number),
    baseline_correct = ifelse(
      gold_is_choice,
      !is.na(baseline_choice) & !is.na(gold_choice) & baseline_choice == gold_choice,
      !is.na(baseline_pred) & !is.na(gold_num) & abs(baseline_pred - gold_num) < 1e-6
    ),
    ablated_correct = ifelse(
      gold_is_choice,
      !is.na(ablated_choice) & !is.na(gold_choice) & ablated_choice == gold_choice,
      !is.na(ablated_pred) & !is.na(gold_num) & abs(ablated_pred - gold_num) < 1e-6
    )
  )

baseline_tbl <- df %>%
  filter(!is.na(step_i)) %>%
  group_by(step_i) %>%
  summarise(
    total = sum(!is.na(baseline_correct)),
    correct = sum(baseline_correct, na.rm = TRUE),
    acc = ifelse(total > 0, correct / total, NA_real_),
    .groups = "drop"
  )

format_bucket <- function(total, correct) {
  acc <- ifelse(total > 0, correct / total, 0)
  sprintf("acc=%.3f (correct=%d/%d)", acc, as.integer(correct), as.integer(total))
}

format_flip <- function(total, w2r, r2w) {
  if (total <= 0) return("flip rate=n/a")
  rate <- (w2r + r2w) / total
  w2r_rate <- w2r / total
  r2w_rate <- r2w / total
  sprintf(
    "flip rate=%.3f (wrong->right=%.3f (%d/%d) right->wrong=%.3f (%d/%d))",
    rate, w2r_rate, as.integer(w2r), as.integer(total),
    r2w_rate, as.integer(r2w), as.integer(total)
  )
}

ablated_stats <- df %>%
  filter(!is.na(step_i)) %>%
  group_by(mode, step_i) %>%
  summarise(
    total = sum(!is.na(ablated_correct)),
    correct = sum(ablated_correct, na.rm = TRUE),
    acc = ifelse(total > 0, correct / total, NA_real_),
    .groups = "drop"
  ) %>%
  left_join(baseline_tbl %>% select(step_i, baseline_acc = acc), by = "step_i") %>%
  mutate(delta_acc = acc - baseline_acc)

flip_stats <- df %>%
  filter(!is.na(step_i), !is.na(baseline_correct), !is.na(ablated_correct)) %>%
  group_by(mode, step_i) %>%
  summarise(
    total = n(),
    correct_to_wrong = sum(baseline_correct & !ablated_correct, na.rm = TRUE),
    wrong_to_correct = sum(!baseline_correct & ablated_correct, na.rm = TRUE),
    flip_rate = ifelse(total > 0, (correct_to_wrong + wrong_to_correct) / total, NA_real_),
    .groups = "drop"
  )

logp_stats <- df %>%
  filter(!is.na(step_i)) %>%
  group_by(mode, step_i) %>%
  summarise(
    mean_delta_seq = mean(teacher_forced_delta_sum, na.rm = TRUE),
    mean_delta_final = mean(delta_logp_final_token, na.rm = TRUE),
    .groups = "drop"
  )

prefix <- tools::file_path_sans_ext(basename(opts$path))
out_dir <- if (is.null(opts$out_dir)) dirname(opts$path) else opts$out_dir
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}
prefix <- file.path(out_dir, prefix)

cat("=== Baseline (per step) ===\n")
if (nrow(baseline_tbl) == 0) {
  cat("acc=n/a\n")
} else {
  for (i in seq_len(nrow(baseline_tbl))) {
    row <- baseline_tbl[i, ]
    cat(sprintf("step=%d: %s\n", as.integer(row$step_i), format_bucket(row$total, row$correct)))
  }
}

cat("\n=== Ablated (greedy accuracy & flips) ===\n")
if (nrow(ablated_stats) > 0) {
  modes_present <- sort(unique(ablated_stats$mode))
  cat("Mode descriptions:\n")
  for (m in modes_present) {
    desc <- MODE_DESCRIPTIONS[[m]]
    if (is.null(desc)) desc <- "N/A"
    cat(sprintf("  %s: %s\n", m, desc))
  }

  merged_stats <- ablated_stats %>%
    left_join(flip_stats, by = c("mode", "step_i")) %>%
    arrange(mode, step_i)
  for (i in seq_len(nrow(merged_stats))) {
    row <- merged_stats[i, ]
    acc_msg <- format_bucket(row$total, row$correct)
    flip_msg <- format_flip(row$total.y, row$wrong_to_correct, row$correct_to_wrong)
    cat(sprintf(
      "%-15s step=%d: %s | %s\n",
      as.character(row$mode), as.integer(row$step_i), acc_msg, flip_msg
    ))
  }
}

if (nrow(logp_stats) > 0) {
  cat("\n=== Teacher-forced logp deltas (base - ablt) ===\n")
  cat("Positive: ablation hurts gold (lower p). Negative: ablation helps gold (higher p).\n")
  logp_print <- logp_stats %>%
    arrange(mode, step_i)
  for (i in seq_len(nrow(logp_print))) {
    row <- logp_print[i, ]
    seq_msg <- sprintf("%+.3f (n=%d)", row$mean_delta_seq, nrow(df %>% filter(mode == row$mode, step_i == row$step_i, !is.na(teacher_forced_delta_sum))))
    fin_msg <- sprintf("%+.3f (n=%d)", row$mean_delta_final, nrow(df %>% filter(mode == row$mode, step_i == row$step_i, !is.na(delta_logp_final_token))))
    cat(sprintf(
      "%-15s step=%d: mean Δlogp_seq=%s, mean Δlogp_final=%s\n",
      as.character(row$mode), as.integer(row$step_i), seq_msg, fin_msg
    ))
  }
}

if (nrow(ablated_stats) > 0) {
  mat <- ablated_stats %>%
    mutate(
      mode = factor(mode),
      step = factor(step_i, levels = sort(unique(step_i))),
      label = ifelse(is.na(delta_acc), "", sprintf("%+.3f", delta_acc)),
      value = delta_acc
    ) %>%
    select(mode, step, value, label)
  plot_heatmap(
    mat,
    "Delta accuracy vs baseline",
    "Δacc",
    paste0(prefix, "_delta_acc_heatmap.png"),
    diverging = TRUE
  )
}

if (nrow(flip_stats) > 0) {
  mat <- flip_stats %>%
    mutate(
      mode = factor(mode),
      step = factor(step_i, levels = sort(unique(step_i))),
      label = ifelse(is.na(flip_rate), "", sprintf("%.2f", flip_rate)),
      value = flip_rate
    ) %>%
    select(mode, step, value, label)
  plot_heatmap(
    mat,
    "Flip rate (w->r + r->w) by mode/step",
    "flip_rate",
    paste0(prefix, "_flip_heatmap.png"),
    diverging = FALSE
  )
}

if (nrow(logp_stats) > 0) {
  mat_seq <- logp_stats %>%
    mutate(
      mode = factor(mode),
      step = factor(step_i, levels = sort(unique(step_i))),
      label = ifelse(is.na(mean_delta_seq), "", sprintf("%+.2f", mean_delta_seq)),
      value = mean_delta_seq
    ) %>%
    select(mode, step, value, label)
  plot_heatmap(
    mat_seq,
    "Mean Δ log p(gold sequence) (base - ablt)",
    "Δlogp_seq",
    paste0(prefix, "_delta_logp_seq_heatmap.png"),
    diverging = TRUE
  )

  mat_final <- logp_stats %>%
    mutate(
      mode = factor(mode),
      step = factor(step_i, levels = sort(unique(step_i))),
      label = ifelse(is.na(mean_delta_final), "", sprintf("%+.2f", mean_delta_final)),
      value = mean_delta_final
    ) %>%
    select(mode, step, value, label)
  plot_heatmap(
    mat_final,
    "Mean Δ log p(final gold token) (base - ablt)",
    "Δlogp_final",
    paste0(prefix, "_delta_logp_final_heatmap.png"),
    diverging = TRUE
  )
}
