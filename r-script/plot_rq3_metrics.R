#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(scales)
})

parse_args <- function(args) {
  opts <- list(metrics_csv = NULL, out_dir = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--metrics_csv") {
      opts$metrics_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$metrics_csv)) {
    stop("Missing required --metrics_csv")
  }
  if (is.null(opts$out_dir)) {
    opts$out_dir <- dirname(opts$metrics_csv)
  }
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

df <- read.csv(opts$metrics_csv, stringsAsFactors = FALSE)
if (!dir.exists(opts$out_dir)) {
  dir.create(opts$out_dir, recursive = TRUE)
}

metric_labels <- c(
  SS = "Superposition Score (SS)",
  SignedDeltaP = "Signed Delta P",
  Entropy = "Entropy",
  DeltaP = "Delta P",
  AFR = "AFR"
)
df$metric <- factor(df$metric, levels = names(metric_labels), labels = metric_labels)

p_main <- ggplot(subset(df, metric != "AFR"), aes(x = step, y = mean)) +
  geom_line(linewidth = 0.9, color = "#1f77b4") +
  geom_point(size = 2, color = "#1f77b4") +
  geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem), alpha = 0.15, fill = "#1f77b4") +
  facet_wrap(~ metric, scales = "free_y") +
  labs(x = "step", y = "mean (± sem)", title = "RQ3 Metrics by Step") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

out_path <- file.path(opts$out_dir, "rq3_metrics_by_step.png")
ggsave(out_path, p_main, width = 8.5, height = 4.8, dpi = 300)
message("Saved: ", out_path)

df_afr <- subset(df, metric == "AFR")
if (nrow(df_afr) > 0) {
  if (length(unique(df_afr$phase)) > 1) {
    p_afr <- ggplot(df_afr, aes(x = step, y = mean, color = phase)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 2) +
      labs(x = "step", y = "AFR", title = "AFR by Ablation Step", color = NULL) +
      theme_minimal(base_size = 12) +
      theme(panel.grid = element_blank(), legend.position = "bottom")
  } else {
    p_afr <- ggplot(df_afr, aes(x = step, y = mean)) +
      geom_line(linewidth = 0.9, color = "#2ca02c") +
      geom_point(size = 2, color = "#2ca02c") +
      labs(x = "step", y = "AFR", title = "AFR by Ablation Step") +
      theme_minimal(base_size = 12) +
      theme(panel.grid = element_blank())
  }
  out_path_afr <- file.path(opts$out_dir, "rq3_afr_by_step.png")
  ggsave(out_path_afr, p_afr, width = 6.8, height = 4.0, dpi = 300)
  message("Saved: ", out_path_afr)
}

summary_path <- file.path(opts$out_dir, "rq3_metrics_summary.json")
if (file.exists(summary_path)) {
  summary <- jsonlite::read_json(summary_path, simplifyVector = TRUE)
  metrics <- c("AFR", "Orthogonality")
  values <- c(as.numeric(summary$afr), as.numeric(summary$orthogonality_mean_dot))
  if (!is.null(summary$decision_step_mean)) {
    metrics <- c(metrics, "DecisionStep")
    values <- c(values, as.numeric(summary$decision_step_mean))
  }
  df_sum <- data.frame(metric = metrics, value = values)
  p2 <- ggplot(df_sum, aes(x = metric, y = value, fill = metric)) +
    geom_col(width = 0.6, color = "white") +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = c("AFR" = "#2ca02c", "Orthogonality" = "#9467bd", "DecisionStep" = "#ff7f0e")) +
    labs(x = NULL, y = "value", title = "RQ3 Summary Metrics") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank(), legend.position = "none")
  out_path2 <- file.path(opts$out_dir, "rq3_metrics_summary.png")
  ggsave(out_path2, p2, width = 6.6, height = 3.6, dpi = 300)
  message("Saved: ", out_path2)
}

tf_csv <- file.path(opts$out_dir, "rq3_metrics_teacher_forced.csv")
if (file.exists(tf_csv)) {
  df_tf <- read.csv(tf_csv, stringsAsFactors = FALSE)
  df_tf$metric <- factor(df_tf$metric, levels = names(metric_labels), labels = metric_labels)
  p_tf <- ggplot(df_tf, aes(x = step, y = mean)) +
    geom_line(linewidth = 0.9, color = "#9467bd") +
    geom_point(size = 2, color = "#9467bd") +
    geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem), alpha = 0.15, fill = "#9467bd") +
    facet_wrap(~ metric, scales = "free_y") +
    labs(x = "step", y = "mean (± sem)", title = "Teacher-forced Competition by Step") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
  out_path_tf <- file.path(opts$out_dir, "rq3_teacher_forced_by_step.png")
  ggsave(out_path_tf, p_tf, width = 8.5, height = 4.8, dpi = 300)
  message("Saved: ", out_path_tf)
}
