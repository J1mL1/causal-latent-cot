#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

parse_args <- function(args) {
  opts <- list(inputs = list(), out_dir = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --input label=dir")
      opts$inputs[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (length(opts$inputs) == 0) stop("At least one --input label=dir is required.")
  opts
}

read_metrics_per_step <- function(dir_path, label) {
  path <- file.path(dir_path, "rq3_metrics_per_step.csv")
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path)
  df$model <- label
  df
}

read_summary <- function(dir_path, label) {
  path <- file.path(dir_path, "rq3_metrics_summary.json")
  if (!file.exists(path)) return(NULL)
  obj <- jsonlite::fromJSON(path)
  data.frame(
    model = label,
    decision_step_mean = as.numeric(obj$decision_step_mean),
    decision_step_std = as.numeric(obj$decision_step_std),
    decision_step_n = as.numeric(obj$decision_step_n)
  )
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  out_dir <- if (is.null(opts$out_dir)) "outputs/plots/rq3-metrics-multi" else opts$out_dir
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  per_step_list <- list()
  summary_list <- list()
  for (label in names(opts$inputs)) {
    dir_path <- opts$inputs[[label]]
    per_step_list[[length(per_step_list) + 1]] <- read_metrics_per_step(dir_path, label)
    summary_list[[length(summary_list) + 1]] <- read_summary(dir_path, label)
  }
  per_step_df <- bind_rows(per_step_list)
  summary_df <- bind_rows(summary_list)

  # Decision step mean bar
  if (nrow(summary_df) > 0) {
    p_bar <- ggplot(summary_df, aes(x = reorder(model, decision_step_mean), y = decision_step_mean, fill = model)) +
      geom_col(width = 0.7) +
      labs(title = "Decision step mean", x = "model", y = "decision_step_mean") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")
    ggsave(file.path(out_dir, "rq3_decision_step_mean_bar.png"), p_bar, width = 7.5, height = 4.5, dpi = 300)
  }

  if (nrow(per_step_df) > 0) {
    entropy_df <- per_step_df %>% filter(metric == "Entropy")
    if (nrow(entropy_df) > 0) {
      p_entropy <- ggplot(entropy_df, aes(x = step, y = mean, color = model)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2.0) +
        labs(title = "Entropy by step", x = "step", y = "Entropy", color = "model") +
        theme_minimal(base_size = 12)
      ggsave(file.path(out_dir, "rq3_entropy_by_step.png"), p_entropy, width = 9, height = 4.5, dpi = 300)
    }

    ss_df <- per_step_df %>% filter(metric == "SS")
    if (nrow(ss_df) > 0) {
      p_ss <- ggplot(ss_df, aes(x = step, y = mean, color = model)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2.0) +
        labs(title = "SS by step", x = "step", y = "SS", color = "model") +
        theme_minimal(base_size = 12)
      ggsave(file.path(out_dir, "rq3_ss_by_step.png"), p_ss, width = 9, height = 4.5, dpi = 300)
    }
  }
}

main()
