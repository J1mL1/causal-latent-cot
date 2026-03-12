#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
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

read_jsonl <- function(path) {
  if (!file.exists(path)) return(data.frame())
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  if (length(lines) == 0) return(data.frame())
  as.data.frame(jsonlite::stream_in(textConnection(lines), verbose = FALSE))
}

summarize_model <- function(dir_path, label) {
  sample_path <- file.path(dir_path, "ambiguous", "ambiguous_samples.jsonl")
  traj_path <- file.path(dir_path, "ambiguous", "ambiguous_trajectories.jsonl")
  samples <- read_jsonl(sample_path)
  if (!nrow(samples)) return(NULL)
  trajs <- read_jsonl(traj_path)

  summary_df <- data.frame(
    model = label,
    n_samples = nrow(samples),
    n_trajectories = nrow(trajs),
    avg_count_A = mean(as.numeric(samples$count_A), na.rm = TRUE),
    avg_count_B = mean(as.numeric(samples$count_B), na.rm = TRUE),
    avg_ratio_B = mean(as.numeric(samples$ratio_B), na.rm = TRUE),
    total_count_A = sum(as.numeric(samples$count_A), na.rm = TRUE),
    total_count_B = sum(as.numeric(samples$count_B), na.rm = TRUE)
  )

  cluster_df <- data.frame(
    model = label,
    cluster = c("A", "B"),
    mean_count = c(summary_df$avg_count_A, summary_df$avg_count_B)
  )

  list(summary = summary_df, cluster = cluster_df)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  out_dir <- if (is.null(opts$out_dir)) "outputs/rq3/plots/rq3-strategyqa-sampling" else opts$out_dir
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  summaries <- list()
  clusters <- list()
  for (label in names(opts$inputs)) {
    res <- summarize_model(opts$inputs[[label]], label)
    if (is.null(res)) next
    summaries[[length(summaries) + 1]] <- res$summary
    clusters[[length(clusters) + 1]] <- res$cluster
  }

  summary_df <- bind_rows(summaries)
  cluster_df <- bind_rows(clusters)
  model_order <- names(opts$inputs)
  if (nrow(summary_df) == 0 || nrow(cluster_df) == 0) {
    stop("No valid ambiguous_samples.jsonl found.")
  }

  summary_df$model <- factor(summary_df$model, levels = model_order)
  cluster_df$model <- factor(cluster_df$model, levels = model_order)
  cluster_df$cluster <- factor(cluster_df$cluster, levels = c("A", "B"))

  write.csv(summary_df, file.path(out_dir, "rq3_strategyqa_sampling_summary.csv"), row.names = FALSE)

  p_cluster <- ggplot(cluster_df, aes(x = model, y = mean_count, fill = cluster)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.65) +
    scale_fill_manual(values = c("A" = "#4c78a8", "B" = "#f58518")) +
    labs(title = "Average sampled trajectories per cluster", x = "model", y = "mean count", fill = "cluster") +
    theme_classic(base_size = 12) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))

  p_samples <- ggplot(summary_df, aes(x = model, y = n_samples, fill = model)) +
    geom_col(width = 0.7) +
    labs(title = "Ambiguous samples (count)", x = "model", y = "n_samples") +
    theme_classic(base_size = 12) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "none")

  combined <- p_cluster + p_samples + plot_layout(ncol = 2, widths = c(1.3, 1))
  ggsave(file.path(out_dir, "rq3_strategyqa_sampling.png"), combined, width = 11, height = 4.2, dpi = 300)
}

main()
