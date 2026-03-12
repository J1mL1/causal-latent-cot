#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(file_arg[1])))
  }
  normalizePath(".")
}

source(file.path(get_script_dir(), "graph_common.R"))

build_latent_aggregates <- function(df, metric) {
  ensure_cols(df, c("step_i", "step_j", metric))
  df <- df %>%
    mutate(step_i = as.character(step_i),
           step_j = as.character(step_j))

  steps <- build_step_levels(df)

  kl_tbl <- df %>%
    group_by(step_i, step_j) %>%
    summarise(kl = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")

  list(steps = steps, kl_tbl = kl_tbl)
}

plot_heatmap <- function(steps, kl_tbl, out_path, metric_label = "kl_mean") {
  mat <- kl_tbl %>%
    mutate(step_i = factor(step_i, levels = steps),
           step_j = factor(step_j, levels = steps)) %>%
    complete(step_i, step_j, fill = list(kl = NA_real_)) %>%
    arrange(step_i, step_j) %>%
    mutate(kl = ifelse(step_i == step_j, NA_real_, kl))

  p <- ggplot(mat, aes(x = step_j, y = step_i, fill = kl)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(is.na(kl), "", sprintf("%.2f", kl))), size = 3) +
    scale_fill_viridis_c(option = "C", na.value = "grey95") +
    coord_equal() +
    labs(x = "step_j", y = "step_i", fill = metric_label) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_blank(),
      panel.grid = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )

  ggsave(out_path, p, width = 7, height = 6, dpi = 300, bg = "white")
}

parse_args <- function(args) {
  opts <- list(input = NULL, out_path = NULL, metric = "kl_mean", max_steps = 6)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      opts$input <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--metric") {
      opts$metric <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--max_steps") {
      opts$max_steps <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  opts
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)
  if (is.null(opts$input) || is.null(opts$out_path)) {
    stop("--input and --out_path are required.")
  }

  df <- read_jsonl_df(opts$input)
  agg <- build_latent_aggregates(df, opts$metric)
  steps <- agg$steps
  if (!is.null(opts$max_steps)) {
    steps <- steps[seq_len(min(length(steps), opts$max_steps))]
  }
  agg$kl_tbl <- agg$kl_tbl %>%
    filter(step_i %in% steps, step_j %in% steps)

  out_dir <- dirname(opts$out_path)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  plot_heatmap(steps, agg$kl_tbl, opts$out_path, metric_label = opts$metric)
}

main()
