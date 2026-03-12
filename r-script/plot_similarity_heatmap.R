#!/usr/bin/env Rscript

# Plot similarity matrix heatmap with the same ggplot style as latent graph heatmaps.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

parse_args <- function(args) {
  opts <- list(input = NULL, out_path = NULL, title = NULL, value_label = NULL,
               split_by_dataset = FALSE)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      opts$input <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--title") {
      opts$title <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--value_label") {
      opts$value_label <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--split_by_dataset") {
      opts$split_by_dataset <- TRUE
      i <- i + 1
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$input) || is.null(opts$out_path)) {
    stop("--input and --out_path are required.")
  }
  if (is.null(opts$value_label)) opts$value_label <- "similarity"
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

mat <- read.csv(opts$input, row.names = 1, check.names = FALSE)
mat <- as.matrix(mat)
labels <- rownames(mat)

make_plot <- function(df, out_path, title) {
  p <- ggplot(df, aes(x = col, y = row, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = label), size = 3) +
    scale_fill_gradient(low = "#f7fbff", high = "#08306b", na.value = "grey95") +
    coord_equal() +
    labs(x = NULL, y = NULL, fill = opts$value_label, title = title) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  ggsave(out_path, p, width = 7, height = 6, dpi = 300, bg = "white")
}

prepare_df <- function(mat, labels) {
  df <- as.data.frame(mat) %>%
    mutate(row = labels) %>%
    pivot_longer(-row, names_to = "col", values_to = "value")
  df$row <- factor(df$row, levels = labels)
  df$col <- factor(df$col, levels = labels)
  df %>% mutate(label = ifelse(is.na(value), "", sprintf("%.2f", value)))
}

out_dir <- dirname(opts$out_path)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (opts$split_by_dataset) {
  datasets <- c("commonsenseqa", "gsm8k")
  for (ds in datasets) {
    keep <- grepl(paste0("^", ds, ":"), labels)
    sub_labels <- labels[keep]
    sub_mat <- mat[sub_labels, sub_labels, drop = FALSE]
    df <- prepare_df(sub_mat, sub_labels)
    out_path <- sub("\\.(pdf|png)$", paste0("_", ds, ".\\1"), opts$out_path)
    title <- if (!is.null(opts$title)) paste(opts$title, "-", ds) else ds
    make_plot(df, out_path, title)
  }
} else {
  df <- prepare_df(mat, labels)
  make_plot(df, opts$out_path, opts$title)
}
