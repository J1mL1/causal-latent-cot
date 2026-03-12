#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

parse_args <- function(args) {
  opts <- list(matrix_csv = NULL, out_path = NULL, title = "Aligned Explicit Graph",
               pct = 70, topk = 0, topk_direction = "out",
               threshold_mode = "percentile", max_ratio = 0.9)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--matrix_csv") {
      opts$matrix_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--title") {
      opts$title <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--pct") {
      opts$pct <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--topk") {
      opts$topk <- as.integer(args[[i + 1]])
      i <- i + 2
    } else if (key == "--topk_direction") {
      opts$topk_direction <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--threshold_mode") {
      opts$threshold_mode <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--max_ratio") {
      opts$max_ratio <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$matrix_csv)) stop("--matrix_csv is required")
  if (is.null(opts$out_path)) stop("--out_path is required")
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

df <- read.csv(opts$matrix_csv, check.names = FALSE)
row_labels <- df[[1]]
mat <- as.matrix(df[, -1, drop = FALSE])
rownames(mat) <- row_labels
colnames(mat) <- colnames(df)[-1]

plot_df <- as.data.frame(as.table(mat))
names(plot_df) <- c("step_i", "step_j", "value")

finite_vals <- plot_df$value[is.finite(plot_df$value)]
abs_vals <- abs(finite_vals)
threshold_mode <- opts$threshold_mode
thr <- -Inf
if (length(abs_vals) == 0) {
  thr <- Inf
} else if (threshold_mode == "max_ratio") {
  thr <- max(abs_vals) * opts$max_ratio
} else if (threshold_mode == "percentile") {
  thr <- as.numeric(quantile(abs_vals, probs = opts$pct / 100, names = FALSE, type = 7))
}

plot_df <- plot_df %>%
  mutate(abs_value = abs(value),
         keep_thr = abs_value >= thr)

if (!is.na(opts$topk) && opts$topk > 0) {
  group_col <- if (opts$topk_direction == "in") "step_j" else "step_i"
  topk_tbl <- plot_df %>%
    group_by(.data[[group_col]]) %>%
    arrange(desc(abs_value), .by_group = TRUE) %>%
    slice_head(n = opts$topk) %>%
    mutate(keep_topk = TRUE) %>%
    ungroup() %>%
    select(step_i, step_j, keep_topk)
  plot_df <- plot_df %>%
    left_join(topk_tbl, by = c("step_i", "step_j")) %>%
    mutate(keep_topk = ifelse(is.na(keep_topk), FALSE, keep_topk)) %>%
    filter(keep_thr | keep_topk)
} else {
  plot_df <- plot_df %>% filter(keep_thr)
}
plot_df <- plot_df %>% select(step_i, step_j, value)

p <- ggplot(plot_df, aes(x = step_j, y = step_i, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c", na.value = "grey90") +
  labs(title = opts$title, x = "explicit step j", y = "explicit step i", fill = "value") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(opts$out_path, p, width = 6.5, height = 5.5, dpi = 300)
