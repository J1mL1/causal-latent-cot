#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

parse_args <- function(args) {
  opts <- list(inputs = list(), out_path = NULL, mode = "probe")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --input label=path")
      opts$inputs[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--mode") {
      opts$mode <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (length(opts$inputs) == 0) stop("Provide at least one --input label=path")
  if (is.null(opts$out_path)) stop("--out_path required")
  opts
}

read_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  if (length(lines) == 0) return(data.frame())
  as.data.frame(jsonlite::stream_in(textConnection(lines), verbose = FALSE))
}

compute_ss_probe <- function(df) {
  if (!nrow(df)) return(df)
  if ("cos_A" %in% names(df) && "cos_B" %in% names(df)) {
    norm_a <- (as.numeric(df$cos_A) + 1) * 0.5
    norm_b <- (as.numeric(df$cos_B) + 1) * 0.5
  } else {
    norm_a <- as.numeric(df$score_A)
    norm_b <- as.numeric(df$score_B)
  }
  ss <- pmin(norm_a, norm_b)
  df$ss <- ss
  df
}

compute_ss_tf <- function(df) {
  if (!nrow(df)) return(df)
  if ("ss" %in% names(df)) {
    df$ss <- as.numeric(df$ss)
  } else if ("s_yes" %in% names(df) && "s_no" %in% names(df)) {
    df$ss <- pmin(as.numeric(df$s_yes), as.numeric(df$s_no))
  } else {
    df$ss <- NA_real_
  }
  df
}

summarize_steps <- function(df, label) {
  df %>%
    group_by(step) %>%
    summarise(
      mean = mean(ss, na.rm = TRUE),
      sem = sd(ss, na.rm = TRUE) / sqrt(sum(is.finite(ss))),
      .groups = "drop"
    ) %>%
    mutate(model = label)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  mode <- tolower(opts$mode)

  color_map <- c(
    "Coconut-GPT2" = "#c6dbef",
    "Coconut-Llama3-1B" = "#6baed6",
    "Coconut-Qwen3-4B" = "#2171b5",
    "CODI-GPT2" = "#fcbba1",
    "CODI-Llama3-1B" = "#fb6a4a",
    "CODI-Qwen3-4B" = "#cb181d"
  )
  model_levels <- names(color_map)

  all_rows <- list()
  for (label in names(opts$inputs)) {
    path <- opts$inputs[[label]]
    df <- read_jsonl(path)
    if (!nrow(df)) next
    if (mode == "probe") {
      df <- compute_ss_probe(df)
    } else {
      df <- compute_ss_tf(df)
    }
    if (!"step" %in% names(df)) next
    df$step <- as.integer(df$step)
    all_rows[[length(all_rows) + 1]] <- summarize_steps(df, label)
  }
  plot_df <- bind_rows(all_rows)
  if (!nrow(plot_df)) stop("No data to plot.")

  plot_df$model <- factor(plot_df$model, levels = model_levels)

  title_label <- if (mode == "probe") "metric: probe" else "metric: teacher-forced logp"
  p <- ggplot(plot_df, aes(x = step, y = mean, color = model)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem, fill = model), alpha = 0.15, color = NA) +
    scale_color_manual(values = color_map, drop = FALSE) +
    scale_fill_manual(values = color_map, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(plot_df$step))) +
    labs(x = "step", y = "superposition score", color = NULL, fill = NULL, title = title_label) +
    theme_classic(base_size = 12) +
    theme(
      axis.line = element_line(linewidth = 0.5, color = "black"),
      axis.ticks = element_line(linewidth = 0.4, color = "black"),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.text = element_text(size = 10),
      plot.margin = margin(6, 8, 6, 8)
    )

  out_dir <- dirname(opts$out_path)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(opts$out_path, p, width = 6.25, height = 4.4, dpi = 300, bg = "white")
}

main()
