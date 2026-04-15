#!/usr/bin/env Rscript
# RQ4 matched arithmetic patch: figures from run_matched_arithmetic_patch.py JSONL.
# Three panels: target gold log-prob drop (matched vs random), source-answer delta, greedy mismatch rate.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(patchwork)
  library(grid)
})

parse_args <- function(args) {
  opts <- list(jsonl = NULL, out_prefix = NULL, summary_json = NULL, width = 18, height = 5.5, dpi = 150)
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--jsonl") {
      opts$jsonl <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--out_prefix") {
      opts$out_prefix <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--summary_json") {
      opts$summary_json <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--width") {
      opts$width <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--height") {
      opts$height <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--dpi") {
      opts$dpi <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(opts$jsonl) || !nzchar(opts$jsonl)) {
    stop("--jsonl path required")
  }
  opts
}

read_jsonl_lines <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nchar(trimws(lines)) > 0]
  if (length(lines) == 0) return(tibble())
  bind_rows(lapply(lines, function(l) {
    tryCatch({
      as_tibble(jsonlite::fromJSON(l))
    }, error = function(e) NULL)
  }))
}

#' Greedy decode (matched) differs from target gold — NOT baseline→intervened flip.
greedy_mismatch_col <- function(df) {
  if ("greedy_matched_neq_target_gold" %in% names(df)) {
    return(as.logical(df$greedy_matched_neq_target_gold))
  }
  if ("flip_matched_wrong_vs_target" %in% names(df)) {
    return(as.logical(df$flip_matched_wrong_vs_target))
  }
  rep(NA, nrow(df))
}

sem <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(0)
  stats::sd(x) / sqrt(length(x))
}

# Tighter titles + margins so 3-column patchwork does not overlap labels.
panel_theme <- function() {
  theme_bw(base_size = 10) +
    theme(
      plot.title = element_text(
        size = 9.5, hjust = 0.5, lineheight = 1.15,
        margin = margin(b = 4, t = 2)
      ),
      plot.margin = margin(6, 10, 6, 10),
      axis.title = element_text(size = 9),
      axis.title.y = element_text(margin = margin(r = 6)),
      legend.position = "top",
      legend.title = element_blank(),
      legend.key.size = unit(0.35, "cm"),
      legend.margin = margin(b = 2)
    )
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  df <- read_jsonl_lines(opts$jsonl)
  if (!nrow(df)) stop("Empty JSONL: ", opts$jsonl)
  if ("error" %in% names(df)) {
    df <- df %>% filter(is.na(.data$error) | .data$error == "")
  }
  if (!"latent_step" %in% names(df)) stop("JSONL missing latent_step")
  df$latent_step <- as.integer(df$latent_step)

  jsonl_path <- opts$jsonl
  if (is.null(opts$out_prefix) || !nzchar(opts$out_prefix)) {
    base <- sub("\\.jsonl$", "", jsonl_path, ignore.case = TRUE)
    opts$out_prefix <- base
  }
  out_dir <- dirname(opts$out_prefix)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  base_name <- basename(opts$out_prefix)

  df$greedy_mismatch <- greedy_mismatch_col(df)

  sum_step <- df %>%
    group_by(latent_step) %>%
    summarise(
      n = n(),
      target_gold_logp_drop_matched_mean = mean(as.numeric(target_gold_logp_drop_matched), na.rm = TRUE),
      target_gold_logp_drop_matched_sem = sem(as.numeric(target_gold_logp_drop_matched)),
      target_gold_logp_drop_random_mean = mean(as.numeric(target_gold_logp_drop_random), na.rm = TRUE),
      target_gold_logp_drop_random_sem = sem(as.numeric(target_gold_logp_drop_random)),
      source_ans_delta_mean = mean(as.numeric(source_ans_logp_increase_matched_minus_random), na.rm = TRUE),
      source_ans_delta_sem = sem(as.numeric(source_ans_logp_increase_matched_minus_random)),
      greedy_mismatch_rate = mean(greedy_mismatch == TRUE, na.rm = TRUE),
      greedy_mismatch_sem = {
        gm <- as.integer(greedy_mismatch == TRUE)
        gm <- gm[is.finite(gm)]
        if (length(gm) < 2) 0 else stats::sd(gm) / sqrt(length(gm))
      },
      .groups = "drop"
    )

  # --- Panel 1: grouped bars ---
  long_drop <- sum_step %>%
    select(
      latent_step,
      target_gold_logp_drop_matched_mean,
      target_gold_logp_drop_matched_sem,
      target_gold_logp_drop_random_mean,
      target_gold_logp_drop_random_sem
    ) %>%
    pivot_longer(
      cols = c(target_gold_logp_drop_matched_mean, target_gold_logp_drop_random_mean),
      names_to = "patch",
      values_to = "mean"
    ) %>%
    mutate(
      patch = ifelse(grepl("matched", patch), "matched source", "random source"),
      sem = ifelse(
        patch == "matched source",
        sum_step$target_gold_logp_drop_matched_sem[match(latent_step, sum_step$latent_step)],
        sum_step$target_gold_logp_drop_random_sem[match(latent_step, sum_step$latent_step)]
      )
    )

  p1 <- ggplot(long_drop, aes(x = factor(latent_step), y = mean, fill = patch)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.8) +
    geom_errorbar(
      aes(ymin = mean - sem, ymax = mean + sem),
      position = position_dodge(width = 0.85),
      width = 0.2
    ) +
    scale_fill_manual(values = c("matched source" = "#2c7fb8", "random source" = "#feb24c")) +
    labs(
      x = "Latent step",
      y = expression(Delta * log * p * " (target gold)"),
      title = "Target gold log-prob drop"
    ) +
    panel_theme() +
    theme(axis.title.y = element_text(size = 9, margin = margin(r = 6)))

  # --- Panel 2: source delta ---
  p2 <- ggplot(sum_step, aes(x = factor(latent_step), y = source_ans_delta_mean,
                             fill = source_ans_delta_mean >= 0)) +
    geom_col(width = 0.55) +
    geom_errorbar(aes(ymin = source_ans_delta_mean - source_ans_delta_sem,
                      ymax = source_ans_delta_mean + source_ans_delta_sem), width = 0.2) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("TRUE" = "#31a354", "FALSE" = "#e34a33"), guide = "none") +
    labs(
      x = "Latent step",
      y = expression(Delta * log * p * " (source gold)"),
      title = "Source-answer log-prob shift"
    ) +
    panel_theme()

  # --- Panel 3: greedy mismatch (not strict baseline flip) ---
  p3 <- ggplot(sum_step, aes(x = factor(latent_step), y = greedy_mismatch_rate)) +
    geom_col(width = 0.55, fill = "#756bb1") +
    geom_errorbar(aes(ymin = pmax(0, greedy_mismatch_rate - greedy_mismatch_sem),
                      ymax = pmin(1, greedy_mismatch_rate + greedy_mismatch_sem)), width = 0.2) +
    scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0)) +
    labs(
      x = "Latent step",
      y = "Rate",
      title = "Greedy pred != target gold"
    ) +
    panel_theme()

  combined <- p1 + p2 + p3 + plot_layout(ncol = 3, widths = c(1, 1, 1))

  pdf_path <- file.path(out_dir, paste0(base_name, "_combined.pdf"))
  png_path <- file.path(out_dir, paste0(base_name, "_combined.png"))

  ggsave(pdf_path, combined, width = opts$width, height = opts$height)
  ggsave(png_path, combined, width = opts$width, height = opts$height, dpi = opts$dpi)

  # Separate single-panel PDFs for slides
  ggsave(file.path(out_dir, paste0(base_name, "_target_gold_drop.pdf")), p1, width = 4.5, height = 4)
  ggsave(file.path(out_dir, paste0(base_name, "_source_ans_delta.pdf")), p2, width = 4.5, height = 4)
  ggsave(file.path(out_dir, paste0(base_name, "_greedy_mismatch_target.pdf")), p3, width = 4.5, height = 4)

  # Summary JSON (per-step means / SEMs)
  summ <- list(
    n_rows_ok = nrow(df),
    by_step = jsonlite::parse_json(jsonlite::toJSON(sum_step, dataframe = "rows", na = "null"))
  )
  summ_path <- if (!is.null(opts$summary_json) && nzchar(opts$summary_json)) {
    opts$summary_json
  } else {
    file.path(out_dir, paste0(base_name, "_summary.json"))
  }
  writeLines(jsonlite::toJSON(summ, pretty = TRUE, auto_unbox = TRUE), summ_path, useBytes = TRUE)

  message("Wrote: ", pdf_path, ", ", png_path, ", ", summ_path)
}

tryCatch(main(), error = function(e) {
  message("ERROR: ", conditionMessage(e))
  quit(status = 1)
})
