#!/usr/bin/env Rscript

# Plot step-wise flip rate bars with error bars for two datasets.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
script_dir <- if (length(script_path) > 0 && nzchar(script_path)) {
  dirname(normalizePath(script_path))
} else {
  normalizePath(".")
}
source(file.path(script_dir, "ablation_parse_utils.R"))

compute_flip_stats <- function(path, label, mode_name = "zero") {
  df <- read_jsonl_df(path)
  if (!all(c("mode", "step") %in% names(df))) return(NULL)

  df <- df %>%
    mutate(
      step_i = suppressWarnings(as.integer(as.character(step))),
      batch_idx = ifelse("batch_idx" %in% names(df), as.integer(batch_idx), 0L),
      gold_raw = ifelse("gold_answer" %in% names(df), as.character(gold_answer), NA_character_),
      baseline_text = {
        b_list <- to_list_col(baseline, nrow(df))
        vapply(seq_len(nrow(df)), function(i) get_text_field(b_list[[i]], batch_idx[[i]]), character(1))
      },
      ablated_text = {
        a_list <- to_list_col(ablated, nrow(df))
        vapply(seq_len(nrow(df)), function(i) get_text_field(a_list[[i]], batch_idx[[i]]), character(1))
      },
      gold_is_choice = grepl("^[A-E]$", as.character(gold_answer)),
      gold = sapply(gold_answer, parse_gold),
      gold_choice = ifelse(gold_is_choice, as.character(gold_answer), NA_character_),
      gold_num = ifelse(!gold_is_choice, suppressWarnings(as.numeric(gold)), NA_real_),
      baseline_choice = sapply(baseline_text, extract_choice),
      ablated_choice = sapply(ablated_text, extract_choice),
      baseline_num = sapply(baseline_text, extract_number),
      ablated_num = sapply(ablated_text, extract_number),
      baseline_correct = ifelse(
        gold_is_choice,
        !is.na(baseline_choice) & !is.na(gold_choice) & baseline_choice == gold_choice,
        !is.na(baseline_num) & !is.na(gold_num) & abs(baseline_num - gold_num) < 1e-6
      ),
      ablated_correct = ifelse(
        gold_is_choice,
        !is.na(ablated_choice) & !is.na(gold_choice) & ablated_choice == gold_choice,
        !is.na(ablated_num) & !is.na(gold_num) & abs(ablated_num - gold_num) < 1e-6
      )
    )

  df_mode <- df %>% filter(mode == mode_name, !is.na(step_i), step_i >= 1, step_i <= 6)

  flip_tbl <- df_mode %>%
    filter(!is.na(baseline_correct), !is.na(ablated_correct)) %>%
    group_by(step_i) %>%
    summarise(
      n = n(),
      flip_rate = mean(baseline_correct != ablated_correct),
      se = ifelse(n > 0, sqrt(flip_rate * (1 - flip_rate) / n), NA_real_),
      .groups = "drop"
    ) %>%
    mutate(
      model = label,
      lower = pmax(flip_rate - 1.96 * se, 0),
      upper = pmin(flip_rate + 1.96 * se, 1)
    )

  flip_tbl
}

parse_args <- function(args) {
  opts <- list(csqa = list(), gsm8k = list(), out_dir = NULL, file_stub = "fliprate_step_ablation", mode = "zero")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--csqa") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --csqa label=path")
      opts$csqa[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--gsm8k") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --gsm8k label=path")
      opts$gsm8k[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--file_stub") {
      opts$file_stub <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--mode") {
      opts$mode <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (length(opts$csqa) == 0 || length(opts$gsm8k) == 0) {
    stop("Provide at least one --csqa and one --gsm8k input.")
  }
  opts
}

make_panel <- function(tbl, panel_title, model_levels, color_map) {
  tbl <- tbl %>%
    mutate(
      step = factor(step_i, levels = 1:6),
      model = factor(model, levels = model_levels),
      family = ifelse(grepl("^Coconut", model), "Coconut", "CODI"),
      step_x = as.numeric(step) * 0.82
    )

  base <- ggplot(tbl, aes(x = step_x, y = flip_rate, fill = model)) +
    geom_col(position = position_dodge(width = 0.55), width = 0.5, color = "black", linewidth = 0.2)

  base +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = 0.55),
      width = 0.12,
      linewidth = 0.3
    ) +
    scale_fill_manual(values = color_map, drop = FALSE, breaks = model_levels) +
    scale_x_continuous(
      breaks = (1:6) * 0.82,
      labels = 1:6,
      expand = expansion(mult = c(0.08, 0.08))
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(x = "Latent step", y = "Flip rate", title = panel_title, fill = NULL) +
    theme_classic(base_size = 13) +
    theme(
      axis.line = element_line(linewidth = 0.5, color = "black"),
      axis.ticks = element_line(linewidth = 0.4, color = "black"),
      plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.text = element_text(size = 8.5),
      legend.key.width = unit(0.9, "lines"),
      legend.key.height = unit(0.9, "lines"),
      plot.margin = margin(3, 4, 3, 4)
    )
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

model_levels <- c(
  "Coconut-GPT2",
  "Coconut-Llama3-1B",
  "Coconut-Qwen3-4B",
  "CODI-GPT2",
  "CODI-Llama3-1B",
  "CODI-Qwen3-4B"
)

color_map <- c(
  "Coconut-GPT2" = "#c6dbef",
  "Coconut-Llama3-1B" = "#6baed6",
  "Coconut-Qwen3-4B" = "#2171b5",
  "CODI-GPT2" = "#fcbba1",
  "CODI-Llama3-1B" = "#fb6a4a",
  "CODI-Qwen3-4B" = "#cb181d"
)

csqa_tbl <- bind_rows(lapply(names(opts$csqa), function(label) {
  compute_flip_stats(opts$csqa[[label]], label, opts$mode)
}))

gsm8k_tbl <- bind_rows(lapply(names(opts$gsm8k), function(label) {
  compute_flip_stats(opts$gsm8k[[label]], label, opts$mode)
}))

if (!nrow(csqa_tbl) || !nrow(gsm8k_tbl)) {
  stop("No usable rows produced from inputs.")
}

csqa_plot <- make_panel(csqa_tbl, "CommonsenseQA", model_levels, color_map)
gsm8k_plot <- make_panel(gsm8k_tbl, "GSM8K", model_levels, color_map)

combined <- csqa_plot + gsm8k_plot +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", panel.spacing = unit(0.2, "lines")) &
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))

out_dir <- if (is.null(opts$out_dir)) "outputs/rq1/plots" else opts$out_dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

pdf_path <- file.path(out_dir, paste0(opts$file_stub, ".pdf"))
svg_path <- file.path(out_dir, paste0(opts$file_stub, ".svg"))

# Save vector outputs (SVG requires svglite)
ggsave(pdf_path, combined, width = 9.6, height = 3.4, dpi = 300, limitsize = FALSE)
message("Saved: ", pdf_path)

if (requireNamespace("svglite", quietly = TRUE)) {
  ggsave(svg_path, combined, width = 9.6, height = 3.4, dpi = 300, limitsize = FALSE)
  message("Saved: ", svg_path)
} else {
  message("svglite not installed; skipped SVG output.")
}
