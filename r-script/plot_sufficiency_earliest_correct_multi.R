#!/usr/bin/env Rscript

# Plot earliest-correct metrics for multiple sufficiency JSONL inputs.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(grid)
  library(gtable)
})

parse_args <- function(args) {
  opts <- list(csqa = list(), gsm8k = list(), out_dir = NULL, max_steps = NULL)
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
    } else if (key == "--max_steps") {
      opts$max_steps <- as.integer(args[[i + 1]])
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

extract_number <- function(text) {
  if (is.null(text)) return(NA_real_)
  text <- gsub(",", "", as.character(text))
  m <- gregexpr("-?\\d+\\.?\\d*", text, perl = TRUE)
  nums <- regmatches(text, m)[[1]]
  if (length(nums) == 0) return(NA_real_)
  suppressWarnings(as.numeric(tail(nums, 1)))
}

extract_choice <- function(text) {
  if (is.null(text)) return(NA_character_)
  text <- as.character(text)
  m <- regexpr("###\\s*([A-E])\\b", text, perl = TRUE)
  if (m[1] != -1) {
    hit <- regmatches(text, m)
    return(toupper(sub(".*([A-E]).*", "\\1", hit)))
  }
  m <- gregexpr("\\b[A-E]\\b", text, perl = TRUE)
  choices <- regmatches(text, m)[[1]]
  if (length(choices) == 0) return(NA_character_)
  toupper(tail(choices, 1))
}

get_decode_texts <- function(rec) {
  if (is.null(rec$decode)) return(NULL)
  if (is.list(rec$decode) && !is.null(rec$decode$text)) return(rec$decode$text)
  NULL
}

read_records <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  lapply(lines, function(line) jsonlite::fromJSON(line, simplifyVector = FALSE))
}

compute_earliest <- function(records, dataset_name, max_steps) {
  use_gsm8k_parse <- identical(dataset_name, "gsm8k")
  earliest_correct <- list()
  all_samples <- character()
  steps_seen <- integer()

  for (rec in records) {
    step <- rec$step
    if (is.null(step)) next
    step_is_baseline <- is.character(step) && identical(step, "baseline")
    step_num <- suppressWarnings(as.integer(step))
    if (step_is_baseline || is.na(step_num)) next
    if (!is.null(max_steps) && step_num > max_steps) next

    if (use_gsm8k_parse) {
      gold_val <- extract_number(rec$gold_answer)
      if (is.na(gold_val)) next
    } else {
      gold_val <- rec$gold_answer
      if (is.null(gold_val)) next
      gold_val <- toupper(trimws(as.character(gold_val)))
      if (!nzchar(gold_val)) next
    }

    sid <- rec$sample_id
    sid_key <- if (is.null(sid)) "NA" else as.character(sid)
    all_samples <- c(all_samples, sid_key)
    steps_seen <- c(steps_seen, step_num)

    decode_texts <- get_decode_texts(rec)
    if (!is.null(decode_texts) && length(decode_texts) > 0) {
      if (use_gsm8k_parse) {
        pred_val <- extract_number(decode_texts[[1]])
        is_correct <- !is.na(pred_val) && abs(pred_val - gold_val) < 1e-6
      } else {
        pred_val <- extract_choice(decode_texts[[1]])
        is_correct <- !is.na(pred_val) && identical(pred_val, gold_val)
      }
      if (is_correct) {
        prev <- earliest_correct[[sid_key]]
        if (is.null(prev) || step_num < prev) {
          earliest_correct[[sid_key]] <- step_num
        }
      }
    }
  }

  list(
    earliest = earliest_correct,
    total_samples = length(unique(all_samples)),
    steps = sort(unique(steps_seen))
  )
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  out_dir <- if (is.null(opts$out_dir)) "outputs/plots/sufficiency" else opts$out_dir
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

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

  build_tables <- function(input_list, dataset_name) {
    model_lines <- list()
    model_stats <- list()
    for (raw_label in names(input_list)) {
      path <- input_list[[raw_label]]
      label <- raw_label
      step_offset <- 0
      m <- regexec("^(.*?)([+-]\\d+)$", raw_label)
      parts <- regmatches(raw_label, m)[[1]]
      if (length(parts) == 3) {
        label <- parts[2]
        step_offset <- as.integer(parts[3])
      }
      records <- read_records(path)
      res <- compute_earliest(records, dataset_name, opts$max_steps)
      if (step_offset != 0) {
        res$steps <- res$steps + step_offset
        res$earliest <- lapply(res$earliest, function(v) v + step_offset)
      }
      if (length(res$steps) == 0 || res$total_samples == 0) next

      vals <- as.numeric(unlist(res$earliest))
      mean_step <- if (length(vals) > 0) mean(vals) else NA_real_
      sd_step <- if (length(vals) > 1) sd(vals) else NA_real_
      sem_step <- if (length(vals) > 1) sd_step / sqrt(length(vals)) else NA_real_
      model_stats[[label]] <- data.frame(
        model = label,
        mean_step = mean_step,
        sd_step = sd_step,
        sem_step = sem_step,
        solved = length(vals),
        total = res$total_samples
      )

      for (s in res$steps) {
        solved <- sum(vals <= s)
        frac <- if (res$total_samples > 0) solved / res$total_samples else 0
        model_lines[[length(model_lines) + 1]] <- data.frame(
          model = label,
          step = s,
          fraction = frac
        )
      }
    }
    list(
      line = bind_rows(model_lines),
      stats = bind_rows(model_stats)
    )
  }

  csqa_tbls <- build_tables(opts$csqa, "commonsenseqa")
  gsm8k_tbls <- build_tables(opts$gsm8k, "gsm8k")

  if (!nrow(csqa_tbls$line) || !nrow(gsm8k_tbls$line)) {
    stop("No usable inputs found.")
  }

  style_line <- function(df, panel_title, show_legend = TRUE) {
    df$model <- factor(df$model, levels = model_levels)
    ggplot(df, aes(x = step, y = fraction, color = model, shape = model)) +
      geom_line(linewidth = 1.1) +
      geom_point(size = 2.8) +
      scale_color_manual(values = color_map, drop = FALSE, breaks = model_levels) +
      scale_shape_manual(
        values = c(
          "Coconut-GPT2" = 17,
          "Coconut-Llama3-1B" = 17,
          "Coconut-Qwen3-4B" = 17,
          "CODI-GPT2" = 16,
          "CODI-Llama3-1B" = 16,
          "CODI-Qwen3-4B" = 16
        ),
        drop = FALSE,
        breaks = model_levels
      ) +
      scale_x_continuous(breaks = sort(unique(df$step))) +
      labs(title = panel_title, x = "step", y = "fraction solved", color = NULL, shape = NULL) +
      theme_classic(base_size = 9) +
      theme(
        axis.line = element_line(linewidth = 0.6, color = "black"),
        axis.ticks = element_line(linewidth = 0.5, color = "black"),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 9),
        plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
        legend.position = if (show_legend) "bottom" else "none",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        legend.text = element_text(size = 7),
        legend.key.height = unit(0.3, "cm"),
        legend.key.width = unit(0.75, "cm"),
        legend.spacing.x = unit(0.2, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.margin = margin(0, 0, 0, 0),
        plot.margin = margin(6, 8, 6, 8)
      )
  }

  extract_legend <- function(p) {
    g <- ggplotGrob(p)
    idx <- which(sapply(g$grobs, function(x) x$name) == "guide-box")
    if (length(idx) == 0) return(NULL)
    g$grobs[[idx[1]]]
  }

  assemble_with_legend <- function(p_left, p_right, legend_plot, legend_nrow = 2) {
    legend <- extract_legend(legend_plot)
    if (is.null(legend)) {
      return(p_left + p_right + plot_layout(guides = "collect") &
        theme(legend.position = "bottom") &
        guides(color = guide_legend(nrow = legend_nrow, byrow = TRUE)))
    }
    main <- p_left + p_right + plot_layout(guides = "keep")
    legend_row <- wrap_elements(legend) + theme(plot.margin = margin(0, 0, 0, 0))
    main / legend_row + plot_layout(heights = c(1, 0.28))
  }

  style_bar <- function(df, panel_title, show_legend = TRUE) {
    df$model <- factor(df$model, levels = model_levels)
    df <- df %>% arrange(mean_step)
    ggplot(df, aes(x = model, y = mean_step, fill = model)) +
      geom_col(width = 0.7, color = "black", linewidth = 0.2) +
      geom_errorbar(
        aes(ymin = mean_step - sem_step, ymax = mean_step + sem_step),
        width = 0.2,
        linewidth = 0.6
      ) +
      scale_fill_manual(values = color_map, drop = FALSE, breaks = model_levels) +
      labs(title = panel_title, x = NULL, y = "mean earliest step", fill = NULL) +
      theme_classic(base_size = 9) +
      theme(
        axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        axis.title = element_text(size = 9),
        axis.line = element_line(linewidth = 0.6, color = "black"),
        axis.ticks = element_line(linewidth = 0.5, color = "black"),
        plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
        legend.position = if (show_legend) "bottom" else "none",
        legend.text = element_text(size = 7),
        legend.key.height = unit(0.3, "cm"),
        legend.key.width = unit(0.75, "cm"),
        plot.margin = margin(6, 8, 6, 8)
      )
  }

  p_line_legend_src <- style_line(csqa_tbls$line, "CommonsenseQA", show_legend = TRUE) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
  p_line_csqa <- p_line_legend_src + theme(legend.position = "none")
  p_line_gsm8k <- style_line(gsm8k_tbls$line, "GSM8K", show_legend = FALSE)
  p_line <- assemble_with_legend(p_line_csqa, p_line_gsm8k, p_line_legend_src, legend_nrow = 2)
  p_line <- p_line + plot_annotation(title = NULL)
  ggsave(file.path(out_dir, "earliest_correct_by_step.png"), p_line, width = 6.25, height = 2.2, dpi = 300)
  ggsave(file.path(out_dir, "earliest_correct_by_step.pdf"), p_line, width = 6.25, height = 2.2)

  p_bar_legend_src <- style_bar(csqa_tbls$stats, "CommonsenseQA", show_legend = TRUE) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE))
  p_bar_csqa <- p_bar_legend_src + theme(legend.position = "none")
  p_bar_gsm8k <- style_bar(gsm8k_tbls$stats, "GSM8K", show_legend = FALSE)
  p_bar <- assemble_with_legend(p_bar_csqa, p_bar_gsm8k, p_bar_legend_src, legend_nrow = 2)
  p_bar <- p_bar + plot_annotation(title = NULL)
  ggsave(file.path(out_dir, "mean_earliest_correct_bar.png"), p_bar, width = 6.25, height = 2.2, dpi = 300)
  ggsave(file.path(out_dir, "mean_earliest_correct_bar.pdf"), p_bar, width = 6.25, height = 2.2)
}

main()
