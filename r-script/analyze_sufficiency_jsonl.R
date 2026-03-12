#!/usr/bin/env Rscript
# Analyzer for step sufficiency JSONL outputs.
# Reports decode accuracy per step and summarizes logit-lens scores if present.
# Designed for GSM8K-style numeric answers (extracts last number).
#
# Usage:
#   Rscript scripts/rq1-latent/sufficiency/analyze_sufficiency_jsonl.py \
#     --path outputs/step_sufficiency/codi_gpt2.jsonl --dataset_name gsm8k

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
})

parse_args <- function(args) {
  opts <- list(path = NULL, dataset_name = "gsm8k", out_dir = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--path") {
      opts$path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--dataset_name") {
      opts$dataset_name <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$path)) {
    stop("Missing required --path")
  }
  opts
}

extract_number <- function(text) {
  if (is.null(text)) {
    return(NA_real_)
  }
  text <- gsub(",", "", as.character(text))
  m <- gregexpr("-?\\d+\\.?\\d*", text, perl = TRUE)
  nums <- regmatches(text, m)[[1]]
  if (length(nums) == 0) {
    return(NA_real_)
  }
  suppressWarnings(as.numeric(tail(nums, 1)))
}

get_decode_texts <- function(rec) {
  if (is.null(rec$decode)) {
    return(NULL)
  }
  if (is.list(rec$decode)) {
    if (!is.null(rec$decode$text)) {
      return(rec$decode$text)
    }
    return(NULL)
  }
  return(NULL)
}

plot_series <- function(steps, values, title, ylabel, out_path) {
  df <- data.frame(step = steps, value = values)
  p <- ggplot(df, aes(x = step, y = value)) +
    geom_line() +
    geom_point() +
    labs(title = title, x = "step", y = ylabel) +
    theme_minimal(base_size = 12)
  ggsave(out_path, p, width = 8, height = 4, dpi = 200)
  message("Saved plot to ", out_path)
}

plot_series_labels <- function(steps, values, labels, title, ylabel, out_path) {
  df <- data.frame(step = steps, value = values, label = labels)
  p <- ggplot(df, aes(x = step, y = value)) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = steps, labels = labels) +
    labs(title = title, x = "step", y = ylabel) +
    theme_minimal(base_size = 12)
  ggsave(out_path, p, width = 8, height = 4, dpi = 200)
  message("Saved plot to ", out_path)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  use_gsm8k_parse <- identical(opts$dataset_name, "gsm8k")

  decode_stats <- list()
  first_token_logprob <- list()
  teacher_logprob <- list()
  earliest_correct <- list()
  all_samples <- character()

  lines <- readLines(opts$path, warn = FALSE)
  for (line in lines) {
    if (nchar(trimws(line)) == 0) {
      next
    }
    rec <- jsonlite::fromJSON(line, simplifyVector = FALSE)
    step <- rec$step
    if (is.null(step)) {
      next
    }

    gold_val <- if (use_gsm8k_parse) extract_number(rec$gold_answer) else NA_real_
    if (is.na(gold_val)) {
      next
    }

    sid <- rec$sample_id
    sid_key <- if (is.null(sid)) "NA" else as.character(sid)
    all_samples <- c(all_samples, sid_key)

    decode_texts <- get_decode_texts(rec)
    step_is_baseline <- is.character(step) && identical(step, "baseline")
    step_num <- suppressWarnings(as.integer(step))
    if (!step_is_baseline && is.na(step_num)) {
      next
    }
    step_key <- as.character(step_num)
    if (!is.null(decode_texts) && length(decode_texts) > 0) {
      pred_val <- extract_number(decode_texts[[1]])
      if (step_is_baseline) {
        if (is.null(decode_stats[["baseline"]])) {
          decode_stats[["baseline"]] <- list(total = 0, correct = 0)
        }
        decode_stats[["baseline"]]$total <- decode_stats[["baseline"]]$total + 1
        if (!is.na(pred_val) && abs(pred_val - gold_val) < 1e-6) {
          decode_stats[["baseline"]]$correct <- decode_stats[["baseline"]]$correct + 1
        }
      } else {
        if (is.null(decode_stats[[step_key]])) {
          decode_stats[[step_key]] <- list(total = 0, correct = 0)
        }
        decode_stats[[step_key]]$total <- decode_stats[[step_key]]$total + 1
        if (!is.na(pred_val) && abs(pred_val - gold_val) < 1e-6) {
          decode_stats[[step_key]]$correct <- decode_stats[[step_key]]$correct + 1
          prev <- earliest_correct[[sid_key]]
          if (is.null(prev) || step_num < prev) {
            earliest_correct[[sid_key]] <- step_num
          }
        }
      }
    }

    lp_first <- rec$gold_first_token_logprob
    if (!is.null(lp_first)) {
      first_token_logprob[[step_key]] <- c(first_token_logprob[[step_key]], as.numeric(lp_first))
    }

    lp_teacher <- rec$logit_lens_teacher_mean_logprob
    if (!is.null(lp_teacher)) {
      teacher_logprob[[step_key]] <- c(teacher_logprob[[step_key]], as.numeric(lp_teacher))
    }
  }

  step_keys <- names(decode_stats)
  step_keys_num <- suppressWarnings(as.integer(step_keys))
  steps_sorted <- sort(step_keys_num[!is.na(step_keys_num)])

  message("=== Sufficiency Decode Accuracy by Step ===")
  for (s in steps_sorted) {
    bucket <- decode_stats[[as.character(s)]]
    total <- bucket$total
    correct <- bucket$correct
    acc <- if (total > 0) correct / total else 0
    message(sprintf("step=%d: acc=%.3f (correct=%d/%d)", s, acc, correct, total))
  }
  if (!is.null(decode_stats[["baseline"]])) {
    bucket <- decode_stats[["baseline"]]
    total <- bucket$total
    correct <- bucket$correct
    acc <- if (total > 0) correct / total else 0
    message(sprintf("baseline: acc=%.3f (correct=%d/%d)", acc, correct, total))
  }

  if (length(earliest_correct) > 0) {
    vals <- as.numeric(unlist(earliest_correct))
    avg_step <- mean(vals)
    message(sprintf(
      "Earliest correct step mean=%.2f (%d/%d samples solved at some step)",
      avg_step,
      length(earliest_correct),
      length(unique(all_samples))
    ))
  } else {
    message("Earliest correct step: none solved.")
  }

  prefix <- tools::file_path_sans_ext(opts$path)
  if (!is.null(opts$out_dir)) {
    if (!dir.exists(opts$out_dir)) {
      dir.create(opts$out_dir, recursive = TRUE)
    }
    prefix <- file.path(opts$out_dir, basename(prefix))
  }
  if (length(steps_sorted) > 0) {
    acc_values <- sapply(steps_sorted, function(s) {
      bucket <- decode_stats[[as.character(s)]]
      if (bucket$total > 0) bucket$correct / bucket$total else 0
    })
    plot_steps <- steps_sorted
    plot_values <- acc_values
    plot_labels <- as.character(steps_sorted)
    if (!is.null(decode_stats[["baseline"]])) {
      bucket <- decode_stats[["baseline"]]
      baseline_acc <- if (bucket$total > 0) bucket$correct / bucket$total else 0
      baseline_step <- max(steps_sorted) + 1
      plot_steps <- c(plot_steps, baseline_step)
      plot_values <- c(plot_values, baseline_acc)
      plot_labels <- c(plot_labels, "baseline")
    }
    plot_series_labels(
      plot_steps,
      plot_values,
      plot_labels,
      "Sufficiency decode accuracy",
      "accuracy",
      paste0(prefix, "_decode_acc.png")
    )
  }

  if (length(first_token_logprob) > 0) {
    steps_lp <- sort(as.integer(names(first_token_logprob)))
    lp_means <- sapply(steps_lp, function(s) mean(first_token_logprob[[as.character(s)]], na.rm = TRUE))
    plot_series(
      steps_lp,
      lp_means,
      "Logit lens (gold first token logprob)",
      "mean logprob",
      paste0(prefix, "_logit_lens_first_token.png")
    )
  }

  if (length(teacher_logprob) > 0) {
    steps_tf <- sort(as.integer(names(teacher_logprob)))
    lp_means <- sapply(steps_tf, function(s) mean(teacher_logprob[[as.character(s)]], na.rm = TRUE))
    plot_series(
      steps_tf,
      lp_means,
      "Teacher-forced gold mean logprob",
      "mean logprob",
      paste0(prefix, "_logit_lens_teacher.png")
    )
  }

  if (length(all_samples) > 0 && length(steps_sorted) > 0) {
    total <- length(unique(all_samples))
    coverage_vals <- sapply(steps_sorted, function(s) {
      solved <- sum(as.numeric(unlist(earliest_correct)) <= s)
      if (total > 0) solved / total else 0
    })
    plot_series(
      steps_sorted,
      coverage_vals,
      "Cumulative solved fraction by step",
      "fraction solved",
      paste0(prefix, "_earliest_correct.png")
    )
  }
}

main()
