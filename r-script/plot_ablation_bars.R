#!/usr/bin/env Rscript

# Plot per-step ablation bars (delta logp, delta acc, flip rate) across models.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

read_jsonl_df <- function(path) {
  con <- file(path, open = "r")
  on.exit(close(con), add = TRUE)
  df <- jsonlite::stream_in(con, verbose = FALSE)
  as_tibble(df)
}

to_list_col <- function(x, n_rows) {
  if (is.data.frame(x)) {
    return(split(x, seq_len(nrow(x))))
  }
  if (is.list(x)) {
    return(x)
  }
  return(rep(list(NULL), n_rows))
}

get_text_field <- function(x, idx = 0) {
  if (is.null(x)) return(NA_character_)
  if (is.data.frame(x) && "text" %in% names(x)) {
    txt <- x[["text"]]
  } else if (is.list(x) && "text" %in% names(x)) {
    txt <- x[["text"]]
  } else {
    return(NA_character_)
  }
  if (is.list(txt)) txt <- unlist(txt)
  if (length(txt) == 0) return(NA_character_)
  pos <- idx + 1
  if (length(txt) >= pos && pos >= 1) {
    return(as.character(txt[[pos]]))
  }
  as.character(txt[[1]])
}

extract_answer_one <- function(text) {
  if (is.null(text) || length(text) != 1 || is.na(text)) return(NA_character_)
  s <- as.character(text)
  if (grepl("###", s, fixed = TRUE)) {
    s <- sub("^.*###", "", s)
  }
  if (grepl("<\\|end-latent\\|>", s)) {
    s <- sub("^.*<\\|end-latent\\|>", "", s)
  }
  if (grepl("The answer is:", s, fixed = TRUE)) {
    s <- sub("^.*The answer is:", "", s)
  }
  s <- gsub("\\r|\\n", " ", s)
  s <- trimws(s)
  if (nchar(s) == 0) return(NA_character_)
  tok <- strsplit(s, "\\s+")[[1]][1]
  toupper(tok)
}

extract_answer <- function(text) {
  vapply(text, extract_answer_one, character(1))
}

extract_number_one <- function(text) {
  if (is.null(text) || length(text) != 1 || is.na(text)) return(NA_real_)
  s <- as.character(text)
  s <- gsub(",", "", s)
  nums <- regmatches(s, gregexpr("-?\\d+\\.?\\d*", s))[[1]]
  if (length(nums) == 0) return(NA_real_)
  suppressWarnings(as.numeric(tail(nums, 1)))
}

extract_number <- function(text) {
  vapply(text, extract_number_one, numeric(1))
}

parse_args <- function(args) {
  opts <- list(inputs = list(), mode = "mean_step", logp_kind = "seq", out_dir = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --input label=path")
      opts$inputs[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--mode") {
      opts$mode <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--logp_kind") {
      opts$logp_kind <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (length(opts$inputs) == 0) stop("At least one --input label=path is required.")
  if (!(opts$logp_kind %in% c("seq", "final"))) stop("--logp_kind must be seq or final")
  opts
}

wrap_title <- function(x, width = 42) {
  if (is.null(x) || is.na(x)) return(x)
  paste(strwrap(x, width = width), collapse = "\n")
}

plot_bars <- function(df, title, ylab) {
  if (!nrow(df)) return(invisible(NULL))
  p <- ggplot(df, aes(x = step, y = value, fill = model)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_hline(yintercept = 0, color = "grey60", linewidth = 0.4) +
    labs(
      title = wrap_title(title),
      x = "step",
      y = ylab,
      fill = "model"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.margin = margin(8, 8, 8, 8)
    )
  p
}

tag_plot <- function(label) {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = label, size = 4, fontface = "bold") +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void()
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

all_rows <- list()
for (model in names(opts$inputs)) {
  path <- opts$inputs[[model]]
  df <- read_jsonl_df(path)
  if (!all(c("mode", "step") %in% names(df))) next

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
      gold_ans = extract_answer(gold_raw),
      baseline_ans = extract_answer(baseline_text),
      ablated_ans = extract_answer(ablated_text),
      gold_num = extract_number(gold_raw),
      baseline_num = extract_number(baseline_text),
      ablated_num = extract_number(ablated_text),
      baseline_correct = ifelse(!is.na(gold_num) & !is.na(baseline_num),
                                abs(baseline_num - gold_num) < 1e-6,
                                !is.na(gold_ans) & !is.na(baseline_ans) & gold_ans == baseline_ans),
      ablated_correct = ifelse(!is.na(gold_num) & !is.na(ablated_num),
                               abs(ablated_num - gold_num) < 1e-6,
                               !is.na(gold_ans) & !is.na(ablated_ans) & gold_ans == ablated_ans)
    )

  baseline_tbl <- df %>%
    filter(!is.na(step_i)) %>%
    group_by(step_i) %>%
    summarise(
      total = sum(!is.na(baseline_correct)),
      correct = sum(baseline_correct, na.rm = TRUE),
      baseline_acc = ifelse(total > 0, correct / total, NA_real_),
      .groups = "drop"
    )

  df_mode <- df %>% filter(mode == opts$mode, !is.na(step_i))

  acc_tbl <- df_mode %>%
    group_by(step_i) %>%
    summarise(
      total = sum(!is.na(ablated_correct)),
      correct = sum(ablated_correct, na.rm = TRUE),
      ablated_acc = ifelse(total > 0, correct / total, NA_real_),
      .groups = "drop"
    ) %>%
    left_join(baseline_tbl, by = "step_i") %>%
    mutate(delta_acc = ablated_acc - baseline_acc)

  flip_tbl <- df_mode %>%
    filter(!is.na(baseline_correct), !is.na(ablated_correct)) %>%
    group_by(step_i) %>%
    summarise(
      total = n(),
      correct_to_wrong = sum(baseline_correct & !ablated_correct, na.rm = TRUE),
      wrong_to_correct = sum(!baseline_correct & ablated_correct, na.rm = TRUE),
      flip_rate = ifelse(total > 0, (correct_to_wrong + wrong_to_correct) / total, NA_real_),
      .groups = "drop"
    )

  logp_tbl <- df_mode %>%
    group_by(step_i) %>%
    summarise(
      mean_delta_seq = mean(teacher_forced_delta_sum, na.rm = TRUE),
      mean_delta_final = mean(delta_logp_final_token, na.rm = TRUE),
      .groups = "drop"
    )

  all_rows[[model]] <- list(
    acc = acc_tbl,
    flip = flip_tbl,
    logp = logp_tbl
  )
}

if (length(all_rows) == 0) {
  stop("No usable inputs found.")
}

all_steps <- sort(unique(unlist(lapply(all_rows, function(x) x$acc$step_i))))
step_factor <- factor(all_steps, levels = all_steps)

  acc_plot <- bind_rows(lapply(names(all_rows), function(model) {
    tbl <- all_rows[[model]]$acc
    if (!nrow(tbl)) return(NULL)
    tibble(model = model, step = factor(tbl$step_i, levels = all_steps), value = abs(tbl$delta_acc))
  }))

  flip_plot <- bind_rows(lapply(names(all_rows), function(model) {
    tbl <- all_rows[[model]]$flip
    if (!nrow(tbl)) return(NULL)
    tibble(model = model, step = factor(tbl$step_i, levels = all_steps), value = abs(tbl$flip_rate))
  }))

  logp_plot <- bind_rows(lapply(names(all_rows), function(model) {
    tbl <- all_rows[[model]]$logp
    if (!nrow(tbl)) return(NULL)
    val <- if (opts$logp_kind == "seq") tbl$mean_delta_seq else tbl$mean_delta_final
    tibble(model = model, step = factor(tbl$step_i, levels = all_steps), value = abs(val))
  }))

out_dir <- if (is.null(opts$out_dir)) "outputs/plots/ablation_bars" else opts$out_dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

p_logp <- plot_bars(
  logp_plot,
  "Delta log(p) of the answer by steps",
  "Δ logp (base - ablt)"
)
p_acc <- plot_bars(
  acc_plot,
  "Delta Accuracy by steps",
  "Δ accuracy"
)
p_flip <- plot_bars(
  flip_plot,
  "Flip rate by step",
  "Flip rate"
)

tag_row <- tag_plot("(a)") + tag_plot("(b)") + tag_plot("(c)")

combined <- (p_logp + p_acc + p_flip) / tag_row +
  plot_layout(guides = "collect", heights = c(1, 0.08)) &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.margin = margin(t = 6)
  )
combined <- combined & guides(fill = guide_legend(nrow = 1, byrow = TRUE))

ggsave(
  file.path(out_dir, "ablation_bars_triptych.png"),
  combined,
  width = 19,
  height = 6.2,
  dpi = 300,
  limitsize = FALSE
)
