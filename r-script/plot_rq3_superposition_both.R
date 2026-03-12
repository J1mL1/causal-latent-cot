#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(grid)
  library(gtable)
})

parse_args <- function(args) {
  opts <- list(
    probe_inputs = list(),
    tf_inputs = list(),
    out_path = NULL
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--probe") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --probe label=path")
      opts$probe_inputs[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--tf") {
      val <- args[[i + 1]]
      parts <- strsplit(val, "=", fixed = TRUE)[[1]]
      if (length(parts) != 2) stop("Expected --tf label=path")
      opts$tf_inputs[[parts[[1]]]] <- parts[[2]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (length(opts$probe_inputs) == 0 || length(opts$tf_inputs) == 0) {
    stop("Provide --probe and --tf inputs.")
  }
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
  df$ss <- pmin(norm_a, norm_b)
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

build_plot <- function(input_map, mode, title_label, color_map) {
  rows <- list()
  for (label in names(input_map)) {
    df <- read_jsonl(input_map[[label]])
    if (!nrow(df)) next
    if (mode == "probe") {
      df <- compute_ss_probe(df)
    } else {
      df <- compute_ss_tf(df)
    }
    if (!"step" %in% names(df)) next
    df$step <- as.integer(df$step)
    rows[[length(rows) + 1]] <- summarize_steps(df, label)
  }
  plot_df <- bind_rows(rows)
  plot_df$model <- factor(plot_df$model, levels = names(color_map))

  ggplot(plot_df, aes(x = step, y = mean, color = model, shape = model)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.8) +
    geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem, fill = model), alpha = 0.15, color = NA) +
    scale_color_manual(values = color_map, drop = FALSE) +
    scale_fill_manual(values = color_map, drop = FALSE, guide = "none") +
    scale_shape_manual(
      values = c(
        "Coconut-GPT2" = 17,
        "Coconut-Llama3-1B" = 17,
        "Coconut-Qwen3-4B" = 17,
        "CODI-GPT2" = 16,
        "CODI-Llama3-1B" = 16,
        "CODI-Qwen3-4B" = 16
      ),
      drop = FALSE
    ) +
    scale_x_continuous(breaks = sort(unique(plot_df$step))) +
    labs(x = "step", y = "SS", color = NULL, shape = NULL, title = title_label) +
    theme_classic(base_size = 9) +
    theme(
      axis.line = element_line(linewidth = 0.6, color = "black"),
      axis.ticks = element_line(linewidth = 0.5, color = "black"),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 9),
      plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
      legend.position = "bottom",
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

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  color_map <- c(
    "Coconut-GPT2" = "#c6dbef",
    "Coconut-Llama3-1B" = "#6baed6",
    "Coconut-Qwen3-4B" = "#2171b5",
    "CODI-GPT2" = "#fcbba1",
    "CODI-Llama3-1B" = "#fb6a4a",
    "CODI-Qwen3-4B" = "#cb181d"
  )

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
        guides(color = guide_legend(nrow = legend_nrow, byrow = TRUE),
               shape = guide_legend(nrow = legend_nrow, byrow = TRUE)))
    }
    main <- p_left + p_right + plot_layout(guides = "keep")
    legend_row <- wrap_elements(legend) + theme(plot.margin = margin(0, 0, 0, 0))
    main / legend_row + plot_layout(heights = c(1, 0.28))
  }

  p_probe_legend_src <- build_plot(opts$probe_inputs, "probe", "metric: probe", color_map) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE),
           shape = guide_legend(nrow = 2, byrow = TRUE))
  p_probe <- p_probe_legend_src + theme(legend.position = "none")
  p_tf <- build_plot(opts$tf_inputs, "teacher", "metric: teacher-forced logp", color_map) +
    theme(legend.position = "none")
  p <- assemble_with_legend(p_probe, p_tf, p_probe_legend_src, legend_nrow = 2)

  out_dir <- dirname(opts$out_path)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(opts$out_path, p, width = 6.25, height = 2.2, dpi = 300, bg = "white")
}

main()
