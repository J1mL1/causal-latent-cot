#!/usr/bin/env Rscript

# Latent causal graph for outputs with kl_mean + delta_logp_final_token.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggforce)
  library(scales)
  library(grid)
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

metric_display_label <- function(metric) {
  if (metric %in% c("grad_logprob", "grad_margin")) {
    return(gsub("_", " ", metric))
  }
  if (metric %in% c("kl_mean", "kl_logit_ht")) {
    return(gsub("_", " ", metric))
  }
  metric
}

format_edge_label <- function(x) {
  ifelse(is.na(x), "",
         ifelse(abs(x) < 1e-3,
                formatC(x, format = "e", digits = 1),
                sprintf("%.2f", x)))
}

format_legend_label <- function(x) {
  ifelse(is.na(x), "",
         ifelse(abs(x) < 1e-3,
                formatC(x, format = "e", digits = 1),
                sprintf("%.2f", x)))
}

detect_dataset_label <- function(prefix, input_path, df = NULL) {
  if (!is.null(df) && "dataset_name" %in% names(df)) {
    ds <- df$dataset_name[!is.na(df$dataset_name)]
    if (length(ds) > 0) {
      return(unique(ds)[1])
    }
  }
  haystack <- paste(prefix, basename(input_path), sep = " ")
  if (grepl("gsm8k", haystack, ignore.case = TRUE)) return("gsm8k")
  if (grepl("commonsenseqa", haystack, ignore.case = TRUE)) return("commonsenseqa")
  if (grepl("csqa", haystack, ignore.case = TRUE)) return("commonsenseqa")
  if (grepl("math", haystack, ignore.case = TRUE)) return("math")
  if (grepl("svamp", haystack, ignore.case = TRUE)) return("svamp")
  if (grepl("aqua", haystack, ignore.case = TRUE)) return("aqua")
  NULL
}

build_latent_aggregates <- function(df, metric, answer_metric) {
  ensure_cols(df, c("step_i", "step_j", metric, answer_metric))

  df <- df %>%
    mutate(step_i = as.character(step_i),
           step_j = as.character(step_j))

  steps <- build_step_levels(df)

  kl_tbl <- df %>%
    group_by(step_i, step_j) %>%
    summarise(kl = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")

  dlogp_tbl <- df %>%
    filter(step_i == step_j) %>%
    group_by(step_i) %>%
    summarise(
      dlogp = mean(.data[[answer_metric]], na.rm = TRUE),
      count = sum(is.finite(.data[[answer_metric]])),
      .groups = "drop"
    )

  list(steps = steps, kl_tbl = kl_tbl, dlogp_tbl = dlogp_tbl)
}

kl_threshold_forward <- function(kl_tbl, pct = 70) {
  v <- kl_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    pull(kl)
  v <- v[is.finite(v)]
  if (length(v) == 0) return(Inf)
  as.numeric(quantile(v, probs = pct / 100, names = FALSE, type = 7))
}

max_ratio_threshold_forward <- function(kl_tbl, ratio = 0.9) {
  v <- kl_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    pull(kl)
  v <- v[is.finite(v)]
  if (length(v) == 0) return(Inf)
  max(v) * ratio
}

topk_outgoing_forward <- function(kl_tbl, topk = 2) {
  kl_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    group_by(step_i) %>%
    arrange(desc(kl), .by_group = TRUE) %>%
    slice_head(n = topk) %>%
    mutate(top_rank = row_number()) %>%
    ungroup()
}

topk_incoming_forward <- function(kl_tbl, topk = 2) {
  kl_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    group_by(step_j) %>%
    arrange(desc(kl), .by_group = TRUE) %>%
    slice_head(n = topk) %>%
    mutate(top_rank = row_number()) %>%
    ungroup()
}

plot_kl_heatmap_no_ans <- function(steps, kl_tbl, out_png, metric_label = "kl_mean") {
  mat <- kl_tbl %>%
    mutate(step_i = factor(step_i, levels = steps),
           step_j = factor(step_j, levels = steps)) %>%
    complete(step_i, step_j, fill = list(kl = NA_real_)) %>%
    arrange(step_i, step_j)

  p <- ggplot(mat, aes(x = step_j, y = step_i, fill = kl)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(is.na(kl), "", sprintf("%.2f", kl))), size = 3) +
    scale_fill_viridis_c(option = "C", na.value = "grey95") +
    coord_equal() +
    labs(
      x = "step_j",
      y = "step_i",
      fill = metric_display_label(metric_label),
      title = paste0(metric_display_label(metric_label), " adjacency heatmap (no Ans; diagonal is ", metric_display_label(metric_label), ")")
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank())

  ggsave(out_png, p, width = 7, height = 6, dpi = 300)
}

plot_latent_graph <- function(steps, kl_tbl, out_png, pct = 70, topk = 2,
                              metric_label = "", title_label = NULL, topk_direction = "out",
                              threshold_mode = "percentile", max_ratio = 0.9) {
  # Always aggregate first to avoid threshold/top-k bias from pre-filtered rows.
  kl_tbl <- kl_tbl %>%
    group_by(step_i, step_j) %>%
    summarise(kl = mean(kl, na.rm = TRUE), .groups = "drop")
  thr <- if (threshold_mode == "max_ratio") {
    max_ratio_threshold_forward(kl_tbl, max_ratio)
  } else {
    kl_threshold_forward(kl_tbl, pct)
  }
  n_steps <- length(steps)

  kl_forward <- kl_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i)

  kl_thresh <- kl_forward %>%
    filter(kl >= thr)

  kl_keep <- if (topk > 0) {
    topk_tbl <- if (topk_direction == "in") {
        topk_incoming_forward(kl_thresh, topk)
      } else {
        topk_outgoing_forward(kl_thresh, topk)
      } %>%
      select(step_i, step_j, kl, top_rank)
    kl_thresh %>%
      left_join(topk_tbl %>% select(step_i, step_j, top_rank),
                by = c("step_i", "step_j")) %>%
      filter(!is.na(top_rank)) %>%
      mutate(etype = "kl", weight = kl) %>%
      select(from = step_i, to = step_j, etype, weight, top_rank)
  } else {
    kl_thresh %>%
      mutate(top_rank = NA_integer_, etype = "kl", weight = kl) %>%
      select(from = step_i, to = step_j, etype, weight, top_rank)
  }

  edges <- kl_keep %>%
    mutate(
      w_abs = abs(weight),
      from_i = to_step_int(from),
      to_i = to_step_int(to),
      span = pmax(to_i - from_i, 1),
      color_group = if (threshold_mode == "max_ratio") {
        paste0("Edge type -> ", metric_display_label(metric_label), "(thr=max*", max_ratio, ")")
      } else {
        paste0("Edge type -> ", metric_display_label(metric_label), "(thr=", pct / 100, ")")
      }
    )

  node_spacing <- 0.5
  nodes <- tibble(
    name = steps,
    idx = seq_along(steps),
    x = seq_along(steps) * node_spacing,
    y = 0.0
  )

  e <- edges %>%
    left_join(nodes %>% select(name, x, y) %>% rename(from = name, x0 = x, y0 = y), by = "from") %>%
    left_join(nodes %>% select(name, x, y) %>% rename(to = name, x2 = x, y2 = y), by = "to") %>%
    mutate(
      mx = (x0 + x2) / 2, my = (y0 + y2) / 2,
      dx = x2 - x0, dy = y2 - y0,
      L = sqrt(dx * dx + dy * dy) + 1e-9,
      nx = -dy / L, ny = dx / L,
      bend = (0.75 + 0.38 * pmax(span - 1, 0)) * ifelse(from_i %% 2 == 1, 1, -1),
      cx = mx + nx * bend,
      cy = my + ny * bend
    )

  e <- e %>%
    group_by(etype) %>%
    mutate(w_cap = pmin(w_abs, as.numeric(quantile(w_abs, 0.90, na.rm = TRUE)))) %>%
    ungroup() %>%
    group_by(etype) %>%
    mutate(edge_width = rescale(w_cap, to = c(0.55, 1.55))) %>%
    ungroup()

  bezier_pts <- bind_rows(
    e %>% transmute(edge_id = row_number(), edge_width, etype, from_i, weight,
                    x = x0, y = y0, t = 1L),
    e %>% transmute(edge_id = row_number(), edge_width, etype, from_i, weight,
                    x = cx, y = cy, t = 2L),
    e %>% transmute(edge_id = row_number(), edge_width, etype, from_i, weight,
                    x = x2, y = y2, t = 3L)
  )

  p <- ggplot() +
    ggforce::geom_bezier(
      data = bezier_pts,
      aes(x = x, y = y, group = edge_id, colour = weight, size = edge_width),
      arrow = grid::arrow(type = "closed", length = unit(3.0, "mm"), angle = 15),
      lineend = "round",
      alpha = 0.88,
      show.legend = TRUE
    ) +
    geom_point(
      data = nodes,
      aes(x = x, y = y),
      shape = 21, size = 9.4, stroke = 1.5,
      fill = "#f7f7f7", colour = "#222",
      show.legend = FALSE
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = name),
      fontface = "bold", size = 7.2, show.legend = FALSE
    ) +
    scale_colour_gradient(
      low = "#9ecae1",
      high = "#1f4e8c",
      name = metric_display_label(metric_label),
      breaks = scales::breaks_extended(n = 4),
      labels = format_legend_label
    ) +
    guides(
      colour = guide_colourbar(
        barheight = unit(12, "mm"),
        barwidth = unit(82, "mm"),
        title.position = "top",
        title.hjust = 0.5,
        label.position = "bottom",
        label.hjust = 0.5
      ),
      size = "none"
    ) +
    coord_cartesian(
      xlim = c(0.3, max(nodes$x) + 0.7),
      ylim = c(-0.85, 1.45),
      expand = FALSE
    ) +
    theme_void(base_size = 14) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.title = element_text(face = "bold", size = 22, margin = margin(b = 6), hjust = 0.5),
      plot.margin = margin(6, 6, 4, 6),
      legend.title = element_text(size = 26),
      legend.text = element_text(size = 20),
      legend.box.spacing = unit(0.05, "lines"),
      legend.margin = margin(0, 0, 0, 0),
      legend.justification = c(0.47, 0.5),
      legend.box.just = "center",
      legend.title.align = 0.5
    ) +
    ggtitle(ifelse(is.null(title_label) || title_label == "",
                   "",
                   title_label))

  if (is.null(title_label) || title_label == "") {
    p <- p + theme(plot.title = element_blank()) + labs(title = NULL)
  }

  ggsave(out_png, p, width = 12.8, height = 4.2, dpi = 320, bg = "white")
}

run_all <- function(jsonl_path, prefix, metric, answer_metric, pct = 70, topk = 2,
                    out_dir = NULL, max_steps = 6, title_label = NULL, topk_direction = "out",
                    threshold_mode = "percentile", max_ratio = 0.9) {
  df <- read_jsonl_df(jsonl_path)
  missing_cols <- setdiff(c("step_i", "step_j", metric, answer_metric), names(df))
  if (length(missing_cols) > 0) {
    warning(paste("Missing required columns:", paste(missing_cols, collapse = ", "), "- skipping metric:", metric))
    return(invisible(NULL))
  }
  agg <- build_latent_aggregates(df, metric, answer_metric)
  steps <- agg$steps
  if (!is.null(max_steps)) {
    steps <- steps[seq_len(min(length(steps), max_steps))]
  }
  agg$kl_tbl <- agg$kl_tbl %>%
    filter(step_i %in% steps, step_j %in% steps)
  agg$dlogp_tbl <- agg$dlogp_tbl %>%
    filter(step_i %in% steps)

  if (is.null(out_dir)) {
    out_dir <- dirname(jsonl_path)
  }
  data_dir <- file.path(out_dir, "data", metric)
  plot_dir <- file.path(out_dir, "plot_img", metric)
  heat_dir <- file.path(out_dir, "heatmap", metric)
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, recursive = TRUE)
  }
  if (!dir.exists(plot_dir)) {
    dir.create(plot_dir, recursive = TRUE)
  }
  if (!dir.exists(heat_dir)) {
    dir.create(heat_dir, recursive = TRUE)
  }

  heat_png <- file.path(heat_dir, paste0(prefix, "_kl_heatmap_noans.png"))
  graph_png <- file.path(plot_dir, paste0(prefix, "_causal_graph_top", topk, ".png"))
  csv_path <- file.path(data_dir, paste0(prefix, "_answer.csv"))

  plot_kl_heatmap_no_ans(steps, agg$kl_tbl, heat_png, metric_label = metric)
  final_title <- if (!is.null(title_label) && title_label != "") title_label else ""
  plot_latent_graph(
    steps,
    agg$kl_tbl,
    graph_png,
    pct = pct,
    topk = topk,
    metric_label = metric,
    title_label = final_title,
    topk_direction = topk_direction,
    threshold_mode = threshold_mode,
    max_ratio = max_ratio
  )
  csv_tbl <- agg$dlogp_tbl %>%
    mutate(metric = answer_metric) %>%
    rename(mean_value = dlogp) %>%
    select(step_i, metric, mean_value, count)
  write.csv(csv_tbl, csv_path, row.names = FALSE)

  message("Saved: ", heat_png)
  message("Saved: ", graph_png)
  message("Saved: ", csv_path)
}

parse_args <- function(args) {
  opts <- list(input = NULL, prefix = NULL, pct = 70, topk = 2, out_dir = NULL,
               metric = "kl_mean", answer_metric = "delta_logp_final_token", max_steps = 6,
               title_label = NULL, topk_direction = NULL, threshold_mode = "percentile",
               max_ratio = 0.9)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      opts$input <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--prefix") {
      opts$prefix <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--pct") {
      opts$pct <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--topk") {
      opts$topk <- as.integer(args[[i + 1]])
      i <- i + 2
    } else if (key == "--metric") {
      opts$metric <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--answer_metric") {
      opts$answer_metric <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--max_steps") {
      opts$max_steps <- as.integer(args[[i + 1]])
      i <- i + 2
    } else if (key == "--out_dir") {
      opts$out_dir <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--title_label") {
      opts$title_label <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--threshold_mode") {
      opts$threshold_mode <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--max_ratio") {
      opts$max_ratio <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--topk_direction") {
      # Reserved for future use; accept to avoid hard failure.
      opts$topk_direction <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }
  if (is.null(opts$input)) {
    stop("Missing required --input")
  }
  if (is.null(opts$prefix)) {
    opts$prefix <- tools::file_path_sans_ext(basename(opts$input))
  }
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)
run_all(
  opts$input,
  opts$prefix,
  metric = opts$metric,
  answer_metric = opts$answer_metric,
  pct = opts$pct,
  topk = opts$topk,
  out_dir = opts$out_dir,
  max_steps = opts$max_steps,
  title_label = opts$title_label,
  topk_direction = ifelse(is.null(opts$topk_direction), "out", opts$topk_direction),
  threshold_mode = opts$threshold_mode,
  max_ratio = opts$max_ratio
)
