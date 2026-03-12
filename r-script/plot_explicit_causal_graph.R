#!/usr/bin/env Rscript

# Explicit causal graph
# - Heatmap: explicit delta_logp metric for i -> j (no Ans)
# - Graph: step edges (i<j) thresholded by quantile/topk; step -> Ans edges unthresholded
# - Labels placed on arcs, with white background

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

build_explicit_aggregates <- function(df, metric, answer_metric) {
  ensure_cols(df, c("step_i", "step_j", metric, answer_metric))

  df <- df %>%
    mutate(step_i = as.character(step_i),
           step_j = as.character(step_j))

  steps <- build_step_levels(df)

  step_tbl <- df %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(!is.na(i), !is.na(j)) %>%
    group_by(step_i, step_j) %>%
    summarise(weight = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")

  ans_tbl <- df %>%
    filter(step_j == "Y") %>%
    group_by(step_i) %>%
    summarise(
      weight = mean(.data[[answer_metric]], na.rm = TRUE),
      count = sum(is.finite(.data[[answer_metric]])),
      .groups = "drop"
    )

  list(steps = steps, step_tbl = step_tbl, ans_tbl = ans_tbl)
}

edge_threshold_forward <- function(step_tbl, pct = 70) {
  v <- step_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    pull(weight)
  v <- v[is.finite(v)]
  if (length(v) == 0) return(Inf)
  as.numeric(quantile(v, probs = pct / 100, names = FALSE, type = 7))
}

max_ratio_threshold_forward <- function(step_tbl, ratio = 0.9) {
  v <- step_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    pull(weight)
  v <- v[is.finite(v)]
  if (length(v) == 0) return(Inf)
  max(v) * ratio
}

topk_outgoing_forward <- function(step_tbl, topk = 2) {
  step_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i) %>%
    group_by(step_i) %>%
    arrange(desc(weight), .by_group = TRUE) %>%
    slice_head(n = topk) %>%
    mutate(top_rank = row_number()) %>%
    ungroup()
}

plot_step_heatmap_no_ans <- function(steps, step_tbl, metric, out_png) {
  mat <- step_tbl %>%
    mutate(step_i = factor(step_i, levels = steps),
           step_j = factor(step_j, levels = steps)) %>%
    complete(step_i, step_j, fill = list(weight = NA_real_)) %>%
    arrange(step_i, step_j)

  p <- ggplot(mat, aes(x = step_j, y = step_i, fill = weight)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(is.na(weight), "", sprintf("%.2f", weight))), size = 3) +
    scale_fill_viridis_c(option = "C", na.value = "grey95") +
    coord_equal() +
    labs(x = "step_j", y = "step_i", fill = metric,
         title = paste0("Explicit adjacency heatmap: ", metric)) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank())

  ggsave(out_png, p, width = 7, height = 6, dpi = 300)
}

plot_explicit_graph <- function(steps, step_tbl, metric, out_png, pct = 70, topk = 2, title_label = NULL,
                                threshold_mode = "percentile", max_ratio = 0.9) {
  # Always aggregate first to avoid threshold/top-k bias from pre-filtered rows.
  step_tbl <- step_tbl %>%
    group_by(step_i, step_j) %>%
    summarise(weight = mean(weight, na.rm = TRUE), .groups = "drop")
  thr <- if (threshold_mode == "max_ratio") {
    max_ratio_threshold_forward(step_tbl, max_ratio)
  } else {
    edge_threshold_forward(step_tbl, pct)
  }
  n_steps <- length(steps)

  step_forward <- step_tbl %>%
    mutate(i = to_step_int(step_i), j = to_step_int(step_j)) %>%
    filter(j > i)

  step_thresh <- step_forward %>%
    filter(weight >= thr)

  step_keep <- if (topk > 0) {
    topk_tbl <- topk_outgoing_forward(step_thresh, topk) %>%
      select(step_i, step_j, weight, top_rank)
    step_thresh %>%
      left_join(topk_tbl %>% select(step_i, step_j, top_rank),
                by = c("step_i", "step_j")) %>%
      filter(!is.na(top_rank)) %>%
      mutate(etype = "step") %>%
      select(from = step_i, to = step_j, etype, weight, top_rank)
  } else {
    step_thresh %>%
      mutate(top_rank = NA_integer_, etype = "step") %>%
      select(from = step_i, to = step_j, etype, weight, top_rank)
  }

  edges <- step_keep %>%
    mutate(
      w_abs = abs(weight),
      from_i = to_step_int(from),
      to_i = to_step_int(to),
      span = pmax(to_i - from_i, 1),
      color_group = if (threshold_mode == "max_ratio") {
        paste0("Edge type -> KL(thr=max*", max_ratio, ")")
      } else {
        paste0("Edge type -> KL(thr=", pct / 100, ")")
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
      name = metric,
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
      xlim = c(0.3, max(nodes$x) + 0.6),
      ylim = c(-1.45, 1.45),
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
      legend.box.spacing = unit(0.02, "lines"),
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
                    out_dir = NULL, max_steps = 6, title_label = NULL,
                    threshold_mode = "percentile", max_ratio = 0.9) {
  df <- read_jsonl_df(jsonl_path)
  agg <- build_explicit_aggregates(df, metric, answer_metric)
  steps <- agg$steps
  if (!is.null(max_steps)) {
    steps <- steps[seq_len(min(length(steps), max_steps))]
  }
  agg$step_tbl <- agg$step_tbl %>%
    filter(step_i %in% steps, step_j %in% steps)
  agg$ans_tbl <- agg$ans_tbl %>%
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

  heat_png <- file.path(heat_dir, paste0(prefix, "_explicit_heatmap.png"))
  graph_png <- file.path(plot_dir, paste0(prefix, "_explicit_graph.png"))
  csv_path <- file.path(data_dir, paste0(prefix, "_explicit_answer.csv"))

  plot_step_heatmap_no_ans(steps, agg$step_tbl, metric, heat_png)
  final_title <- if (!is.null(title_label) && title_label != "") title_label else ""
  plot_explicit_graph(
    steps,
    agg$step_tbl,
    metric,
    graph_png,
    pct = pct,
    topk = topk,
    title_label = final_title,
    threshold_mode = threshold_mode,
    max_ratio = max_ratio
  )
  write.csv(
    agg$ans_tbl %>% mutate(metric = answer_metric) %>% rename(mean_value = weight) %>% select(step_i, metric, mean_value, count),
    csv_path,
    row.names = FALSE
  )

  message("Saved: ", heat_png)
  message("Saved: ", graph_png)
  message("Saved: ", csv_path)
}

parse_args <- function(args) {
  opts <- list(input = NULL, prefix = NULL, pct = 70, topk = 2, out_dir = NULL,
               metric = "delta_logp_seq", answer_metric = "delta_logp_last",
               max_steps = 6, title_label = NULL,
               threshold_mode = "percentile", max_ratio = 0.9)
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
  threshold_mode = opts$threshold_mode,
  max_ratio = opts$max_ratio
)
