#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(patchwork)
})

parse_args <- function(args) {
  opts <- list(
    latent_csv = NULL,
    explicit_csv = NULL,
    out_path = NULL,
    dataset = "gsm8k",
    metric = "kl_mean"
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--latent_csv") {
      opts$latent_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--explicit_csv") {
      opts$explicit_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_path") {
      opts$out_path <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--dataset") {
      opts$dataset <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--metric") {
      opts$metric <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  opts
}

map_latent <- function(df) {
  df %>%
    mutate(
      method = case_when(
        grepl("^coconut_", model) ~ "coconut",
        grepl("^codi_", model) ~ "codi",
        TRUE ~ NA_character_
      ),
      backbone = case_when(
        grepl("gpt2", model) ~ "gpt2",
        grepl("llama", model) ~ "llama1b",
        grepl("qwen3_4b", model) ~ "qwen3-4b",
        TRUE ~ NA_character_
      )
    )
}

map_explicit <- function(df) {
  df %>%
    mutate(
      method = "explicit",
      backbone = case_when(
        grepl("gpt2", model) ~ "gpt2",
        grepl("llama1b", model) ~ "llama1b",
        grepl("qwen3_4b", model) ~ "qwen3-4b",
        TRUE ~ NA_character_
      )
    )
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.null(opts$latent_csv) || is.null(opts$explicit_csv) || is.null(opts$out_path)) {
    stop("--latent_csv, --explicit_csv, --out_path are required.")
  }

  latent <- read.csv(opts$latent_csv)
  explicit <- read.csv(opts$explicit_csv)

  latent <- latent %>%
    filter(dataset == opts$dataset, metric == opts$metric, mode == "zero") %>%
    map_latent() %>%
    filter(!is.na(method), !is.na(backbone))

  explicit <- explicit %>%
    filter(dataset == opts$dataset, metric == opts$metric, mode == "zero") %>%
    map_explicit() %>%
    filter(!is.na(method), !is.na(backbone))

  combined <- bind_rows(latent, explicit)

  metrics_long <- combined %>%
    select(method, backbone, locality, span, early_out, late_in) %>%
    pivot_longer(cols = c(locality, span, early_out, late_in),
                 names_to = "struct_metric",
                 values_to = "value")

  plot_tbl <- metrics_long %>%
    mutate(
      method = factor(method, levels = c("coconut", "codi", "explicit")),
      backbone = factor(backbone, levels = c("gpt2", "llama1b", "qwen3-4b")),
      model_label = factor(
        paste(method, gsub("-", "_", backbone), sep = "_"),
        levels = c(
          "coconut_gpt2", "coconut_llama1b", "coconut_qwen3_4b",
          "codi_gpt2", "codi_llama1b", "codi_qwen3_4b",
          "explicit_gpt2", "explicit_llama1b", "explicit_qwen3_4b"
        )
      ),
      method_label = factor(
        case_when(
          method == "coconut" ~ "Coconut",
          method == "codi" ~ "CODI",
          method == "explicit" ~ "CoT",
          TRUE ~ NA_character_
        ),
        levels = c("Coconut", "CODI", "CoT")
      ),
      struct_metric = factor(
        struct_metric,
        levels = c("locality", "span", "early_out", "late_in"),
        labels = c("(a) locality", "(b) span", "(c) early-out", "(d) late-in")
      )
    ) %>%
    filter(!is.na(value), !is.na(model_label))

  # De-duplicate in case multiple jsonl files map to the same model/backbone.
  plot_tbl <- plot_tbl %>%
    group_by(model_label, struct_metric, method_label) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

  label_map <- c(
    "coconut_gpt2" = "Coconut-GPT2",
    "coconut_llama1b" = "Coconut-Llama3-1B",
    "coconut_qwen3_4b" = "Coconut-Qwen3-4B",
    "codi_gpt2" = "CODI-GPT2",
    "codi_llama1b" = "CODI-Llama3-1B",
    "codi_qwen3_4b" = "CODI-Qwen3-4B",
    "explicit_gpt2" = "CoT-GPT2",
    "explicit_llama1b" = "CoT-Llama3-1B",
    "explicit_qwen3_4b" = "CoT-Qwen3-4B"
  )

  color_map <- c(
    "coconut_gpt2" = "#9ecae1",
    "coconut_llama1b" = "#6baed6",
    "coconut_qwen3_4b" = "#2171b5",
    "codi_gpt2" = "#fcbba1",
    "codi_llama1b" = "#fb6a4a",
    "codi_qwen3_4b" = "#cb181d",
    "explicit_gpt2" = "#fee391",
    "explicit_llama1b" = "#fec44f",
    "explicit_qwen3_4b" = "#d95f0e"
  )

  base_plot <- function(df) {
    ggplot(df, aes(x = method_label, y = value, fill = model_label)) +
      geom_col(
        width = 0.7,
        color = "black",
        linewidth = 0.1,
        position = position_dodge2(width = 0.82, preserve = "single")
      ) +
      scale_fill_manual(
        values = color_map,
        labels = label_map,
        drop = FALSE,
        breaks = names(label_map)
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
      facet_wrap(~struct_metric, ncol = 4, scales = "free_y", strip.position = "bottom") +
      labs(x = NULL, y = NULL, fill = NULL) +
      theme_classic(base_size = 7) +
      theme(
        axis.line = element_line(linewidth = 0.25, color = "black"),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(size = 6),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, "lines"),
        legend.key.height = unit(0.3, "lines"),
        legend.key.width = unit(0.45, "lines"),
        legend.margin = margin(6, 0, 0, 0),
        legend.box.margin = margin(0, 0, 0, 0),
        legend.spacing.y = unit(4, "pt"),
        legend.box.spacing = unit(0, "pt"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 7.5),
        strip.placement = "outside",
        panel.spacing = unit(10, "pt"),
        aspect.ratio = 0.45,
        plot.margin = margin(0, 0, 0, 0)
    )
  }

  p <- base_plot(plot_tbl)
  p <- p + guides(fill = guide_legend(nrow = 1, byrow = TRUE)) + theme(legend.position = "bottom")

  out_dir <- dirname(opts$out_path)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  ggsave(opts$out_path, p, width = 8.6, height = 2.2, dpi = 300, bg = "white")
}

main()
