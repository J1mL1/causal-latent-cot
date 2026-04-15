#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})

read_jsonl_df <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  df <- jsonlite::stream_in(textConnection(lines), verbose = FALSE)
  as_tibble(df)
}

to_step_int <- function(x) suppressWarnings(as.integer(as.character(x)))

ensure_cols <- function(df, cols) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(paste("Missing required columns:", paste(missing, collapse = ", ")))
  }
}

build_step_levels <- function(df) {
  # Union numeric steps from both axes: step_i alone misses the last latent (e.g. edges (5,6)
  # have step_i=5, step_j=6; with --no-include_self there may be no row with step_i=6).
  si <- df %>% mutate(v = to_step_int(step_i)) %>% filter(!is.na(v)) %>% distinct(v) %>% pull(v)
  sj <- df %>% mutate(v = to_step_int(step_j)) %>% filter(!is.na(v)) %>% distinct(v) %>% pull(v)
  steps <- sort(unique(c(si, sj)))
  as.character(steps)
}
