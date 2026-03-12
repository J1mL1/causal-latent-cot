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
  steps <- df %>%
    mutate(step_i_int = to_step_int(step_i)) %>%
    filter(!is.na(step_i_int)) %>%
    distinct(step_i_int) %>%
    arrange(step_i_int) %>%
    pull(step_i_int)
  as.character(steps)
}
