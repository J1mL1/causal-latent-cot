#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
})

read_jsonl_df <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  df <- jsonlite::stream_in(textConnection(lines), verbose = FALSE)
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

extract_number <- function(text) {
  if (is.null(text)) return(NA_real_)
  s <- as.character(text)
  if (grepl("<\\|end-latent\\|>", s)) {
    s <- sub("^.*<\\|end-latent\\|>", "", s)
  }
  s <- gsub(",", "", s)
  nums <- regmatches(s, gregexpr("-?\\d+\\.?\\d*", s))[[1]]
  if (length(nums) == 0) return(NA_real_)
  suppressWarnings(as.numeric(tail(nums, 1)))
}

extract_choice <- function(text) {
  if (is.null(text)) return(NA_character_)
  s <- as.character(text)
  if (grepl("<\\|end-latent\\|>", s)) {
    s <- sub("^.*<\\|end-latent\\|>", "", s)
  }
  matches <- regmatches(s, gregexpr("\\b[A-E]\\b", s))
  if (length(matches) && length(matches[[1]]) > 0) {
    return(tail(matches[[1]], 1))
  }
  NA_character_
}

parse_gold <- function(gold_answer) {
  if (is.null(gold_answer)) return(NA)
  g <- as.character(gold_answer)
  if (grepl("^[A-E]$", g)) {
    return(g)
  }
  extract_number(g)
}
