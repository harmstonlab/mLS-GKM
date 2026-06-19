#!/usr/bin/env Rscript
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)
library(purrr)
library(tidyr)
library(forcats)

# Set working directory to the script's directory.
# If running inside RStudio, use the rstudioapi; otherwise infer from commandArgs.
set_script_wd <- function() {
    script_dir <- NULL
    if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
        path <- rstudioapi::getActiveDocumentContext()$path
        if (nzchar(path)) {
            script_dir <- dirname(path)
        }
    } else {
        args <- commandArgs(trailingOnly = FALSE)
        file_arg <- "--file="
        matches <- grep(file_arg, args)
        if (length(matches) > 0) {
            path <- sub("^--file=", "", args[matches][1])
            script_dir <- dirname(normalizePath(path))
        } else {
            # Fallback: use current working directory
            script_dir <- getwd()
        }
    }
    if (!is.null(script_dir) && dir.exists(script_dir)) {
        setwd(script_dir)
        invisible(script_dir)
    } else {
        invisible(NULL)
    }
}
set_script_wd()

# Set these paths before running in an interactive R session.
base_dir <- "./Speed_EVAL"
out_dir <- file.path(base_dir, "plots")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

resource_files <- list.files(
    path = base_dir,
    pattern = "resource_usage\\.csv$",
    recursive = TRUE,
    full.names = TRUE
)

if (length(resource_files) == 0) {
    stop(sprintf("No resource_usage.csv files found under: %s", base_dir))
}

parse_file_meta <- function(path, base_dir) {
    rel <- gsub("\\\\", "/", sub(paste0("^", normalizePath(base_dir), "/?"), "", normalizePath(path)))

    top_match <- str_match(rel, "^(Predict|Explain)_(mLS-GKM|LS-GKM)/")
    mode <- top_match[, 2]
    version <- top_match[, 3]

    thread_match <- str_match(rel, "THREADS_([0-9]+)")
    rpt_match <- str_match(rel, "RPT_([0-9]+)")

    threads <- suppressWarnings(as.integer(thread_match[, 2]))
    rpt <- suppressWarnings(as.integer(rpt_match[, 2]))

    if (is.na(threads) && !is.na(mode) && mode == "Explain" && version == "LS-GKM") {
        threads <- 0L
    }

    config_label <- case_when(
        mode == "Explain" & version == "LS-GKM" ~ "LS-GKM_no_threads",
        TRUE ~ paste0(version, "_T", threads)
    )

    tibble(
        path = path,
        rel = rel,
        mode = mode,
        version = version,
        threads = threads,
        rpt = rpt,
        config_label = config_label
    )
}

summarise_run <- function(path, mode, version, threads, rpt, config_label, rel) {
    dat <- read_csv(path, show_col_types = FALSE)

    required_cols <- c("elapsed_time", "cpu", "mem_real", "mem_virtual")
    missing_cols <- setdiff(required_cols, colnames(dat))

    if (length(missing_cols) > 0) {
        stop(sprintf("Missing columns in %s: %s", path, paste(missing_cols, collapse = ", ")))
    }

    tibble(
        path = path,
        rel = rel,
        mode = mode,
        version = version,
        threads = threads,
        rpt = rpt,
        config_label = config_label,
        elapsed_seconds = max(dat$elapsed_time, na.rm = TRUE),
        cpu_mean = mean(dat$cpu, na.rm = TRUE),
        cpu_peak = max(dat$cpu, na.rm = TRUE),
        mem_real_peak_mb = max(dat$mem_real, na.rm = TRUE),
        mem_virtual_peak_mb = max(dat$mem_virtual, na.rm = TRUE)
    )
}

file_meta <- map_dfr(resource_files, parse_file_meta, base_dir = base_dir) %>%
    filter(!is.na(mode), !is.na(version), !is.na(rpt), !is.na(threads))

if (nrow(file_meta) == 0) {
    stop("No valid files matched Predict/Explain LS-GKM/mLS-GKM layout.")
}

run_summary <- pmap_dfr(file_meta, summarise_run)

metric_info <- tibble(
    metric = c("elapsed_seconds", "cpu_peak", "mem_real_peak_mb", "mem_virtual_peak_mb"),
    y_label = c("Elapsed Time (s)", "Peak CPU (%)", "Peak RSS Memory (MB)", "Peak Virtual Memory (MB)")
)

get_config_order <- function(df_mode) {
    df_mode %>%
        distinct(mode, config_label, version, threads) %>%
        mutate(
            sort_threads = if_else(mode == "Explain" & version == "LS-GKM", -1L, threads),
            sort_version = if_else(version == "LS-GKM", 0L, 1L)
        ) %>%
        arrange(sort_threads, sort_version) %>%
        pull(config_label)
}

plot_mode_metric <- function(df_mode, mode_name, metric_name, y_label, out_dir) {
    cfg_order <- get_config_order(df_mode)

    df_mode <- df_mode %>%
        mutate(config_label = factor(config_label, levels = cfg_order))

    avg_df <- df_mode %>%
        group_by(mode, version, threads, config_label) %>%
        summarise(value = mean(.data[[metric_name]], na.rm = TRUE), .groups = "drop")

    p_bar <- ggplot(avg_df, aes(x = config_label, y = value)) +
        geom_col(fill = "#4c78a8", colour = "black", linewidth = 0.2) +
        labs(
            title = paste(mode_name, "LS-GKM vs mLS-GKM -", metric_name, "(Average of 5 repeats)"),
            x = "Configuration",
            y = y_label
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(face = "bold")
        )

    p_box <- ggplot(df_mode, aes(x = config_label, y = .data[[metric_name]])) +
        geom_boxplot(fill = "#4c78a8", alpha = 0.9, outlier.shape = 21, outlier.stroke = 0.2) +
        labs(
            title = paste(mode_name, "LS-GKM vs mLS-GKM -", metric_name, "(Distribution across repeats)"),
            x = "Configuration",
            y = y_label
        ) +
        theme_minimal(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(face = "bold")
        )

    out_stub <- paste0(tolower(mode_name), "_", metric_name)

    ggsave(file.path(out_dir, paste0(out_stub, "_bar_avg.png")), p_bar, width = 10, height = 6, dpi = 300)
    ggsave(file.path(out_dir, paste0(out_stub, "_boxplot.png")), p_box, width = 10, height = 6, dpi = 300)
}

for (i in seq_len(nrow(metric_info))) {
    metric_name <- metric_info$metric[i]
    y_label <- metric_info$y_label[i]

    for (mode_name in c("Predict", "Explain")) {
        df_mode <- run_summary %>% filter(mode == mode_name)
        if (nrow(df_mode) == 0) {
            next
        }
        plot_mode_metric(df_mode, mode_name, metric_name, y_label, out_dir)
    }
}

write_csv(run_summary, file.path(out_dir, "run_level_summary.csv"))

group_average_summary <- run_summary %>%
    group_by(mode, version, threads, config_label) %>%
    summarise(
        elapsed_seconds_mean = mean(elapsed_seconds, na.rm = TRUE),
        elapsed_seconds_sd = sd(elapsed_seconds, na.rm = TRUE),
        cpu_peak_mean = mean(cpu_peak, na.rm = TRUE),
        mem_real_peak_mb_mean = mean(mem_real_peak_mb, na.rm = TRUE),
        mem_virtual_peak_mb_mean = mean(mem_virtual_peak_mb, na.rm = TRUE),
        n_runs = n(),
        .groups = "drop"
    ) %>%
    arrange(mode, threads, version)

write_csv(group_average_summary, file.path(out_dir, "group_average_summary.csv"))

message("Done.")
message(sprintf("Input directory: %s", normalizePath(base_dir)))
message(sprintf("Output directory: %s", normalizePath(out_dir)))
message("Generated:")
message(" - run_level_summary.csv")
message(" - group_average_summary.csv")
message(" - bar and boxplot PNGs for each metric in Predict and Explain")
