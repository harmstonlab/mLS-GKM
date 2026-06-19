#### Supplemental Figures ####
# S1 – gkmpredict: Peak Memory (RSS)
# S2 – Synthetic test motifs: A_0, B_0, C_0
#
# Requires: ggplot2, dplyr, readr, patchwork, universalmotif

library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)
library(universalmotif)

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

#### 1. File Paths ####
PATH_SPEED  <- "./SPEEDUP/ENCODE_DATA/Speed_EVAL/plots/run_level_summary.csv"
PATH_MEME_A <- "./multiclass/synthetic/gkmexplain/BASE_PARAMS/AvsB_P/AvsB_P_patterns.meme"
PATH_MEME_B <- "./multiclass/synthetic/gkmexplain/BASE_PARAMS/BvsC_D/BvsC_D_patterns.meme"
PATH_MEME_C <- "./multiclass/synthetic/gkmexplain/BASE_PARAMS/CvsB_D/CvsB_D_patterns.meme"
OUT_DIR     <- "."
OUT_DPI     <- 600

#### 2. Shared Theme & Colour Palette ####
COL_OLD <- "#4477AA"
COL_NEW <- "#EE6677"
COL_VERSION <- c("LS-GKM" = COL_OLD, "mLS-GKM" = COL_NEW)
COL_DNA <- c(A = "#2CA02C", C = "#1F77B4", G = "#DDAA33", `T` = "#BB5566")

LABEL_OLD <- "LS-GKM"
LABEL_NEW <- "mLS-GKM"

THEME_BASE <- theme_classic(base_size = 10) +
  theme(
    plot.title       = element_text(face = "bold", size = 10),
    axis.title       = element_text(size = 9),
    axis.text        = element_text(size = 8),
    legend.title     = element_text(size = 8),
    legend.text      = element_text(size = 8),
    legend.key.size  = unit(0.4, "cm"),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

#### S1 — gkmpredict: Peak Memory (RSS) ####
df_speed <- read_csv(PATH_SPEED, show_col_types = FALSE)

df_pred <- df_speed |>
  filter(mode == "Predict") |>
  mutate(
    version_label = factor(if_else(version == "LS-GKM", LABEL_OLD, LABEL_NEW),
                           levels = c(LABEL_OLD, LABEL_NEW))
  )

# bar ordering: LS-GKM (T1, T4, T16) then mLS-GKM (T1–T128)
old_configs <- df_pred |> filter(version == "LS-GKM") |>
  distinct(config_label, threads) |> arrange(threads)
new_configs <- df_pred |> filter(version == "mLS-GKM") |>
  distinct(config_label, threads) |> arrange(threads)

bar_order  <- bind_rows(old_configs, new_configs)
n_old_bars <- nrow(old_configs)
label_map  <- setNames(as.character(bar_order$threads), bar_order$config_label)

df_pred_mem <- df_pred |>
  group_by(config_label, version_label) |>
  summarise(
    mean_val = mean(mem_real_peak_mb),
    sd_val   = sd(mem_real_peak_mb),
    .groups  = "drop"
  ) |>
  mutate(config_label = factor(config_label, levels = bar_order$config_label))

sfig1 <- ggplot(df_pred_mem,
                aes(x = config_label, y = mean_val, fill = version_label)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(
    aes(ymin = pmax(mean_val - sd_val, 0), ymax = mean_val + sd_val),
    width = 0.25, linewidth = 0.5
  ) +
  geom_vline(xintercept = n_old_bars + 0.5,
             linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  scale_x_discrete(labels = label_map) +
  scale_fill_manual(values = COL_VERSION, name = "Version") +
  labs(title = "gkmpredict: Peak Memory (RSS)",
       x = "Threads", y = "Peak RSS (MB)") +
  THEME_BASE

ggsave(file.path(OUT_DIR, "sfig_predict_memory.pdf"), sfig1,
       width = 5, height = 4, device = "pdf")
ggsave(file.path(OUT_DIR, "sfig_predict_memory.png"), sfig1,
       width = 5, height = 4, dpi = OUT_DPI, device = "png")
message("Saved: sfig_predict_memory.pdf / .png")

#### S2 — Synthetic Test Motifs ####
logo_colours <- c(A = COL_DNA[["A"]], C = COL_DNA[["C"]],
                  G = COL_DNA[["G"]], `T` = COL_DNA[["T"]])

motifs_A <- read_meme(PATH_MEME_A)
motifs_B <- read_meme(PATH_MEME_B)
motifs_C <- read_meme(PATH_MEME_C)

all_motifs <- c(motifs_A[[1]], motifs_B[[1]], motifs_C[[1]])
all_motifs[[1]]["name"] <- "A_0"
all_motifs[[2]]["name"] <- "B_0"
all_motifs[[3]]["name"] <- "C_0"

sfig2 <- view_motifs(
    trim_motifs(all_motifs, min.ic = 0.1),
    use.type            = "ICM",
    show.positions.once = TRUE,
    show.names          = TRUE,
    tryRC               = FALSE,
    relative_entropy    = TRUE,
    min.overlap         = 50,
    colour.scheme       = logo_colours
  ) +
  labs(title = "Recovered Synthetic Motifs",
       x = "Position", y = "IC (bits)") +
  THEME_BASE +
  theme(legend.position = "none")

ggsave(file.path(OUT_DIR, "sfig_synthetic_motifs.pdf"), sfig2,
       width = 7, height = 9, device = "pdf")
ggsave(file.path(OUT_DIR, "sfig_synthetic_motifs.png"), sfig2,
       width = 7, height = 9, dpi = OUT_DPI, device = "png")
message("Saved: sfig_synthetic_motifs.pdf / .png")
