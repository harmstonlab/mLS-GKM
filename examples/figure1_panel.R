#### Figure 1: lsgkm Performance & Speed Benchmarking — 2x3 Panel Plot ####
# Panels:
#   A  – ROC AUC correlation (original vs. modified gkmpredict)
#   B  – Run time: gkmpredict (bar + SD error bars, original-first layout)
#   C  – Run time: gkmexplain (bar + SD error bars, original-first layout)
#   D  – Peak RSS memory: gkmexplain (bar + SD error bars)
#   E  – One-vs-Rest ROC curves: 3-class regulatory element prediction
#   F  – ICM sequence logo: enhancer_vs_promoter motif
#
# Requires: ggplot2, dplyr, tidyr, readr, pROC, patchwork, stringr, scales,
#           TFBSTools, universalmotif

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(pROC)
library(patchwork)
library(stringr)
library(scales)
library(TFBSTools)
library(universalmotif)

#### 1. File Paths ####
PATH_ROC_AUC    <- "./SPEEDUP/ENCODE_Data/Predictions/metrics_summary/summary_roc_auc.csv"
PATH_SPEED      <- "./SPEEDUP/ENCODE_Data/Speed_EVAL/plots/run_level_summary.csv"
PATH_MULTICLASS <- "./multiclass/regulatory_features/predictions_P/combined_test_scores.txt"
PATH_MEME       <- "./multiclass/regulatory_features/gkmexplain/BASE_PARAMS/enhancer_vs_promoter_D/enhancer_vs_promoter_D_patterns.meme"

# Output
OUT_STEM   <- "./figure1_panel"
OUT_WIDTH  <- 14     # inches
OUT_HEIGHT <- 9      # inches
OUT_DPI    <- 600    # dpi

#### 2. Plot Titles & Labels ####
TITLE_A <- "gkmpredict: ROC AUC Correlation"
TITLE_B <- "gkmpredict: Run Time"
TITLE_C <- "gkmexplain: Run Time"
TITLE_D <- "gkmexplain: Peak Memory (RSS)"
TITLE_E <- "Multi-class ROC (OvR): Regulatory Elements"
TITLE_F <- "Enhancer Recovered Motif (FOS::JUN)"

MOTIF_NAME <- "enhancer_vs_promoter_D_pattern_0"
LABEL_OLD  <- "LS-GKM"
LABEL_NEW  <- "mLS-GKM"

CLASS_LEVELS_E <- c("enhancer", "promoter", "CTCF_Binding_Site")

#### 3. Shared Theme & Colour Palette ####
COL_OLD <- "#4477AA"   # LS-GKM
COL_NEW <- "#EE6677"   # mLS-GKM
COL_VERSION <- c("LS-GKM" = COL_OLD, "mLS-GKM" = COL_NEW)

COL_CLASS <- c(
  "enhancer"          = "#228B22",
  "promoter"          = "#CC3311",
  "CTCF_Binding_Site" = "#0077BB"
)

COL_DNA <- c(A = "#2CA02C", C = "#1F77B4", G = "#DDAA33", `T` = "#BB5566")

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

#### 4. Helper Functions ####

map_to_class <- function(labels, class_levels) {
  vapply(labels, function(lbl) {
    if (is.na(lbl)) return(NA_character_)
    hit <- which(str_starts(tolower(lbl), tolower(class_levels)))
    if (length(hit) == 0) NA_character_ else class_levels[hit[1]]
  }, character(1), USE.NAMES = FALSE)
}

make_speed_summary <- function(df, mode_filter, value_col) {
  df_mode <- df |>
    filter(mode == mode_filter) |>
    mutate(
      threads       = if_else(version == "LS-GKM" & threads == 0L, NA_integer_, threads),  # threads=0 means serial
      version_label = factor(if_else(version == "LS-GKM", LABEL_OLD, LABEL_NEW),
                             levels = c(LABEL_OLD, LABEL_NEW))
    ) |>
    mutate(
      config_key    = case_when(
        version == "LS-GKM" & is.na(threads) ~ "LS-GKM_serial",
        version == "LS-GKM"                  ~ paste0("LS-GKM_", threads, "T"),
        TRUE                                 ~ paste0("mLS-GKM_", threads, "T")
      ),
      display_label = case_when(
        version == "LS-GKM" & is.na(threads) ~ "1",
        TRUE                              ~ as.character(threads)
      )
    )

  old_order <- df_mode |>
    filter(version == "LS-GKM") |>
    distinct(config_key, display_label, threads) |>
    arrange(threads)

  new_order <- df_mode |>
    filter(version == "mLS-GKM") |>
    distinct(config_key, display_label, threads) |>
    arrange(threads)

  bar_order  <- bind_rows(old_order, new_order)
  n_old_bars <- nrow(old_order)

  label_map <- setNames(bar_order$display_label, bar_order$config_key)

  df_sum <- df_mode |>
    group_by(config_key, display_label, version_label) |>
    summarise(mean_val = mean(.data[[value_col]]),
              sd_val   = sd(.data[[value_col]]),
              .groups  = "drop") |>
    mutate(config_key = factor(config_key, levels = bar_order$config_key))

  list(data = df_sum, n_old_bars = n_old_bars, label_map = label_map)
}

#### Panel A — ROC AUC Correlation ####
df_auc <- read_csv(PATH_ROC_AUC, show_col_types = FALSE)
colnames(df_auc) <- c("experiment", "auc_old", "auc_new")

auc_range <- range(c(df_auc$auc_old, df_auc$auc_new))
auc_lo    <- floor(auc_range[1] * 20) / 20
auc_hi    <- ceiling(auc_range[2] * 20) / 20
r_pearson <- cor(df_auc$auc_old, df_auc$auc_new, method = "pearson")

panel_A <- ggplot(df_auc, aes(x = auc_old, y = auc_new)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(size = 1.5, alpha = 0.7, colour = COL_OLD) +
  annotate("text",
           x = auc_lo + 0.005, y = auc_hi - 0.005,
           label = sprintf("r = %.4f", r_pearson),
           hjust = 0, vjust = 1, size = 3.5) +
  scale_x_continuous(limits = c(auc_lo, auc_hi), breaks = pretty_breaks(5)) +
  scale_y_continuous(limits = c(auc_lo, auc_hi), breaks = pretty_breaks(5)) +
  coord_fixed() +
  labs(title = TITLE_A,
       x     = paste(LABEL_OLD, "ROC AUC"),
       y     = paste(LABEL_NEW, "ROC AUC")) +
  THEME_BASE

#### Panels B, C, D — Speed & Memory ####
df_speed <- read_csv(PATH_SPEED, show_col_types = FALSE)

add_version_separator <- function(p, n_old_bars) {
  p + geom_vline(xintercept = n_old_bars + 0.5,
                 linetype = "dashed", colour = "grey50", linewidth = 0.5)
}

# --- Panel B: gkmpredict run time ---
pred_res  <- make_speed_summary(df_speed, "Predict", "elapsed_seconds")
df_pred   <- pred_res$data

panel_B <- ggplot(df_pred, aes(x = config_key, y = mean_val, fill = version_label)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = pmax(mean_val - sd_val, 0), ymax = mean_val + sd_val),
                width = 0.25, linewidth = 0.5) +
  scale_x_discrete(labels = pred_res$label_map) +
  scale_fill_manual(values = COL_VERSION, name = "Version") +
  labs(title = TITLE_B, x = "Threads", y = "Elapsed Time (s)") +
  THEME_BASE
panel_B <- add_version_separator(panel_B, pred_res$n_old_bars)

# --- Panel C: gkmexplain run time ---
expl_res  <- make_speed_summary(df_speed, "Explain", "elapsed_seconds")
df_expl   <- expl_res$data

panel_C <- ggplot(df_expl, aes(x = config_key, y = mean_val, fill = version_label)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = pmax(mean_val - sd_val, 0), ymax = mean_val + sd_val),
                width = 0.25, linewidth = 0.5) +
  scale_x_discrete(labels = expl_res$label_map) +
  scale_y_continuous(limits = c(0, 30000)) +
  scale_fill_manual(values = COL_VERSION, name = "Version") +
  labs(title = TITLE_C, x = "Threads", y = "Elapsed Time (s)") +
  THEME_BASE
panel_C <- add_version_separator(panel_C, expl_res$n_old_bars)

# --- Panel D: gkmexplain peak RSS memory (bar + SD) ---
mem_res <- make_speed_summary(df_speed, "Explain", "mem_real_peak_mb")
df_mem  <- mem_res$data

panel_D <- ggplot(df_mem, aes(x = config_key, y = mean_val, fill = version_label)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = pmax(mean_val - sd_val, 0), ymax = mean_val + sd_val),
                width = 0.25, linewidth = 0.5) +
  scale_x_discrete(labels = mem_res$label_map) +
  scale_y_continuous(limits = c(0, 1500)) +
  scale_fill_manual(values = COL_VERSION, name = "Version") +
  labs(title = TITLE_D, x = "Threads", y = "Peak RSS (MB)") +
  THEME_BASE
panel_D <- add_version_separator(panel_D, mem_res$n_old_bars)

#### Panel E — Multi-class One-vs-Rest ROC ####
df_mc <- read_tsv(PATH_MULTICLASS, col_names = FALSE, show_col_types = FALSE)
colnames(df_mc) <- c("seq_id", paste0("prob_", CLASS_LEVELS_E))

df_mc <- df_mc |>
  mutate(
    true_label_raw = str_extract(seq_id, "(?<=source=)[^|\\s]+"),
    true_label     = map_to_class(true_label_raw, CLASS_LEVELS_E)
  ) |>
  filter(!is.na(true_label))

scores_mat           <- as.matrix(df_mc[, paste0("prob_", CLASS_LEVELS_E)])
colnames(scores_mat) <- CLASS_LEVELS_E

roc_list <- lapply(CLASS_LEVELS_E, function(cls) {
  pROC::roc(response  = as.integer(df_mc$true_label == cls),
            predictor = scores_mat[, cls],
            quiet = TRUE, direction = "<")
})
names(roc_list) <- CLASS_LEVELS_E
auc_vals <- sapply(roc_list, function(r) as.numeric(r$auc))

roc_df <- bind_rows(lapply(CLASS_LEVELS_E, function(cls) {
  r <- roc_list[[cls]]
  tibble(class = cls, fpr = 1 - r$specificities, tpr = r$sensitivities)
})) |>
  mutate(class = factor(class, levels = CLASS_LEVELS_E))

class_legend_labels <- setNames(
  sprintf("%s  (AUC = %.3f)", CLASS_LEVELS_E, auc_vals),
  CLASS_LEVELS_E
)

panel_E <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = class)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey60", linewidth = 0.5) +
  scale_colour_manual(values = COL_CLASS, labels = class_legend_labels, name = NULL) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(title = TITLE_E, x = "False Positive Rate", y = "True Positive Rate") +
  THEME_BASE +
  theme(
    legend.position   = c(0.62, 0.16),
    legend.background = element_rect(fill = "white", colour = "grey80", linewidth = 0.3)
  )

#### Panel F — ICM Sequence Logo ####
meme_motifs  <- read_meme(PATH_MEME)
motif_names  <- sapply(meme_motifs, function(m) m["name"])
motif_idx    <- which(motif_names == MOTIF_NAME)
if (length(motif_idx) == 0)
  stop(sprintf("Motif '%s' not found in %s", MOTIF_NAME, PATH_MEME))
target_motif <- meme_motifs[[motif_idx[1]]]

logo_colours <- c(A = COL_DNA[["A"]], C = COL_DNA[["C"]],
                  G = COL_DNA[["G"]], T = COL_DNA[["T"]])

panel_F <- view_motifs(
    trim_motifs(target_motif, min.ic = 0.1),
    use.type            = "ICM",
    show.positions.once = TRUE,
    show.names          = FALSE,
    tryRC               = TRUE,
    min.overlap         = 50,
    relative_entropy    = TRUE,
    colour.scheme       = logo_colours
  ) +
  labs(title = TITLE_F, x = "Position", y = "Information Content (bits)") +
  THEME_BASE +
  theme(legend.position = "none")

#### Assemble 2x3 Panel ####
panel_plot <- (panel_A | panel_B | panel_C) /
              (panel_D | panel_E | panel_F) +
  plot_annotation(
    tag_levels = "A",
    theme      = theme(plot.background = element_rect(fill = "white", colour = NA))
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))
panel_plot
ggsave(paste0(OUT_STEM, ".pdf"), panel_plot,
       width = OUT_WIDTH, height = OUT_HEIGHT,
       device = "pdf")

ggsave(paste0(OUT_STEM, ".png"), panel_plot,
       width = OUT_WIDTH, height = OUT_HEIGHT,
       dpi = OUT_DPI, device = "png")

message("Saved: ", OUT_STEM, ".pdf / .png")
