library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)
library(glue)
library(pROC)
library(tibble)
library(scales)
library(patchwork)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Derive class levels from the model file name (e.g. A_vs_B_vs_C.t3.model.txt)
model_file <- list.files(".", pattern = "_vs_.*\\.model\\.txt$", full.names = FALSE)[1]
if (is.na(model_file)) stop("No model file matching *_vs_*.model.txt found in script directory")
run_name <- sub("\\..*", "", model_file)

CLASS_LEVELS <- str_split(run_name, "_vs_", simplify = TRUE)
CLASS_LEVELS <- CLASS_LEVELS[CLASS_LEVELS != ""]
CLASS_LEVELS <- str_replace(CLASS_LEVELS, "_VARIABLE$", "")
if (length(CLASS_LEVELS) < 2) stop(glue("Could not derive >=2 classes from model name: {run_name}"))

comb_path <- "./predictions_P/combined_test_scores.txt"
out_dir   <- "./predictions_P"

##### Helpers #####
canon_label <- function(x) {
  y <- trimws(as.character(x))
  y[y == ""] <- NA_character_
  y
}

detect_truth_column <- function(df) {
  cand <- names(df)
  truth_names <- c("true_label", "label", "class", "target", "y")
  idx <- match(tolower(truth_names), tolower(cand))
  idx <- idx[!is.na(idx)]
  if (length(idx) > 0) return(cand[idx[1]])
  if ("seq_id" %in% names(df)) return("__derive_from_seq_id__")
  NA_character_
}

map_to_class <- function(labels, class_levels) {
  vapply(labels, function(lbl) {
    if (is.na(lbl)) return(NA_character_)
    hit <- which(str_starts(tolower(lbl), tolower(class_levels)))
    if (length(hit) == 0) NA_character_ else class_levels[hit[1]]
  }, character(1), USE.NAMES = FALSE)
}

extract_source_from_seq <- function(seq_id) {
  src <- str_extract(seq_id, "source=[^|]+")
  src <- str_replace(src, "source=", "")
  ifelse(is.na(src), NA_character_, src)
}

detect_score_columns <- function(df, class_levels) {
  nms <- names(df)
  find_all <- function(levels) {
    idx <- match(tolower(levels), tolower(nms))
    if (any(is.na(idx))) return(NULL)
    nms[idx]
  }
  by_name <- find_all(class_levels)
  if (!is.null(by_name)) return(by_name)
  num_cols <- vapply(df, function(col) is.numeric(col) || is.integer(col), logical(1))
  exclude <- tolower(nms) %in% c("true_label", "label", "class", "target", "y", "seq_id")
  num_cols[exclude] <- FALSE
  idx <- which(num_cols)
  if (length(idx) < length(class_levels)) stop("Could not detect enough score columns in combined_test_scores.txt")
  nms[idx[1:length(class_levels)]]
}

compute_pr_df <- function(scores, labels) {
  labels <- as.integer(labels)
  pos_total <- sum(labels == 1L)
  neg_total <- sum(labels == 0L)
  if (pos_total == 0 || neg_total == 0) {
    return(tibble(recall = numeric(0), precision = numeric(0), threshold = numeric(0)))
  }
  ord <- order(scores, decreasing = TRUE)
  lab_ord <- labels[ord]
  scr_ord <- scores[ord]
  cum_tp <- cumsum(lab_ord == 1L)
  cum_fp <- cumsum(lab_ord == 0L)
  recall <- cum_tp / pos_total
  precision <- cum_tp / (cum_tp + cum_fp)
  tibble(recall = recall, precision = precision, threshold = scr_ord)
}

mcc_per_class_from_cm <- function(cm_mat) {
  s <- sum(cm_mat)
  t <- rowSums(cm_mat); p <- colSums(cm_mat)
  d <- diag(cm_mat)
  TP <- d; FP <- p - d; FN <- t - d; TN <- s - TP - FP - FN
  denom <- sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
  mcc <- (TP*TN - FP*FN) / denom
  mcc[denom == 0] <- NA_real_
  stats::setNames(mcc, rownames(cm_mat))
}

multiclass_mcc_from_cm <- function(cm_mat) {
  C <- matrix(as.numeric(cm_mat), nrow = nrow(cm_mat), ncol = ncol(cm_mat), dimnames = dimnames(cm_mat))
  s <- sum(C); c <- sum(diag(C))
  t_k <- rowSums(C); p_k <- colSums(C)
  numerator <- (s * c) - sum(t_k * p_k)
  denom <- sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
  if (!is.finite(denom) || denom == 0) return(NA_real_)
  as.numeric(numerator / denom)
}

##### MAIN SCRIPT #####
message(glue("Run: {run_name}"))
message(glue("Classes: {paste(CLASS_LEVELS, collapse = ', ')}"))
message(glue("Predictions: {comb_path}"))

if (!file.exists(comb_path)) stop(glue("Predictions file not found: {comb_path}"))

df <- read_tsv(comb_path, col_names = FALSE, show_col_types = FALSE)
if (ncol(df) < (1 + length(CLASS_LEVELS))) stop(glue("Not enough columns for {length(CLASS_LEVELS)} classes"))
names(df)[1] <- "seq_id"

truth_col <- detect_truth_column(df)
if (is.na(truth_col)) stop("No true label found and cannot derive from seq_id")
if (truth_col == "__derive_from_seq_id__") {
  df$true_label_raw <- extract_source_from_seq(df$seq_id)
} else {
  df$true_label_raw <- df[[truth_col]]
}
df$true_label <- map_to_class(canon_label(df$true_label_raw), CLASS_LEVELS)
if (any(is.na(df$true_label))) {
  n_na <- sum(is.na(df$true_label))
  warning(glue("{n_na} rows have unknown true_label; dropping them"))
  df <- df |> filter(!is.na(true_label))
}
df$true_label <- factor(df$true_label, levels = CLASS_LEVELS)

score_cols <- detect_score_columns(df, CLASS_LEVELS)
scores_raw <- df[, score_cols]
colnames(scores_raw) <- CLASS_LEVELS
scores_mat <- as.matrix(scores_raw[, CLASS_LEVELS])

pred_idx <- max.col(scores_mat, ties.method = "first")
df$pred_label <- factor(CLASS_LEVELS[pred_idx], levels = CLASS_LEVELS)

cm_tbl <- table(Reference = df$true_label, Prediction = df$pred_label)
cm_tbl <- cm_tbl[CLASS_LEVELS, CLASS_LEVELS, drop = FALSE]
overall_acc <- sum(diag(cm_tbl)) / sum(cm_tbl)
overall_mcc <- multiclass_mcc_from_cm(cm_tbl)

cm_df <- as.data.frame(cm_tbl) |> as_tibble() |> rename(n = Freq)

per_class <- cm_df |>
  group_by(Reference) |>
  summarise(TP = sum(n[Prediction == Reference]), FN = sum(n[Prediction != Reference]), .groups = "drop") |>
  left_join(
    cm_df |> group_by(Prediction) |> summarise(FP = sum(n[Reference != Prediction]), .groups = "drop"),
    by = c("Reference" = "Prediction")
  ) |>
  mutate(
    TN        = sum(cm_df$n) - TP - FP - FN,
    Precision = ifelse(TP + FP > 0, TP / (TP + FP), NA_real_),
    Recall    = ifelse(TP + FN > 0, TP / (TP + FN), NA_real_),
    F1        = ifelse(!is.na(Precision) & !is.na(Recall) & (Precision + Recall) > 0,
                       2 * Precision * Recall / (Precision + Recall), NA_real_)
  ) |>
  rename(Class = Reference)

per_class_mcc <- mcc_per_class_from_cm(cm_tbl)
per_class$MCC <- per_class_mcc[as.character(per_class$Class)]

roc_list <- lapply(CLASS_LEVELS, function(cls) {
  pROC::roc(response = as.integer(df$true_label == cls), predictor = scores_mat[, cls],
            quiet = TRUE, direction = "<")
})
names(roc_list) <- CLASS_LEVELS
auc_per_class <- sapply(roc_list, function(r) as.numeric(r$auc))
mc_auc <- tryCatch(
  as.numeric(pROC::multiclass.roc(df$true_label, scores_mat, levels = CLASS_LEVELS)$auc),
  error = function(e) NA_real_
)

pr_df <- bind_rows(lapply(CLASS_LEVELS, function(cls) {
  d <- compute_pr_df(scores_mat[, cls], as.integer(df$true_label == cls))
  if (nrow(d) == 0) return(NULL)
  d$class <- cls; d
}))
ap_per_class <- sapply(CLASS_LEVELS, function(cls) {
  pr <- compute_pr_df(scores_mat[, cls], as.integer(df$true_label == cls))
  if (nrow(pr) == 0) return(NA_real_)
  sum(diff(c(0, pr$recall)) * pr$precision)
})

# --- CSV outputs ---
write_csv(as.data.frame(cm_tbl),
          file.path(out_dir, glue("{run_name}_multiclass_confusion_matrix.csv")))
write_csv(per_class,
          file.path(out_dir, glue("{run_name}_per_class_metrics.csv")))

summary_df <- tibble(run = run_name, overall_accuracy = overall_acc,
                     multiclass_auc = mc_auc, multiclass_mcc = overall_mcc) |>
  bind_cols(as_tibble_row(setNames(as.list(auc_per_class),  glue("auc_{tolower(CLASS_LEVELS)}")))) |>
  bind_cols(as_tibble_row(setNames(as.list(ap_per_class),   glue("ap_{tolower(CLASS_LEVELS)}")))) |>
  bind_cols(as_tibble_row(setNames(as.list(per_class_mcc[CLASS_LEVELS]), glue("mcc_{tolower(CLASS_LEVELS)}"))))
write_csv(summary_df, file.path(out_dir, glue("{run_name}_summary_overview.csv")))

# --- Plots ---
base_colours  <- c("#00BA38", "#619CFF", "#F8766D", "#B79F00", "#00BFC4", "#F564E3")
class_colours <- setNames(rep(base_colours, length.out = length(CLASS_LEVELS)), CLASS_LEVELS)

CM_PLOT_DF <- as.data.frame(cm_tbl) |> as_tibble() |> rename(n = Freq) |>
  mutate(Reference = factor(Reference, levels = CLASS_LEVELS),
         Prediction = factor(Prediction, levels = CLASS_LEVELS)) |>
  group_by(Reference) |>
  mutate(row_total = sum(n), pct = ifelse(row_total > 0, n / row_total, NA_real_)) |>
  ungroup()

CM_PLOT <- ggplot(CM_PLOT_DF, aes(x = Reference, y = Prediction, fill = pct)) +
  geom_tile(colour = "grey85") +
  geom_text(aes(label = glue("{n}\n{scales::percent(pct, accuracy = 0.1)}")), lineheight = 0.9) +
  scale_fill_gradient(low = "white", high = "#009194", limits = c(0, 1),
                      labels = scales::percent_format(accuracy = 1)) +
  labs(title = glue("Confusion Matrix — Multi-class ({run_name})"),
       x = "Reference class", y = "Prediction", fill = "% of Class") +
  theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank(), panel.border = element_blank(),
        plot.background = element_rect(fill = "white", colour = NA))
ggsave(file.path(out_dir, glue("{run_name}_CM_multiclass.png")), CM_PLOT, width = 12, height = 6, scale = 1.5)

ROC_PLOT <- pROC::ggroc(roc_list, linewidth = 2) +
  labs(title = glue("OvR ROC — Multi-class ({run_name})"),
       x = "False Positive Rate", y = "True Positive Rate") +
  scale_colour_manual(values = class_colours) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white", colour = NA)) +
  geom_text(
    data = data.frame(
      label = glue("{CLASS_LEVELS} (AUC={sprintf('%.3f', auc_per_class)})"),
      x = 0.65,
      y = seq(0.25, 0.25 + 0.05 * (length(CLASS_LEVELS) - 1), by = 0.05)
    ),
    aes(x = x, y = y, label = label), inherit.aes = FALSE, hjust = 0, size = 10
  )
ggsave(file.path(out_dir, glue("{run_name}_ROC_multiclass.png")), ROC_PLOT, width = 12, height = 6, scale = 1.5)

if (nrow(pr_df) > 0) {
  pr_df$class <- factor(
    pr_df$class, levels = CLASS_LEVELS,
    labels = glue("{CLASS_LEVELS} (AP={sprintf('%.3f', ap_per_class)})")
  )
  PR_PLOT <- ggplot(pr_df, aes(x = recall, y = precision,
                                colour = class)) +
    geom_path(linewidth = 2, alpha = 0.9) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = glue("OvR Precision–Recall — Multi-class ({run_name})"),
         x = "Recall", y = "Precision") +
    scale_colour_manual(values = setNames(
      rep(base_colours, length.out = length(CLASS_LEVELS)),
      glue("{CLASS_LEVELS} (AP={sprintf('%.3f', ap_per_class)})")
    )) +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white", colour = NA))
  ggsave(file.path(out_dir, glue("{run_name}_PR_multiclass.png")), PR_PLOT, width = 12, height = 6, scale = 1.5)
}

src <- extract_source_from_seq(df$seq_id)
if (any(!is.na(src))) {
  src_df <- tibble(source = src, correct = as.integer(df$true_label == df$pred_label)) |>
    filter(!is.na(source)) |>
    group_by(source) |>
    summarise(n = n(), accuracy = mean(correct), .groups = "drop") |>
    arrange(desc(accuracy))
  write_csv(src_df, file.path(out_dir, glue("{run_name}_per_source_accuracy.csv")))
  SRC_PLOT <- ggplot(src_df, aes(x = reorder(source, accuracy), y = accuracy)) +
    geom_col(fill = "#009194") + coord_flip() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(title = glue("Per-source Accuracy — {run_name}"), x = "Source", y = "Accuracy") +
    theme_minimal(base_size = 13) +
    theme(plot.background = element_rect(fill = "white", colour = NA))
  ggsave(file.path(out_dir, glue("{run_name}_PerSource_Accuracy.png")), SRC_PLOT, width = 12, height = 6, scale = 1.5)
}

message(glue("Completed evaluation for {run_name}. Outputs written to {out_dir}"))
