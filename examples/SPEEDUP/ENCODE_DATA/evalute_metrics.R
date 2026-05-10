library(dplyr)
library(readr)
library(tibble)
library(pROC)
library(ggplot2)

PRED_ROOT <- "./Predictions"
MLS_GKM_DIR <- file.path(PRED_ROOT, "mLS-GKM_Models")
LS_GKM_DIR <- file.path(PRED_ROOT, "LS-GKM_Models")
BIGBED_LIST <- "./bigbed_list.tsv"

OUT_DIR <- file.path(PRED_ROOT, "metrics_summary")
dir.create(OUT_DIR, showWarnings = TRUE, recursive = TRUE)


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

compute_binary_metrics <- function(pred_path) {
    if (!file.exists(pred_path)) {
        return(tibble(
            roc_auc = NA_real_,
            pr_auc = NA_real_,
            mcc = NA_real_,
            accuracy = NA_real_,
            precision = NA_real_,
            recall = NA_real_,
            f1 = NA_real_
        ))
    }

    df <- read_tsv(pred_path, col_names = c("seq_id", "score"), show_col_types = FALSE)
    label <- ifelse(startsWith(df$seq_id, "pos_"), 1L,
        ifelse(startsWith(df$seq_id, "neg_"), 0L, NA_integer_)
    )
    keep <- !is.na(label) & !is.na(df$score)
    df <- df[keep, , drop = FALSE]
    label <- label[keep]

    pos_total <- sum(label == 1L)
    neg_total <- sum(label == 0L)
    if (pos_total == 0 || neg_total == 0) {
        return(tibble(
            roc_auc = NA_real_,
            pr_auc = NA_real_,
            mcc = NA_real_,
            accuracy = NA_real_,
            precision = NA_real_,
            recall = NA_real_,
            f1 = NA_real_
        ))
    }

    score <- df$score
    pred_label <- ifelse(score > 0, 1L, 0L)

    tp <- sum(pred_label == 1L & label == 1L)
    tn <- sum(pred_label == 0L & label == 0L)
    fp <- sum(pred_label == 1L & label == 0L)
    fn <- sum(pred_label == 0L & label == 1L)

    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
    f1 <- ifelse(!is.na(precision) & !is.na(recall) & (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        NA_real_
    )

    denom <- sqrt(
        as.numeric(tp + fp) * as.numeric(tp + fn) *
        as.numeric(tn + fp) * as.numeric(tn + fn)
    )
    mcc <- ifelse(denom > 0, (tp * tn - fp * fn) / denom, NA_real_)

    roc_auc <- tryCatch({
        as.numeric(pROC::auc(pROC::roc(response = label, predictor = score, quiet = TRUE, direction = "<")))
    }, error = function(e) NA_real_)

    pr_df <- compute_pr_df(score, label)
    pr_auc <- if (nrow(pr_df) > 0) {
        sum(diff(c(0, pr_df$recall)) * pr_df$precision)
    } else {
        NA_real_
    }

    tibble(
        roc_auc = roc_auc,
        pr_auc = pr_auc,
        mcc = mcc,
        accuracy = accuracy,
        precision = precision,
        recall = recall,
        f1 = f1
    )
}

exp_list <- read_tsv(BIGBED_LIST, show_col_types = FALSE)
exp_names <- unique(exp_list$EXP_NAME)

metrics_ls_gkm <- lapply(exp_names, function(exp_name) {
    pred_path <- file.path(LS_GKM_DIR, exp_name, "all_predictions.txt")
    compute_binary_metrics(pred_path)
})

metrics_mls_gkm <- lapply(exp_names, function(exp_name) {
    pred_path <- file.path(MLS_GKM_DIR, exp_name, "all_predictions.txt")
    compute_binary_metrics(pred_path)
})

ls_gkm_df <- bind_rows(metrics_ls_gkm)
mls_gkm_df <- bind_rows(metrics_mls_gkm)

metric_names <- c("roc_auc", "pr_auc", "mcc", "accuracy", "precision", "recall", "f1")

for (metric_name in metric_names) {
    summary_df <- tibble(
        EXP_NAME = exp_names,
        LS_GKM_METRIC = ls_gkm_df[[metric_name]],
        MLS_GKM_METRIC = mls_gkm_df[[metric_name]]
    )
    out_path <- file.path(OUT_DIR, paste0("summary_", metric_name, ".csv"))
    write_csv(summary_df, out_path)
}

plot_metric_scatter <- function(metric_name) {
    in_path <- file.path(OUT_DIR, paste0("summary_", metric_name, ".csv"))
    if (!file.exists(in_path)) {
        warning(paste("Missing metrics file:", in_path))
        return(NULL)
    }

    df <- read_csv(in_path, show_col_types = FALSE)
    df <- df[is.finite(df$LS_GKM_METRIC) & is.finite(df$MLS_GKM_METRIC), , drop = FALSE]
    if (nrow(df) < 2) {
        warning(paste("Not enough data to plot:", metric_name))
        return(NULL)
    }

    fit <- lm(MLS_GKM_METRIC ~ LS_GKM_METRIC, data = df)
    r2 <- summary(fit)$r.squared

    p <- ggplot(df, aes(x = LS_GKM_METRIC, y = MLS_GKM_METRIC)) +
        geom_point(alpha = 0.7, size = 2) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
        labs(
            title = paste0(metric_name, ": LS-GKM vs mLS-GKM"),
            x = "LS_GKM_METRIC",
            y = "MLS_GKM_METRIC"
        ) +
        annotate("text", x = Inf, y = -Inf, label = sprintf("R2 = %.4f", r2),
            hjust = 1.05, vjust = -0.4, size = 4
        ) +
        theme_minimal()

    out_path <- file.path(OUT_DIR, paste0("scatter_", metric_name, ".png"))
    ggsave(out_path, p, width = 6, height = 6, dpi = 150)
    p
}

for (metric_name in metric_names) {
    plot_metric_scatter(metric_name)
}


