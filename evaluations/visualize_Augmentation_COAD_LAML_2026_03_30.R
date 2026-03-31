library(tidyverse)
library(ggplot2)
library(ggh4x)

## 配置 -----------------------------------------------------------

# 可视化指标开关
METRIC_TO_PLOT <- c(
    "cARI_real_gen_consistency",
    "ARI_group_consistency",
    "ccc_log2FC",
    "ccc_log10pvalue"
)

metrics <- METRIC_TO_PLOT

# 根据 metric 数量自适应图像高度
metric_count <- length(metrics)
height_unit <- 20 / 4.4
plot_height <- height_unit * (metric_count + 0.4)

# 输入结果文件（单一 CSV，包含 with_AE_head 列）
results_file <- "evaluations/Augmentation-COAD-LAML-2026-03-30/COAD-LAML_Augmentation-2026-03-30_evaluation_results.csv"
output_dir <- "evaluations/Augmentation-COAD-LAML-2026-03-30"

# Group variable 与样本量数据路径
case_base <- "syng_bts/data/case"
group_var_list <- c(
    COAD = "tumor_status",
    LAML = "blast_count"
)
positive_files <- c(
    COAD = file.path(case_base, "COAD_5-2", "COADPositive_5-2.csv"),
    LAML = file.path(case_base, "LAML_5-2", "LAMLPositive_5-2.csv")
)

get_group_counts <- function(cancer) {
    path <- positive_files[cancer]
    if (is.na(path) || !file.exists(path)) return(c(n0 = NA_integer_, n1 = NA_integer_))
    df <- read.csv(path, check.names = FALSE)
    if (!("groups" %in% colnames(df))) return(c(n0 = NA_integer_, n1 = NA_integer_))
    g <- as.character(as.integer(round(as.numeric(df$groups))))
    c(n0 = sum(g == "0", na.rm = TRUE), n1 = sum(g == "1", na.rm = TRUE))
}

## 读取结果 -------------------------------------------------------

if (!file.exists(results_file)) {
    stop("找不到结果文件: ", results_file)
}

cat("读取结果文件:", results_file, "\n")
all_results <- read.csv(results_file, check.names = FALSE)
cat("数据维度:", dim(all_results), "\n")

required_cols <- c(metrics, "model", "norm", "data_type", "cancer", "with_AE_head")
missing_cols <- setdiff(required_cols, colnames(all_results))
if (length(missing_cols) > 0) {
    stop("结果文件缺少必需列: ", paste(missing_cols, collapse = ", "))
}

## 按癌种拆分并逐个绘图 -----------------------------------------------

unique_cancers <- sort(unique(all_results$cancer))
cat("检测到的癌种:", paste(unique_cancers, collapse = ", "), "\n")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

for (cancer in unique_cancers) {
    cat("\n===== 开始绘制癌种:", cancer, "=====\n")

    df_cancer <- all_results %>%
        filter(cancer == !!cancer)

    if (nrow(df_cancer) == 0) {
        cat("癌种", cancer, "没有数据，跳过。\n")
        next
    }

    ## 整理长表 ---------------------------------------------------

    df_long <- df_cancer %>%
        pivot_longer(
            cols = all_of(metrics),
            names_to = "metric",
            values_to = "value"
        ) %>%
        filter(data_type == "miRNA")

    cat("癌种", cancer, "转换为长表后维度:", dim(df_long), "\n")

    # 标准化顺序
    df_long$norm <- as.character(df_long$norm)
    df_long$norm[df_long$norm == ""] <- "Raw"
    df_long$norm[tolower(df_long$norm) == "raw"] <- "Raw"
    df_long$norm <- factor(df_long$norm, levels = c("Raw", "TC", "DESeq"))

    # with_AE_head 统一为左右两个 panel
    with_ae_raw <- as.character(df_long$with_AE_head)
    with_ae_lc <- tolower(with_ae_raw)
    is_with_ae <- with_ae_lc %in% c("true", "t", "1", "yes")
    df_long$ae_panel <- ifelse(is_with_ae, "With AE head", "Without AE head")
    df_long$ae_panel <- factor(df_long$ae_panel, levels = c("Without AE head", "With AE head"))

    model_order <- c("CVAE10-1", "CVAE1-1", "CVAE1-10", "CVAE1-50", "CVAE1-100", "CVAE1-200")
    model_rest <- setdiff(sort(unique(df_long$model)), model_order)
    df_long$model <- factor(df_long$model, levels = c(model_order, model_rest))
    df_long$metric <- factor(df_long$metric, levels = metrics)

    ## 去掉在某个 (model, norm, metric, ae_panel) 下完全不变的指标 ----------

    df_box <- df_long %>%
        group_by(model, norm, metric, ae_panel) %>%
        mutate(is_constant = (sd(value, na.rm = TRUE) == 0)) %>%
        ungroup() %>%
        filter(!is_constant)

    cat("癌种", cancer, "过滤掉常数指标后维度:", dim(df_box), "\n")

    if (nrow(df_box) == 0) {
        cat("癌种", cancer, "过滤后没有可绘制的数据，跳过。\n")
        next
    }

    ## Group variable 与样本量（用于 subtitle）
    group_var <- if (cancer %in% names(group_var_list)) group_var_list[[cancer]] else "groups"
    counts <- get_group_counts(cancer)
    subtitle_str <- paste0(
        "Group variable: ", group_var,
        " | Group 0: n = ", counts["n0"], ", Group 1: n = ", counts["n1"]
    )

    ## 绘图：左右 panel 对比 with/without AE head --------------------------

    dodge_w <- 0.7
    p <- ggplot(df_box, aes(x = norm, y = value, fill = model)) +
        geom_boxplot(
            width = 0.6,
            outlier.size = 0.5,
            color = "black",
            position = position_dodge(width = dodge_w)
        ) +
        facet_grid(metric ~ ae_panel, scales = "free_y") +
        facetted_pos_scales(
            y = list(
                metric == "cARI_real_gen_consistency" ~ scale_y_continuous(limits = c(0.5, 1), breaks = seq(0.5, 1, 0.1)),
                metric == "ARI_group_consistency"      ~ scale_y_continuous(limits = c(0, 0.5), breaks = seq(0, 0.5, 0.1)),
                metric == "ccc_log2FC"                 ~ scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)),
                metric == "ccc_log10pvalue"            ~ scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))
            )
        ) +
        theme_classic(base_size = 16) +
        theme(
            strip.text       = element_text(size = 12),
            strip.background = element_blank(),
            axis.title.x     = element_blank(),
            axis.title.y     = element_text(size = 14),
            axis.text.x      = element_text(angle = 45, hjust = 1),
            legend.position  = "right",
            panel.border     = element_rect(colour = "black", fill = NA, linewidth = 0.8),
            panel.spacing.y  = unit(0.6, "lines")
        ) +
        labs(
            title = paste0(cancer, " Augmentation (2026-03-30)"),
            subtitle = subtitle_str,
            x = NULL,
            y = "Evaluation Score",
            fill = "Model"
        )

    ## 保存 -------------------------------------------------------

    output_plot <- file.path(
        output_dir,
        paste0(cancer, "_Augmentation-2026-03-30_evaluation_boxplot_AEhead_panels.png")
    )

    cat("保存图像到:", output_plot, "\n")
    ggsave(output_plot, plot = p, width = 14, height = plot_height, dpi = 300)

    if (file.exists(output_plot)) {
        file_size <- file.info(output_plot)$size
        cat("图已保存: ", output_plot, "，文件大小: ", file_size, " bytes\n", sep = "")
    } else {
        cat("图像保存失败\n")
    }
}

cat("\nCOAD-LAML Augmentation 可视化全部完成。\n")

