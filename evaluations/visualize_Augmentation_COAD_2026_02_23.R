library(tidyverse)
library(ggplot2)
library(ggh4x)

## 配置 -----------------------------------------------------------

# 可视化的指标开关：在这里控制要画哪些 metric
METRIC_TO_PLOT <- c(
    "cARI_real_gen_consistency",
    "ARI_group_consistency",
    "ccc_log2FC",
    "ccc_log10pvalue"
)

cancer <- "COAD"
metrics <- METRIC_TO_PLOT

results_file <- "evaluations/Augmentation-2026-02-23/COAD_Augmentation-2026-02-23_evaluation_results.csv"
output_plot <- "evaluations/Augmentation-2026-02-23/COAD_Augmentation-2026-02-23_evaluation_boxplot.png"

# Group variable 与样本量（与 group_variable_info.txt 一致）
group_var <- "tumor_status"
case_base <- "syng_bts/data/case"
positive_file <- file.path(case_base, "COAD_5-2", "COADPositive_5-2.csv")
n_group0 <- NA_integer_
n_group1 <- NA_integer_
if (file.exists(positive_file)) {
    pos_df <- read.csv(positive_file, check.names = FALSE)
    if ("groups" %in% colnames(pos_df)) {
        g <- as.character(as.integer(round(as.numeric(pos_df$groups))))
        n_group0 <- sum(g == "0", na.rm = TRUE)
        n_group1 <- sum(g == "1", na.rm = TRUE)
    }
}
subtitle_str <- paste0(
    "Group variable: ", group_var,
    " | Group 0: n = ", n_group0, ", Group 1: n = ", n_group1
)

## 读取结果 -------------------------------------------------------

if (!file.exists(results_file)) {
    stop("找不到结果文件: ", results_file)
}

cat("📖 读取结果文件:", results_file, "\n")
all_results <- read.csv(results_file, check.names = FALSE)
cat("📊 数据维度:", dim(all_results), "\n")

required_cols <- c(metrics, "model", "norm", "data_type")
missing_cols <- setdiff(required_cols, colnames(all_results))
if (length(missing_cols) > 0) {
    stop("结果文件缺少必需列: ", paste(missing_cols, collapse = ", "))
}

## 整理长表 -------------------------------------------------------

df_long <- all_results %>%
    pivot_longer(
        cols = all_of(metrics),
        names_to = "metric",
        values_to = "value"
    )

cat("📊 转换为长表后维度:", dim(df_long), "\n")

# X 轴按 Raw / TC / DESeq 分块，顺序固定（无数据的 norm 不会出现但顺序一致）
df_long$norm <- as.character(df_long$norm)
df_long$norm[df_long$norm == ""] <- "Raw"
df_long$norm[tolower(df_long$norm) == "raw"] <- "Raw"
df_long$norm <- factor(df_long$norm, levels = c("Raw", "TC", "DESeq"))

df_long$data_type <- factor(df_long$data_type, levels = c("miRNA", "RNA"))
# 将 miRNA / RNA 的显示标签改为 normal / wider
levels(df_long$data_type) <- c("normal", "wider")

model_order <- c("CVAE10-1", "CVAE1-1", "CVAE1-10", "CVAE1-50", "CVAE1-100", "CVAE1-200")
model_rest <- setdiff(sort(unique(df_long$model)), model_order)
df_long$model <- factor(df_long$model, levels = c(model_order, model_rest))
df_long$metric <- factor(df_long$metric, levels = metrics)

## 去掉在某个 (model, norm, data_type, metric) 下完全不变的指标 ---------

df_box <- df_long %>%
    group_by(model, norm, data_type, metric) %>%
    mutate(is_constant = (sd(value, na.rm = TRUE) == 0)) %>%
    ungroup() %>%
    filter(!is_constant)

cat("📊 过滤掉常数指标后维度:", dim(df_box), "\n")

if (nrow(df_box) == 0) {
    stop("过滤后没有可绘制的数据，请检查结果文件。")
}

## 绘图：X 轴 = norm（Raw / TC / DESeq）分块，横向按 miRNA | RNA 分面；y 轴由 facetted_pos_scales 强制限制 -------

dodge_w <- 0.7
p <- ggplot(df_box, aes(x = norm, y = value, fill = model)) +
    geom_boxplot(
        width = 0.6,
        outlier.size = 0.5,
        color = "black",
        position = position_dodge(width = dodge_w)
    ) +
    facet_grid(metric ~ data_type, scales = "free_y") +
    facetted_pos_scales(
        y = list(
            metric == "cARI_real_gen_consistency" ~ scale_y_continuous(limits = c(0.5, 1), breaks = seq(0.5, 1, 0.1)),
            metric == "ARI_group_consistency"      ~ scale_y_continuous(limits = c(0, 0.5), breaks = seq(0, 0.5, 0.1)),
            metric == "ccc_log2FC"                ~ scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)),
            metric == "ccc_log10pvalue"           ~ scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))
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
        title = paste0(cancer, " Augmentation (2026-02-23)"),
        subtitle = subtitle_str,
        x = NULL,
        y = "Evaluation Score",
        fill = "Model"
    )

## 保存 -----------------------------------------------------------

cat("💾 保存图像到:", output_plot, "\n")
# 图像尺寸放大一倍（原先约为 10x10）
ggsave(output_plot, plot = p, width = 20, height = 20, dpi = 300)

if (file.exists(output_plot)) {
    file_size <- file.info(output_plot)$size
    cat("✅ 已保存:", output_plot, "，文件大小:", file_size, "bytes\n")
} else {
    cat("❌ 图像保存失败\n")
}

cat("🎉 COAD Augmentation 可视化完成。\n")

