library(tidyverse)
library(ggplot2)
library(ggh4x)

## 配置 -----------------------------------------------------------

# 可视化的指标开关：在这里控制要画哪些 metric（与 COAD 脚本保持一致）
METRIC_TO_PLOT <- c(
    "cARI_real_gen_consistency",
    "ARI_group_consistency",
    "ccc_log2FC"
)

metrics <- METRIC_TO_PLOT

# 根据 metric 数量自适应图像高度。
# 约定：当 metric = 4 时，高度为 20（当前尺寸）：
#   20 = 4 * x + 0.4 * x = 4.4x  =>  x = 20 / 4.4
# 对于一般的 metric = n，高度设为：height = n * x + 0.4 * x
metric_count <- length(metrics)
height_unit <- 20 / 4.4
plot_height <- height_unit * (metric_count + 0.4)

# 汇总结果文件（五个癌种的合并结果）
results_file <- "evaluations/Augmentation-FiveSubtypes-2026-02-24/FiveSubtypes_Augmentation-2026-02-24_evaluation_results.csv"

# 每个癌种单独输出一张图像，文件名中带癌症名称
output_dir <- "evaluations/Augmentation-FiveSubtypes-2026-02-24"

# Group variable 与样本量数据路径（与 group_variable_info.txt 一致）
case_base <- "syng_bts/data/case"
group_var_list <- c(
    COAD = "tumor_status",
    LAML = "blast_count",
    PAAD = "ajcc_nodes_pathologic_pn",
    READ = "ajcc_tumor_pathologic_pt",
    SKCM = "breslow_thickness_at_diagnosis"
)
positive_files <- setNames(
    file.path(case_base, paste0(names(group_var_list), "_5-2"), paste0(names(group_var_list), "Positive_5-2.csv")),
    names(group_var_list)
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

cat("📖 读取结果文件:", results_file, "\n")
all_results <- read.csv(results_file, check.names = FALSE)
cat("📊 数据维度:", dim(all_results), "\n")

required_cols <- c(metrics, "model", "norm", "data_type", "cancer")
missing_cols <- setdiff(required_cols, colnames(all_results))
if (length(missing_cols) > 0) {
    stop("结果文件缺少必需列: ", paste(missing_cols, collapse = ", "))
}

## 按癌症拆分并逐个绘图 -----------------------------------------------

unique_cancers <- sort(unique(all_results$cancer))
cat("🧬 检测到的癌种:", paste(unique_cancers, collapse = ", "), "\n")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

for (cancer in unique_cancers) {
    cat("\n===== 开始绘制癌种:", cancer, "=====\n")

    df_cancer <- all_results %>%
        filter(cancer == !!cancer)

    if (nrow(df_cancer) == 0) {
        cat("⚠️ 癌种", cancer, "没有数据，跳过。\n")
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

    cat("📊 癌种", cancer, "转换为长表后维度:", dim(df_long), "\n")

    # X 轴按 Raw / TC / DESeq 分块，顺序固定（无数据的 norm 不会出现但顺序一致）
    df_long$norm <- as.character(df_long$norm)
    df_long$norm[df_long$norm == ""] <- "Raw"
    df_long$norm[tolower(df_long$norm) == "raw"] <- "Raw"
    df_long$norm <- factor(df_long$norm, levels = c("Raw", "TC", "DESeq"))

    df_long$data_type <- factor(df_long$data_type, levels = c("miRNA", "RNA"))
    # 将 miRNA / RNA 的显示标签改为 normal / wider（与 COAD 图保持一致）
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

    cat("📊 癌种", cancer, "过滤掉常数指标后维度:", dim(df_box), "\n")

    if (nrow(df_box) == 0) {
        cat("⚠️ 癌种", cancer, "过滤后没有可绘制的数据，跳过。\n")
        next
    }

    ## Group variable 与样本量（用于 subtitle）
    group_var <- if (cancer %in% names(group_var_list)) group_var_list[[cancer]] else "groups"
    counts <- get_group_counts(cancer)
    subtitle_str <- paste0(
        "Group variable: ", group_var,
        " | Group 0: n = ", counts["n0"], ", Group 1: n = ", counts["n1"]
    )

    ## 绘图：X 轴 = norm（Raw / TC / DESeq）分块，横向按 data_type 分面 -------

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
            title = paste0(cancer, " Augmentation (2026-02-24)"),
            subtitle = subtitle_str,
            x = NULL,
            y = "Evaluation Score",
            fill = "Model"
        )

    ## 保存 -------------------------------------------------------

    output_plot <- file.path(
        output_dir,
        paste0(cancer, "_Augmentation-2026-02-24_evaluation_boxplot.png")
    )

    cat("💾 癌种", cancer, "保存图像到:", output_plot, "\n")
    ggsave(output_plot, plot = p, width = 10, height = plot_height, dpi = 300)

    if (file.exists(output_plot)) {
        file_size <- file.info(output_plot)$size
        cat("✅ 癌种 ", cancer, " 图已保存，文件大小: ", file_size, " bytes\n", sep = "")
    } else {
        cat("❌ 癌种", cancer, "图像保存失败\n")
    }
}

cat("\n🎉 FiveSubtypes Augmentation 可视化全部完成。\n")

