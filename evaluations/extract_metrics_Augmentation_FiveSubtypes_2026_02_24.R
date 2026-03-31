## 提取 2026-02-24 FiveSubtypes 增强数据的评价指标

## 加载本地 evaluation 函数（与原始用法一致）
source("/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/evaluations_functions.r")
source("/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/get_evaluation.r")

## 基本参数配置 -----------------------------------------------------------

# FiveSubtypes 根目录：下面有 COAD_5-2 / LAML_5-2 / PAAD_5-2 / READ_5-2 / SKCM_5-2
root_path <- "/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/syng_bts/data/case/Augmentation-FiveSubtypes-2026-02-24"

# 癌症前缀（与目录名 + 文件名前缀一致）
cancer_prefix_list <- c("COAD_5-2", "LAML_5-2", "PAAD_5-2", "READ_5-2", "SKCM_5-2")

# 三种标准化：与目录 raw / TC / DESeq 一致
norm_list <- c("raw", "TC", "DESeq")

# batch 范围（多写一些没关系，不存在的文件会被跳过）
batch_list <- 1:20

# 数据类型：目前目录中主要是 miRNA；RNA 若不存在会自动跳过
data_type_list <- c("miRNA", "RNA")

# 评价时是否视为已经 log2（真实数据会在这里做 log2(x+1) 预处理）
log_input <- TRUE

# zero-variance 特征的处理方式
failure_mode <- "replace"

# ccpos 的坐标信息（目前只对 miRNA 有意义，这里先不使用）
coords_miRNA <- NULL

## 自动探测一次模型列表（从某癌种某 norm 的 batch_1 目录下读） ----------------

detect_models <- function(base_path, norm) {
    batch1_dir <- file.path(base_path, norm, "batch_1")
    if (!dir.exists(batch1_dir)) {
        warning("目录不存在，无法自动探测模型: ", batch1_dir)
        return(character(0))
    }
    subdirs <- list.dirs(batch1_dir, full.names = FALSE, recursive = FALSE)
    # 过滤掉真实数据文件等非模型子目录
    subdirs[!(subdirs %in% c("", ".", ".."))]
}

## 主循环：遍历 cancer_prefix / norm / batch / model / data_type --------------

all_results <- list()

for (cancer_prefix in cancer_prefix_list) {
    # 提取癌症简写（用于结果中的 cancer 列，如 COAD / LAML 等）
    cancer <- sub("_.*$", "", cancer_prefix)

    # 该癌种的根路径：root_path/COAD_5-2 等，下面是 raw / TC / DESeq
    base_path <- file.path(root_path, cancer_prefix)

    if (!dir.exists(base_path)) {
        warning("找不到癌种目录，跳过: ", base_path)
        next
    }

    cat("=== 开始评估癌种:", cancer, " (前缀:", cancer_prefix, ") ===\n")

    for (norm in norm_list) {
        # 每个 norm 下单独探测模型列表（不同 norm 可能略有差异）
        model_list <- detect_models(base_path, norm)
        if (length(model_list) == 0) {
            warning("癌种 = ", cancer, ", norm = ", norm, " 下未探测到模型子目录，跳过。")
            next
        }
        cat("癌种 = ", cancer, ", norm = ", norm, " 将评估的模型: ",
            paste(model_list, collapse = ", "), "\n", sep = "")

        for (batch in batch_list) {
            batch_dir <- file.path(base_path, norm, paste0("batch_", batch))

            # 构造真实数据路径
            real_file <- file.path(
                batch_dir,
                sprintf("%s_%s_batch_%d_test.csv", cancer_prefix, norm, batch)
            )

            if (!file.exists(real_file)) {
                next
            }

            cat("发现癌种 ", cancer, " batch ", batch,
                " 的真实数据文件: ", real_file, "\n", sep = "")

            # 读取真实数据
            real_df <- read.csv(real_file, check.names = FALSE)
            if ("samples" %in% colnames(real_df)) {
                real_df$samples <- NULL
            }

            group_col <- "groups"
            if (!(group_col %in% colnames(real_df))) {
                stop("真实数据中找不到 'groups' 列: ", real_file)
            }

            # 将 groups 列统一为整型 0/1（用于与 generated 最后一列一致，便于 log2FC 等计算）
            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                real_df[[group_col]] <- as.integer(ifelse(real_df[[group_col]] == "YES", 1L, 0L))
            } else {
                real_df[[group_col]] <- as.integer(as.character(real_df[[group_col]]))
            }

            # 对表达矩阵做 log2(x+1) 以匹配 get_eval 中 log=TRUE 的设定
            expr_cols <- setdiff(colnames(real_df), group_col)
            real_df[expr_cols] <- log2(real_df[expr_cols] + 1)

            for (model in model_list) {
                for (data_type in data_type_list) {
                    # 生成数据所在目录：<norm>/batch_k/<model>/<data_type>/
                    gen_dir <- file.path(batch_dir, model, data_type)
                    gen_file <- file.path(
                        gen_dir,
                        sprintf(
                            "%s_%s_batch_%d_train_epochES_batch01_%s_generated.csv",
                            cancer_prefix, norm, batch, model
                        )
                    )

                    if (!file.exists(gen_file)) {
                        next
                    }

                    cat("Processing: cancer = ", cancer,
                        ", norm = ", norm,
                        ", batch = ", batch,
                        ", model = ", model,
                        ", data_type = ", data_type,
                        " -> ",
                        basename(real_file), " + ", basename(gen_file),
                        "\n", sep = "")

                    # 读取生成数据（无表头）
                    generated_df <- read.csv(gen_file, header = FALSE, check.names = FALSE)
                    if (nrow(generated_df) == 0L) {
                        warning("生成数据为空，跳过: ", gen_file)
                        next
                    }

                    # 严格约定：test 为 n 列时，去掉 samples 后为 (n-1) 列 = (n-2) 表达列 + 1 列 groups；
                    # generated 必须为 (n-2) 表达列 + 1 列 group = (n-1) 列，即 ncol(generated) == ncol(real_df)
                    expr_cols <- setdiff(colnames(real_df), group_col)
                    n_expr <- length(expr_cols)
                    n_real <- ncol(real_df)
                    n_gen <- nrow(generated_df)
                    n_gen_col <- ncol(generated_df)

                    expected_gen_cols <- n_expr + 1L
                    if (n_gen_col != expected_gen_cols) {
                        stop(
                            "生成数据列数不符合约定。test 去掉 samples 后为 ", n_real, " 列 (", n_expr, " 表达 + 1 groups)，",
                            "generated 必须为 ", expected_gen_cols, " 列 (", n_expr, " 表达 + 1 group)。",
                            "当前 generated 列数: ", n_gen_col, "。文件: ", gen_file
                        )
                    }
                    if (length(generated_df[[n_gen_col]]) != n_gen) {
                        stop(
                            "生成数据最后一列（group）长度与行数不一致。行数: ", n_gen, "，最后一列长度: ",
                            length(generated_df[[n_gen_col]]), "。文件: ", gen_file
                        )
                    }

                    gen_groups <- as.integer(round(generated_df[[n_gen_col]]))
                    generated_df <- generated_df[, seq_len(n_expr), drop = FALSE]
                    colnames(generated_df) <- expr_cols
                    generated_df[[group_col]] <- gen_groups

                    # miRNA 可以在将来接入 coords，这里先统一为 NULL
                    coords_used <- if (data_type == "miRNA") coords_miRNA else NULL

                    # 调用核心评估函数（draw 设为 1，避免重复抽样）
                    result_df <- get_eval(
                        real_df = real_df,
                        generated_df = generated_df,
                        model_name = model,
                        log = log_input,
                        failure = failure_mode,
                        poly = TRUE,
                        coords = coords_used,
                        draw = 1,
                        group_col = group_col,
                        plot_first = FALSE
                    )

                    # 添加元信息；subtype 改名为 data_type；增加 cancer 信息
                    result_df$cancer <- cancer
                    result_df$data_type <- data_type
                    result_df$epoch <- "ES"
                    result_df$batch <- batch
                    result_df$norm <- norm

                    all_results[[length(all_results) + 1]] <- result_df
                }
            }
        }
    }
}

if (length(all_results) == 0) {
    stop("没有成功评估到任何配置，请检查数据路径和命名是否正确。")
}

final_df <- do.call(rbind, all_results)

## 保存结果 ---------------------------------------------------------------

output_dir <- "/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/Augmentation-FiveSubtypes-2026-02-24"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "FiveSubtypes_Augmentation-2026-02-24_evaluation_results.csv")
write.csv(final_df, output_file, row.names = FALSE)

cat("共评估了", nrow(final_df), "个配置组合。\n")
cat("结果已保存到:", output_file, "\n")

