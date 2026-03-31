## 提取 2026-03-30 COAD-LAML 增强数据的评价指标

## 加载本地 evaluation 函数（与原始用法一致）
source("/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/evaluations_functions.r")
source("/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/get_evaluation.r")

## 基本参数配置 -----------------------------------------------------------

# COAD-LAML 根目录：下面有 COAD_5-2 / LAML_5-2
root_path <- "/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/syng_bts/data/case/Augmentation-COAD-LAML-2026-03-30"

# 癌症前缀（与目录名 + 文件名前缀一致）
cancer_prefix_list <- c("COAD_5-2", "LAML_5-2")

# 三种标准化：与目录 raw / TC / DESeq 一致
norm_list <- c("raw", "TC", "DESeq")

# off_aug 目录名
off_aug_list <- c("offaug_none", "offaug_AE_head")

# batch 范围
batch_list <- 1:20

# 数据类型：当前该 batch 为 miRNA
data_type_list <- c("miRNA")

# 评价时是否视为已经 log2（真实数据会在这里做 log2(x+1) 预处理）
log_input <- TRUE

# zero-variance 特征的处理方式
failure_mode <- "replace"

# ccpos 的坐标信息（目前先不使用）
coords_miRNA <- NULL

## 自动探测一次模型列表（从某癌种某 norm 的 batch_1/offaug_none 下读） ----------

detect_models <- function(base_path, norm) {
    probe_dirs <- c(
        file.path(base_path, norm, "batch_1", "offaug_none"),
        file.path(base_path, norm, "batch_1")
    )
    batch1_dir <- probe_dirs[dir.exists(probe_dirs)][1]
    if (is.na(batch1_dir)) {
        warning("目录不存在，无法自动探测模型: ", paste(probe_dirs, collapse = " or "))
        return(character(0))
    }
    subdirs <- list.dirs(batch1_dir, full.names = FALSE, recursive = FALSE)
    subdirs[!(subdirs %in% c("", ".", ".."))]
}

## 主循环：遍历 cancer_prefix / norm / batch / off_aug / model / data_type -----

all_results <- list()

for (cancer_prefix in cancer_prefix_list) {
    cancer <- sub("_.*$", "", cancer_prefix)
    base_path <- file.path(root_path, cancer_prefix)

    if (!dir.exists(base_path)) {
        warning("找不到癌种目录，跳过: ", base_path)
        next
    }

    cat("=== 开始评估癌种:", cancer, " (前缀:", cancer_prefix, ") ===\n")

    for (norm in norm_list) {
        model_list <- detect_models(base_path, norm)
        if (length(model_list) == 0) {
            warning("癌种 = ", cancer, ", norm = ", norm, " 下未探测到模型子目录，跳过。")
            next
        }
        cat("癌种 = ", cancer, ", norm = ", norm, " 将评估的模型: ",
            paste(model_list, collapse = ", "), "\n", sep = "")

        for (batch in batch_list) {
            batch_dir <- file.path(base_path, norm, paste0("batch_", batch))
            real_file <- file.path(
                batch_dir,
                sprintf("%s_%s_batch_%d_test.csv", cancer_prefix, norm, batch)
            )

            if (!file.exists(real_file)) {
                next
            }

            cat("发现癌种 ", cancer, " batch ", batch,
                " 的真实数据文件: ", real_file, "\n", sep = "")

            real_df <- read.csv(real_file, check.names = FALSE)
            if ("samples" %in% colnames(real_df)) {
                real_df$samples <- NULL
            }

            group_col <- "groups"
            if (!(group_col %in% colnames(real_df))) {
                stop("真实数据中找不到 'groups' 列: ", real_file)
            }

            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                real_df[[group_col]] <- as.integer(ifelse(real_df[[group_col]] == "YES", 1L, 0L))
            } else {
                real_df[[group_col]] <- as.integer(as.character(real_df[[group_col]]))
            }

            expr_cols <- setdiff(colnames(real_df), group_col)
            real_df[expr_cols] <- log2(real_df[expr_cols] + 1)

            for (off_aug in off_aug_list) {
                for (model in model_list) {
                    for (data_type in data_type_list) {
                        gen_dir <- file.path(batch_dir, off_aug, model, data_type)

                        # 兼容两种命名：
                        # 1) offaug_none: ..._train_epochES_batch01_<model>_generated.csv
                        # 2) offaug_AE_head: ..._train_AEhead_epochES_batch01_<model>_generated.csv
                        gen_file_candidates <- c(
                            file.path(
                                gen_dir,
                                sprintf(
                                    "%s_%s_batch_%d_train_epochES_batch01_%s_generated.csv",
                                    cancer_prefix, norm, batch, model
                                )
                            ),
                            file.path(
                                gen_dir,
                                sprintf(
                                    "%s_%s_batch_%d_train_AEhead_epochES_batch01_%s_generated.csv",
                                    cancer_prefix, norm, batch, model
                                )
                            )
                        )
                        existing_idx <- which(file.exists(gen_file_candidates))
                        if (length(existing_idx) == 0) {
                            next
                        }
                        gen_file <- gen_file_candidates[existing_idx[1]]

                        cat("Processing: cancer = ", cancer,
                            ", norm = ", norm,
                            ", batch = ", batch,
                            ", off_aug = ", off_aug,
                            ", model = ", model,
                            ", data_type = ", data_type,
                            " -> ",
                            basename(real_file), " + ", basename(gen_file),
                            "\n", sep = "")

                        generated_df <- read.csv(gen_file, header = FALSE, check.names = FALSE)
                        if (nrow(generated_df) == 0L) {
                            warning("生成数据为空，跳过: ", gen_file)
                            next
                        }

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

                        coords_used <- if (data_type == "miRNA") coords_miRNA else NULL

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

                        result_df$cancer <- cancer
                        result_df$data_type <- data_type
                        result_df$with_AE_head <- (off_aug == "offaug_AE_head")
                        result_df$epoch <- "ES"
                        result_df$batch <- batch
                        result_df$norm <- norm

                        all_results[[length(all_results) + 1]] <- result_df
                    }
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

output_dir <- "/Users/yanjiechen/Documents/Github/SyNG-BTS_2.6/evaluations/Augmentation-COAD-LAML-2026-03-30"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "COAD-LAML_Augmentation-2026-03-30_evaluation_results.csv")
write.csv(final_df, output_file, row.names = FALSE)

cat("共评估了", nrow(final_df), "个配置组合。\n")
cat("结果已保存到:", output_file, "\n")

