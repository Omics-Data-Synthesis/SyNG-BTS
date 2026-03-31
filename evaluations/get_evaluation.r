get_eval <- function(real_df, generated_df,
                     model_name = "Model",
                     log = TRUE,
                     failure = c("replace", "remove"),
                     poly = FALSE,
                     coords = NULL,
                     draw = 5,
                     group_col = "groups",
                     plot_first = FALSE) {
    failure <- match.arg(failure)

    # 统计真实数据中按 group_col（通常为 0/1）划分的样本量，方便在结果表中记录
    n_group0_real <- NA_integer_
    n_group1_real <- NA_integer_
    if (!is.null(group_col) && group_col %in% colnames(real_df)) {
        g <- as.character(real_df[[group_col]])
        tab <- table(g)
        if ("0" %in% names(tab)) n_group0_real <- as.integer(tab[["0"]])
        if ("1" %in% names(tab)) n_group1_real <- as.integer(tab[["1"]])
    }

    df_index_store <- data.frame(
        model = rep(model_name, draw),
        draw = 1:draw,
        ccc_log10pvalue = NA,
        ccc_log2FC = NA,
        cARI_real_gen_consistency = NA,
        ARI_group_consistency = NA,
        fail_prop = NA,
        ccc_pos = NA,
        ks_mean = NA,
        ks_sd = NA,
        ks_zero = NA,
        ks_cv = NA,
        n_group0_real = n_group0_real,
        n_group1_real = n_group1_real
    )

    # 提取表达列名（不包含 group 列）
    cols_to_use <- setdiff(colnames(real_df), group_col)

    # 提取 real 的表达数据和 groups
    dat_real <- as.matrix(real_df[, cols_to_use])
    groups_real <- if (!is.null(group_col) && group_col %in% colnames(real_df)) {
        real_df[[group_col]]
    } else {
        NULL
    }

    for (i in 1:draw) {
        # 从 generated 中分层抽样
        sampled_generated <- stratified_sample(
            real_df = real_df,
            generated_df = generated_df,
            group_col = if (group_col %in% colnames(real_df)) group_col else NULL,
            replace = FALSE,
            seed = 123 + i
        )

        # 提取 generated 的表达数据和 groups
        dat_generated <- as.matrix(sampled_generated[, cols_to_use])
        groups_generated <- if (!is.null(group_col) && group_col %in% colnames(sampled_generated)) {
            sampled_generated[[group_col]]
        } else {
            NULL
        }

        # 可视化（仅第一次）
        if (i == 1 && plot_first) {
            try(print(heatmap_eval(real_df, sampled_generated, group_col = group_col, log = log)), silent = TRUE)
            try(
                {
                    umap_out <- UMAP_eval(real_df, sampled_generated, group_col = group_col, log = log, failure = failure)
                    print(umap_out$p_umap)
                },
                silent = TRUE
            )
        }

        # cARI_real_gen_consistency = 1 - |ARI(datatype, clusters)|：用「Real vs Generated 不可区分性」计算（不传 groups），越高越好
        try(
            {
                ari_indist <- cluster_eval(
                    dat_real, dat_generated,
                    groups_real = NULL,
                    groups_generated = NULL,
                    log = log, failure = failure
                )
                df_index_store[i, "cARI_real_gen_consistency"] <- ari_indist
            },
            silent = TRUE
        )

        # ARI_group_consistency = ARI(groups, clusters)：聚类与生物学 group 标签一致性，需与 DEA 同映射
        ari_bio_orig <- NA_real_
        ari_bio_flip <- NA_real_
        g_gen_int <- if (!is.null(groups_generated)) as.integer(groups_generated) else NULL
        groups_generated_flipped <- if (!is.null(g_gen_int)) (1L - g_gen_int) else NULL
        try(
            {
                ari_bio_orig <- cluster_eval(
                    dat_real, dat_generated,
                    groups_real = groups_real,
                    groups_generated = groups_generated,
                    log = log, failure = failure
                )
            },
            silent = TRUE
        )
        if (!is.null(groups_generated_flipped)) {
            try(
                {
                    ari_bio_flip <- cluster_eval(
                        dat_real, dat_generated,
                        groups_real = groups_real,
                        groups_generated = groups_generated_flipped,
                        log = log, failure = failure
                    )
                },
                silent = TRUE
            )
        }

        # 列映射仅根据 ccc_log2FC 较高者选择；原映射与反转映射下各算 DEA 两个 metric
        orig_vals <- c(ccc_log10pvalue = NA_real_, ccc_log2FC = NA_real_)
        flip_vals <- c(ccc_log10pvalue = NA_real_, ccc_log2FC = NA_real_)

        try(
            {
                de_res <- DEA_eval(
                    dat_real, dat_generated,
                    groups_real = groups_real,
                    groups_generated = groups_generated,
                    log = log, failure = failure
                )
                orig_vals["ccc_log10pvalue"] <- de_res[["ccc_log10pvalue"]]
                orig_vals["ccc_log2FC"] <- de_res[["ccc_log2FC"]]
            },
            silent = TRUE
        )
        if (!is.null(groups_generated_flipped)) {
            try(
                {
                    de_res_flip <- DEA_eval(
                        dat_real, dat_generated,
                        groups_real = groups_real,
                        groups_generated = groups_generated_flipped,
                        log = log, failure = failure
                    )
                    flip_vals["ccc_log10pvalue"] <- de_res_flip[["ccc_log10pvalue"]]
                    flip_vals["ccc_log2FC"] <- de_res_flip[["ccc_log2FC"]]
                },
                silent = TRUE
            )
        }

        # 仅根据 ccc_log2FC 较高者选择映射（NA 时视为更差）
        o_fc <- orig_vals["ccc_log2FC"]
        f_fc <- flip_vals["ccc_log2FC"]
        use_flip <- FALSE
        if (!is.na(f_fc) && (is.na(o_fc) || f_fc > o_fc)) use_flip <- TRUE

        vals <- if (use_flip) flip_vals else orig_vals
        other_vals <- if (use_flip) orig_vals else flip_vals
        df_index_store[i, "ccc_log10pvalue"] <- vals["ccc_log10pvalue"]
        df_index_store[i, "ccc_log2FC"] <- vals["ccc_log2FC"]
        ari_bio_chosen <- if (use_flip) ari_bio_flip else ari_bio_orig
        if (!is.na(ari_bio_chosen)) df_index_store[i, "ARI_group_consistency"] <- ari_bio_chosen

        # 若当前映射导致 ccc_log10pvalue 低于另一映射，打印具体表现
        o_p <- orig_vals["ccc_log10pvalue"]
        f_p <- flip_vals["ccc_log10pvalue"]
        if (!is.na(o_p) && !is.na(f_p)) {
            chosen_p <- vals["ccc_log10pvalue"]
            other_p <- other_vals["ccc_log10pvalue"]
            if (chosen_p < other_p) {
                mapping_lab <- if (use_flip) "反转" else "原"
                cat(
                    "此映射方案（", mapping_lab, "）导致 ccc_log10pvalue 得分较低：当前 = ", chosen_p,
                    "，另一映射 = ", other_p, "\n",
                    sep = ""
                )
            }
        }

        # Failure features
        try(
            {
                df_index_store[i, "fail_prop"] <- fail_features_eval(dat_real, dat_generated)
            },
            silent = TRUE
        )

        # ccpos
        try(
            {
                df_index_store[i, "ccc_pos"] <- ccpos_eval(
                    dat_real, dat_generated,
                    failure = failure, log = log,
                    coords = coords, poly = poly, thres = 32
                )
            },
            silent = TRUE
        )

        # Summary
        try(
            {
                summary_out <- summary_eval(dat_real, dat_generated, log = log, failure = failure)
                df_index_store[i, c("ks_mean", "ks_sd", "ks_zero", "ks_cv")] <- unlist(summary_out)
            },
            silent = TRUE
        )
    }

    return(df_index_store)
}

get_eval_all_configs <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                 base_path_real, base_path_generated,
                                 log = TRUE,
                                 failure = "replace",
                                 poly = TRUE,
                                 coords = NULL,
                                 draw = 5,
                                 group_col = "groups",
                                 plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构造文件路径
                        # 处理模型名称映射
                        model_mapping <- list(
                            "CVAE1-10" = "CVAE1-10",
                            "AE-CVAE1-10" = "AEhead_CVAE1-10"
                        )
                        mapped_model <- model_mapping[[model]]
                        
                        # 构建批次目录路径
                        batch_dir <- file.path(base_path_real, cancer, norm, paste0("batch_", batch))
                        
                        # 构建文件路径
                        real_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 根据模型类型构建生成文件路径
                        if (model == "AE-CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_AEhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_", mapped_model, "_generated.csv"))
                        }

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )

                        # 添加元信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm

                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

# 新的函数适配 optimal_config_offline_aug_test 数据格式
get_eval_all_configs_optimal <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                        base_path_real, base_path_generated,
                                        log = TRUE,
                                        failure = "replace",
                                        poly = TRUE,
                                        coords = NULL,
                                        draw = 5,
                                        group_col = "groups",
                                        plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构建批次目录路径 - 适配新的文件夹结构
                        batch_dir <- file.path(base_path_real, cancer, norm, paste0("batch_", batch))
                        
                        # 构建文件路径
                        real_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 根据模型类型构建生成文件路径 - 适配新的文件命名格式
                        if (model == "AEhead_CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_AEhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else if (model == "Gaussianhead_CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_Gaussianhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else {
                            # CVAE1-10 基础版本
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        }

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )
                        
                        # 添加配置信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm
                        
                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

# 新的函数适配 RNAseq_augmentation_data 数据格式
get_eval_all_configs_rnaseq <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                        base_path_real, base_path_generated,
                                        log = TRUE,
                                        failure = "replace",
                                        poly = TRUE,
                                        coords = NULL,
                                        draw = 5,
                                        group_col = "groups",
                                        plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构建目录路径 - 适配 RNAseq_augmentation_data 新的文件夹结构
                        # 格式: RNAseq_augmentation_data/{癌症类型}_5-2/{标准化方法}/batch_{数字}/{模型}/
                        model_dir <- file.path(base_path_real, paste0(cancer, "_5-2"), norm, paste0("batch_", batch), model)
                        
                        # 构建文件路径
                        # 测试文件: {癌症类型}_5-2_{标准化方法}_batch_{数字}_test.csv
                        real_file <- file.path(model_dir, paste0(cancer, "_5-2_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 生成文件: {癌症类型}_5-2_{标准化方法}_batch_{数字}_train_epoch{epoch}_batch01_{模型}_generated.csv
                        generated_file <- file.path(model_dir, paste0(cancer, "_5-2_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_", model, "_generated.csv"))

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )
                        
                        # 添加配置信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm
                        
                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}
