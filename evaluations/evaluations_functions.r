library(ggplot2)
library(Rtsne)
library(tidyverse)
library(cowplot)
library(aricode)
library(RColorBrewer)
library(DANA)
library(dgof)
library(limma)
library(epiR)
library(umap)
library(foreach)
library(doParallel)

listClusters <- function(geneClusters) {
    geneClusters <- as.factor(geneClusters)
    clusters <- list()
    clusters <- lapply(
        levels(geneClusters),
        function(x) names(geneClusters)[geneClusters == x]
    )
    names(clusters) <- levels(geneClusters)
    return(clusters)
}

compute.cc <- function(rawCor, normCor, clusters) {
    # remove genes that are not present in both models
    rawCor <- rawCor[
        !is.na(match(colnames(rawCor), colnames(normCor))),
        !is.na(match(colnames(rawCor), colnames(normCor)))
    ]
    normCor <- normCor[
        !is.na(match(colnames(normCor), colnames(rawCor))),
        !is.na(match(colnames(normCor), colnames(rawCor)))
    ]

    # consider only cluster with multiple genes
    clusters <- clusters[lengths(clusters) > 1]
    clusters.raw <- c()
    clusters.norm <- c()
    for (clust in clusters) {
        # only consider cluster genes that are positive controls
        clust.genes <- clust[stats::na.omit(match(colnames(rawCor), clust))]
        if (length(clust.genes) < 2) {
            next # disregard clusters with less than 2 genes
        }
        # subset of correlations in the cluster
        rawCor.clust <- rawCor[clust.genes, clust.genes]
        clusters.raw <- c(clusters.raw, rawCor.clust[upper.tri(rawCor.clust)])
        normCor.clust <- normCor[clust.genes, clust.genes]
        clusters.norm <- c(clusters.norm, normCor.clust[upper.tri(normCor.clust)])
    }

    cc <- DescTools::CCC(as.vector(clusters.raw), as.vector(clusters.norm))$rho.c$est

    return(cc)
}

stratified_sample <- function(real_df, generated_df, group_col = "groups", replace = FALSE, seed = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }

    real_df[[group_col]] <- as.character(real_df[[group_col]])
    generated_df[[group_col]] <- as.character(generated_df[[group_col]])

    group_counts <- table(real_df[[group_col]])

    sampled_df <- generated_df %>%
        group_by(!!sym(group_col)) %>%
        group_modify(~ {
            group_id <- as.character(.y[[group_col]])
            target_n <- group_counts[group_id]
            if (is.na(target_n)) {
                stop(paste("Group", group_id, "not found in real_df"))
            }
            sample_n(.x, size = target_n, replace = replace || nrow(.x) < target_n)
        }) %>%
        ungroup()

    return(sampled_df)
}

ccpos_eval <- function(dat_real, dat_generated, log = TRUE, failure = c("replace", "remove"), coords, poly = FALSE, thres = NULL) {
    # This function performs DANA comparison of concordance correlation which measures
    # the preservation of biological signals in real versus generated data.
    # @param: dat_real - real data matrix (samples x features)
    # @param: dat_generated - generated data matrix (samples x features)
    # @param: log - whether the data are log2 transformed
    # @param: failure - how to deal with failure genes: "replace" or "remove"
    # @param: coords - coordinates data frame based on miRBase miRNA definitions
    # @param: poly - whether all features are positive controls
    # @param: thres - threshold for defining positive control features

    if (is.null(coords)) {
        cc <- NA
    } else {
        if (log) {
            # 防止产生无穷值
            dat_real <- 2^pmin(dat_real, 20) - 1
            dat_real[dat_real < 0] <- 0
            dat_real[!is.finite(dat_real)] <- 0
            dat_real <- round(dat_real)

            dat_generated <- 2^pmin(dat_generated, 20) - 1
            dat_generated[dat_generated < 0] <- 0
            dat_generated[!is.finite(dat_generated)] <- 0
            dat_generated <- round(dat_generated)
        }

        if (failure == "remove") {
            keep_cols <- apply(dat_generated, 2, sd) != 0
            dat_real <- dat_real[, keep_cols, drop = FALSE]
            dat_generated <- dat_generated[, keep_cols, drop = FALSE]
        } else if (failure == "replace") {
            zero_sd_cols <- apply(dat_generated, 2, sd) == 0
            dat_generated[, zero_sd_cols] <- dat_real[, zero_sd_cols]
        }

        # Align row and column names
        colnames(dat_generated) <- colnames(dat_real)
        rownames(dat_generated) <- rownames(dat_real)

        if (nrow(dat_real) + nrow(dat_generated) <= 2) {
            cc <- NA
        } else {
            clusters <- DANA::defineClusters(
                genes = coords$name,
                chr = coords$chr,
                pos = (coords$start + coords$end) / 2
            )
            if (poly) {
                posControls <- colnames(dat_real)
            } else {
                posControls <- DANA::defineControls(
                    raw = t(dat_real),
                    tZero = 2,
                    tPoor = 10,
                    tWell = thres,
                    clusters = clusters
                )$posControls
            }

            if (!is.null(posControls)) {
                invisible(capture.output(
                    corPos_real <- DANA::partialCor(t(dat_real[, posControls]), scale = TRUE)
                ))
                invisible(capture.output(
                    corPos_generated <- DANA::partialCor(t(dat_generated[, posControls]), scale = TRUE)
                ))

                cc <- compute.cc(corPos_real, corPos_generated, listClusters(as.factor(clusters)))
            } else {
                cc <- NA
            }
        }
    }
    return(cc)
}

summary_eval <- function(dat_real, dat_generated,
                         log = FALSE,
                         failure = c("replace", "remove")) {
    # dat_real, dat_generated: matrices or data.frames with samples in rows and features in columns
    # log: whether the input is already log2 transformed (if not, log2(x + 1) will be applied)
    # failure: how to handle zero-variance features in generated data

    failure <- match.arg(failure)

    if (!log) {
        dat_real <- log2(dat_real + 1)
        dat_generated <- log2(dat_generated + 1)
    }

    # failure mode
    if (failure == "remove") {
        keep_cols <- apply(dat_generated, 2, sd) != 0
        dat_real <- dat_real[, keep_cols, drop = FALSE]
        dat_generated <- dat_generated[, keep_cols, drop = FALSE]
    } else if (failure == "replace") {
        zero_sd_cols <- apply(dat_generated, 2, sd) == 0
        dat_generated[, zero_sd_cols] <- dat_real[, zero_sd_cols]
    }

    if ((nrow(dat_real) + nrow(dat_generated)) <= 2) {
        return(NA)
    }

    # compute summary statistics
    mu_real <- apply(dat_real, 2, mean)
    sigma_real <- apply(dat_real, 2, sd)
    zero_real <- apply(dat_real, 2, function(x) mean(x == 0))
    cv_real <- sigma_real / mu_real

    mu_generated <- apply(dat_generated, 2, mean)
    sigma_generated <- apply(dat_generated, 2, sd)
    zero_generated <- apply(dat_generated, 2, function(x) mean(x == 0))
    cv_generated <- sigma_generated / mu_generated

    ks_mean <- mad(mu_generated - mu_real)
    ks_sigma <- mad(sigma_generated - sigma_real)
    ks_zero <- mad(zero_generated - zero_real)
    ks_cv <- mad(cv_generated - cv_real)

    return(list(
        ks_mean = ks_mean,
        ks_sigma = ks_sigma,
        ks_zero = ks_zero,
        ks_cv = ks_cv
    ))
}

DEA_eval <- function(dat_real, dat_generated,
                     groups_real = NULL, groups_generated = NULL,
                     log = TRUE, failure = c("replace", "remove"),
                     plot = FALSE) {
    genes <- colnames(dat_real)

    if (log) {
        # 防止产生无穷值
        dat_real <- 2^pmin(dat_real, 20) - 1
        dat_real[dat_real < 0] <- 0
        dat_real[!is.finite(dat_real)] <- 0
        dat_real <- round(dat_real)

        dat_generated <- 2^pmin(dat_generated, 20) - 1
        dat_generated[dat_generated < 0] <- 0
        dat_generated[!is.finite(dat_generated)] <- 0
        dat_generated <- round(dat_generated)
    }

    if (failure == "remove") {
        keep_cols <- apply(dat_generated, 2, sd) != 0
        dat_real <- dat_real[, keep_cols, drop = FALSE]
        dat_generated <- dat_generated[, keep_cols, drop = FALSE]
        genes <- colnames(dat_real)
    } else if (failure == "replace") {
        zero_sd_cols <- apply(dat_generated, 2, sd) == 0
        dat_generated[, zero_sd_cols] <- dat_real[, zero_sd_cols]
    }

    if ((nrow(dat_real) + nrow(dat_generated)) <= 2) {
        return(NA)
    }

    if (is.null(groups_real) || is.null(groups_generated)) {
        groups_real <- rep("A", nrow(dat_real))
        groups_real[sample(1:nrow(dat_real), round(0.5 * nrow(dat_real)))] <- "B"
        groups_generated <- groups_real
    }

    dat_real <- as.matrix(dat_real)
    dat_generated <- as.matrix(dat_generated)

    # Real
    design_real <- model.matrix(~ factor(groups_real))
    v_real <- limma::voom(t(dat_real), design_real)
    fit_real <- limma::lmFit(v_real, design_real)
    fit_real <- limma::eBayes(fit_real)
    DEA_real <- list(
        p.val = fit_real$p.value[, 2],
        log2.FC = fit_real$coefficients[, 2],
        id.list = rownames(fit_real)
    )

    # Generated
    design_generated <- model.matrix(~ factor(groups_generated))
    v_generated <- limma::voom(t(dat_generated), design_generated)
    fit_generated <- limma::lmFit(v_generated, design_generated)
    fit_generated <- limma::eBayes(fit_generated)
    DEA_generated <- list(
        p.val = fit_generated$p.value[, 2],
        log2.FC = fit_generated$coefficients[, 2],
        id.list = rownames(fit_generated)
    )

    # 修正 -log10(p) 中无穷大问题
    DEA_real$p.val[!is.finite(log10(DEA_real$p.val))] <- min(DEA_real$p.val[is.finite(log10(DEA_real$p.val))])
    DEA_generated$p.val[!is.finite(log10(DEA_generated$p.val))] <- min(DEA_generated$p.val[is.finite(log10(DEA_generated$p.val))])

    # 计算 concordance correlation
    res <- c(
        ccc_log10pvalue = round(epiR::epi.ccc(-log10(DEA_real$p.val[genes]), -log10(DEA_generated$p.val[genes]))$rho.c$est, 4),
        ccc_log2FC = round(epiR::epi.ccc(DEA_real$log2.FC[DEA_real$id.list], DEA_generated$log2.FC[DEA_real$id.list])$rho.c$est, 4)
    )

    if (plot) {
        layout(matrix(c(1, 2, 3), nrow = 1, byrow = TRUE))

        plot(apply(dat_real, 2, mean), apply(dat_real, 2, sd),
            main = "mean vs sd", xlab = "Feature mean", ylab = "Feature std"
        )
        points(apply(dat_generated, 2, mean), apply(dat_generated, 2, sd), col = "red")
        legend("topright", pch = c(1, 1), col = c("black", "red"), legend = c("Real", "Generated"))

        plot(-log10(DEA_real$p.val[genes]), -log10(DEA_generated$p.val[genes]),
            xlab = "Real", ylab = "Generated",
            main = paste("-log10 pvalues, ccc=", res["ccc_log10pvalue"])
        )
        abline(a = 0, b = 1)

        plot(DEA_real$log2.FC[DEA_real$id.list], DEA_generated$log2.FC[DEA_real$id.list],
            xlab = "Real", ylab = "Generated",
            main = paste("log2FC of real DE genes, ccc =", res["ccc_log2FC"])
        )
        abline(a = 0, b = 1)

        layout(1)
    }

    return(res)
}

UMAP_eval <- function(real_df, generated_df,
                      group_col = NULL,
                      log = TRUE,
                      failure = c("replace", "remove")) {
    failure <- match.arg(failure)

    # Extract group labels (if provided)
    if (!is.null(group_col) && group_col %in% colnames(real_df)) {
        groups_real <- as.character(real_df[[group_col]])
        groups_generated <- as.character(generated_df[[group_col]])
        dat_real <- real_df[, setdiff(colnames(real_df), group_col), drop = FALSE]
        dat_generated <- generated_df[, setdiff(colnames(generated_df), group_col), drop = FALSE]
    } else {
        dat_real <- real_df
        dat_generated <- generated_df
        groups_real <- rep(NA, nrow(dat_real))
        groups_generated <- rep(NA, nrow(dat_generated))
    }

    # Log transform if not already
    if (!log) {
        dat_real <- log2(dat_real + 1)
        dat_generated <- log2(dat_generated + 1)
    }

    # Handle zero-SD features in generated data
    if (failure == "remove") {
        keep_cols <- apply(dat_generated, 2, sd) != 0
        dat_real <- dat_real[, keep_cols, drop = FALSE]
        dat_generated <- dat_generated[, keep_cols, drop = FALSE]
    } else if (failure == "replace") {
        zero_sd_cols <- apply(dat_generated, 2, sd) == 0
        dat_generated[, zero_sd_cols] <- dat_real[, zero_sd_cols]
    }

    # Combine and run UMAP
    dat_combine <- rbind(dat_real, dat_generated)
    datatype <- rep(c("Real", "Generated"), times = c(nrow(dat_real), nrow(dat_generated)))
    groups_combine <- c(groups_real, groups_generated)

    umap_fit <- umap::umap(as.matrix(dat_combine))

    plot_df <- as.data.frame(umap_fit$layout)
    colnames(plot_df) <- c("UMAP1", "UMAP2")
    plot_df$datatype <- datatype
    plot_df$groups <- groups_combine

    p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2, color = datatype)) +
        geom_point(aes(shape = groups)) +
        labs(title = "UMAP plot of Real vs Generated") +
        theme_bw() +
        theme(legend.position = "bottom")

    return(list(plot_df = plot_df, p_umap = p))
}

cluster_eval <- function(dat_real, dat_generated,
                         groups_real = NULL, groups_generated = NULL,
                         log = TRUE, failure = c("replace", "remove"),
                         plot = FALSE) {
    if (!log) {
        dat_real <- log2(dat_real + 1)
        dat_generated <- log2(dat_generated + 1)
    }

    if (failure == "remove") {
        keep_cols <- apply(dat_generated, 2, sd) != 0
        dat_real <- dat_real[, keep_cols, drop = FALSE]
        dat_generated <- dat_generated[, keep_cols, drop = FALSE]
    } else if (failure == "replace") {
        zero_sd_cols <- apply(dat_generated, 2, sd) == 0
        dat_generated[, zero_sd_cols] <- dat_real[, zero_sd_cols]
    }

    dat_combine <- rbind(dat_real, dat_generated)
    datatype <- rep(c("Real", "Generated"), c(nrow(dat_real), nrow(dat_generated)))

    # 构造组标签
    if (is.null(groups_real) || is.null(groups_generated)) {
        no_clusters <- 2
        groups_combine <- NULL
    } else {
        groups_combine <- factor(c(as.character(groups_real), as.character(groups_generated)))
        no_clusters <- length(unique(groups_combine))
    }

    d <- dist(dat_combine, method = "euclidean")
    fit <- hclust(d, method = "ward.D2")
    clusters <- cutree(fit, k = no_clusters)

    if (is.null(groups_combine)) {
        if (plot) {
            plot(fit, labels = FALSE, main = "Hierarchical Clustering Dendrogram")
            rect.hclust(fit, k = no_clusters, border = "red")
            df <- data.frame(dataset = datatype, cluster = as.factor(clusters), sample = 1:nrow(dat_combine))
            print(
                ggplot(df, aes(x = sample, y = dataset, color = cluster)) +
                    geom_point() +
                    labs(x = "Sample index", y = "Dataset")
            )
        }
        return(1 - abs(round(aricode::ARI(datatype, clusters), 4)))
    } else {
        if (plot) {
            plot(fit, labels = FALSE, main = "Hierarchical Clustering Dendrogram")
            rect.hclust(fit, k = no_clusters, border = "red")
            df <- data.frame(
                groups = as.factor(groups_combine), dataset = datatype,
                cluster = as.factor(clusters), sample = 1:nrow(dat_combine)
            )
            print(
                ggplot(df, aes(x = sample, y = groups, color = cluster)) +
                    geom_point() +
                    facet_wrap(vars(dataset)) +
                    scale_y_discrete(name = "Groups") + # 👈 强制设为离散
                    labs(x = "Sample index")
            )
        }
        return(round(aricode::ARI(groups_combine, clusters), 4))
    }
}

heatmap_eval <- function(dat_real, dat_generated,
                         group_col = "groups",
                         log = TRUE,
                         main = "Heatmap of Real vs Generated") {
    # Apply log2(x + 1) transform if needed
    if (!log) {
        dat_real <- dat_real %>% mutate(across(-all_of(group_col), ~ log2(. + 1)))
        dat_generated <- dat_generated %>% mutate(across(-all_of(group_col), ~ log2(. + 1)))
    }

    # Get feature column names (excluding group column)
    expr_cols <- setdiff(colnames(dat_real), group_col)

    # Prepare data for plotting
    mat_real <- dat_real[, expr_cols]
    mat_generated <- dat_generated[, expr_cols]

    mat_real$dataset <- "Real"
    mat_generated$dataset <- "Generated"
    mat_real$sample_id <- seq_len(nrow(mat_real))
    mat_generated$sample_id <- seq_len(nrow(mat_generated))

    combined <- bind_rows(mat_real, mat_generated)

    dat_long <- combined %>%
        pivot_longer(cols = all_of(expr_cols), names_to = "gene", values_to = "expression")

    p <- ggplot(dat_long, aes(x = gene, y = sample_id, fill = expression)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "deeppink") +
        facet_wrap(vars(dataset), scales = "free_y") +
        theme_minimal() +
        labs(title = main, x = "Genes", y = "Samples", fill = "expression") +
        theme(
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank()
        )

    return(p)
}
