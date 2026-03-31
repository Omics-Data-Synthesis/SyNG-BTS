Cinput_files <- c(
  COAD = "syng_bts/data/case/COAD_5-2/COADPositive_5-2.csv",
  LAML = "syng_bts/data/case/LAML_5-2/LAMLPositive_5-2.csv",
  PAAD = "syng_bts/data/case/PAAD_5-2/PAADPositive_5-2.csv",
  READ = "syng_bts/data/case/READ_5-2/READPositive_5-2.csv",
  SKCM = "syng_bts/data/case/SKCM_5-2/SKCMPositive_5-2.csv"
)

summary_list <- lapply(names(input_files), function(project_id) {
  file_path <- input_files[[project_id]]
  data_mat <- read.csv(file_path, check.names = FALSE)

  data.frame(
    Project_ID = project_id,
    n_features = ncol(data_mat),
    n_samples = nrow(data_mat),
    ratio = ncol(data_mat) / nrow(data_mat),
    stringsAsFactors = FALSE
  )
})

summary_df <- do.call(rbind, summary_list)

output_path <- "evaluations/case_dimensions_summary.csv"
write.csv(summary_df, output_path, row.names = FALSE)

message("Summary file saved to: ", output_path)

max_features_row <- summary_df[which.max(summary_df$n_features), ]
min_features_row <- summary_df[which.min(summary_df$n_features), ]
max_samples_row <- summary_df[which.max(summary_df$n_samples), ]
min_samples_row <- summary_df[which.min(summary_df$n_samples), ]
max_ratio_row <- summary_df[which.max(summary_df$ratio), ]
min_ratio_row <- summary_df[which.min(summary_df$ratio), ]

message(
  "Highest n_features: ",
  max_features_row$Project_ID,
  " (",
  max_features_row$n_features,
  ")"
)
message(
  "Lowest n_features: ",
  min_features_row$Project_ID,
  " (",
  min_features_row$n_features,
  ")"
)
message(
  "Highest n_samples: ",
  max_samples_row$Project_ID,
  " (",
  max_samples_row$n_samples,
  ")"
)
message(
  "Lowest n_samples: ",
  min_samples_row$Project_ID,
  " (",
  min_samples_row$n_samples,
  ")"
)
message(
  "Highest ratio (n_features/n_samples): ",
  max_ratio_row$Project_ID,
  " (",
  round(max_ratio_row$ratio, 4),
  ")"
)
message(
  "Lowest ratio (n_features/n_samples): ",
  min_ratio_row$Project_ID,
  " (",
  round(min_ratio_row$ratio, 4),
  ")"
)
