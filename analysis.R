library(GEOquery)
library(data.table)
library(xml2)
library(dplyr)
library(tidyverse)
library(limma)


# 1. LOAD GEO SERIES MATRIX (GSE42133)

gse <- getGEO("GSE42133", GSEMatrix = TRUE)[[1]]

print(gse)

expr <- exprs(gse)
print(expr)

summary(as.vector(expr))
hist(as.vector(expr), breaks = 100,
     main = "Distribution of Expression Values",
     xlab = "Expression")

meta <- pData(gse)
print(meta)

head(meta[, c("title", "source_name_ch1", "characteristics_ch1.1")])



# 2. PROCESS RAW GSM EXPRESSION FILES

annot_file <- "data/GPL10558-50081.txt"

annot_df <- fread(annot_file, sep = "\t", skip = "#", data.table = FALSE)
annot_df <- annot_df[, c("ID", "Symbol")]

gsm_files <- list.files(
  "data/GSE42133_family",
  pattern = "GSM.*-tbl-1.txt$",
  full.names = TRUE
)

cat("Found", length(gsm_files), "GSM files\n")

expr_list <- list()

for (f in gsm_files) {
  expr_df <- fread(f, sep = "\t", header = FALSE)
  colnames(expr_df) <- c("Probe_ID", "Expression")

  merged_df <- merge(expr_df, annot_df,
                     by.x = "Probe_ID", by.y = "ID", all.x = TRUE)

  gsm_id <- sub("-tbl-1.txt", "", basename(f))

  merged_df <- merged_df[, c("Probe_ID", "Symbol", "Expression")]
  colnames(merged_df)[3] <- gsm_id

  expr_list[[gsm_id]] <- merged_df
}

# Combine all samples
final_df <- expr_list[[1]]

for (i in 2:length(expr_list)) {
  final_df <- merge(final_df, expr_list[[i]],
                    by = c("Probe_ID", "Symbol"), all = TRUE)
}

write.csv(final_df, "output/expression_matrix_all_samples.csv", row.names = FALSE)
cat("Saved combined expression matrix.\n")



# 3. PARSE GEO XML → DIAGNOSIS LABELS (ASD / Control)

parse_geo_xml <- function(xml_file) {
  doc <- read_xml(xml_file)
  ns <- xml_ns(doc)

  samples <- xml_find_all(doc, "//d1:Sample", ns = ns)

  results <- lapply(samples, function(sample) {
    gsm_id <- xml_text(xml_find_first(sample, "./d1:Accession", ns))

    diagnosis <- xml_text(
      xml_find_first(sample,
                     './/d1:Characteristics[@tag="dx (diagnosis)"]', ns)
    ) |> trimws()

    if (!is.na(diagnosis) && diagnosis != "")
      data.frame(GSM_ID = gsm_id, Diagnosis = diagnosis)
    else
      NULL
  })

  results <- do.call(rbind, results[!sapply(results, is.null)])
  return(results)
}

xml_file <- "data/GSE42133_family/GSE42133_family.xml"
samples <- parse_geo_xml(xml_file)

print(samples)
print(samples |> count(Diagnosis))

write.csv(samples, "output/GSM_diagnosis.csv", row.names = FALSE)
cat("Saved GSM diagnosis mapping.\n")



# 4. SPLIT SAMPLES INTO ASD / CONTROL FOLDERS

annot_df <- fread("data/GPL10558-50081.txt", sep = "\t", skip = "#", data.table = FALSE)
annot_df <- annot_df[, c("ID", "Symbol")]

mapping <- fread("output/GSM_diagnosis.csv")

dir.create("data/ASD", showWarnings = FALSE)
dir.create("data/Controls", showWarnings = FALSE)

input_folder <- "data/GSE42133_family"

for (i in 1:nrow(mapping)) {
  gsm_id <- mapping$GSM_ID[i]
  diagnosis <- mapping$Diagnosis[i]

  filename <- file.path(input_folder, paste0(gsm_id, "-tbl-1.txt"))
  if (!file.exists(filename)) next

  out_folder <- ifelse(diagnosis == "ASD", "data/ASD", "data/Controls")
  out_path <- file.path(out_folder, paste0("expression_data_", gsm_id, ".csv"))

  expr_df <- fread(filename, sep = "\t", header = FALSE)
  colnames(expr_df) <- c("Probe_ID", "Expression")

  merged_df <- merge(expr_df, annot_df,
                     by.x = "Probe_ID", by.y = "ID", all.x = TRUE)

  fwrite(merged_df, out_path)
  cat("Saved:", out_path, "\n")
}



# 5. MERGE ASD + CONTROL INTO ONE LONG FORMAT FILE

metadata <- read_csv("output/GSM_diagnosis.csv")

load_sample <- function(gsm, dx) {
  folder <- ifelse(dx == "ASD", "data/ASD", "data/Controls")
  file_path <- file.path(folder, paste0("expression_data_", gsm, ".csv"))

  if (!file.exists(file_path)) return(NULL)

  read_csv(file_path, show_col_types = FALSE) |>
    mutate(Sample = gsm, Condition = dx)
}

long_list <- lapply(1:nrow(metadata), function(i) {
  load_sample(metadata$GSM_ID[i], metadata$Diagnosis[i])
})

merged_df <- bind_rows(long_list) |>
  select(Probe_ID, Symbol, Expression, Sample, Condition)

write_csv(merged_df, "output/merged_data_long_format.csv")
cat("Saved merged long-format dataset.\n")



# 6. DEG ANALYSIS WITH LIMMA + ML READY DATA

df <- read_csv("output/merged_data_long_format.csv")

df_expr <- df |>
  filter(!is.na(Symbol)) |>
  group_by(Symbol, Sample) |>
  summarise(Expression = mean(Expression, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = Sample, values_from = Expression) |>
  column_to_rownames("Symbol")

conditions <- df |> distinct(Sample, Condition) |> deframe()
df_expr <- df_expr[, names(conditions)]

colData <- data.frame(
  Sample = names(conditions),
  Condition = factor(conditions, levels = c("Control", "ASD"))
)

design <- model.matrix(~ Condition, data = colData)

fit <- lmFit(df_expr, design) |> eBayes()
limma_results <- topTable(fit, coef = "ConditionASD", number = Inf) |>
  rownames_to_column("Gene") |>
  select(Gene, logFC, P.Value, adj.P.Val) |>
  rename(Log2FC = logFC, p_value = P.Value, adj_pval = adj.P.Val)

write_csv(limma_results, "output/DEG_results_limma.csv")


#  Volcano Plot
volcano_data <- limma_results |>
  mutate(significant = adj_pval < 0.05)

ggplot(volcano_data, aes(Log2FC, -log10(adj_pval), color = significant)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "blue") +
  theme_minimal() +
  labs(title = "Volcano Plot (limma)",
       x = "Log2 Fold Change",
       y = "-log10(Adjusted P-value)")

ggsave("output/volcano_plot_limma.png", width = 8, height = 6)

#  ML Dataset
filtered <- limma_results |>
  filter(abs(Log2FC) > 0.2, adj_pval < 0.1)

genes <- unique(filtered$Gene)

expr_mat <- as.matrix(df_expr)[genes, ]
expr_for_ml <- t(expr_mat)

ml_data <- data.frame(Sample = rownames(expr_for_ml),
                      expr_for_ml,
                      Condition = colData$Condition)

write_csv(ml_data, "output/ML_dataset.csv")

cat("Pipeline complete.\n")
