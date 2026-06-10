
# AI-Driven Prediction of Autism Spectrum Disorder Using Transcriptional Biomarkers

Find genes whose RNA-Seq expression profiles serve as biomarkers for distinguishing ASD vs. Control individuals and build a reproducible classifier using transcriptomic signatures.




## Authors

- [@Sohel](https://github.com/Sohel-404)


## Dataset

[@GSE42133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42133) - Gene Expression Omnibus (GEO)

This dataset contains whole blood transcriptome profiles for ASD and matched Control subjects.
## Workflow Overview

1. Data Retrieval & Preprocessing (R)
- Download GSE42133 from GEO and extract expression + metadata.
- Load GPL10558 annotation and map probe IDs to gene symbols.
- Parse XML to extract ASD/Control diagnosis for each GSM sample.
- Process raw GSM expression files and attach gene annotations.
- Save per-sample annotated expression tables for downstream merging.

2. Data Integration
- Merge all GSM files into a unified expression matrix.
- Combine gene, probe, sample ID, and condition (ASD/Control).
- Convert to long-format structure for downstream transformation.
- Generate final gene × sample matrix

3. Differential Expression Analysis (R)
- Aggregate probe-level values to gene-level means.
- Build design matrix comparing Control vs ASD groups.
- Run limma to compute Log2FC and adjusted p-values.
- Export DEG tables and volcano plots, and filter genes for ML dataset.

4. Machine Learning
- Load ML dataset and perform train–test split.
- Apply normalization (z-score), variance filtering, and feature selection.
- Address class imbalance using SMOTE to balance ASD and Control samples.
- Optimize XGBoost hyperparameters with Hyperopt and train final model.

5. Model Interpretation
- Generate SHAP values using TreeExplainer for trained XGBoost model.
- Produce SHAP summary, feature impact, and importance visualizations.
- Identify top ASD-associated biomarker genes contributing to classification.
- Support biological interpretation through model-transparent gene scoring.




## Requirements

**R Dependencies**
- GEOquery
- data.table
- xml2
- dplyr
- readr
- tidyverse
- limma

```bash
install.packages(c(
  "GEOquery", "data.table", "xml2", "dplyr", "readr", "tidyverse", "limma"
))

```

**Python Dependencies:**
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- imbalanced-learn
- hyperopt
- shap

```bash
pip install -r requirements.txt
```


