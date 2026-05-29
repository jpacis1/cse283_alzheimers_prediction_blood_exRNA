# Classifier to Predict Alzheimer's Disease through Extracellular RNA from Patient Blood

Predicting Alzheimer's Disease (AD) using blood extracellular RNA sequencing (SILVER-seq) data.

---

## Dataset & Gene Lists

The `ad_gwas_hits` directory contains two distinct gene lists curated by the **Alzheimer's Disease Sequencing Project (ADSP) Gene Verification Committee (GVC)**. 

> **Source:** Gene lists were downloaded from the [ADSP GVC Top Hits List](https://adsp.niagads.org/gvc-top-hits-list/).

### File Descriptions

* **`AD_GWAS_hits_gene_verified.csv`** Contains genes compiled by the ADSP with evidence suggesting that the gene or genetic signal influences or causes AD. 
    * *Note: The quality of evidence presented in the associated publications is highly variable.*
    
* **`AD_GWAS_hits_risk.csv`** Contains validated causal genes for AD risk or protection, identified via a rigorous literature review by the ADSP GVC.

---

## Sample Sequencing Data (`silver_seq`)

This folder contains the core expression data and clinical metadata required for building the classifier.

### 1. Count Matrix (`silver_seq_counts.txt`)
* **Dimensions:** 60,675 genes × 115 samples.
* **Format:** Raw integer counts generated from `featureCounts`.
* **Note on sparsity:** Approximately 50% of the entries are zero. High sparsity is typical for blood RNA-seq data since most genes are either lowly expressed or completely absent in any given sample.
* **Library Sizes:** Varies significantly, ranging from **102K to 899K reads**. 
    > **TO CONSIDER:** Due to the wide variance in library sizes, proper normalization is essential prior to performing any comparative analysis or modeling.

### 2. Metadata (`silver_seq_metadata.xlsx`)
Consists of 115 rows (one per sample) and 10 clinical fields. The primary fields of interest for the classifier are `donor_group` (AD vs. N), `braak_stage`, and the three **ApoE** fields.

#### Key Modeling & Data Considerations:

* **Ordinal Labels (`donor_status_score`):** This field features a finer clinical gradient (`N--`, `N-`, `N`, `A`, `AD+`, `AD++`). We can possibly consider utilizing this as a soft label or an ordinal covariate rather than relying strictly on the binary AD/N classification. But, this may also just be doing too much, N/AD seems good enough for the scope of this project. 
* **Longitudinal Design & Correlation:** The dataset is longitudinal; individual donors contribute between **3 and 8 samples** collected across multiple years (2000–2014). We should consider that samples from the same donor are statistically correlated. **Do not treat them as independent data points** during cross-validation or modeling.
* **ApoE Imbalance:** The ApoE distribution is heavily imbalanced: 33 of 74 AD samples carry ApoE4, compared to only 9 of 41 normal samples. While we should expect this when we consider AD biology (ApoE4 is the strongest genetic risk factor), including ApoE status as a feature requires caution so it does not dominate the model and mask the underlying transcriptomic signal.
* **Missing Demographics:** We **do not have sex or age columns** in this metadata file, which is a limitation of the dataset we have access to.

--- 
## AD classifer pipeline
Run the pipeline through the following command:

(this is a call i ran to make sure the pipeline was working, RESULTS ARE BAD lol but they are in the results section of the pipeline directory)
```
python ad_classifier_pipeline.py \
    --counts   ../silver_seq/silver_seq_counts.txt \
    --metadata ../silver_seq/silver_seq_metadata.xlsx \
    --gwas     ../ad_gwas_hits/AD_GWAS_hits.csv \
    --output_dir results/ \
    --norm_method cpm_log2 \
    --classifier logistic
```

## Pipeline Execution Steps

1. **GWAS Parsing:** Reads your CSV with `header=1` (skipping the table title row), then auto-detects the gene column by searching for "gene" in the header. Multi-gene cells (e.g., `HLA-DRB1/HLA-DRB5` or `MS4A6A/MS4A4E`) are automatically split on `/` and `,`.
2. **Low-Count Filter:** Removes genes with fewer than 10 reads in at least 10% of samples ($\approx$ 12 samples here). This filters the matrix down from 60,675 to **~22,150 genes** before normalization—a standard practice to eliminate background noise from very sparse genes.
3. **Normalization:** Defaults to $log_2(\text{CPM}+1)$, which is fast and robust. Pass `--norm_method vst` to use `PyDESeq2`'s variance-stabilizing transform instead, which handles the count-mean relationship more effectively but increases processing time.
4. **ENSEMBL → Symbol Mapping:** Calls the `mygene.info` REST API in batches to translate ENSEMBL IDs to HGNC symbols for matching against GWAS gene names. 
    * *Note: Requires internet access on the first run. If the API is unavailable, the pipeline falls back to using all expressed genes with a warning. If this happens, it is recommended to pre-map your GWAS ENSEMBL IDs manually and pass them in the gene column instead.*
5. **Feature Matrix Construction:** Combines normalized GWAS gene expression (columns prefixed with `expr_`) with available clinical covariates: `apoe4_carrier`, `apoe4_dose`, `year_sample`, and `braak_stage`. `sex` and `age` will be picked up automatically if those column names exist in your metadata.
6. **Leave-Donor-Out Cross-Validation (CV):** Iterates over all 24 donors, holding each out in turn. This is the mathematically correct CV scheme for longitudinal data, as it prevents samples from the same donor from leaking into both train and test sets. Each fold reports AUC and average precision (AP).

---

## Outputs

All outputs are written to the directory specified by `--output_dir` (default is `results/`):

| File Name | Contents |
| :--- | :--- |
| `silver_metrics.csv` | Pooled AUC, AP, and mean/std fold AUC |
| `silver_fold_metrics.csv` | Per-fold AUC and AP metrics |
| `silver_predictions.csv` | Per-sample predicted probabilities |
| `silver_feature_importances.csv` | Mean absolute coefficient per feature |
| `silver_roc_pr.png` | Plot of ROC + PR curves |
| `silver_fold_aucs.png` | Box/strip plot of per-fold AUCs |
| `silver_top_features.png` | Visualization of top 30 features (expression vs. clinical) |
| `silver_donor_scores.png` | Per-donor mean predicted scores |
| `silver_library_sizes.png` | QC plot: library size distribution |

> **NOTE ON STATISTICS:** With only 24 donors (15 AD, 9 N), per-fold AUCs will naturally be noisy since some folds will only contain 3–5 test samples. The **pooled AUC** across all held-out samples serves as the more reliable summary metric.

---

## Conda environment
I can attach a yml file for my conda environment that I used to run the pipeline, but I made my own conda environment to run this pipeline and installed the necessary packages

Once I made the environment, I think this is all that is needed to be installed to run the classifier pipeline.
```
mamba install numpy pandas matplotlib seaborn scipy scikit-learn -c conda-forge
mamba install openpyxl
```
