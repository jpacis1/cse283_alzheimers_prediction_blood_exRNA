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

## Sample Sequencing Data (`silver_seq/`)

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