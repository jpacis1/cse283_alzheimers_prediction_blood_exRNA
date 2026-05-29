"""
AD Blood exRNA Classifier Pipeline
===================================
Preprocessing and classification pipeline for predicting Alzheimer's Disease
from blood exRNA (SILVER-seq) data, integrating AD GWAS gene sets with
clinical covariates (ApoE status, age, sex when available).

Usage
-----
    python ad_classifier_pipeline.py \
        --counts silver_seq_counts.txt \
        --metadata silver_seq_metadata.xlsx \
        --gwas ad_gwas_hits.csv \
        [--output_dir results/] \
        [--norm_method vst]          # vst | cpm_log2
        [--classifier logistic]      # logistic | rf | elasticnet
        [--min_count 10] \
        [--min_samples_frac 0.1] \
        [--seed 42]

GWAS CSV format expected
------------------------
The file must contain at minimum a gene-name column. The parser looks for
a column whose header contains "gene" (case-insensitive), e.g.:
    "Reported Gene/ Closest gene"  →  parsed as gene names
Multiple genes per cell separated by "/" or "," are split automatically.
ENSEMBL IDs in the count matrix are mapped to HGNC symbols via a bundled
lookup; if no match, the gene is silently dropped from the feature set.
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  GWAS gene-set loading
# ═══════════════════════════════════════════════════════════════════════════

def load_gwas_genes(gwas_csv: str) -> set:
    """
    Read a pre-wrangled GWAS CSV with a 'gene_symbol' column
    (output of load_gwas_combined).
    """
    log.info(f"Loading GWAS gene list from: {gwas_csv}")
    df = pd.read_csv(gwas_csv)
    
    if "gene_symbol" not in df.columns:
        raise ValueError(
            f"Expected a 'gene_symbol' column but found: {df.columns.tolist()}\n"
            "Make sure you're passing the wrangled output CSV, not the raw ADSP file."
        )
    
    genes = set(df["gene_symbol"].dropna().str.strip())
    genes.discard("")
    log.info(f"  Loaded {len(genes)} unique GWAS gene symbols")
    return genes


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Count matrix loading and sample alignment
# ═══════════════════════════════════════════════════════════════════════════

def load_counts(counts_path: str) -> pd.DataFrame:
    """
    Load featureCounts output.
    Rows = ENSEMBL gene IDs, columns = sample IDs.
    Returns counts as integers.
    """
    log.info(f"Loading count matrix from: {counts_path}")
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    log.info(f"  Count matrix: {counts.shape[0]} genes × {counts.shape[1]} samples")
    return counts.astype(int)


def load_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load sample metadata from Excel.
    Returns DataFrame indexed by sample_id_alias.
    """
    log.info(f"Loading metadata from: {meta_path}")
    meta = pd.read_excel(meta_path)
    meta = meta.set_index("sample_id_alias")
    log.info(f"  Metadata: {meta.shape[0]} samples, columns: {meta.columns.tolist()}")
    return meta


def align_samples(counts: pd.DataFrame, meta: pd.DataFrame):
    """
    Keep only samples present in both count matrix and metadata,
    in the same order.
    """
    common = counts.columns.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between count matrix and metadata!")
    n_drop = counts.shape[1] - len(common)
    if n_drop > 0:
        log.warning(f"  Dropping {n_drop} samples not in metadata")
    counts = counts[common]
    meta = meta.loc[common]
    log.info(f"  Aligned: {len(common)} samples retained")
    return counts, meta


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Low-count gene filtering
# ═══════════════════════════════════════════════════════════════════════════

def filter_low_counts(counts: pd.DataFrame,
                      min_count: int = 10,
                      min_samples_frac: float = 0.10) -> pd.DataFrame:
    """
    Remove genes where fewer than `min_samples_frac` fraction of samples
    have >= `min_count` reads. This is the standard edgeR/DESeq2 pre-filter.
    """
    n_samples = counts.shape[1]
    min_samples = max(1, int(np.ceil(min_samples_frac * n_samples)))
    mask = (counts >= min_count).sum(axis=1) >= min_samples
    filtered = counts.loc[mask]
    log.info(
        f"  Low-count filter (>= {min_count} in >= {min_samples} samples): "
        f"{counts.shape[0]} → {filtered.shape[0]} genes retained"
    )
    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_cpm_log2(counts: pd.DataFrame,
                       pseudo: float = 1.0) -> pd.DataFrame:
    """
    CPM normalisation followed by log2(CPM + pseudo).
    Simple and effective; pseudo-count avoids log(0).
    Genes on rows, samples on columns.
    """
    lib_sizes = counts.sum(axis=0)
    cpm = counts.divide(lib_sizes, axis=1) * 1e6
    log2cpm = np.log2(cpm + pseudo)
    log.info("  Normalisation: log2(CPM + 1)")
    return log2cpm


def normalize_vst(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    DESeq2-style variance-stabilising transformation via PyDESeq2.
    Falls back to log2-CPM if PyDESeq2 is unavailable or fails.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference

        log.info("  Normalisation: DESeq2 VST (via PyDESeq2)")
        # PyDESeq2 expects samples on rows
        counts_T = counts.T.copy()
        counts_T.index.name = "sample"

        meta_sub = meta[["donor_group"]].copy()
        meta_sub.index.name = "sample"

        dds = DeseqDataSet(
            counts=counts_T,
            metadata=meta_sub,
            design_factors="donor_group",
            refit_cooks=True,
            inference=DefaultInference(n_cpus=1),
        )
        dds.deseq2()
        dds.vst(use_design=False)
        vst_mat = dds.layers["vst_counts"]           # samples × genes
        vst_df = pd.DataFrame(
            vst_mat.T,                               # back to genes × samples
            index=counts.index,
            columns=counts.columns,
        )
        return vst_df
    except Exception as e:
        log.warning(f"  VST failed ({e}); falling back to log2-CPM")
        return normalize_cpm_log2(counts)


def normalize(counts: pd.DataFrame, meta: pd.DataFrame,
              method: str = "cpm_log2") -> pd.DataFrame:
    if method == "vst":
        return normalize_vst(counts, meta)
    return normalize_cpm_log2(counts)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ENSEMBL → HGNC symbol mapping
# ═══════════════════════════════════════════════════════════════════════════

def build_ensembl_to_symbol_map(ensembl_ids: pd.Index) -> dict:
    """
    Map ENSEMBL gene IDs to HGNC symbols using the official mygene client.
    The client uses POST and handles batching internally, avoiding the URL
    length limits that broke a previous GET-based implementation.
    Returns an empty dict if the API is unavailable (offline run).
    """
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        ids = list(ensembl_ids)
        results = mg.querymany(
            ids, scopes="ensembl.gene", fields="symbol",
            species="human", returnall=False, verbose=False,
        )
        mapping = {
            r["query"]: r["symbol"]
            for r in results
            if "symbol" in r and "notfound" not in r
        }
        log.info(f"  Mapped {len(mapping)} / {len(ids)} ENSEMBL IDs to symbols")
        return mapping
    except Exception as e:
        log.warning(f"  ENSEMBL→symbol lookup failed ({e}); using ENSEMBL IDs directly")
        return {}


def gwas_filter_expression(norm_expr: pd.DataFrame,
                            gwas_genes: set,
                            ensembl_map: dict) -> pd.DataFrame:
    """
    Subset normalised expression to rows matching any GWAS gene.
    Matching is done symbol-first (via ensembl_map), then raw index
    (in case the index already contains symbols).
    Returns a genes × samples DataFrame.
    """
    # Build reverse: gene symbol → list of ENSEMBL IDs
    symbol_to_ensembl = {}
    for eid, sym in ensembl_map.items():
        symbol_to_ensembl.setdefault(sym, []).append(eid)

    selected_rows = []
    matched_symbols = set()

    for gene in gwas_genes:
        eids = symbol_to_ensembl.get(gene, [])
        for eid in eids:
            if eid in norm_expr.index:
                selected_rows.append(eid)
                matched_symbols.add(gene)
        # Also check if the gene symbol is already the index (non-ENSEMBL matrix)
        if gene in norm_expr.index:
            selected_rows.append(gene)
            matched_symbols.add(gene)

    selected_rows = list(dict.fromkeys(selected_rows))  # deduplicate, preserve order
    log.info(
        f"  GWAS gene filter: {len(matched_symbols)} / {len(gwas_genes)} "
        f"GWAS genes found in expression matrix → {len(selected_rows)} feature rows"
    )

    if len(selected_rows) == 0:
        log.warning(
            "  No GWAS genes matched expression matrix. "
            "The ENSEMBL→symbol mapping may have failed. "
            "Using full normalised matrix as fallback (not recommended)."
        )
        return norm_expr

    return norm_expr.loc[selected_rows]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Clinical covariate encoding
# ═══════════════════════════════════════════════════════════════════════════

def encode_covariates(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Encode clinical covariates. Braak stage is EXCLUDED — it is a 
    post-mortem neuropathological measure that directly defines the 
    AD/N label and would constitute target leakage.
    """
    cov = pd.DataFrame(index=meta.index)

    if "apoe_carrier" in meta.columns:
        cov["apoe4_carrier"] = (meta["apoe_carrier"] == "apoe4").astype(float)

    if "apoe_dose" in meta.columns:
        dose_map = {"no_apoe4": 0, "apoe4": 1, "apoe44": 2}
        cov["apoe4_dose"] = meta["apoe_dose"].map(dose_map).fillna(0).astype(float)

    if "sex" in meta.columns:
        cov["sex_male"] = meta["sex"].str.lower().map(
            {"m": 1, "male": 1, "f": 0, "female": 0}
        ).fillna(0).astype(float)

    if "age" in meta.columns:
        cov["age"] = pd.to_numeric(meta["age"], errors="coerce")
        cov["age"] = cov["age"].fillna(cov["age"].mean())

    # year_sample intentionally excluded — collection-time batch variable,
    # not biology. AD samples skew later (mean 2006.8 vs 2004.6 for N), so
    # including it leaks cohort timing into the classifier.

    # braak_stage intentionally excluded — direct proxy for the label

    log.info(f"  Encoded covariates: {cov.columns.tolist()}")
    return cov.astype("float32")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Feature matrix assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_matrix(gwas_expr: pd.DataFrame,
                         covariates: pd.DataFrame) -> pd.DataFrame:
    """
    Transpose expression (genes → columns) and join clinical covariates.
    Returns a samples × features DataFrame.
    """
    # Samples × genes
    X_expr = gwas_expr.T.copy()
    X_expr.columns = [f"expr_{c}" for c in X_expr.columns]

    # Join covariates
    X = X_expr.join(covariates, how="inner")
    log.info(
        f"  Feature matrix assembled: {X.shape[0]} samples × {X.shape[1]} features "
        f"({X_expr.shape[1]} expression + {covariates.shape[1]} clinical)"
    )
    return X


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Leave-donor-out cross-validation
# ═══════════════════════════════════════════════════════════════════════════

def make_classifier(name: str, seed: int = 42):
    """
    Return a scikit-learn Pipeline with StandardScaler + chosen classifier.

    logistic   : L1-penalised logistic regression (sparse, interpretable)
    elasticnet : ElasticNet logistic (L1+L2 mix) — more stable than pure L1
    rf         : Random forest — non-parametric, handles interactions
    """
    if name == "logistic":
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            max_iter=1000, random_state=seed
        )
    elif name == "elasticnet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", C=0.1,
            l1_ratio=0.5, max_iter=2000, random_state=seed
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=3, random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier: {name}. Choose logistic | elasticnet | rf")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def leave_donor_out_cv(X, y, groups, classifier_name="logistic", seed=42, min_test_donors=2):
    """
    Leave-donor-out CV. To ensure each test fold contains both classes,
    we group donors rather than holding out one at a time. With 15 AD and
    9 N donors, we use n_splits=9 (matching the minority class size) so
    each fold gets ~1-2 donors per class in the test set.
    """
    donors = groups.unique()
    n_donors = len(donors)

    # Compute donor-level labels to determine safe n_splits
    donor_labels = y.groupby(groups).first()
    n_minority = donor_labels.value_counts().min()
    # n_splits capped at minority class donor count to guarantee both 
    # classes appear in every test fold
    n_splits = min(n_minority, 9)
    log.info(
        f"  {n_donors} donors ({(donor_labels==1).sum()} AD, {(donor_labels==0).sum()} N) "
        f"→ using {n_splits}-fold leave-donor-group-out CV"
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_y_true, all_y_prob, all_donors = [], [], []
    fold_aucs, fold_aps = [], []
    coef_accum = np.zeros(X.shape[1])
    # Per-fold signed coefficients (linear models only); rows = folds.
    # Stays empty for non-linear models like RF.
    signed_coefs_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        donor_test = groups.iloc[test_idx]

        model = make_classifier(classifier_name, seed)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        n_classes_in_test = len(np.unique(y_test))
        if n_classes_in_test < 2:
            auc = float("nan")
            ap  = float("nan")
            log.warning(
                f"    Fold {fold_i+1:2d} | single-class test set "
                f"— AUC/AP undefined, excluded from fold summary"
            )
        else:
            auc = roc_auc_score(y_test, y_prob)
            ap  = average_precision_score(y_test, y_prob)
            log.info(
                f"    Fold {fold_i+1:2d} | donors={donor_test.nunique()} "
                f"n_test={len(y_test)} | AUC={auc:.3f}  AP={ap:.3f}"
            )

        fold_aucs.append(auc)
        fold_aps.append(ap)
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_donors.extend(donor_test.tolist())

        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            signed = clf.coef_[0]
            coef_accum += np.abs(signed)
            signed_coefs_per_fold.append(signed)
        elif hasattr(clf, "feature_importances_"):
            coef_accum += clf.feature_importances_

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    overall_auc = roc_auc_score(all_y_true, all_y_prob)
    overall_ap  = average_precision_score(all_y_true, all_y_prob)

    valid_fold_aucs = [a for a in fold_aucs if not np.isnan(a)]

    log.info(f"\n  === CV Results ===")
    log.info(f"  {len(valid_fold_aucs)}/{n_splits} folds had ≥2 classes in test set")

    if valid_fold_aucs:
        log.info(
            f"  Per-fold AUC: mean={np.mean(valid_fold_aucs):.3f}  "
            f"std={np.std(valid_fold_aucs):.3f}  "
            f"[{np.min(valid_fold_aucs):.3f}, {np.max(valid_fold_aucs):.3f}]"
        )
    else:
        log.warning("  No valid per-fold AUCs — all test folds were single-class")

    log.info(f"  Pooled AUC (all test predictions): {overall_auc:.3f}")
    log.info(f"  Pooled Average Precision:          {overall_ap:.3f}")

    mean_coef = coef_accum / n_splits
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_coef": mean_coef,
    }).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)

    # Sign-stability table for linear models: did each feature's coefficient
    # point the same direction in every fold? A feature that flips signs is
    # picking up donor-specific noise rather than disease signal.
    sign_df = None
    if signed_coefs_per_fold:
        signed_mat = np.vstack(signed_coefs_per_fold)        # (n_folds, n_features)
        n_pos = (signed_mat > 0).sum(axis=0)
        n_neg = (signed_mat < 0).sum(axis=0)
        n_nz  = n_pos + n_neg
        # Fraction of nonzero folds that agree with the majority sign.
        # 1.0 = perfectly consistent direction; 0.5 = coin-flip.
        majority = np.maximum(n_pos, n_neg)
        with np.errstate(divide="ignore", invalid="ignore"):
            sign_consistency = np.where(n_nz > 0, majority / n_nz, np.nan)
        sign_df = pd.DataFrame({
            "feature":          X.columns,
            "mean_signed_coef": signed_mat.mean(axis=0),
            "mean_abs_coef":    mean_coef,
            "n_folds_pos":      n_pos,
            "n_folds_neg":      n_neg,
            "n_folds_nonzero":  n_nz,
            "sign_consistency": sign_consistency,
        }).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)

        # Log diagnostic on the top features
        top = sign_df.head(15)
        log.info("\n  === Sign stability of top 15 features (by mean |coef|) ===")
        log.info(f"  {'feature':<25} {'mean_signed':>12} {'+folds':>7} {'-folds':>7} {'consistency':>12}")
        for _, r in top.iterrows():
            log.info(
                f"  {r['feature']:<25} {r['mean_signed_coef']:>12.4f} "
                f"{int(r['n_folds_pos']):>7d} {int(r['n_folds_neg']):>7d} "
                f"{r['sign_consistency']:>12.2f}"
            )

    results = {
        "fold_aucs":     fold_aucs,
        "fold_aps":      fold_aps,
        "overall_auc":   overall_auc,
        "overall_ap":    overall_ap,
        "n_folds":       n_splits,
        "n_valid_folds": len(valid_fold_aucs),
    }
    return results, all_y_true, all_y_prob, np.array(all_donors), coef_df, sign_df


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Visualization
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = {"AD": "#D85A30", "N": "#1D9E75"}


def plot_roc_pr(y_true, y_prob, output_dir, prefix="silver"):
    """ROC and PR curves on a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color="#3266ad", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.08, color="#3266ad")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC — leave-donor-out CV")
    axes[0].legend(loc="lower right", fontsize=10)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    axes[1].plot(rec, prec, color="#533AB7", lw=2, label=f"AP = {ap:.3f}")
    axes[1].axhline(baseline, color="k", lw=0.8, ls="--", alpha=0.5,
                    label=f"Baseline = {baseline:.2f}")
    axes[1].fill_between(rec, prec, alpha=0.08, color="#533AB7")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision–Recall — leave-donor-out CV")
    axes[1].legend(loc="upper right", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}_roc_pr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


def plot_fold_aucs(fold_aucs, output_dir, prefix="silver"):
    """Strip + box plot of per-fold AUCs."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(fold_aucs))
    ax.scatter(np.ones(len(fold_aucs)) + jitter, fold_aucs,
               color="#3266ad", alpha=0.7, s=40, zorder=3)
    bp = ax.boxplot(fold_aucs, positions=[1], widths=0.25,
                    patch_artist=True, zorder=2,
                    boxprops=dict(facecolor="#B5D4F4", alpha=0.6),
                    medianprops=dict(color="#185FA5", lw=2),
                    whiskerprops=dict(color="#888780"),
                    capprops=dict(color="#888780"),
                    flierprops=dict(marker="o", color="#888780", alpha=0.5))
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5, label="Random (0.5)")
    ax.set_xticks([1]); ax.set_xticklabels(["Leave-donor-out"])
    ax.set_ylabel("AUC (per donor fold)")
    ax.set_title("Per-fold AUC distribution")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}_fold_aucs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


def plot_top_features(coef_df, n=30, output_dir=".", prefix="silver"):
    """Horizontal bar chart of top feature importances."""
    top = coef_df.head(n).copy()
    top["label"] = top["feature"].str.replace("expr_", "", regex=False)

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.28)))
    colors = ["#D85A30" if "expr_" in f else "#185FA5"
              for f in top["feature"]]
    ax.barh(top["label"][::-1], top["mean_abs_coef"][::-1],
            color=colors[::-1], height=0.7)
    ax.set_xlabel("Mean |coefficient| / importance across folds")
    ax.set_title(f"Top {n} features")
    ax.spines[["top", "right"]].set_visible(False)

    patches = [
        mpatches.Patch(color="#D85A30", label="Expression (GWAS gene)"),
        mpatches.Patch(color="#185FA5", label="Clinical covariate"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}_top_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


def plot_prediction_scores(y_true, y_prob, donors, output_dir, prefix="silver"):
    """
    Dot plot of predicted AD probability per donor (mean of test samples),
    coloured by true label.
    """
    df = pd.DataFrame({"donor": donors, "y_true": y_true, "y_prob": y_prob})
    donor_agg = df.groupby("donor").agg(
        mean_prob=("y_prob", "mean"),
        label=("y_true", "first")
    ).reset_index()
    donor_agg["group"] = donor_agg["label"].map({1: "AD", 0: "N"})
    donor_agg = donor_agg.sort_values(["group", "mean_prob"])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for grp, col in PALETTE.items():
        sub = donor_agg[donor_agg["group"] == grp]
        ax.scatter(range(len(sub)), sub["mean_prob"], color=col,
                   label=grp, s=60, alpha=0.85, zorder=3)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Donor (sorted within group)")
    ax.set_ylabel("Mean predicted AD probability")
    ax.set_title("Predicted scores per donor (leave-donor-out test set)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}_donor_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


def plot_library_sizes(counts: pd.DataFrame, meta: pd.DataFrame,
                       output_dir=".", prefix="silver"):
    """Library size distribution coloured by group."""
    lib = counts.sum(axis=0).rename("lib_size").to_frame()
    lib = lib.join(meta[["donor_group"]])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for grp, col in PALETTE.items():
        sub = lib[lib["donor_group"] == grp]["lib_size"]
        ax.hist(sub, bins=20, alpha=0.6, color=col, label=grp, edgecolor="none")
    ax.set_xlabel("Library size (total mapped reads)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Library size distribution by group")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}_library_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  Save outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results, coef_df, all_y_true, all_y_prob, all_donors,
                 output_dir, prefix="silver", sign_df=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary metrics
    metrics = {
        "overall_auc": results["overall_auc"],
        "overall_ap":  results["overall_ap"],
        "mean_fold_auc": float(np.nanmean(results["fold_aucs"])),
        "std_fold_auc":  float(np.nanstd(results["fold_aucs"])),
        "n_folds": results["n_folds"],
    }
    pd.DataFrame([metrics]).to_csv(out / f"{prefix}_metrics.csv", index=False)

    # Per-fold AUCs
    pd.DataFrame({
        "fold": range(1, len(results["fold_aucs"]) + 1),
        "auc":  results["fold_aucs"],
        "ap":   results["fold_aps"],
    }).to_csv(out / f"{prefix}_fold_metrics.csv", index=False)

    # Feature importances
    coef_df.to_csv(out / f"{prefix}_feature_importances.csv", index=False)

    # Signed-coefficient sign-stability table (linear models only)
    if sign_df is not None:
        sign_df.to_csv(out / f"{prefix}_coef_sign_stability.csv", index=False)

    # Per-sample predictions
    pd.DataFrame({
        "donor":  all_donors,
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }).to_csv(out / f"{prefix}_predictions.csv", index=False)

    log.info(f"  Saved all results to: {out}/")


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    counts_path: str,
    meta_path: str,
    gwas_csv: str,
    output_dir: str = "results",
    norm_method: str = "cpm_log2",
    classifier_name: str = "logistic",
    min_count: int = 10,
    min_samples_frac: float = 0.10,
    seed: int = 42,
    prefix: str = "silver",
):
    np.random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("AD CLASSIFIER PIPELINE")
    log.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    log.info("\n[1/8] Loading data")
    counts  = load_counts(counts_path)
    meta    = load_metadata(meta_path)
    gwas_genes = load_gwas_genes(gwas_csv)

    # ── Step 2: Align samples ──────────────────────────────────────────────
    log.info("\n[2/8] Aligning samples")
    counts, meta = align_samples(counts, meta)

    # ── Step 3: QC plot ────────────────────────────────────────────────────
    log.info("\n[3/8] QC — library sizes")
    plot_library_sizes(counts, meta, output_dir, prefix)

    # ── Step 4: Filter low-count genes ────────────────────────────────────
    log.info("\n[4/8] Filtering low-count genes")
    counts_filt = filter_low_counts(counts, min_count, min_samples_frac)

    # ── Step 5: Normalise ─────────────────────────────────────────────────
    log.info(f"\n[5/8] Normalising ({norm_method})")
    norm_expr = normalize(counts_filt, meta, method=norm_method)

    # ── Step 6: GWAS feature selection ────────────────────────────────────
    log.info("\n[6/8] GWAS gene filtering + feature assembly")
    ensembl_map = build_ensembl_to_symbol_map(norm_expr.index)
    gwas_expr   = gwas_filter_expression(norm_expr, gwas_genes, ensembl_map)
    covariates  = encode_covariates(meta)
    X           = build_feature_matrix(gwas_expr, covariates)

    # Align X and meta to same sample order
    common_samples = X.index.intersection(meta.index)
    X    = X.loc[common_samples]
    meta = meta.loc[common_samples]

    # Labels (1 = AD, 0 = N)
    y      = (meta["donor_group"] == "AD").astype(int).rename("label")
    groups = meta["donor_id_alias"]          # donor grouping for CV

    # ── Step 7: Classification ────────────────────────────────────────────
    log.info(f"\n[7/8] Leave-donor-out CV with '{classifier_name}' classifier")
    results, y_true, y_prob, donors, coef_df, sign_df = leave_donor_out_cv(
        X, y, groups, classifier_name, seed
    )

    # ── Step 8: Save & plot ───────────────────────────────────────────────
    log.info("\n[8/8] Saving results and plots")
    save_results(results, coef_df, y_true, y_prob, donors, output_dir, prefix,
                 sign_df=sign_df)
    plot_roc_pr(y_true, y_prob, output_dir, prefix)
    plot_fold_aucs(results["fold_aucs"], output_dir, prefix)
    plot_top_features(coef_df, n=min(30, len(coef_df)), output_dir=output_dir, prefix=prefix)
    plot_prediction_scores(y_true, y_prob, donors, output_dir, prefix)

    log.info("\n" + "=" * 60)
    log.info(f"DONE  |  Pooled AUC = {results['overall_auc']:.3f}  "
             f"|  AP = {results['overall_ap']:.3f}")
    log.info(f"Results written to: {output_dir}/")
    log.info("=" * 60)

    return results, coef_df


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AD blood exRNA classifier — SILVER-seq pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--counts",    required=True, help="Path to raw count matrix (TSV, genes × samples)")
    p.add_argument("--metadata",  required=True, help="Path to metadata Excel (.xlsx)")
    p.add_argument("--gwas",      required=True, help="Path to AD GWAS hits CSV (ADSP GVC format)")
    p.add_argument("--output_dir", default="results", help="Directory for outputs")
    p.add_argument("--prefix",    default="silver", help="Filename prefix for all outputs")
    p.add_argument("--norm_method", default="cpm_log2", choices=["cpm_log2", "vst"],
                   help="Normalisation method")
    p.add_argument("--classifier", default="logistic",
                   choices=["logistic", "elasticnet", "rf"],
                   help="Classifier type")
    p.add_argument("--min_count",  type=int,   default=10,
                   help="Minimum read count for low-count filter")
    p.add_argument("--min_samples_frac", type=float, default=0.10,
                   help="Minimum fraction of samples with min_count reads")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        counts_path      = args.counts,
        meta_path        = args.metadata,
        gwas_csv         = args.gwas,
        output_dir       = args.output_dir,
        norm_method      = args.norm_method,
        classifier_name  = args.classifier,
        min_count        = args.min_count,
        min_samples_frac = args.min_samples_frac,
        seed             = args.seed,
        prefix           = args.prefix,
    )