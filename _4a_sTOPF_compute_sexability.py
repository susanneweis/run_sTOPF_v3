import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from _util_glass_brains import create_glassbrains


def compute_cohens_d(group1, group2):
    """
    Cohen's d for group1 - group2.
    Kept here from the original script although the new main analysis
    uses one-sample d within each sex.
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = np.std(group1, ddof=1)
    s2 = np.std(group2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return np.nan

    d = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return d


def compute_one_sample_d(values):
    """
    One-sample Cohen's d against zero.
    This is mean(values) / sd(values).
    """
    values = pd.Series(values).dropna()

    if len(values) < 2:
        return np.nan

    sd = np.std(values, ddof=1)
    if sd == 0:
        return np.nan

    return np.mean(values) / sd


def reshape_pc_scores(df):
    """
    Convert long format (Region, PC_score_1) into wide format: timepoints x regions.
    Falls back to a wide dataframe if regions are already columns.
    """
    if {"Region", "PC_score_1"}.issubset(df.columns):
        return df.pivot(columns="Region", values="PC_score_1")

    if {"region", "PC_score_1"}.issubset(df.columns):
        return df.pivot(columns="region", values="PC_score_1")

    # already wide or in another reasonable format
    non_region_cols = {"TR", "tr", "time", "Time", "Unnamed: 0"}
    keep_cols = [c for c in df.columns if c not in non_region_cols]
    return df[keep_cols]


def compute_template_similarity_by_region(movie, pca_base_path, method="corr", nn=3):
    """
    Compute regionwise similarity between female and male typical responses.

    Assumes movie-specific files:
      results_PCA_per_sex/<movie>/PC1_scores_female_allROI.csv
      results_PCA_per_sex/<movie>/PC1_scores_male_allROI.csv

    method:
      - "corr": Pearson correlation
      - "mi": mutual information estimated with mutual_info_regression
    """
    female_file = os.path.join(pca_base_path, movie, "PC1_scores_female_allROI.csv")
    male_file = os.path.join(pca_base_path, movie, "PC1_scores_male_allROI.csv")

    if not os.path.exists(female_file) or not os.path.exists(male_file):
        raise FileNotFoundError(
            f"Could not find template files for movie {movie}:\n"
            f"{female_file}\n{male_file}"
        )

    female_df = pd.read_csv(female_file)
    male_df = pd.read_csv(male_file)

    female_wide = reshape_pc_scores(female_df)
    male_wide = reshape_pc_scores(male_df)

    common_regions = [c for c in female_wide.columns if c in male_wide.columns]

    rows = []
    for region in common_regions:
        x = female_wide[region].values
        y = male_wide[region].values

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            sim = np.nan
        else:
            if method == "corr":
                sim = np.corrcoef(x, y)[0, 1]
            elif method == "mi":
                sim = mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    n_neighbors=nn,
                    random_state=0
                )[0]
            else:
                raise ValueError(f"Unknown method: {method}")

        rows.append({
            "region": region,
            "template_similarity": sim
        })

    return pd.DataFrame(rows)


def save_and_plot(df, value_col, out_csv, title, name_str,
                  roi_names, atlas_path, results_glass_path,
                  mask_thresh=None):
    df.to_csv(out_csv, index=False)

    create_glassbrains(
        out_csv,
        value_col,
        "region",
        roi_names,
        atlas_path,
        title,
        results_glass_path,
        name_str,
        "continuous"
    )

    if mask_thresh is not None:
        masked_df = df.copy()
        masked_df.loc[masked_df[value_col].abs() < mask_thresh, value_col] = 0

        masked_csv = out_csv.replace(".csv", "_masked.csv")
        masked_df.to_csv(masked_csv, index=False)

        create_glassbrains(
            masked_csv,
            value_col,
            "region",
            roi_names,
            atlas_path,
            f"{title} masked",
            results_glass_path,
            f"{name_str}_masked",
            "continuous"
        )

        print(f"Saved: {masked_csv}")
    else:
        print(f"Saved: {out_csv}")



def main(base_path, proj, code, nn):

    results_in_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}"

    results_out = f"{results_in_path}/sexability_nn{nn}_balanced_mask08"
    os.makedirs(results_out, exist_ok=True)

    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    roi_name_file = f"{data_path}/ROI_names.csv"
    roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()

    ind_exp_file = f"{results_in_path}/individual_expression_all_nn{nn}.csv"
    ind_exp_data = pd.read_csv(ind_exp_file)

    pca_base_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_PCA_per_sex"

    mask_thresh = 0.8
    metrics = ["corr", f"mi_nn{nn}"]
    movies = ind_exp_data["movie"].unique()

    for metric in metrics:

        results_out_path = f"{results_out}/{metric}"
        os.makedirs(results_out_path, exist_ok=True)

        results_glass_path = f"{results_out_path}/figures"
        os.makedirs(results_glass_path, exist_ok=True)

        if metric == "corr":
            female_col = "correlation_female"
            male_col = "correlation_male"
        else:
            female_col = "fem_mi"
            male_col = "mal_mi"

        for movie in movies:

            df_movie = ind_exp_data[ind_exp_data["movie"] == movie]
            regions = df_movie["region"].unique()
            rows = []

            for region in regions:
                df_r = df_movie[df_movie["region"] == region].copy()

                female_scores = (
                    df_r[df_r["sex"] == "female"][female_col]
                    - df_r[df_r["sex"] == "female"][male_col]
                ).dropna()

                male_scores = (
                    df_r[df_r["sex"] == "male"][male_col]
                    - df_r[df_r["sex"] == "male"][female_col]
                ).dropna()

                d_female = compute_one_sample_d(female_scores)
                d_male = compute_one_sample_d(male_scores)
                d_balanced = np.nanmean([d_female, d_male])

                rows.append({
                    "region": region,
                    "cohens_d_female_own_minus_other": d_female,
                    "cohens_d_male_own_minus_other": d_male,
                    "cohens_d_balanced": d_balanced,
                    "mean_female_own_minus_other": np.mean(female_scores) if len(female_scores) > 0 else np.nan,
                    "mean_male_own_minus_other": np.mean(male_scores) if len(male_scores) > 0 else np.nan,
                    "n_female": len(female_scores),
                    "n_male": len(male_scores)
                })

            out_df = pd.DataFrame(rows)

            out_path = os.path.join(
                results_out_path,
                f"cohens_d_female_own_minus_other_{movie}_{metric}.csv"
            )
            save_and_plot(
                out_df[["region", "cohens_d_female_own_minus_other"]].copy(),
                "cohens_d_female_own_minus_other",
                out_path,
                f"Sexability {movie} {metric} female own-other d",
                f"Sexability_{movie}_{metric}_female_own_minus_other",
                roi_names,
                atlas_path,
                results_glass_path,
                mask_thresh=mask_thresh
            )

            out_path = os.path.join(
                results_out_path,
                f"cohens_d_male_own_minus_other_{movie}_{metric}.csv"
            )
            save_and_plot(
                out_df[["region", "cohens_d_male_own_minus_other"]].copy(),
                "cohens_d_male_own_minus_other",
                out_path,
                f"Sexability {movie} {metric} male own-other d",
                f"Sexability_{movie}_{metric}_male_own_minus_other",
                roi_names,
                atlas_path,
                results_glass_path,
                mask_thresh=mask_thresh
            )

            out_path = os.path.join(
                results_out_path,
                f"cohens_d_balanced_{movie}_{metric}.csv"
            )
            save_and_plot(
                out_df[["region", "cohens_d_balanced"]].copy(),
                "cohens_d_balanced",
                out_path,
                f"Sexability {movie} {metric} balanced d",
                f"Sexability_{movie}_{metric}_balanced",
                roi_names,
                atlas_path,
                results_glass_path,
                mask_thresh=mask_thresh
            )

            full_out_path = os.path.join(
                results_out_path,
                f"sexability_balanced_summary_{movie}_{metric}.csv"
            )
            out_df.to_csv(full_out_path, index=False)
            print(f"Saved: {full_out_path}")

            if metric == "corr":
                template_df = compute_template_similarity_by_region(
                    movie, pca_base_path, method="corr", nn=nn
                )

                out_path = os.path.join(
                    results_out_path,
                    f"template_similarity_corr_{movie}.csv"
                )
                save_and_plot(
                    template_df,
                    "template_similarity",
                    out_path,
                    f"Template similarity corr {movie}",
                    f"Template_similarity_corr_{movie}",
                    roi_names,
                    atlas_path,
                    results_glass_path,
                    mask_thresh=None
                )

                weighted_df = out_df[["region", "cohens_d_balanced"]].merge(
                    template_df, on="region", how="left"
                )
                weighted_df["cohens_d_balanced_weighted"] = (
                    weighted_df["cohens_d_balanced"]
                    * (1 - weighted_df["template_similarity"])
                )

                out_path = os.path.join(
                    results_out_path,
                    f"cohens_d_balanced_weighted_by_template_dissimilarity_{movie}_{metric}.csv"
                )
                save_and_plot(
                    weighted_df[["region", "cohens_d_balanced_weighted"]].copy(),
                    "cohens_d_balanced_weighted",
                    out_path,
                    f"Sexability {movie} {metric} balanced d weighted by template dissimilarity",
                    f"Sexability_{movie}_{metric}_balanced_weighted_by_template_dissimilarity",
                    roi_names,
                    atlas_path,
                    results_glass_path,
                    mask_thresh=mask_thresh
                )

                weighted_summary_out = os.path.join(
                    results_out_path,
                    f"sexability_balanced_weighted_summary_{movie}_{metric}.csv"
                )
                weighted_df.to_csv(weighted_summary_out, index=False)
                print(f"Saved: {weighted_summary_out}")

            elif metric.startswith("mi_nn"):
                template_df = compute_template_similarity_by_region(
                    movie, pca_base_path, method="mi", nn=nn
                )

                out_path = os.path.join(
                    results_out_path,
                    f"template_similarity_mi_{movie}_{metric}.csv"
                )
                save_and_plot(
                    template_df,
                    "template_similarity",
                    out_path,
                    f"Template similarity MI {movie} {metric}",
                    f"Template_similarity_MI_{movie}_{metric}",
                    roi_names,
                    atlas_path,
                    results_glass_path,
                    mask_thresh=None
                )


# Execute script
if __name__ == "__main__":
    # Example:
    # main(
    #     base_path="/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies",
    #     proj="v4",
    #     code="v3",
    #     nn=17,
    # )
    pass
