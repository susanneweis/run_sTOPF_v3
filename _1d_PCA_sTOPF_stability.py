import os
import numpy as np
import pandas as pd
from glob import glob


def reshape_pc1(df):
    """
    Convert long format:
        Region | PC_score_1
    into wide format:
        timepoints x regions
    """
    df = df.copy()
    df["timepoint"] = df.groupby("Region").cumcount()
    return df.pivot(index="timepoint", columns="Region", values="PC_score_1")


def compute_regionwise_stability(full_df, loo_df, use_abs=True):
    """
    Compute correlation between full-group PC1 and LOO PC1 for each region.
    Returns:
        corrs: array of correlations
        region_names: list of region names in matching order
    """
    common_cols = full_df.columns.intersection(loo_df.columns)
    full_df = full_df[common_cols]
    loo_df = loo_df[common_cols]

    corrs = []

    for col in full_df.columns:
        x = full_df[col].values
        y = loo_df[col].values

        if len(x) != len(y) or np.std(x) == 0 or np.std(y) == 0:
            corrs.append(np.nan)
        else:
            r = np.corrcoef(x, y)[0, 1]
            corrs.append(abs(r) if use_abs else r)

    return np.array(corrs), list(full_df.columns)


def main(base_path, proj, code, movies_properties):

    out_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    base_full = f"{out_path}/results_PCA_per_sex"
    base_loo = f"{out_path}/results_PCA_loo"
    
    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    sex_info_file = f"{data_path}/Participant_sex_info.csv"
    sex_info = pd.read_csv(sex_info_file)

    sex_mapping = {1: 'male', 2: 'female'} 
    sex_lookup = {
        str(row["subject_ID"]): sex_mapping.get(row["gender"], "unknown")
        for _, row in sex_info.iterrows()
    }

    # change later

    #mv = ["DD", "S","DPS"]
    mv = list(movies_properties.keys())
    mv.append("concat")
    #mv = mv[:-2]
    
    movies = [
        os.path.join(base_full, m)
        for m in mv
        if os.path.isdir(os.path.join(base_full, m))
    ]

    #movies = sorted(
    #    [p for p in glob(os.path.join(base_full, "*")) if os.path.isdir(p)]
    #)

    results = []

    for movie_path in movies:
        movie = os.path.basename(os.path.normpath(movie_path))

        full_female_file = os.path.join(movie_path, "PC1_scores_female_allROI.csv")
        full_male_file = os.path.join(movie_path, "PC1_scores_male_allROI.csv")

        if not os.path.exists(full_female_file) or not os.path.exists(full_male_file):
            print(f"Skipping {movie}: missing full-group files")
            continue

        full_female = reshape_pc1(pd.read_csv(full_female_file))
        full_male = reshape_pc1(pd.read_csv(full_male_file))

        loo_movie_path = os.path.join(base_loo, movie)
        if not os.path.isdir(loo_movie_path):
            print(f"Skipping {movie}: no matching LOO movie folder")
            continue

        subjects = sorted(
            [p for p in glob(os.path.join(loo_movie_path, "*")) if os.path.isdir(p)]
        )

        corr_female_all = []
        corr_male_all = []

        for subj_path in subjects:
            subj = os.path.basename(os.path.normpath(subj_path))

            if subj not in sex_lookup:
                print(f"Skipping {movie} / {subj}: subject not found in sex info")
                continue

            subj_gender = sex_lookup[subj]

            loo_female_file = os.path.join(subj_path, "PC1_scores_female_allROI.csv")
            loo_male_file = os.path.join(subj_path, "PC1_scores_male_allROI.csv")

            if not os.path.exists(loo_female_file) or not os.path.exists(loo_male_file):
                print(f"Skipping {movie} / {subj}: missing LOO files")
                continue

            # Female stability: only across female left-out subjects
            if subj_gender == "female":
                loo_female = reshape_pc1(pd.read_csv(loo_female_file))
                corr_female, region_names_f = compute_regionwise_stability(
                    full_female, loo_female, use_abs=True
                )
                corr_female_all.append(corr_female)

            # Male stability: only across male left-out subjects
            elif subj_gender == "male":
                loo_male = reshape_pc1(pd.read_csv(loo_male_file))
                corr_male, region_names_m = compute_regionwise_stability(
                    full_male, loo_male, use_abs=True
                )
                corr_male_all.append(corr_male)

            else:
                print(f"Skipping {movie} / {subj}: unknown gender code {subj_gender}")

        if len(corr_female_all) == 0 and len(corr_male_all) == 0:
            print(f"Skipping {movie}: no valid LOO subjects")
            continue

        mean_female = std_female = mean_male = std_male = None
        region_names = None

        if len(corr_female_all) > 0:
            corr_female_all = np.array(corr_female_all)
            mean_female = np.nanmean(corr_female_all, axis=0)
            std_female = np.nanstd(corr_female_all, axis=0)
            region_names = region_names_f

        if len(corr_male_all) > 0:
            corr_male_all = np.array(corr_male_all)
            mean_male = np.nanmean(corr_male_all, axis=0)
            std_male = np.nanstd(corr_male_all, axis=0)
            if region_names is None:
                region_names = region_names_m

        for i, region in enumerate(region_names):
            results.append({
                "movie": movie,
                "region": region,
                "mean_stability_female": mean_female[i] if mean_female is not None else np.nan,
                "std_stability_female": std_female[i] if std_female is not None else np.nan,
                "n_female_subjects_used": len(corr_female_all) if isinstance(corr_female_all, list) is False else 0,
                "mean_stability_male": mean_male[i] if mean_male is not None else np.nan,
                "std_stability_male": std_male[i] if std_male is not None else np.nan,
                "n_male_subjects_used": len(corr_male_all) if isinstance(corr_male_all, list) is False else 0,
            })

    out_df = pd.DataFrame(results)

    out_path = os.path.join(out_path, "PC1_stability_summary.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")


# Execute script
if __name__ == "__main__":
    main()
