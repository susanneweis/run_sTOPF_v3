import os
import pandas as pd
import numpy as np

def main(base_path,proj,code,nn): 
    """
    For each movie, compute mean pairwise correlations for:
      - female-female pairs
      - male-male pairs
      - female-male pairs

    Uses the same number of males and females per movie,
    defined as the minimum available number across sexes.    
    """

    all_res_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}"
    results_in_path = f"{all_res_path}/pairwise_subject_similarity_nn{nn}"
    output_dir = f"{all_res_path}/within_and_between_similarity_nn{nn}"

    os.makedirs(output_dir, exist_ok=True)

    movies = sorted([
        folder
        for folder in os.listdir(results_in_path)
        if os.path.isdir(os.path.join(results_in_path, folder))
    ])

    all_movie_summaries = []

    for movie in movies:
        print(f"Processing {movie}")

        female_file = f"{results_in_path}/{movie}/{movie}_female_used_subjects.csv"
        male_file = f"{results_in_path}/{movie}/{movie}_male_used_subjects.csv"
        corr_file = f"{results_in_path}/{movie}/{movie}_all_corr_long.csv"

        female_subjects_all = (
            pd.read_csv(female_file)["subject"]
            .astype(str)
            .tolist()
        )

        male_subjects_all = (
            pd.read_csv(male_file)["subject"]
            .astype(str)
            .tolist()
        )

        n_use = min(len(female_subjects_all), len(male_subjects_all))

        female_subjects = female_subjects_all[:n_use]
        male_subjects = male_subjects_all[:n_use]

        female_set = set(female_subjects)
        male_set = set(male_subjects)
        selected_subjects = female_set | male_set

        corr_df = pd.read_csv(corr_file)
        corr_df["subject_1"] = corr_df["subject_1"].astype(str)
        corr_df["subject_2"] = corr_df["subject_2"].astype(str)

        corr_df = corr_df[
            corr_df["subject_1"].isin(selected_subjects)
            & corr_df["subject_2"].isin(selected_subjects)
        ].copy()

        def classify_pair(row):
            s1 = row["subject_1"]
            s2 = row["subject_2"]

            if s1 in female_set and s2 in female_set:
                return "FF"
            elif s1 in male_set and s2 in male_set:
                return "MM"
            elif (
                (s1 in female_set and s2 in male_set)
                or (s1 in male_set and s2 in female_set)
            ):
                return "FM"
            else:
                return pd.NA

        corr_df["pair_type"] = corr_df.apply(classify_pair, axis=1)
        corr_df = corr_df.dropna(subset=["pair_type"])

        # Fisher-z transform correlations before averaging
        eps = 1e-10
        corr_df["value_z"] = np.arctanh(
            np.clip(corr_df["value"], -1 + eps, 1 - eps)
        )

        summary = (
            corr_df
            .groupby(["region", "pair_type"])["value_z"]
            .mean()
            .unstack("pair_type")
            .reset_index()
        )

        # Transform back to correlation space
        summary["FF"] = np.tanh(summary["FF"])
        summary["MM"] = np.tanh(summary["MM"])
        summary["FM"] = np.tanh(summary["FM"])



        summary = summary.rename(columns={
            "FF": "mean_female_female_corr",
            "MM": "mean_male_male_corr",
            "FM": "mean_female_male_corr",
        })

        for col in [
            "mean_female_female_corr",
            "mean_male_male_corr",
            "mean_female_male_corr",
        ]:
            if col not in summary.columns:
                summary[col] = pd.NA

        summary = summary[
            [
                "region",
                "mean_female_female_corr",
                "mean_male_male_corr",
                "mean_female_male_corr",
            ]
        ]

        summary.insert(0, "movie", movie)
        summary.insert(1, "n_female_available", len(female_subjects_all))
        summary.insert(2, "n_male_available", len(male_subjects_all))
        summary.insert(3, "n_per_sex_used", n_use)

        out_file = os.path.join(
            output_dir,
            f"{movie}_mean_pairwise_corr_by_sex_equalN.csv"
        )
        summary.to_csv(out_file, index=False)

        all_movie_summaries.append(summary)

        print(f"Saved: {out_file}")

    combined = pd.concat(all_movie_summaries, ignore_index=True)

    combined_file = os.path.join(
        output_dir,
        "ALL_movies_mean_pairwise_corr_by_sex_equalN.csv"
    )
    combined.to_csv(combined_file, index=False)

    print(f"Saved combined file: {combined_file}")


