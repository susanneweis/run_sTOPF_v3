import os
import re
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def symmetric_mutual_information(x, y, nn, random_state=42):
    """
    Symmetric MI between two continuous 1D signals.
    Uses sklearn's kNN-based mutual information estimator and
    averages MI(x->y) and MI(y->x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    # sklearn expects X to be 2D
    mi_xy = mutual_info_regression(
        x.reshape(-1, 1),
        y,
        discrete_features=False,
        n_neighbors=nn,
        random_state=random_state
    )[0]

    mi_yx = mutual_info_regression(
        y.reshape(-1, 1),
        x,
        discrete_features=False,
        n_neighbors=nn,
        random_state=random_state
    )[0]

    return 0.5 * (mi_xy + mi_yx)


def compute_pairwise_similarity_long(
    df_sub,
    region_cols,
    metric="corr",
    mi_n_neighbors=5,
    random_state=0
):
    """
    Parameters
    ----------
    df_sub : DataFrame
        Must contain columns: subject, timepoint, region columns
    region_cols : list
        Region columns
    metric : str
        'corr' or 'mi'

    Returns
    -------
    long_df : DataFrame
        Columns: region, subject_1, subject_2, value
    """
    subjects = sorted(df_sub["subject"].unique())
    pairs = list(itertools.combinations(subjects, 2))

    rows = []

    for region in region_cols:
        pivot = df_sub.pivot(index="timepoint", columns="subject", values=region)
        pivot = pivot.sort_index()

        # added this late, possibly delete again
        # standardize per subject time series
        pivot = pivot.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col, axis=0)

        # keep only subjects that actually exist in this pivot
        available_subjects = [s for s in subjects if s in pivot.columns]

        for s1, s2 in itertools.combinations(available_subjects, 2):
            x = pivot[s1].values
            y = pivot[s2].values

            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if len(x) < 3:
                value = np.nan
            else:
                if metric == "corr":
                    # Pearson correlation
                    if np.std(x) == 0 or np.std(y) == 0:
                        value = np.nan
                    else:
                        value = np.corrcoef(x, y)[0, 1]

                elif metric == "mi":
                    value = symmetric_mutual_information(
                        x, y,
                        nn=mi_n_neighbors,
                        random_state=random_state
                    )
                else:
                    raise ValueError("metric must be 'corr' or 'mi'")

            rows.append({
                "region": region,
                "subject_1": s1,
                "subject_2": s2,
                "value": value
            })

    long_df = pd.DataFrame(rows)
    return long_df

def main(base_path,proj,code,movies_properties,nn): 

#     movie_file,
#     movie_timepoints_file,
#     sex_info_file,
#     valid_subjects,
#     excluded_participants=None,
#     output_dir="subject_pair_similarity",
#     mi_n_neighbors=5,
#     random_state=0

#     For one movie file, computes 6 long-format output files:
#       1) corr_all
#       2) corr_male
#       3) corr_female
#       4) mi_all
#       5) mi_male
#       6) mi_female

#     Sex coding:
#       1 = male
#       2 = female
#     """

    data_path = f"{base_path}/data_run_sTOPF_{proj}"

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}/pairwise_subject_similarity_nn{nn}/"
    os.makedirs(results_path, exist_ok=True)
 
    phenotype_path = f"{data_path}/Participant_sex_info.csv"
    complete_participants_path = f"{data_path}/complete_participants.csv"
    excluded_participants_path = f"{data_path}/excluded_participants.csv"

    movies = list(movies_properties.keys())

    # Load phenotype data (assumed to be a CSV with a subject ID and gender columns)
    phenotypes = pd.read_csv(phenotype_path)
    sex_mapping = {1: 'male', 2: 'female'}

    # Load list of complete participants (verified list with participants_verification.py)
    complete_participants = set(pd.read_csv(complete_participants_path)['subject'].astype(str))
    excluded_participants = set(pd.read_csv(excluded_participants_path)['subject'].astype(str))

    # Load list of excluded subjects (hormonal outlier detection with hormone_outlier_detection_SD.py)
    # not yet relevant here 
    # exclude_df = pd.read_csv(exclude_path, sep=',')
    # excluded_subjects = set(exclude_df['PCode'].astype(str))

    # Get valid subjects and exclude outliers
    phenotype_subjects = set(phenotypes['subject_ID'].astype(str))
    valid_subjects = complete_participants.intersection(phenotype_subjects)
    valid_subjects = valid_subjects.difference(excluded_participants)

    print(f"Number of included valid subjects after exclusion: {len(valid_subjects)}")

    movies = list(movies_properties.keys())

    for curr_mov in movies: 

        output_dir = f"{results_path}/{curr_mov}/"
        os.makedirs(output_dir, exist_ok=True) 

        # load files
        dataset = f"BOLD_Schaefer_436_2025_mean_aggregation_task-{curr_mov}_MOVIES.tsv"
        movie_path =  f"{data_path}/fMRIdata/{dataset}" # Path to fMRI data
        
        movie_data = pd.read_csv(movie_path, sep="\t")
        properties = movies_properties[curr_mov] 
        
        min_tp = properties["min_timepoint"]
        max_tp = properties["max_timepoint"]

        # Filter timepoints based on movie properties
        #movie_data = movie_data[
        #    (movie_data["timepoint"] >= properties["min_timepoint"]) & 
        #    (movie_data["timepoint"] <= properties["max_timepoint"])
        #] 

        sex_df = pd.read_csv(f"{data_path}/Participant_sex_info.csv")

        # --------------------------------------------------
        # standardize sex file columns
        # expects: subject_ID, gender
        # gender: 1 = male, 2 = female
        # --------------------------------------------------
        sex_df = sex_df.rename(columns={"subject_ID": "subject", "gender": "sex"})
        sex_df["subject"] = sex_df["subject"].astype(str)
        sex_df['sex'] = sex_df['sex'].replace(sex_mapping)

        male_subjects = set(sex_df.loc[sex_df["sex"] == "male", "subject"])
        female_subjects = set(sex_df.loc[sex_df["sex"] == "female", "subject"])

        # --------------------------------------------------
        # filter data
        # --------------------------------------------------
        movie_data["subject"] = movie_data["subject"].astype(str)
        movie_data = movie_data[movie_data["subject"].isin(valid_subjects)].copy()
        movie_data = movie_data[(movie_data["timepoint"] >= min_tp) & (movie_data["timepoint"] <= max_tp)].copy()
        movie_data = movie_data.sort_values(["subject", "timepoint"])

        # --------------------------------------------------
        # define region columns
        # --------------------------------------------------
        non_region_cols = ["subject", "timepoint"]
        region_cols = [c for c in movie_data.columns if c not in non_region_cols]

        # --------------------------------------------------
        # subsets
        # --------------------------------------------------
        subset_dict = {
            "all": movie_data.copy(),
            "male": movie_data[movie_data["subject"].isin(male_subjects)].copy(),
            "female": movie_data[movie_data["subject"].isin(female_subjects)].copy()
        }

        # --------------------------------------------------
        # output folder
        # --------------------------------------------------

        # Save used subjects per subset
        for subset_name, df_sub in subset_dict.items():
            used_subjects = sorted(df_sub["subject"].unique())
            pd.DataFrame({"subject": used_subjects}).to_csv(
                os.path.join(output_dir, f"{curr_mov}_{subset_name}_used_subjects.csv"),
                index=False
            )

        # --------------------------------------------------
        # compute and save 6 files
        # --------------------------------------------------
        for subset_name, df_sub in subset_dict.items():
            n_sub = df_sub["subject"].nunique()

            if n_sub < 2:
                print(f"Skipping {curr_mov} / {subset_name}: fewer than 2 subjects.")
                continue

            # Correlation
            corr_df = compute_pairwise_similarity_long(
                df_sub=df_sub,
                region_cols=region_cols,
                metric="corr"
            )
            corr_df.to_csv(
                os.path.join(output_dir, f"{curr_mov}_{subset_name}_corr_long.csv"),
                index=False
            )

            # Mutual information
            mi_df = compute_pairwise_similarity_long(
                df_sub=df_sub,
                region_cols=region_cols,
                metric="mi",
                mi_n_neighbors=nn,
                random_state=42
            )
            mi_df.to_csv(
                os.path.join(output_dir, f"{curr_mov}_{subset_name}_mi_nn{nn}_long.csv"),
                index=False
            )

            print(f"Saved {curr_mov} / {subset_name}: corr + mi")

        print(f"Finished movie: {curr_mov}")


# Execute script
if __name__ == "__main__":
    main()
