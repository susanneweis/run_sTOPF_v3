import os
import pandas as pd
import numpy as np


def main(base_path, proj, code):

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    base_dir = f"{results_path}/results_PCA_per_sex"
    
    out_dir = f"{results_path}/sex_separability"
    os.makedirs(out_dir, exist_ok=True)

    eps = 1e-8

    all_movies_results = []

    # --- loop over movie folders ---

    movie_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    for movie in movie_dirs:
        movie_path = os.path.join(base_dir, movie)

        if not os.path.isdir(movie_path):
            continue

        try:
            # --- file paths ---
            ev_male_file = os.path.join(movie_path, "explained_variance_1_male_allROI.csv")
            ev_female_file = os.path.join(movie_path, "explained_variance_1_female_allROI.csv")
            pc_male_file = os.path.join(movie_path, "PC1_scores_male_allROI.csv")
            pc_female_file = os.path.join(movie_path, "PC1_scores_female_allROI.csv")

            # skip if any file missing
            if not all(os.path.exists(f) for f in [ev_male_file, ev_female_file, pc_male_file, pc_female_file]):
                print(f"Skipping {movie} (missing files)")
                continue

            # --- load ---
            ev_male = pd.read_csv(ev_male_file)
            ev_female = pd.read_csv(ev_female_file)
            pc_male = pd.read_csv(pc_male_file)
            pc_female = pd.read_csv(pc_female_file)

            # --- rename (adjust if needed) ---
            ev_male = ev_male.rename(columns={"explained_variance_1": "EV_male"})
            ev_female = ev_female.rename(columns={"explained_variance_1": "EV_female"})

            pc_male = pc_male.rename(columns={"PC_score_1": "pc_male"})
            pc_female = pc_female.rename(columns={"PC_score_1": "pc_female"})

            # --- compute correlation per region ---
            regions = sorted(set(pc_male["Region"]).intersection(pc_female["Region"]))

            rows = []

            for r in regions:
                m = pc_male[pc_male["Region"] == r]["pc_male"].values
                f = pc_female[pc_female["Region"] == r]["pc_female"].values

                n = min(len(m), len(f))
                m = m[:n]
                f = f[:n]

                if n < 2:
                    corr = np.nan
                else:
                    corr = np.corrcoef(f, m)[0, 1]

                rows.append({"Region": r, "corr_female_male": corr})

            corr_df = pd.DataFrame(rows)

            # --- merge EV ---
            df = corr_df.merge(ev_female[["Region", "EV_female"]], on="Region", how="left")
            df = df.merge(ev_male[["Region", "EV_male"]], on="Region", how="left")

            # --- compute score ---
            df["score"] = (
                (1 - df["corr_female_male"])
                /
                ((1 - df["EV_female"]) + (1 - df["EV_male"]) + eps)
            )

            df["movie"] = movie

            # --- save per movie ---
            out_file = os.path.join(out_dir, f"{movie}_sex_diff_score.csv")
            df.to_csv(out_file, index=False)

            all_movies_results.append(df)

            print(f"Processed {movie}")

        except Exception as e:
            print(f"Error in {movie}: {e}")


    # --- concatenate all movies ---
    if len(all_movies_results) > 0:
        all_df = pd.concat(all_movies_results, ignore_index=True)

        out_all = os.path.join(out_dir, "ALL_movies_sex_diff_scores.csv")
        all_df.to_csv(out_all, index=False)

        print(f"\nSaved combined file: {out_all}")