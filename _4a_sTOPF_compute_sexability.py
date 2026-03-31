import os
import numpy as np
import pandas as pd
from _util_glass_brains import create_glassbrains


def compute_cohens_d(group1, group2):

    n1, n2 = len(group1), len(group2)
        
    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = np.std(group1, ddof=1)
    s2 = np.std(group2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return np.nan
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return d

def main(base_path, proj, code, nn):
    

    results_in_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}"
      
    results_out = f"{results_in_path}/sexability_nn{nn}"
    os.makedirs(results_out, exist_ok=True)
  

    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    roi_name_file = f"{data_path}/ROI_names.csv"
    roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()
 
    ind_exp_file = f"{results_in_path}/individual_expression_all_nn{nn}.csv"

    ind_exp_data = pd.read_csv(ind_exp_file)

    for sex in ["female", "male", "fem_male_diff"]:

        # =========================
        # SETTINGS
        # =========================
        metrics = ["corr", f"mi_nn{nn}"]

        #sex_mapping = {1: 'male', 2: 'female'}
        #male_label = [k for k, v in sex_mapping.items() if v == 'male'][0]
        #female_label = [k for k, v in sex_mapping.items() if v == 'female'][0]

        # =========================
        # MAIN LOOP
        # =========================
        movies = ind_exp_data["movie"].unique()
        regions = ind_exp_data["region"].unique()

        for metric in metrics:

            # if metric == "corr":
            #     met = "fem_vs_mal_corr"
            # else:
            #     met = "fem_vs_mal_mi"
          
            if metric == "corr":
                results_out_path = f"{results_out}/corr/"
                os.makedirs(results_out_path, exist_ok=True)
                if sex == "female":
                    met = "correlation_female"
                elif sex == "male":
                    met = "correlation_male"
                else:
                    met = "fem_vs_mal_corr"
            else:
                results_out_path = f"{results_out}/nn{nn}/"
                os.makedirs(results_out_path, exist_ok=True)
                if sex == "female":
                    met = "fem_mi"
                elif sex == "male":
                    met = "mal_mi"
                else:
                    met = "fem_vs_mal_mi"
            
            results_glass_path = f"{results_out_path}/figures"
            os.makedirs(results_glass_path, exist_ok=True)

            for movie in movies:
                
                df_movie = ind_exp_data[ind_exp_data["movie"] == movie]
                rows = []

                for region in regions:
                    df_r = df_movie[df_movie["region"] == region]

                    males = df_r[df_r["sex"] == "male"][met].dropna()
                    females = df_r[df_r["sex"] == "female"][met].dropna()

                    # if sex == "female":
                    #     d = compute_cohens_d(females, males)
                    # else:
                    #     d = compute_cohens_d(males, females)
                    d = compute_cohens_d(females, males)

                    rows.append({
                        "region": region,
                        "cohens_d": d
                    })

                out_df = pd.DataFrame(rows)

                # SAVE FILE
                out_path = os.path.join(
                    results_out_path,
                    f"cohens_d_{movie}_{metric}_{sex}.csv"
                )
                out_df.to_csv(out_path, index=False)

                cluster_assign_file = out_path
                title = f"Sexability {movie} {metric} {sex}"
                name_str = f"Sexability_{movie}_{metric}_{sex}"

                create_glassbrains(cluster_assign_file, "cohens_d", "region",roi_names, atlas_path, title, results_glass_path, name_str,"continuous")

                if sex == "male":
                    # expected negative 
                    out_df.loc[out_df["cohens_d"] > 0, "cohens_d"] = 0
                else:
                    out_df.loc[out_df["cohens_d"] < 0, "cohens_d"] = 0
                    

                #out_df.loc[out_df["cohens_d"] < 0, "cohens_d"] = 0
                out_path = os.path.join(
                    results_out_path,
                    f"cohens_d_{movie}_{metric}_{sex}_masked.csv"
                )
                out_df.to_csv(out_path, index=False)
                cluster_assign_file = out_path
                title = f"Sexability {movie} {metric} {sex} masked"
                name_str = f"Sexability_{movie}_{metric}_{sex}_masked"

                create_glassbrains(cluster_assign_file, "cohens_d", "region",roi_names, atlas_path, title, results_glass_path, name_str,"continuous")



                print(f"Saved: {out_path}")