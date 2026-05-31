import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression

def  comp_exp(pca_s_fem,pca_s_mal,reg,nn, sub_movie):

    pca_fem = pca_s_fem.loc[pca_s_fem["Region"] == reg, "PC_score_1"]
    pca_mal = pca_s_mal.loc[pca_s_mal["Region"] == reg, "PC_score_1"]

    # standardize
    y = (sub_movie[reg] - np.mean(sub_movie[reg])) / np.std(sub_movie[reg])
    xf = (pca_fem - np.mean(pca_fem)) / np.std(pca_fem)
    xm = (pca_mal - np.mean(pca_mal)) / np.std(pca_mal)

    rf, p = pearsonr(y, xf)
    rm, p = pearsonr(y, xm)

    diff = np.arctanh(rf) - np.arctanh(rm)
    diff = np.tanh(diff)

    # design matrix
    X = np.column_stack([xf, xm])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    beta_f, beta_m = model.params[1], model.params[2]
    fem_similarity = (beta_f - beta_m) / (abs(beta_f) + abs(beta_m))

    # mutual information
  
    X = np.column_stack([xm, xf])  # shape (T, 2)
  
    mi = mutual_info_regression(X, y, n_neighbors=nn, random_state = 42)
    mi_m = mi[0]
    mi_f = mi[1]

    diff_mi = mi_f - mi_m

    return  rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi



def main(base_path,proj,code,nn_mi,movies_properties): 
    # Local setup for testing 
    # for Juseless Version see Kristina's code: PCA_foreachsex_allROI_latestversion.py

    data_path = f"{base_path}/data_run_sTOPF_{proj}"

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    results_out_path = f"{results_path}/results_nn{nn_mi}"

    ind_path = f"{results_out_path}/individual_expressions_nn{nn_mi}"
    os.makedirs(ind_path, exist_ok=True)

    valid_subjects_path = f"{data_path}/valid_subjects_balanced_sex.csv"
    if not os.path.exists(valid_subjects_path):
        raise FileNotFoundError(
            f"Balanced valid subject list not found: {valid_subjects_path}\n"
            f"Run create_balanced_valid_subject_list(base_path, proj) first."
        )
    
    valid_subjects_df = pd.read_csv(valid_subjects_path)
    valid_subjects_df["subject_ID"] = valid_subjects_df["subject_ID"].astype(str)
    if "subject_ID" in valid_subjects_df.columns:
        valid_subjects = set(valid_subjects_df["subject_ID"].astype(str))
    elif "subject" in valid_subjects_df.columns:
        valid_subjects = set(valid_subjects_df["subject"].astype(str))
    else:
        raise ValueError("Valid subject file must contain either a 'subject_ID' or 'subject' column.")

    #sex_mapping = {1: 'male', 2: 'female'}
    #subs_sex = pd.read_csv(phenotype_path)
    #phenotypes = subs_sex

    #subs_sex['gender'] = subs_sex['gender'].replace(sex_mapping)
    #phenotypes.columns = ['subject_ID', 'gender']

    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]

    movies = list(movies_properties.keys())

    print(f"Number of included balanced valid subjects: {len(valid_subjects)}")

    loo_results_all = []

    for subj in valid_subjects:

        loo_results_subj = []
        sub_movie_data_concat = pd.DataFrame()

        for curr_mov in movies:
            dataset = f"BOLD_Schaefer_436_2025_mean_aggregation_task-{curr_mov}_MOVIES.tsv"
            movie_path =  f"{data_path}/fMRIdata/{dataset}" # Path to fMRI data

            properties = movies_properties[curr_mov] # Get timepoint properties for the movie
            
            # Load fMRI data
            movie_data = pd.read_csv(movie_path, sep="\t")
            if "Unnamed: 0" in movie_data.columns:
                movie_data = movie_data.drop(columns=["Unnamed: 0"]) # Drop unnecessary columns
                
            # Define column names and brain regions
            brain_regions = movie_data.columns[2:]  # Extract all brain region columns (assuming the first two columns are not brain regions) 

            # Filter timepoints based on movie properties
            movie_data = movie_data[
                (movie_data["timepoint"] >= properties["min_timepoint"]) & 
                (movie_data["timepoint"] <= properties["max_timepoint"])
            ] 
            print(f"movie properties {curr_mov}", movie_data["timepoint"].min(), movie_data["timepoint"].max(),"\n") 
            
            subj_movie_data = movie_data.loc[movie_data["subject"] == subj].copy()

            # exclude REST1 und REST2
            if curr_mov not in ["REST1", "REST2"]:
                sub_movie_data_concat = pd.concat([sub_movie_data_concat, subj_movie_data], axis=0, ignore_index=True)

            # Define the output directory
            # if hostname == "cpu44":
            #   output_dir =r_rootdir # Remote root directory
            #else:

            pca_path = f"{results_path}/results_PCA_loo/{curr_mov}/{subj}"
            pca_scores_female = pd.read_csv(f"{pca_path}/PC1_scores_female_allROI.csv")
            pca_scores_male=  pd.read_csv(f"{pca_path}/PC1_scores_male_allROI.csv")

            for region in brain_regions:        

                rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi = comp_exp(pca_scores_female,pca_scores_male,region,nn_mi, subj_movie_data)
               
                #sub_sex = valid_subjects.loc[valid_subjects["subject_ID"] == subj, "gender"].iloc[0]
                sub_sex = valid_subjects_df.loc[valid_subjects_df["subject_ID"] == subj, "gender"].iloc[0]

                loo_results_all.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})
                loo_results_subj.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})


        # concatenated movie
        pca_path = f"{results_path}/results_PCA_loo/concat/{subj}"
        pca_scores_female = pd.read_csv(f"{pca_path}/PC1_scores_female_allROI.csv")
        pca_scores_male=  pd.read_csv(f"{pca_path}/PC1_scores_male_allROI.csv")

        for region in brain_regions:        

            rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi = comp_exp(pca_scores_female,pca_scores_male,region,nn_mi, sub_movie_data_concat)
               
            #sub_sex = valid_subjects.loc[valid_subjects["subject_ID"] == subj, "gender"].iloc[0]
            sub_sex = valid_subjects_df.loc[valid_subjects_df["subject_ID"] == subj, "gender"].iloc[0]
            loo_results_all.append({"subject": subj, "sex": sub_sex, "movie": "concat", "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})
            loo_results_subj.append({"subject": subj, "sex": sub_sex, "movie": "concat", "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})

        out_df = pd.DataFrame(loo_results_subj, columns=["subject","sex","movie","region","correlation_female","correlation_male","fem_vs_mal_corr","fem_vs_mal_regr","fem_mi","mal_mi","fem_vs_mal_mi"])
        out_csv = f"{ind_path}/individual_expression_{subj}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")


    out_df = pd.DataFrame(loo_results_all, columns=["subject","sex","movie","region","correlation_female","correlation_male","fem_vs_mal_corr","fem_vs_mal_regr","fem_mi","mal_mi","fem_vs_mal_mi"])
    out_csv = f"{results_out_path}/individual_expression_all_nn{nn_mi}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# Execute script
if __name__ == "__main__":
    main()