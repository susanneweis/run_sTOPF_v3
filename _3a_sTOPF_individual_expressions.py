import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression

def  comp_exp(pca_s_fem,pca_s_mal,reg,nn, sub_movie):

    pca_fem = pca_s_fem.loc[pca_s_fem["Region"] == reg, "PC_score_1"]
    pca_mal = pca_s_mal.loc[pca_s_mal["Region"] == reg, "PC_score_1"]

    rf, p = pearsonr(sub_movie[reg], pca_fem)
    rm, p = pearsonr(sub_movie[reg], pca_mal)

    diff = np.arctanh(rf) - np.arctanh(rm)
    diff = np.tanh(diff)

    # standardize
    y = (sub_movie[reg] - np.mean(sub_movie[reg])) / np.std(sub_movie[reg])
    xf = (pca_fem - np.mean(pca_fem)) / np.std(pca_fem)
    xm = (pca_mal - np.mean(pca_mal)) / np.std(pca_mal)

    # design matrix
    X = np.column_stack([xf, xm])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    beta_f, beta_m = model.params[1], model.params[2]
    fem_similarity = (beta_f - beta_m) / (abs(beta_f) + abs(beta_m))

    # mutual information
    df_y = pd.DataFrame(y)
    df_xf = pd.DataFrame(xf)
    df_xm = pd.DataFrame(xm)

    df_y = (df_y - df_y.mean()) / df_y.std(ddof=1)
    df_xf = (df_xf - df_xf.mean()) / df_xf.std(ddof=1)
    df_xm = (df_xm - df_xm.mean()) / df_xm.std(ddof=1)

    X = np.column_stack([df_xm, df_xf])  # shape (T, 2)
    y = df_y                       # shape (T,)

    mi = mutual_info_regression(X, y, n_neighbors=nn, random_state = 42)
    mi_m = mi[0]
    mi_f = mi[1]

    diff_mi = mi_f - mi_m

    return  rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi



def main(base_path,proj,nn_mi,movies_properties): 
    # Local setup for testing 
    # for Juseless Version see Kristina's code: PCA_foreachsex_allROI_latestversion.py

    data_path = f"{base_path}/data_run_sTOPF_{proj}"

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}"
    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"


    ind_path = f"{results_out_path}/individual_expressions_nn{nn_mi}"
    os.makedirs(ind_path, exist_ok=True)

    phenotype_path = f"{data_path}/Participant_sex_info.csv"
    complete_participants_path = f"{data_path}/complete_participants.csv"
    excluded_participants_path = f"{data_path}/excluded_participants.csv"

    # not relevant yet, as currently not considering hormones
    # exclude_path = f"{base_path}/results_pipeline/excluded_subjects.csv"

    sex_mapping = {1: 'male', 2: 'female'}
    subs_sex = pd.read_csv(phenotype_path)
    phenotypes = subs_sex

    subs_sex['gender'] = subs_sex['gender'].replace(sex_mapping)
    phenotypes.columns = ['subject_ID', 'gender']

    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]

    movies = list(movies_properties.keys())

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

    # not yet relevant here
    # valid_subjects = valid_subjects.difference(excluded_subjects)

    print(f"Number of included valid subjects after exclusion: {len(valid_subjects)}")

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
            sub_movie_data_concat = pd.concat([sub_movie_data_concat, subj_movie_data], axis=0, ignore_index=True)

            # Define the output directory
            # if hostname == "cpu44":
            #   output_dir =r_rootdir # Remote root directory
            #else:

            pca_path = f"{results_path}/results_PCA/{curr_mov}/{subj}"
            pca_scores_female = pd.read_csv(f"{pca_path}/PC1_scores_female_allROI.csv")
            pca_scores_male=  pd.read_csv(f"{pca_path}/PC1_scores_male_allROI.csv")

            for region in brain_regions:        

                rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi = comp_exp(pca_scores_female,pca_scores_male,region,nn_mi, subj_movie_data)
               
                sub_sex = subs_sex.loc[subs_sex["subject_ID"] == subj, "gender"].iloc[0]

                loo_results_all.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})
                loo_results_subj.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "fem_vs_mal_corr": diff, "fem_vs_mal_regr": fem_similarity, "fem_mi": mi_f, "mal_mi": mi_m,"fem_vs_mal_mi": diff_mi})


        # concatenated movie
        pca_path = f"{results_path}/results_PCA/concatenated_PCA/{subj}"
        pca_scores_female = pd.read_csv(f"{pca_path}/PC1_scores_female_allROI.csv")
        pca_scores_male=  pd.read_csv(f"{pca_path}/PC1_scores_male_allROI.csv")

        for region in brain_regions:        

            rf, rm, diff, fem_similarity, mi_f, mi_m,diff_mi = comp_exp(pca_scores_female,pca_scores_male,region,nn_mi, sub_movie_data_concat)
               
            sub_sex = subs_sex.loc[subs_sex["subject_ID"] == subj, "gender"].iloc[0]

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