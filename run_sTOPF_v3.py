import pandas as pd
import numpy as np
import os
import sys
import socket

import _1a_sTOPF_PCA_all
import _1b_sTOPF_PCA_per_sex
import _1c_sTOPF_loo_PCA
import _1d_PCA_sTOPF_stability
import _1e_PCA_sTOPF_stability_plot
import _1f_PC1_var_expl_brains
import _2a_sTOPF_pairw_subj_similarity
import _2b_sTOPF_compare_similarity_topf
import _3a_sTOPF_individual_expressions
import _3b_sTOPF_ind_exp_scatter_plot
import _4a_sTOPF_sex_separation
import _4b_sTOPF_sex_separation_network_summary

#import _4b_sTOPF_sex_separation_movie_region_summary_with_glassbrain_1std
#import _4b_sTOPF_sex_separation_movie_region_summary_with_3Dbrain_1std
#import _4d_sTOPF_sex_separation_sub_specific
#import _4e_sTOPF_sex_separation_sub_specific_networks

#import _4a_sTOPF_compute_sexability
#import _4a2_sTOPF_compute_sexability
#import _4a2_sTOPF_compute_sexability_balanced
#import _4b_sTOPF_sexability_shared_and_specific
#import _4c_sTOPF_sexability_detailed
#import _5a_plot_high_low_stability_tcs

# Setup for paths
hostname = socket.gethostname()
if "cpu" in hostname: # Run on Juseless

    # Arguments 

    base_path = sys.argv[1]
    project_ext = sys.argv[2]
    code_ext = sys.argv[3]

    # Parameter for Mutual Information Estimation
    nn_mi = int(sys.argv[4])

    # wkdir = sys.argv[1] # Project directory
    # r_rootdir = sys.argv[2] # Result root directory
    # phenotype = sys.argv[3]  # Phenotype file 
    # complete_participants = sys.argv[4] # Complete participants file
    # excluded_subjects = sys.argv[5] # Exclusion file due to hormonal outliers
    # dataset = sys.argv[6] 
    
    # dataset_list = dataset.split(",") # Split dataset into a list
    # print(f"Dataset list: {dataset_list}")
    # number_of_movies = len(dataset_list) # Number of movies
    # print(f"number of movies {number_of_movies}")
    
    # # Define paths and Check if they exist
    # base_path = f"{wkdir}/data"
    # movie_path =  f"{base_path}/{dataset_list[0]}.csv" # Path to fMRI data - first movie
    # phenotype_path = f"{wkdir}/data/{phenotype}.csv"
    # complete_participants_path = f"{wkdir}/data/{complete_participants}.csv"
    # exclude_path = f"{wkdir}/data/{excluded_subjects}.csv"

else:
    # Local setup for testing 
    
    # dataset_list = ["BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7", "BOLD_Schaefer400_subcor36_mean_task-tgtbtu_MOVIES_INM7"] # only 2 movies
    # dataset = "BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7.csv" 
    # base_path =  "/Users/kbauer/Desktop/master thesis/codes/fMRIdata" 
    # movie_path =  f"{base_path}/{dataset}" # Path to fMRI data
    # phenotype_path = f"{base_path}/movies_phenotype_results.csv"
    # complete_participants_path = f"{base_path}/complete_participants.csv"
    # exclude_path = f"{base_path}/outlier_results/excluded_subjects.csv"
    # Parameter for Mutual Information Estimation

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    project_ext = "v4"
    code_ext = "v3"
    nn_mi = 17

# mov_prop are new read in from file
data_path = f"{base_path}/data_run_sTOPF_{project_ext}"
mov_prop_file = f"{data_path}/movie_timepoints.csv"
mov_prop_df = pd.read_csv(mov_prop_file, index_col="movie")
mov_prop = mov_prop_df.to_dict(orient="index")

atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
roi_name_file = f"{data_path}/ROI_names.csv"
roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()

TR = 0.980  # seconds

for path in [base_path]:
    if not os.path.exists(path): 
        print(f"File not found: {path}")
        raise FileNotFoundError
# print(f"\nPath and Files found: \n - {movie_path}\n - {phenotype_path} \n - {complete_participants_path}\n {exclude_path}\n")    
print(f"\n Path and Files found: \n - {base_path}\n")    


#_1a_sTOPF_PCA_all.main(base_path, project_ext, code_ext, mov_prop)
#_1b_sTOPF_PCA_per_sex.main(base_path, project_ext, code_ext, mov_prop)
#_1c_sTOPF_loo_PCA.main(base_path, project_ext, code_ext, mov_prop)
#_1d_PCA_sTOPF_stability.main(base_path, project_ext, code_ext, mov_prop)
#_1e_PCA_sTOPF_stability_plot.main(base_path, project_ext, code_ext)
#_1f_PC1_var_expl_brains.main(base_path, project_ext, code_ext, mov_prop)
#_2a_sTOPF_pairw_subj_similarity.main(base_path, project_ext, code_ext, mov_prop,nn_mi)
#_2b_sTOPF_compare_similarity_topf.main(base_path, project_ext, code_ext, mov_prop,nn_mi)
#_3a_sTOPF_individual_expressions.main(base_path, project_ext, code_ext, nn_mi, mov_prop)
#_3b_sTOPF_ind_exp_scatter_plot.main(base_path, project_ext, code_ext, nn_mi)
_4a_sTOPF_sex_separation.main(base_path, project_ext, code_ext)
_4b_sTOPF_sex_separation_network_summary.main(base_path, project_ext, code_ext, roi_names, atlas_path)


#_4b_sTOPF_sex_separation_movie_region_summary_with_glassbrain_1std.main(base_path, project_ext, code_ext, roi_names, atlas_path)
#_4b_sTOPF_sex_separation_movie_region_summary_with_3Dbrain_1std.main(base_path, project_ext, code_ext, roi_names, atlas_path)
#_4c_sTOPF_sex_separation_movie_network_summary_with_3Dbrain_1std.main(base_path, project_ext, code_ext, roi_names, atlas_path)
#_4d_sTOPF_sex_separation_sub_specific.main(base_path, project_ext, code_ext, nn_mi)
#_4e_sTOPF_sex_separation_sub_specific_networks.main(base_path, project_ext, code_ext, nn_mi)

 
# _4a_sTOPF_compute_sexability.main(base_path, project_ext, code_ext, nn_mi)


#_4a2_sTOPF_compute_sexability.main(base_path, project_ext, code_ext, nn_mi)
#_4a2_sTOPF_compute_sexability_balanced.main(base_path, project_ext, code_ext, nn_mi)

#_4c_sTOPF_sexability_detailed.main(base_path, project_ext, code_ext, nn_mi)


#_4b_sTOPF_sexability_shared_and_specific.main(base_path, project_ext, code_ext,nn_mi, mov_prop, atlas_path, roi_names)
#top_reg = 10
#_5a_plot_high_low_stability_tcs.main(base_path, project_ext, code_ext, nn_mi, top_reg)