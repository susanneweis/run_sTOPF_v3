import pandas as pd
import numpy as np
import os
import sys
import socket

import _1a_sTOPF_PCA_all
import _1b_sTOPF_PCA_per_sex
import _1c_sTOPF_loo_PCA
import _2a_sTOPF_pairw_subj_similarity
import _2b_sTOPF_compare_similarity_topf
import _3a_sTOPF_individual_expressions
import _4a_sTOPF_compute_sexability
import _4b_sTOPF_sexability_shared_and_specific

# import _2a_sTOPF_result_full_group_PCA
# import _2b_sTOPF_individual_expressions
# import _3_sTOPF_analyse_results
# import _4a_sTOPF_visualize_group_glass_brains
# import _4b_sTOPF_visualize_individual_glass_brains
# import _5b_ind_classification
# import _6b_ind_classification_CV
# import _7b_ind_classification_CV_clustered
# import _7c_ind_classification_CV_clustered_HDBSCAN
# import _7d_ind_classification_CV_clustered_UMAP_HDBSCAN
# import _8b_sTOPF_visualize_cluster_glass_brains
# import _8c_sTOPF_visualize_cluster_HDBSCAN_glass_brains
# import _8d_sTOPF_visualize_cluster_UMAP_HDBSCAN_glass_brains
# import _9d_cluster_archetypes_UMAP_HDBSCAN
# import _9e_cluster_networks_UMAP_HDBSCAN
# import _10d_archetype_brains_UMAP_HDBSCAN
# import _10e_network_brains_UMAP_HDBSCAN
# import _11_sti_gradients
# import _12a_analyse_movie_shared_and_specific
# import _12a2_analyse_movie_shared_and_specific
# import _12b_vis_movie_shared_and_specific
# import _12c_vis_movie_shared_and_specific_netw
# import _12b2_vis_movie_shared_and_specific


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
    
    # Define movie timepoint parameters
    #mov_prop = {
    #    "DD": {"min_timepoint": 6, "max_timepoint": 463},
    #    "S": {"min_timepoint": 6, "max_timepoint": 445},
    #    "DPS": {"min_timepoint": 6, "max_timepoint": 479},
    #    "FG": {"min_timepoint": 6, "max_timepoint": 591},
    #    "DMV": {"min_timepoint": 6, "max_timepoint": 522},
    #    "LIB": {"min_timepoint": 6, "max_timepoint": 454},
    #    "TGTBTU": {"min_timepoint": 6, "max_timepoint": 512},
    #    "SS": {"min_timepoint": 6, "max_timepoint": 642},
    #    "REST1": {"min_timepoint": 6, "max_timepoint": 499},
    #    "REST2": {"min_timepoint": 6, "max_timepoint": 499}
    #}

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
#_2a_sTOPF_pairw_subj_similarity.main(base_path, project_ext, code_ext, mov_prop,nn_mi)
#_2b_sTOPF_compare_similarity_topf.main(base_path, project_ext, code_ext, mov_prop,nn_mi)
#_3a_sTOPF_individual_expressions.main(base_path, project_ext, code_ext, nn_mi, mov_prop)
#_4a_sTOPF_compute_sexability.main(base_path, project_ext, code_ext, nn_mi)
_4b_sTOPF_sexability_shared_and_specific.main(base_path, project_ext, code_ext,nn_mi, mov_prop, atlas_path, roi_names)

# _2a_sTOPF_result_full_group_PCA.main(base_path, project_ext, nn_mi, mov_prop)
# _2b_sTOPF_individual_expressions.main(base_path, project_ext, nn_mi, mov_prop)
# _3_sTOPF_analyse_results.main(base_path, project_ext, nn_mi, mov_prop)
#_4a_sTOPF_visualize_group_glass_brains.main(base_path, project_ext, nn_mi, mov_prop)
#_4b_sTOPF_visualize_individual_glass_brains.main(base_path, project_ext, nn_mi, mov_prop)
# for top_reg in [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]: 
#     _5b_ind_classification.main(base_path, project_ext, nn_mi, mov_prop,top_reg)
# highest stability 
#nn_values = [15, 17]
# models = ["svm", "rf"]
#models = ["rf"]
#for nn_mi in nn_values:
#    for model in models:
#        for top_reg in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: 
#            _6b_ind_classification_CV.main(base_path, project_ext, nn_mi, mov_prop,top_reg,model) 
# for cluster_num in [10, 20, 30, 40]: 
#     _7b_ind_classification_CV_clustered.main(base_path, project_ext, nn_mi, mov_prop, cluster_num)
# _7c_ind_classification_CV_clustered_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
# _7d_ind_classification_CV_clustered_UMAP_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
# for cluster_num in [10, 20, 30, 40, 50]: 
#     _8b_sTOPF_visualize_cluster_glass_brains.main(base_path,project_ext,nn_mi,mov_prop,cluster_num)
# _8c_sTOPF_visualize_cluster_HDBSCAN_glass_brains.main(base_path,project_ext,nn_mi,mov_prop)
# _8d_sTOPF_visualize_cluster_UMAP_HDBSCAN_glass_brains.main(base_path,project_ext,nn_mi,mov_prop)
# _9d_cluster_archetypes_UMAP_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
# _9e_cluster_networks_UMAP_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
# _10d_archetype_brains_UMAP_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
# _10e_network_brains_UMAP_HDBSCAN.main(base_path, project_ext, nn_mi, mov_prop)
#_11_sti_gradients.main(base_path, project_ext, nn_mi, mov_prop)
#max_reg = 30
#_12a_analyse_movie_shared_and_specific.main(base_path, project_ext, nn_mi, mov_prop, max_reg)
#_12a2_analyse_movie_shared_and_specific.main(base_path, project_ext, nn_mi, mov_prop, max_reg, atlas_path, roi_names)
#_12b_vis_movie_shared_and_specific.main(base_path, project_ext, nn_mi, atlas_path, roi_names)
#_12b2_vis_movie_shared_and_specific.main(base_path, project_ext, nn_mi, atlas_path, roi_names)
#_12c_vis_movie_shared_and_specific_netw.main(base_path, project_ext, nn_mi)
