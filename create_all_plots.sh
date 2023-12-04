# The main plots
python another_make_plot.py --dataset cc3m
python another_make_plot.py --dataset cc6m
## Clean with slight poison
python another_make_plot.py --dataset cc3m --clean_with_slight_poison
python another_make_plot.py --dataset cc6m --clean_with_slight_poison
## Analysis experiments
python another_make_plot.py --dataset cc6m --clean_with_heavy_regularization
python another_make_plot.py --dataset cc6m --clean_with_shrink_and_perturb
python another_make_plot.py --dataset cc6m --deep_clustering_experiment
## Ablation experiments
python another_make_plot.py --dataset cc6m-200k --side-by-side-cleaning-100-and-200k
python another_make_plot.py --dataset cc6m --side_by_side_cleaning_20_epochs_100_epochs
python another_make_plot.py --dataset cc6m --plot_ssl_weight
## Trajectory plots for the main paper
python another_make_plot.py --dataset cc6m --make_plot_with_increasing_epochs
## Trajectory plots for all the runs in the appendix
python another_make_plot.py --dataset cc6m --make_plot_with_increasing_epochs_all_runs
## Accuracy ASR dynamics for pre-training
python another_make_plot.py --dataset cc6m --accuracy_and_asr_original_pre_training --pre_training_objective 'mmcl'
python another_make_plot.py --dataset cc6m --accuracy_and_asr_original_pre_training --pre_training_objective 'mmcl_ssl'
