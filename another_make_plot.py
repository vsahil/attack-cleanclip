import seaborn as sns
sns.set_context("talk", font_scale=1)
sns.set_style("whitegrid")
import wandb, os, copy


import warnings   
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 


def plot_best_runs(args, highest_accuracy_runs, accuracy_values, asr_values, training_method, cleaning_method, poisoned_examples_here, start_accuracy, start_asr):
    # import ipdb; ipdb.set_trace()
    ## Here we need to plot the change in ASR and accuracy for the first run as the number of epochs change
    # highest_accuracy_runs = highest_accuracy_runs[:1]     ## we only want to keep the top 5 runs

    import matplotlib.pyplot as plt
    import numpy as np
    # Define a list of colors and markers for different lines
    # colors = ['blue', 'yellow', 'red', 'purple', 'orange']  # Add more colors if needed
    
    # Creating a scatter plot for the color mapping
    # Use the first dataset to create a color mapping for the epochs
    # first_run = next(iter(datasets.values()))
    # epochs = len(accuracy_values[highest_accuracy_runs[0][-1]])
    # scatter = plt.scatter([0], [0], c=[0], cmap='Blues', norm=plt.Normalize(vmin=1, vmax=epochs), s=0)
    sns.set_context("poster", font_scale=1)
    sns.set_style("whitegrid")
    if args.make_plot_with_increasing_epochs:
        assert len(highest_accuracy_runs) == 3, f'{len(highest_accuracy_runs)}'
        fig, plots = plt.subplots(1, 3, figsize=(45, 12), sharex=True, sharey=True)
    elif args.make_plot_with_increasing_epochs_all_runs:
        assert len(highest_accuracy_runs) > 3
        ## Here we need to make the plot with all the runs, so very big figure
        if len(highest_accuracy_runs) == 10:
            fig, plots = plt.subplots(2, 5, figsize=(50, 20), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.08, top=0.98, right=0.98, wspace=0.05, hspace=0.05, left=0.04)
        elif len(highest_accuracy_runs) == 40:
            fig, plots = plt.subplots(8, 5, figsize=(50, 80), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.02, top=0.99, right=0.98, wspace=0.05, hspace=0.05, left=0.04)
        elif len(highest_accuracy_runs) == 56:
            fig, plots = plt.subplots(8, 7, figsize=(70, 80), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.02, top=0.99, right=0.98, wspace=0.05, hspace=0.05, left=0.04)
        elif len(highest_accuracy_runs) == 66:
            fig, plots = plt.subplots(11, 6, figsize=(60, 110), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.02, top=0.99, right=0.98, wspace=0.05, hspace=0.05, left=0.04)
        elif len(highest_accuracy_runs) == 73:
            fig, plots = plt.subplots(11, 7, figsize=(70, 110), sharex=True, sharey=True)
            plt.subplots_adjust(bottom=0.02, top=0.99, right=0.98, wspace=0.05, hspace=0.05, left=0.04)
        else:
            raise NotImplementedError


    for seq_idx, this_run in enumerate(highest_accuracy_runs[:]):
        this_run_name = this_run[3]
        lr_this_run = this_run[4]
        ssl_weight_this_run = this_run[5]
        num_epochs_this_run = this_run[6]
        ## now we have the highest accuracy, start fr
        if 'poisoned_mmcl_clean' in this_run_name if poisoned_examples_here == 1500 else 'poisoned_mmcl_5000poison_clean' in this_run_name if poisoned_examples_here == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in this_run_name:
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.76
                start_asr = 99.98
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError
        
        elif 'poisoned_mmcl_ssl_clean' in this_run_name if poisoned_examples_here == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in this_run_name if poisoned_examples_here == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in this_run_name:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

        else: raise ValueError('Unknown training paradigm')

        if max(asr_values[this_run_name]) <= 1:
            ## multiply accuracy and asr values by 100
            accuracy_values[this_run_name] = [100.0 * accuracy for accuracy in accuracy_values[this_run_name]]
            asr_values[this_run_name] = [100.0 * asr for asr in asr_values[this_run_name]]

        accuracy_values[this_run_name].insert(0, start_accuracy)
        asr_values[this_run_name].insert(0, start_asr)

        if args.make_plot_with_increasing_epochs:
            markers = ['o', 's', '^', 'D', 'x']  # Different marker styles
        else:
            markers = ['o']
        
        plt.style.use('seaborn-v0_8-colorblind')
        
        # Assuming accuracy_values and asr_values are defined and this_run_name is set
        epochs = len(accuracy_values[this_run_name])

        # Sizes and colors
        sizes = np.linspace(80, 300, epochs)  # Marker sizes
        # colors = np.linspace(0.3, 1, epochs)  # Color intensities
        intensity = np.linspace(0.3, 1., epochs)  # Color intensities
        # intensity = np.insert(intensity, 0, 0)  # Color intensities     ## for starting we are adding the red star, so we do not need to add the intensity for that point

        if args.make_plot_with_increasing_epochs:
            axes_here = plots[seq_idx]
        elif args.make_plot_with_increasing_epochs_all_runs:
            if len(highest_accuracy_runs) in [10, 40]:
                axes_here = plots[seq_idx // 5][seq_idx % 5]
            elif len(highest_accuracy_runs) in [57, 66]:
                axes_here = plots[seq_idx // 6][seq_idx % 6]
            elif len(highest_accuracy_runs) in [56, 73]:
                axes_here = plots[seq_idx // 7][seq_idx % 7]
            else: raise NotImplementedError
        if seq_idx == 0:
            scatter = axes_here.scatter([0], [0], c=[0], cmap='Blues', norm=plt.Normalize(vmin=1, vmax=epochs), s=0)
            ## add the start accuracy and asr values with a red large star marker
            # plots[seq_idx].plot(start_asr, start_accuracy, marker='*', markersize=20, c='red')
        
        marker_here = markers[seq_idx % len(markers)]
        print("Marker here: ", marker_here, seq_idx)
        # Plotting with connecting lines and circular markers
        for i in range(epochs - 1):
            sns.lineplot(x=asr_values[this_run_name][i:i+2], y=accuracy_values[this_run_name][i:i+2], ax=axes_here, marker=marker_here, markersize=sizes[i] / 10, alpha=intensity[i], color='blue', linewidth=8)
        
        axes_here.scatter(start_asr, start_accuracy, marker='*', color='red', s=500)
        # print(max(accuracy_values[this_run_name]), accuracy_values[this_run_name])
        if training_method == 'mmcl':
            axes_here.set_ylim([19, max(accuracy_values[this_run_name]) + 1])
        elif training_method == 'mmcl_ssl':
            axes_here.set_ylim([3, max(accuracy_values[this_run_name]) + 1])
        else:
            raise NotImplementedError
        
        if args.make_plot_with_increasing_epochs:
            axes_here.set_ylabel('Top-1 ImageNet Zeroshot accuracy (in %)', fontsize=36)
            axes_here.set_xlabel('ASR (in %)', fontsize=36)
        elif args.make_plot_with_increasing_epochs_all_runs:
            if len(highest_accuracy_runs) > 30:
                fig.text(0.001, 0.5, 'Top-1 ImageNet Zeroshot accuracy (in %)', va='center', rotation='vertical', fontsize=100)
            else:
                fig.text(0.001, 0.5, 'Top-1 ImageNet Zeroshot accuracy (in %)', va='center', rotation='vertical', fontsize=42)
            axes_here.set_xlabel('ASR (in %)', fontsize=36)
            axes_here.set_ylim([0, 27])
            ## add the legend to this plot which displays the lr, ssl weight, and number of epochs. We want to add these to all the subplots, hence it is specific to the axes_here
            # axes_here.legend(f'({lr_this_run}, {ssl_weight_this_run}, {num_epochs_this_run})', loc='lower right', fontsize=36)
            legend_string = f'Hyperparams: {lr_this_run}, {ssl_weight_this_run}, {num_epochs_this_run}'
            # legend_string = f'({lr_this_run}, {ssl_weight_this_run}, {num_epochs_this_run})'
            axes_here.legend([legend_string], loc='lower right', fontsize=32)
        else:
            raise NotImplementedError

        axes_here.tick_params(axis='both', labelsize=36)

    # Adding a color bar to the right of the subplots
    colorbar = fig.colorbar(scatter, ax=plots, orientation='vertical', fraction=0.015, pad=0.02)
    if len(highest_accuracy_runs) > 30:
        colorbar.set_label('Finetuning Epochs', fontsize=80)
    else:
        colorbar.set_label('Finetuning Epochs', fontsize=36)
    # Set the ticks to integer values. Adjust the range and step as needed.
    colorbar.set_ticks([])  # Replace min_value and max_value with your range
    # plt.tight_layout()
    if args.make_plot_with_increasing_epochs:
        fig.set_constrained_layout(True)
    # plt.title('Accuracy vs ASR over finetuning epochs for the 3 runs with highest accuracy', fontsize=16)
    if args.make_plot_with_increasing_epochs:
        plt.savefig(f'results_plots/individual_run_data/accuracy_vs_asr_over_epochs_{args.dataset}_training_{training_method}_{poisoned_examples_here}_poison_cleaning_{cleaning_method}.pdf')
    elif args.make_plot_with_increasing_epochs_all_runs:
        plt.savefig(f'results_plots/individual_run_data/accuracy_vs_asr_over_epochs_{args.dataset}_training_{training_method}_{poisoned_examples_here}_poison_cleaning_{cleaning_method}_all_runs.pdf')



def process_best_runs(args, accuracy_values, asr_values):
    ## we need to filter the runs who have best accuracy for ASR values less than 5%. So first remove any runs that have no asr values less than 5%.
    cleaning_paradigms = ['mmcl_ssl']
    training_paradigms = ['mmcl', 'mmcl_ssl'] #, 'mmcl_ssl']
    poisoned_examples_here = 1500 if args.dataset == 'cc3m' else 3000 if args.dataset == 'cc6m' else 5000 if args.dataset == 'laion400m' else None

    for training_method in training_paradigms:
        if training_method == 'mmcl':
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
            elif args.dataset == 'cc6m':
                start_accuracy = 23.76
                start_asr = 99.98
            else: raise NotImplementedError
        elif training_method == 'mmcl_ssl':
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
            elif args.dataset == 'cc6m':
                start_accuracy = 23.86
                start_asr = 99.45
            else: raise NotImplementedError
        else: raise NotImplementedError

        if not args.make_plot_with_increasing_epochs_all_runs:
            asr_boundary = 0.05 if training_method == 'mmcl_ssl' else 0.1 if training_method == 'mmcl' else None
        else:
            # asr_boundary = 0.3 if training_method == 'mmcl_ssl' else 0.5 if training_method == 'mmcl' else None        ## this will not work, there is bug in the code
            asr_boundary = 1
        for cleaning_method in cleaning_paradigms:
            print(training_method, asr_boundary, len(asr_values.keys()))
            keys_with_no_asr_less_than_5 = []
            for key in asr_values.keys():
                if min(asr_values[key]) > asr_boundary:
                    keys_with_no_asr_less_than_5.append(key)
                # print(f'{key}: {min(asr_values[key])}')
            
            for key in keys_with_no_asr_less_than_5:
                del asr_values[key]
                del accuracy_values[key]

            asr_less_than5_accuracy_asr_pairs = {}    
            highest_accuracy_runs = []      ## This list maintains the list of runs that had asr less than 5% and sorted in decreasing order of accuracy.
            ## now we have the runs that have asr values less than 5%. Now we need to find the runs that have the best accuracy for them
            # import ipdb; ipdb.set_trace()
            for this_run in accuracy_values.keys():
                if args.make_plot_with_increasing_epochs:
                    if "save_this_cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean_mmcl_ssl_lr_" in this_run or 'cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean_mmcl_ssl_lr_0.0003' in this_run or 'cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean_mmcl_ssl_lr_2e-4_higher_epochs' in this_run or 'cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean_mmcl_ssl_lr_1e-4_higher_epochs' in this_run:
                        continue        ## do not count the runs that were started with mmcl ssl finetuning after stopping starting from cleaned point as they are stopped from start. 
                assert min(asr_values[this_run]) <= asr_boundary
                ## remove any runs that are not of this training and cleaning method, ##cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean_mmcl_ssl_lr_
                training_method_here = this_run.split(f'poisoned_{args.dataset}_')[1].split(f'_{poisoned_examples_here}')[0]
                cleaning_method_here = this_run.split('_clean_')[1].split('_lr_')[0]
                assert training_method_here in ['mmcl', 'mmcl_ssl'] and cleaning_method_here in ['mmcl', 'ssl', 'mmcl_ssl'], f'{training_method_here}, {cleaning_method_here}'
                if training_method_here != training_method or cleaning_method_here != cleaning_method:
                    continue
                lr_this_run = float(this_run.split(f'lr_')[1].split('_')[0])
                assert lr_this_run < 1e-2 and lr_this_run >= 1e-9, f'{lr_this_run}'
                if "_ssl_weight" in this_run:
                    ssl_weight_this_run = int(this_run.split('_ssl_weight_')[1].split('_')[0])
                    assert ssl_weight_this_run in [1, 2, 4, 6, 8]
                else:
                    ssl_weight_this_run = 1 if "ssl" in cleaning_method_here else 0
                ## now we need to pair the asr and accuracy values for this run, and then find the pairs where asr is less than 5%
                if len(asr_values[this_run]) == 40:       ## remember that every epoch had 2 measurements at steps -- so we need to only consider the points at the epoch boundaries
                    ## here we only consider skip the first, third, fifth and so on
                    accuracy_values_here = [accuracy_values[this_run][i] for i in range(len(accuracy_values[this_run])) if i % 2 == 1]
                    accuracy_values[this_run] = accuracy_values_here
                    asr_values_here = [asr_values[this_run][i] for i in range(len(asr_values[this_run])) if i % 2 == 1]
                    asr_values[this_run] = asr_values_here
                    asr_less_than5_accuracy_asr_pairs[this_run] = [(accuracy_values_here[epoch], asr_values_here[epoch], epoch, this_run, lr_this_run, ssl_weight_this_run, len(asr_values_here)) for epoch in range(len(asr_values_here)) if asr_values_here[epoch] <= asr_boundary]
                else:
                    if len(asr_values[this_run]) < 20:      ## these are the failed runs, we do not want to plot them
                        continue
                    asr_less_than5_accuracy_asr_pairs[this_run] = [(accuracy_values[this_run][epoch], asr_values[this_run][epoch], epoch, this_run, lr_this_run, ssl_weight_this_run, len(asr_values[this_run])) for epoch in range(len(asr_values[this_run])) if asr_values[this_run][epoch] <= asr_boundary]
                ## we want to append the pair with the highest accuracy to the highest_accuracy_runs list. We want to have atmost three pairs in that list across the runs.
                ## so we need to sort the pairs and then append the highest accuracy pair to the list.
                asr_less_than5_accuracy_asr_pairs[this_run].sort(key=lambda x: x[0], reverse=True)      ## This code is just used for finding the runs with the highest accuracy with ASR less than the asr_boundary, it is not used for plotting -- for that all epochs are used. 
                highest_accuracy_runs.extend(asr_less_than5_accuracy_asr_pairs[this_run])
                highest_accuracy_runs.sort(key=lambda x: x[0], reverse=True)
            
            ## remove the duplicate runs from the highest_accuracy_runs list, based on the run name, which is the last element in the tuple in the list of tuples in highest_accuracy_runs. Remmber that highest_accuracy_runs is a list
            run_names_seen = set()
            remove_indices = []
            for idx, tuples in enumerate(highest_accuracy_runs):
                run_name = tuples[3]
                if run_name in run_names_seen:
                    # highest_accuracy_runs.remove(tuples)
                    remove_indices.append(idx)
                else:
                    run_names_seen.add(run_name)
            
            for idx in remove_indices[::-1]:
                highest_accuracy_runs.pop(idx)

            print(len(highest_accuracy_runs), "SEEEE")
            ## now we have the highest accuracy runs. print them
            if args.make_plot_with_increasing_epochs:
                highest_accuracy_runs = highest_accuracy_runs[:3]     ## we only want to keep the top 5 runs
            elif args.make_plot_with_increasing_epochs_all_runs:
                if training_method == 'mmcl_ssl':
                    highest_accuracy_runs = highest_accuracy_runs[:56]
                elif training_method == 'mmcl':
                    highest_accuracy_runs = highest_accuracy_runs[:10]
                else:   raise NotImplementedError
                ## once over here, we can sort the highest_accuracy_runs with lr, ssl_weights, and the number of epochs -- we will give highest priority to lr, then ssl_weight, and then number of epochs and we want to sort in increasing order of these values
                highest_accuracy_runs.sort(key=lambda x: (x[6], x[5], x[4]))
            else:   raise NotImplementedError
            
            print(f"{args.dataset}, {training_method}, {cleaning_method}, {len(highest_accuracy_runs)}")
            
            for i in range(len(highest_accuracy_runs)):
                print(f'Run {i}: {highest_accuracy_runs[i]}')

            print("Plotting the change in ASR and accuracy with epochs for the best runs.")
            plot_best_runs(args, highest_accuracy_runs, accuracy_values, asr_values, training_method, cleaning_method, poisoned_examples_here, start_accuracy, start_asr)
        

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'cc6m', 'cc6m-200k', 'laion400m'])
parser.add_argument('--side-by-side-cleaning-100-and-200k', action='store_true')
parser.add_argument('--clean_with_slight_poison', action='store_true')
parser.add_argument('--plot_with_increasing_epochs', action='store_true')       ## If this is true, we plot the accuracy and ASR values with change in epochs. 
parser.add_argument('--plot_ssl_weight', action='store_true')           ## If this is true, we get the two_plots = True and plot the ssl weights separately. Till now their default value has been zero. 
parser.add_argument('--deep_clustering_experiment', action='store_true', help='This is the plots for deep clustering experiments.')
parser.add_argument('--deep_clustering_cheating_experiment', action='store_true', help='This is the plots for deep clustering cheating experiments.')
parser.add_argument('--make_plot_with_increasing_epochs', action='store_true', help='These are the plots for selected runs for the models that have highest accuracy with ASR < 5%')
parser.add_argument('--make_plot_with_increasing_epochs_all_runs', action='store_true', help='These are the plots for all the runs')
parser.add_argument('--clean_with_heavy_regularization', action='store_true', help='This is the plots for runs with heavy regularization in the finetuning process which we did for MMCL + SSL trained model')
parser.add_argument('--clean_with_shrink_and_perturb', action='store_true', help='This will plot the runs when cleaning is done with shrinking and perturbing the model parameters')
parser.add_argument('--side_by_side_cleaning_20_epochs_100_epochs', action='store_true', help='This plot will make the side by side plot of cleaning with 20 epochs and 100 epochs for the MMCL + SSL pre-trained model. ')
parser.add_argument('--accuracy_and_asr_original_pre_training', action='store_true', help='This will plot the accuracy and ASR values for the original pre-trained model.')
parser.add_argument('--pre_training_objective', type=str, default=None, choices=['mmcl', 'mmcl_ssl'])
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np


if args.make_plot_with_increasing_epochs or args.make_plot_with_increasing_epochs_all_runs:
    import json
    if args.dataset == 'cc3m':
        if args.clean_with_slight_poison:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC3M_pretrained_1500_cleaned_100k_poisoned.json'
        elif args.plot_ssl_weight:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC3M_pretrained_1500_higher_ssl_weight.json'
        else:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC3M_pretrained_1500.json'
    elif args.dataset == 'cc6m':
        if args.clean_with_slight_poison:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_cleaned_100k_poisoned.json'
        elif args.plot_ssl_weight:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_higher_ssl_weight.json'
        elif args.deep_clustering_experiment:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering.json'
        elif args.deep_clustering_cheating_experiment:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering_cheating.json'
        else:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000.json'
    elif args.dataset == 'cc6m-200k':
        store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_cleaned_200k.json'
    elif args.dataset == 'laion400m':
        store_file_name = 'results_plots/cleaning_plot_data_400M_pretrained_1500.json'
    else:
        raise NotImplementedError

    with open(store_file_name, 'r') as f:
        individual_run_data = json.load(f)
    accuracy_values = {}
    asr_values = {}
    for key in individual_run_data.keys():
        accuracy_values[key] = individual_run_data[key]['accuracy']
        asr_values[key] = individual_run_data[key]['asr']
    
    process_best_runs(args, accuracy_values, asr_values)
    exit()


if args.accuracy_and_asr_original_pre_training:
    import json
    file_name = 'results_plots/accuracy_and_asr_values_original_pre_training.json'
    
    if not os.path.exists(file_name):
        data = {}
        with open(file_name, 'w') as f:
            json.dump(data, f)
    
    this_run_key = args.dataset + "_" + args.pre_training_objective
    with open(file_name, 'r') as f:
        data = json.load(f)

    if not this_run_key in data.keys():
        print("Fetching the accuracy and ASR values for the original pre-trained model from wandb.")
        api = wandb.Api()
        if args.dataset == 'cc6m':
            project_path = "vsahil/clip-cc6m-pre-training"
            # Fetch all runs from the specified project
            runs = api.runs(project_path)
            if args.pre_training_objective == 'mmcl':
                # run_path = "vsahil/clip-cc6m-pre-training/train_cc6m_poison_mmcl_1e_3"
                run_id = "sxfmf305"
                specific_runs = [api.run(f"{project_path}/{run_id}")]
            elif args.pre_training_objective == 'mmcl_ssl':
                # run_path = "vsahil/clip-cc6m-pre-training/train_cc6m_poison_mmcl_ssl_1e_3_batch1024"
                run_id1 = "v15rlm43"
                run_id2 = "scpwxhdy"
                specific_runs = [api.run(f"{project_path}/{run_id1}"), api.run(f"{project_path}/{run_id2}")]
            else:
                raise NotImplementedError
        elif args.dataset == 'cc3m':
            project_path = "vsahil/clip-cc3m-pre-training"
            ## we didn't track the accuracy and ASR at each epoch for these models. 
            raise NotImplementedError

        # # Print details of each run
        # for run in runs:
        #     print(f"Run ID: {run.id}, Name: {run.name}, State: {run.state}")

        # Initialize lists to store values
        accuracy_values = []
        asr_values = []
        val_accuracy = None
        val_asr = None

        for specific_run in specific_runs:
            # Iterate through the run history
            for row in specific_run.scan_history():
                # Check if the accuracy key exists in this row
                if 'evaluation/zeroshot_top1' in row:
                    val_accuracy = row['evaluation/zeroshot_top1']
                
                # Check if the ASR key exists in this row
                if 'evaluation/asr_top1' in row:
                    val_asr = row['evaluation/asr_top1']
                
                if val_accuracy is not None:
                    accuracy_values.append(val_accuracy)
                    val_accuracy = None
                
                if val_asr is not None:
                    asr_values.append(val_asr)
                    val_asr = None

            assert len(accuracy_values) == len(asr_values)
        
        if args.pre_training_objective == 'mmcl_ssl' and args.dataset == 'cc6m':
            accuracy_values.insert(0, 0.09902)
            asr_values.insert(0, 0.9448)
            
        elif args.pre_training_objective == 'mmcl' and args.dataset == 'cc6m':
            accuracy_values.insert(0, 0.07852)
            asr_values.insert(0, 0.9356)

        accuracy_values.insert(0, 0.0)
        asr_values.insert(0, 0.0)

        accuracy_values = [100.0 * accuracy for accuracy in accuracy_values]
        asr_values = [100.0 * asr for asr in asr_values]

        print("Accuracy Values:", len(accuracy_values), accuracy_values, "\n")
        print("ASR Values:", len(asr_values), asr_values)

        ## save the accuracy and asr values in a json file. This json file will have the accuracy and ASR dynamic for all the pre-trained models, and hence we add to this file rather than overwriting it.
        import json
        file_name = 'results_plots/accuracy_and_asr_values_original_pre_training.json'
        with open(file_name, 'r') as f:
            data = json.load(f)
        data[args.dataset + "_" + args.pre_training_objective] = {'accuracy': accuracy_values, 'asr': asr_values}
        with open(file_name, 'w') as f:
            json.dump(data, f)
        exit()
    else:
        accuracy_values = data[this_run_key]['accuracy']
        asr_values = data[this_run_key]['asr']

        if args.pre_training_objective == 'mmcl':
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.756
                start_asr = 99.98
            else: raise NotImplementedError
        elif args.pre_training_objective == 'mmcl_ssl':
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.862
                start_asr = 99.45
            else: raise NotImplementedError
        else: raise NotImplementedError

        # import ipdb; ipdb.set_trace()
        assert start_accuracy in accuracy_values, f'{start_accuracy}, {accuracy_values}'
        start_accuracy_index = accuracy_values.index(start_accuracy)
        assert start_asr == round(asr_values[start_accuracy_index], 2), f'{start_asr}, {asr_values[start_accuracy_index]}'

        print("Plotting the accuracy and ASR values for the original pre-trained model using the data from the json file.")
        ## now we have the accuracy and asr values, we need to plot them in two subplots. The first plot will show the change in accuracy with epochs, and the second plot will show the change in ASR with epochs. We need to make a line plot for both of them.
        import matplotlib.pyplot as plt
        sns.set_context("poster", font_scale=1)
        sns.set_style("whitegrid")
        fig, plots = plt.subplots(1, 2, figsize=(30, 12), sharex=True, sharey=False)        ## since accuracy and ASR are on very different scales, we do not want to share the y-axis
        epochs = len(accuracy_values)
        ## epochs on x-axis, and accuracy/ASR on y-axis. Since epochs is on the x-axis, we do not need increasing size of markers and increasing intensity of colors -- just use the same marker and color for all the points.
        sns.lineplot(x=range(epochs), y=accuracy_values, ax=plots[0], marker='o', markersize=10, markerfacecolor='black', color='steelblue', linewidth=8, alpha=0.9)
        sns.lineplot(x=range(epochs), y=asr_values, ax=plots[1], marker='o', markersize=10, markerfacecolor='black', color='steelblue', linewidth=8, alpha=0.9)
        ## mark the model with the start accuracy and ASR values with a red star marker
        plots[0].plot(start_accuracy_index, start_accuracy, marker='*', color='red', markersize=40, label='Selected Model\'s Accuracy')
        plots[1].plot(start_accuracy_index, start_asr, marker='*', color='red', markersize=40, label='Selected Model\'s ASR')
        plots[0].set_xlabel('Pre-training Epochs', fontsize=36)
        plots[1].set_xlabel('Pre-training Epochs', fontsize=36)
        plots[0].set_ylabel('Top-1 ImageNet Zeroshot accuracy (in %)', fontsize=36)
        plots[1].set_ylabel('ASR (in %)', fontsize=36)
        plots[0].tick_params(axis='both', labelsize=36)
        plots[1].tick_params(axis='both', labelsize=36)
        plots[0].set_title('Accuracy vs Epochs', fontsize=36)
        plots[1].set_title('ASR vs Epochs', fontsize=36)
        plots[0].legend(loc='lower right', fontsize=36)
        plots[1].legend(loc='lower right', fontsize=36)
        pre_training_objective = 'MMCL + SSL' if args.pre_training_objective == 'mmcl_ssl' else 'MMCL'
        training_dataset = 'CC6M' if args.dataset == 'cc6m' else 'CC3M'
        plt.suptitle(f'Accuracy and ASR values vs Epochs for the {pre_training_objective} pre-trained model on {training_dataset} dataset', fontsize=36)
        plt.tight_layout()
        plt.savefig(f'accuracy_vs_asr_over_epochs_{args.dataset}_pre_training_{args.pre_training_objective}.pdf')
    
    exit()
        

## read combinations from json file
import json
if args.dataset == 'cc3m':
    poisoned_examples = [1500]
    if args.clean_with_slight_poison:
        with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500_cleaned_100k_poisoned.json', 'r') as f:
            combinations = json.load(f)
    elif args.plot_ssl_weight:
        with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500_higher_ssl_weight.json', 'r') as f:
            combinations = json.load(f)
    else:
        with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500.json', 'r') as f:
            combinations = json.load(f)
elif args.dataset == 'cc6m':
    poisoned_examples = [3000]
    if args.clean_with_slight_poison:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_100k_poisoned.json', 'r') as f:
            combinations = json.load(f)
    elif args.plot_ssl_weight:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_higher_ssl_weight.json', 'r') as f:
            combinations = json.load(f)
    elif args.deep_clustering_experiment:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering.json', 'r') as f:
            combinations = json.load(f)
    elif args.deep_clustering_cheating_experiment:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering_cheating.json', 'r') as f:
            combinations = json.load(f)
    elif args.clean_with_heavy_regularization:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_heavy_regularization.json', 'r') as f:
            combinations = json.load(f)
    elif args.clean_with_shrink_and_perturb:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_shrink_and_perturb.json', 'r') as f:
            combinations = json.load(f)
    elif args.side_by_side_cleaning_20_epochs_100_epochs:
        with open('results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000.json', 'r') as f:
            combinations = json.load(f)
    else:
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000.json', 'r') as f:
            combinations = json.load(f)
elif args.dataset == 'cc6m-200k':
    poisoned_examples = [3000]
    with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_200k.json', 'r') as f:
        combinations = json.load(f)
elif args.dataset == 'laion400m':
    poisoned_examples = [1500, 5000]
    with open('results_plots/cleaning_plot_data_400M_pretrained_1500.json', 'r') as f:
        combinations = json.load(f)
else:
    raise NotImplementedError


if args.dataset == 'cc6m-200k':
    training_paradigms = ['mmcl_ssl']
else:
    training_paradigms = ['mmcl', 'mmcl_ssl']

if args.clean_with_slight_poison or args.plot_with_increasing_epochs:
    cleaning_paradigms = ['mmcl_ssl']           ## to demonstrate the robustness of CleanCLIP to slight poison we only clean with mmcl_ssl, and to see the change in accuracy and ASR with epochs, we only care about the CleanCLIP objective. 
else:
    cleaning_paradigms = ['mmcl', 'ssl', 'mmcl_ssl']


def extract_plot(combinations, combination, plt, filtering=None, marker='o', alpha=0.8, start_asr=None, start_accuracy=None, color_here=None, legend_label=None, skip_legend=False, skip_y_label=False):
    if len(combinations[combination]) == 0:
        print(f'No data for {combination}')
        return
    # print(combinations[combination])
    asr_values = np.array(combinations[combination][0])
    accuracy_values = np.array(combinations[combination][1])
    assert asr_values.shape[0] == accuracy_values.shape[0]  #== 21 * 8
    plt.set_xlabel('ASR (in %)', fontsize=28)
    plt.tick_params(axis='both', labelsize=28)
    if not skip_y_label:
        plt.set_ylabel('Top-1 ImageNet Zeroshot accuracy (in %)', fontsize=28)
    if "cleaningdata_poison" in combination:
        cleaningdata_poison_number = int(combination.split('_')[-1])
        assert cleaningdata_poison_number in [0, 1, 2, 3, 4, 5, 10, 25]
        if cleaningdata_poison_number == 0:
            cleaning_label = 'No poison'
            color = 'darkmagenta'
            marker='s'
        elif cleaningdata_poison_number == 1:
            color = 'navy'
            cleaning_label = '1 poison'
            marker = 'v'
        elif cleaningdata_poison_number == 2:
            color = 'maroon'
            cleaning_label = '2 poisons'
            marker = "<"
        elif cleaningdata_poison_number == 3:
            color = 'black'
            cleaning_label = '3 poisons'
            marker = '>'
        elif cleaningdata_poison_number == 4:
            color = 'green'
            cleaning_label = '4 poisons'
            marker = 'd'
        elif cleaningdata_poison_number == 5: 
            color = 'black'
            cleaning_label = '5 poisons'
            marker = 'o'
        elif cleaningdata_poison_number == 10:
            color = 'maroon'
            cleaning_label = '10 poisons'
            marker = '^'
            alpha = 0.5 if "clean_mmcl_lr" in combination else 0.8
        elif cleaningdata_poison_number == 25:
            color = 'blue'
            cleaning_label = '25 poisons'
            marker = 'o'
        else:
            raise NotImplementedError
        if cleaningdata_poison_number > 0:
            start_accuracy = None
            start_asr = None
        
        exclude_from_plot_list = [1, 2, 3, 4, 25]
        if cleaningdata_poison_number in exclude_from_plot_list:
            return      ## let's only plot 0, 5, 10, and 25 poisons
         
    elif "_ssl_weight" in combination:
        try:
            this_ssl_weight = int(combination.split('_ssl_weight_')[1].split("_")[0])
        except ValueError:
            import ipdb; ipdb.set_trace()
        assert this_ssl_weight in [1, 2, 4, 6, 8]
        if this_ssl_weight == 1:
            color = 'darkmagenta'    #'black'
            cleaning_label = 'SSL weight 1'
            marker = 's'
        elif this_ssl_weight == 2:
            color = 'navy'
            cleaning_label = 'SSL weight 2'
            marker = 'o'
        elif this_ssl_weight == 4:
            color = 'maroon'
            cleaning_label = 'SSL weight 4'
            marker = "p"
        elif this_ssl_weight == 6:
            color = 'red'
            cleaning_label = 'SSL weight 6'
            marker = 'X'
        elif this_ssl_weight == 8:
            color = 'black'
            cleaning_label = 'SSL weight 8'
            marker = 'D'
        else:
            raise NotImplementedError
        
        if this_ssl_weight > 1:
            start_accuracy = None
            start_asr = None
        
        exclude_from_plot_list = []
        if this_ssl_weight in exclude_from_plot_list:
            return

    else:
        ## make sure we have the color right
        if 'clean_mmcl_lr' in combination:
            color='black'
            marker = 'o'
            cleaning_label = 'Finetuning with MMCL loss'
        elif 'clean_ssl_lr' in combination:
            color = 'maroon'
            marker='X'
            cleaning_label = 'Finetuning with SSL loss'
        elif 'clean_mmcl_ssl_lr' in combination:
            color = 'darkmagenta'
            marker='s'
            cleaning_label = 'Finetuning with MMCL + SSL loss'
        else:
            raise ValueError('Unknown cleaning paradigm')
    
    # if filtering == 'asr10_acc55':
    #     ## remove asr, accuracy pairs where asr is less than 0.9 and accuracy is less than 0.55
    # print the max accuracy for asr less than 5% and 10%
    # print(combination, end='\n')
    try:
        print("best acuracy less than 5%:", max(accuracy_values[asr_values <= 0.05]))
    except ValueError:
        print('No model with ASR less than 5%')
    # # try:
    # #     print("best acuracy less than 10%:", max(accuracy_values[asr_values <= 0.1]))
    # # except ValueError:
    # #     print('No model with ASR less than 10%')
    # print("\n")
    # return

    ## make sure the axis axes ranges are the same
    # print(min(asr_values), combination, "\n")
    asr_values *= 100
    accuracy_values *= 100
    plt.set_xlim([-10, 110])
    # plt.set_ylim([-2, 27])
    # sns.set_context("talk", font_scale=1)
    # sns.set_style("whitegrid")
    if skip_legend:
        cleaning_label = None
    sns.scatterplot(x=asr_values, y=accuracy_values, label=cleaning_label if legend_label is None else legend_label, color=color if color_here is None else color_here, ax=plt, marker=marker, alpha=alpha)
    ## change the border of the plot
    for spine in plt.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    ## add the start point and add the legend
    if start_asr is not None and start_accuracy is not None and not skip_legend:
        plt.scatter(start_asr, start_accuracy, marker='*', color='red', s=500, label='Pre-trained model')


allsix = False    ## this is when we make 6 plots, 2 pre-training schemes and 3 cleaning schemes

if not args.side_by_side_cleaning_100_and_200k and not args.clean_with_slight_poison and not args.plot_with_increasing_epochs and not args.plot_ssl_weight and not args.deep_clustering_experiment and not args.deep_clustering_cheating_experiment and not args.clean_with_heavy_regularization and not args.clean_with_shrink_and_perturb and not args.side_by_side_cleaning_20_epochs_100_epochs:
    two_plots = True  ## this is when we make 2 plots, each for the 2 pre-training schemes -- and 3 cleaning schemes are overlaid
else:
    two_plots = False

one_plot = False  ## this is when we make 1 plot when we only have done cleaning for only one-pretrained model, for example the CC6M model cleaned with 200k datapoints

# if args.deep_clustering_experiment or args.deep_clustering_cheating_experiment:
if args.clean_with_heavy_regularization or args.clean_with_shrink_and_perturb:
    one_plot = True


if allsix:
    fig, plots = plt.subplots(2, 3, figsize=(20, 30), sharex=True, sharey=True)
    for combination in combinations.keys():
        if 'poisoned_mmcl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
            elif args.dataset == 'cc6m':
                start_accuracy = 23.76
                start_asr = 99.98
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[0][0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[0][0].set_title('Model trained with MMCL, cleaned with MMCL', fontsize=28)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[0][1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[0][1].set_title('Model trained with MMCL, cleaned with SSL', fontsize=28)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[0][2], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                plots[0][2].set_title('Model trained with MMCL, cleaned with MMCL+SSL', fontsize=28)
        elif 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
            elif args.dataset == 'cc6m':
                start_accuracy = 23.86
                start_asr = 99.45
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[1][0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[1][0].set_title('Model trained with MMCL+SSL, cleaned with MMCL', fontsize=28)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[1][1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[1][1].set_title('Model trained with MMCL+SSL, cleaned with SSL', fontsize=28)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[1][2], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                plots[1][2].set_title('Model trained with MMCL+SSL, cleaned with MMCL+SSL', fontsize=28)
        else: raise ValueError('Unknown training paradigm')
        # plt.legend(loc='lower right')
        ## set legend of all subplots to lower right
        plots[0][0].legend(loc='lower right')
        plots[0][1].legend(loc='lower right')
        plots[0][2].legend(loc='lower right')
        plots[1][0].legend(loc='lower right')
        plots[1][1].legend(loc='lower right')
        plots[1][2].legend(loc='lower right')

        plt.tight_layout()
        if args.dataset == 'cc3m':
            plt.savefig(f'all_six_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}.pdf')
        elif args.dataset == 'cc6m':
            plt.savefig(f'all_six_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.pdf')
        elif args.dataset == 'laion400m':
            plt.savefig(f'all_six_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
        else:
            raise NotImplementedError

# import ipdb; ipdb.set_trace()
if two_plots:
    fig, plots = plt.subplots(1, 2, figsize=(30, 10), sharey=True, sharex=True)
    smaller_font = False
    # import ipdb; ipdb.set_trace()
    for combination in combinations.keys():
        if 'poisoned_mmcl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.76
                start_asr = 99.98
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations, combination, plots[0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[0], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[0], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Model pre-trained with MMCL objective {dataset_text}', fontsize=21 if smaller_font else 29) #, cleaning with different paradigms for {poisoned_examples[0]} poisoned examples')
        
        elif 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations, combination, plots[1], alpha=0.5, skip_y_label=True)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[1], skip_y_label=True)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[1], start_asr=start_asr, start_accuracy=start_accuracy, skip_y_label=True)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}', fontsize=21 if smaller_font else 29) #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        
        else: raise ValueError('Unknown training paradigm')
        ## set legend of all subplots to lower right
    
    plots[0].legend(loc='lower right', fontsize=21 if smaller_font else 28)
    plots[1].legend(loc='lower right', fontsize=21 if smaller_font else 28)

    plt.tight_layout()
    if args.dataset == 'cc3m':
        plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}.png')
    elif args.dataset == 'cc6m':
        plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.png')
    elif args.dataset == 'cc6m-200k':
        plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_cleaned_200k.pdf')
    elif args.dataset == 'laion400m':
        plt.savefig(f'two_plots_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
    else:
        raise NotImplementedError


if one_plot:        ## this is for the CC6M model cleaned with 200k datapoints
    fig, plots = plt.subplots(1, 1, figsize=(15, 10))
    for combination in combinations.keys():
        if combinations[combination] == []:
            print(f'No data for {combination}')
            continue
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations,  combination, plots, legend_label='Finetuning with MMCL + $\ell_2$' if args.clean_with_heavy_regularization else None, alpha=0.5) 
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations,  combination, plots, legend_label='Finetuning with SSL + $\ell_2$' if args.clean_with_heavy_regularization else None)
            elif 'clean_mmcl_ssl_lr' in combination:
                # extract_plot(combination, plots, marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                extract_plot(combinations, combination, plots, start_asr=start_asr, start_accuracy=start_accuracy, legend_label='Finetuning with MMCL + SSL + $\ell_2$' if args.clean_with_heavy_regularization else None)
            if args.deep_clustering_experiment:
                plots.set_title(f'Model pre-trained with MMCL + SSL objective, cleaning with deep clustering', fontsize=29)
            elif args.deep_clustering_cheating_experiment:
                plots.set_title(f'Model pre-trained with MMCL + SSL objective, cleaning with deep clustering true labels', fontsize=29)
            elif args.clean_with_heavy_regularization or args.clean_with_shrink_and_perturb:
                plots.set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}', fontsize=29)
            else:
                plots.set_title(f'Model pre-trained with MMCL + SSL objective', fontsize=29) #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError(f'Unknown training paradigm: {combination}')

                ## set legend of all subplots to lower right
        plots.legend(loc='lower right', fontsize=28)
        # plots[1].legend(loc='lower right')
        plt.tight_layout()
        if args.deep_clustering_experiment:
            if args.dataset == 'cc3m':
                plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_deep_clustering.png')
            elif args.dataset == 'cc6m':
                plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_deep_clustering.png')
            else:
                raise NotImplementedError
        elif args.deep_clustering_cheating_experiment:
            if args.dataset == 'cc3m':
                plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_deep_clustering_cheating_experiment.png')
            elif args.dataset == 'cc6m':
                plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_deep_clustering_cheating_experiment.png')
            else:
                raise NotImplementedError
        elif args.clean_with_heavy_regularization:
            if args.dataset == 'cc3m':
                plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_heavy_regularization_clean_mmcl_ssl.png')
            elif args.dataset == 'cc6m':
                plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_heavy_regularization_clean_mmcl_ssl.pdf')
            else:
                raise NotImplementedError
        elif args.clean_with_shrink_and_perturb:
            if args.dataset == 'cc3m':
                plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_shrink_and_perturb_clean_mmcl_ssl.png')
            elif args.dataset == 'cc6m':
                plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_shrink_and_perturb_clean_mmcl_ssl.pdf')
            else:
                raise NotImplementedError
        else:
            if args.dataset == 'cc3m':
                plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}.pdf')
            elif args.dataset == 'cc6m':
                plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.pdf')
            elif args.dataset == 'cc6m-200k':
                plt.savefig(f'one_plot_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_cleaned_200k.pdf')
            elif args.dataset == 'laion400m':
                plt.savefig(f'two_plots_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
            else:
                raise NotImplementedError

   
if args.side_by_side_cleaning_100_and_200k:
    del combinations
    combinations_100 = json.load(open('results_plots/cleaning_plot_data_CC6M_pretrained_3000.json', 'r'))
    combinations_200 = json.load(open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_200k.json', 'r'))
    ## drop the mmcl model cleaning from combinations_100 keys
    combinations_100 = {k:v for k,v in combinations_100.items() if 'poisoned_cc6m_mmcl_3000poison_clean' not in k}
    ## combination_100 will have the cleaning plot for cleaning of the MMCL+SSL model with 100k datapoints
    ## combination_200 will have the cleaning plot for cleaning of the MMCL+SSL model with 200k datapoints
    fig, plots = plt.subplots(1, 2, figsize=(30, 10), sharex=True, sharey=True)
    # plt.subplots_adjust(wspace=)
    for combination in combinations_100.keys():
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations_100, combination, plots[0], alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations_100, combination, plots[0]) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations_100, combination, plots[0], start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Models pre-trained with MMCL + SSL, cleaned with 100k datapoints {dataset_text}', fontsize=27) #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError(f'Unknown training paradigm: {combination}')
            
                    ## set legend of all subplots to lower right

    for combination in combinations_200.keys():
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations_200, combination, plots[1],  alpha=0.5, skip_y_label=True)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations_200, combination, plots[1], skip_y_label=True)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations_200, combination, plots[1], start_asr=start_asr, start_accuracy=start_accuracy, skip_y_label=True)
            plots[1].set_title(f'Models pre-trained with MMCL + SSL, cleaned with 200k datapoints {dataset_text}', fontsize=27)
        else: raise ValueError(f'Unknown training paradigm: {combination}')
    plots[0].legend(loc='lower right', fontsize=28)
    plots[1].legend(loc='lower right', fontsize=28)
    plt.tight_layout()
    if args.dataset == 'cc3m':
        plt.savefig(f'side_by_side_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}.pdf')
    elif args.dataset == 'cc6m':
        plt.savefig(f'side_by_side_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.pdf')
    elif args.dataset == 'cc6m-200k':
        plt.savefig(f'side_by_side_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_cleaned_200k.pdf')
    elif args.dataset == 'laion400m':
        plt.savefig(f'side_by_side_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
    else:
        raise NotImplementedError


if args.side_by_side_cleaning_20_epochs_100_epochs:
    ## we will take the combinations with higher_epochs from the combinations
    combinations_higher_epochs = {k:v for k,v in combinations.items() if 'higher_epochs' in k}
    combinations_lower_epochs = {k:v for k,v in combinations.items() if 'higher_epochs' not in k}
    del combinations

    ## merge the accuracy and asr keys in a list
    for combination in combinations_higher_epochs.keys():
        combinations_higher_epochs[combination] = [combinations_higher_epochs[combination]['asr'], combinations_higher_epochs[combination]['accuracy']]     ## asr will be at index 0, accuracy will be at index 1
    for combination in combinations_lower_epochs.keys():
        combinations_lower_epochs[combination] = [combinations_lower_epochs[combination]['asr'], combinations_lower_epochs[combination]['accuracy']]

    fig, plots = plt.subplots(1, 2, figsize=(30, 10), sharex=True, sharey=True)
    skip_legend_lower_epochs = False
    skip_legend_higher_epochs = False
    # import ipdb; ipdb.set_trace()
    for combination in combinations_lower_epochs.keys():        ## we only want the mmcl_ssl cleaning here
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                pass
                # extract_plot(combinations_lower_epochs, combination, plots[0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                pass
                # extract_plot(combinations_lower_epochs, combination, plots[0], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                if "_ssl_weight" in combination:
                    ssl_weight_here = int(combination.split('_ssl_weight_')[1].split("_")[0])
                    if ssl_weight_here > 1:     ## we only want to compare the ssl weight 1. 
                        continue
                extract_plot(combinations_lower_epochs, combination, plots[0], skip_legend=skip_legend_lower_epochs)
                if not skip_legend_lower_epochs:
                    skip_legend_lower_epochs = True
            plots[0].set_title(f'Models pre-trained with MMCL + SSL, finetuned to 20 epochs {dataset_text}', fontsize=29)
        elif 'poisoned_cc6m_mmcl_3000poison' in combination:        ## we only want the mmcl_ssl cleaning here
            pass
        else: raise ValueError(f'Unknown training paradigm: {combination}')
            
                    ## set legend of all subplots to lower right
    plots[0].scatter(start_asr, start_accuracy, marker='*', color='red', s=500, label='Pre-trained model')
    
    for combination in combinations_higher_epochs.keys():
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                # extract_plot(combinations_higher_epochs, combination, plots[1], marker='o', alpha=0.5)
                pass
            elif 'clean_ssl_lr' in combination:
                # extract_plot(combinations_higher_epochs, combination, plots[1], marker='s')
                pass
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations_higher_epochs, combination, plots[1], skip_legend=skip_legend_higher_epochs, skip_y_label=True)
                if not skip_legend_higher_epochs:
                    skip_legend_higher_epochs = True
            plots[1].set_title(f'Models pre-trained with MMCL + SSL, finetuned to 100 epochs {dataset_text}', fontsize=29)
        elif 'poisoned_cc6m_mmcl_3000poison' in combination:        ## we only want the mmcl_ssl cleaning here
            pass
        else: raise ValueError(f'Unknown training paradigm: {combination}')
    
    plots[1].scatter(start_asr, start_accuracy, marker='*', color='red', s=500, label='Pre-trained model')
    plots[0].legend(loc='lower right', fontsize=28)
    plots[1].legend(loc='lower right', fontsize=28)
    plt.tight_layout()
    
    if args.dataset == 'cc6m':
        plt.savefig(f'side_by_side_cleaning_20_epochs_100_epochs_plot_CC6M_pretrained_{poisoned_examples[0]}.pdf')
    else:
        raise NotImplementedError


if args.clean_with_slight_poison:
    ## In this case we would need to read two files, one where we have data for cleaning with 1, 2, 3, 4, 5, 10, and 25 poisons in 100K cleaning dataset for CC6M dataset, and the second file where we have cleaning with 0 poison, we need to combine information from these two files for cleaning with 'mmcl_ssl' scheme, and make one plot. The x-axis will ASR, the y-axis with be accuracy, and the poisons will be denoted by different markers with different colors.
    del combinations
    here_dataset = 'CC6M' if args.dataset == 'cc6m' else 'CC3M' if args.dataset == 'cc3m' else None
    here_poisoned_examples = 1500 if args.dataset == 'cc3m' else 3000 if args.dataset == 'cc6m' else None
    combinations_no_poison = json.load(open(f'results_plots/cleaning_plot_data_{here_dataset}_pretrained_{here_poisoned_examples}.json', 'r'))
    combinations_slight_poison = json.load(open(f'results_plots/cleaning_plot_data_{here_dataset}_pretrained_{here_poisoned_examples}_cleaned_100k_poisoned.json', 'r'))
    ## only keep the cleaning with mmcl_ssl for the no_poison combinations
    combinations_no_poison = {k:v for k,v in combinations_no_poison.items() if 'clean_mmcl_ssl_lr' in k}
    assert len(combinations_no_poison.keys()) == 2
    ## rename the keys of the combination with no poison to add 0 poison to their name
    new_combinations_no_poison = {}
    # import ipdb; ipdb.set_trace()
    for key in combinations_no_poison.keys():
        if args.dataset == 'cc3m':
            assert "poison_" not in key     ## This is the original name which did not have the poison number in it ('cleaning_poisoned_mmcl_ssl_clean_mmcl_ssl_lr_)
            pre_training_objective = key.split("_clean_")[0].split("poisoned_")[1]
            assert pre_training_objective in ['mmcl', 'mmcl_ssl']
            cleaning_method = key.split("_clean_")[1].split("_lr")[0]
            assert cleaning_method in ['mmcl', 'ssl', 'mmcl_ssl']
            new_key_name = f'cleaning_poisoned_cc3m_{pre_training_objective}_{here_poisoned_examples}poison_clean_{cleaning_method}_lr_cleaningdata_poison_0'
            new_combinations_no_poison[new_key_name] = combinations_no_poison[key]
        else:    
            new_combinations_no_poison[f'{key}cleaningdata_poison_0'] = combinations_no_poison[key]
    
    original_length = copy.deepcopy(len(combinations_slight_poison.keys()))
    assert original_length == 14
    combinations_slight_poison = {k:v for k,v in combinations_slight_poison.items() if 'clean_mmcl_ssl_lr' in k}
    assert len(combinations_slight_poison.keys()) == original_length  == 14 ## since all models have been cleaned with mmcl_ssl, the length remains the same
    # now merge the two dictionaries
    combinations = {**new_combinations_no_poison, **combinations_slight_poison}
    assert len(combinations.keys()) == 16

    # import ipdb; ipdb.set_trace()
    ## We will make two plots, left with be the model pre-trained with MMCL and right will be the model pre-trained with mmcl+SSL. The cleaning will only be mmcl+ssl
    fig, plots = plt.subplots(1, 2, figsize=(30, 10), sharey=True, sharex=True)
    for combination in combinations.keys():
        ## we want to make sure we give different colors to the different number of cleaning poisons
        if 'poisoned_cc3m_mmcl_1500poison_clean' in combination if here_poisoned_examples == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination if here_poisoned_examples == 3000 else None:
            if args.dataset == 'cc3m':
                start_accuracy_mmcl = 16.00
                start_asr_mmcl = 99.88
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy_mmcl = 23.76
                start_asr_mmcl = 99.98
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination or 'clean_ssl_lr' in combination:
                raise NotImplementedError
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[0], start_asr=None, start_accuracy=None)
            plots[0].set_title(f'Model pre-trained with MMCL objective {dataset_text}', fontsize=29)
        
        elif 'poisoned_cc3m_mmcl_ssl_1500poison_clean' in combination if here_poisoned_examples == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination if here_poisoned_examples == 3000 else None:
            if args.dataset == 'cc3m':
                start_accuracy_mmcl_ssl = 17.04
                start_asr_mmcl_ssl = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy_mmcl_ssl = 23.86
                start_asr_mmcl_ssl = 99.45
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination or 'clean_ssl_lr' in combination:
                raise NotImplementedError
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[1], start_asr=None, start_accuracy=None, skip_y_label=True)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}', fontsize=29) #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')

        else: raise ValueError(f'Unknown training paradigm {combination}')

    plots[0].scatter(start_asr_mmcl, start_accuracy_mmcl, marker='*', color='red', s=500, label='Pre-trained model')
    plots[1].scatter(start_asr_mmcl_ssl, start_accuracy_mmcl_ssl, marker='*', color='red', s=500, label='Pre-trained model')
    ## set legend of all subplots to lower right
    plots[0].legend(loc='lower right', fontsize=28)
    plots[1].legend(loc='lower right', fontsize=28)
    plt.tight_layout()
    ## share the x axes
    plt.savefig(f'two_plots_cleaning_plot_{here_dataset}_pretrained_{here_poisoned_examples}_cleaned_100k_poisoned.pdf')


if args.plot_with_increasing_epochs:
    # import ipdb; ipdb.set_trace()
    del combinations
    combinations_no_poison = json.load(open('results_plots/cleaning_plot_data_CC6M_pretrained_3000.json', 'r'))
    combinations_slight_poison = json.load(open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_100k_poisoned.json', 'r'))
    ## only keep the cleaning with mmcl_ssl for the no_poison combinations
    combinations_no_poison = {k:v for k,v in combinations_no_poison.items() if 'clean_mmcl_ssl_lr' in k and 'cc6m_mmcl_ssl_3000poison' in k}
    assert len(combinations_no_poison.keys()) == 1
    ## rename the keys of the combination with no poison to add 0 poison to their name
    new_combinations_no_poison = {}
    for key in combinations_no_poison.keys():
        new_combinations_no_poison[f'{key}cleaningdata_poison_0'] = combinations_no_poison[key]
    
    ## let's iterate over the combinations - we have combined all learning rates and epochs in one. So let's split them by 20 lengths -- and we can only split them if they are of length 20, if not then it represents unfinished runs which we don't want to include. 

    ## first let's split the combinations_no_poison
    combinations_no_poison_split = {}
    for combination in combinations_no_poison.keys():
        if 'poisoned_mmcl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.76
                start_asr = 99.98
                dataset_text = '(CC6M dataset)'
                total_runs = 12
                panel_length = 3
            else: raise NotImplementedError
        
        elif 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M dataset)'
                total_runs = 16
                panel_length = 4
            else: raise NotImplementedError
        
        else: raise ValueError('Unknown training paradigm')

        asr_values = np.array(combinations_no_poison[combination][0])
        accuracy_values = np.array(combinations_no_poison[combination][1])
        assert asr_values.shape[0] == accuracy_values.shape[0]
        ## these asr_values and accuracy_values should be divisible by 20
        assert asr_values.shape[0] % total_runs == 0
        ## now let's split them
        asr_values_split = np.split(asr_values, total_runs)
        accuracy_values_split = np.split(accuracy_values, total_runs)
        ## so each split represents one learning rate, and the 20 values represent the changing values over epochs. 
        ## now let's plot the asr and accuracy values for each split on different plots. The x-axis will be ASR, y-axis will be accuracy, and the color will be the epochs. We will not use 20 colors, but we will use the density of the color to represent the epochs.
        for i in range(len(asr_values_split)):
            combinations_no_poison_split[f'{combination}_split_{i}'] = [asr_values_split[i], accuracy_values_split[i]]

        ## plot it. The number of plots is the number of splits == len(asr_values_split)
        fig, plots = plt.subplots(panel_length, total_runs // panel_length, figsize=(10 * panel_length, 10 * total_runs // panel_length))
        for i in range(len(asr_values_split)):
            asr_values_here = np.array(combinations_no_poison_split[f'{combination}_split_{i}'][0]) * 100.0
            accuracy_values_here = np.array(combinations_no_poison_split[f'{combination}_split_{i}'][1]) * 100.0
            plot_row = i // (total_runs // panel_length)        # 5 // 4 = 1
            plot_column = i % (total_runs // panel_length)       # 5 % 4 = 1
            plot_here = plots[plot_row][plot_column]
            plot_here.set_xlim([-10, 110])
            # plt.set_ylim([-2, 27])
            sns.set_context("talk", font_scale=1)
            sns.set_style("whitegrid")
            total_epochs = list(range(len(asr_values_here)))
            if len(asr_values_here) == 40:
                # Creating a list of epochs where each epoch is repeated twice --- this happens when we evaluated the model twice in an epoch. 
                num_epochs = len(asr_values_here) // 2
                total_epochs = [epoch+1 for epoch in range(num_epochs) for _ in range(2)]
            elif len(asr_values_here) == 20:
                total_epochs = [epoch+1 for epoch in range(len(asr_values_here))] 
            else:
                raise NotImplementedError
            sns.scatterplot(x=asr_values_here, y=accuracy_values_here, label='Finetuning with MMCL + SSL loss', color='blue', ax=plot_here, marker='s', alpha=0.8, hue=total_epochs, palette="viridis")
            plot_here.scatter(start_asr, start_accuracy, marker='*', color='red', s=500, label='Pre-trained model')
            plot_here.set_title(f'Run {i+1}', fontsize=29)
            plot_here.legend(loc='lower right', fontsize=28)

            ## change the border of the plot
            for spine in plot_here.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.5)
            print("added run number:", i)

    ## set legend of all subplots to lower right
    plt.tight_layout()
    plt.savefig(f'plot_with_increasing_epochs_pre_training_mmcl_ssl_dataset_{args.dataset}_cleaning_100k.png')


if args.plot_ssl_weight:
    # fig, plots = plt.subplots(1, 2, figsize=(20, 10))
    fig, plots = plt.subplots(1, 1, figsize=(15, 10))       ## let make only the plot of the model pre-trained with MMCL + SSL objective
    # import ipdb; ipdb.set_trace()
    for combination in combinations.keys():
        if 'poisoned_mmcl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination:
            continue        ## we don't want to plot this
            if args.dataset == 'cc3m':
                start_accuracy = 16.00
                start_asr = 99.88
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.76
                start_asr = 99.98
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination or 'clean_ssl_lr' in combination:
                assert len(combinations[combination]) == 0      ## these should be empty
            #     extract_plot(combinations, combination, plots[0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            # elif 'clean_ssl_lr' in combination:
            #     extract_plot(combinations, combination, plots[0], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                if "_ssl_weight" in combination:
                    this_ssl_weight = int(combination.split("_ssl_weight_")[1].split("_")[0])
                    if this_ssl_weight != 1:    
                        assert len(combinations[combination]) == 0      ## for the model pre-trained with mmcl, we only clean with ssl weight 1. 
                    else:
                        assert "_ssl_weight_1" in combination, f'Here is the {combination}'       
                extract_plot(combinations, combination, plots[0], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Model pre-trained with MMCL objective {dataset_text}') #, cleaning with different paradigms for {poisoned_examples[0]} poisoned examples')
        
        elif 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
                dataset_text = '(CC3M dataset)'
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
                dataset_text = '(CC6M dataset)'
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination or 'clean_ssl_lr' in combination:
                assert len(combinations[combination]) == 0      ## these should be empty
            # if 'clean_mmcl_lr' in combination:
            #     extract_plot(combinations, combination, plots[1], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            # elif 'clean_ssl_lr' in combination:
            #     extract_plot(combinations, combination, plots[1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                # extract_plot(combinations, combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                extract_plot(combinations, combination, plots, marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots.set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}', fontsize=29) #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError('Unknown training paradigm')
        ## set legend of all subplots to lower right
    
    # plots[0].legend(loc='lower right')
    # plots[1].legend(loc='lower right')
    plots.legend(loc='lower right', fontsize=28)
    plt.tight_layout()
    if args.dataset == 'cc3m':
        plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_just_higher_ssl_weight.pdf')
    elif args.dataset == 'cc6m':
        plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_just_higher_ssl_weight.pdf')
    # elif args.dataset == 'cc6m-200k':
    #     plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_cleaned_200k_just_higher_ssl_weight.pdf')
    # elif args.dataset == 'laion400m':
    #     plt.savefig(f'two_plots_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
    else:
        raise NotImplementedError


# import ipdb; ipdb.set_trace()
if args.deep_clustering_experiment or args.deep_clustering_cheating_experiment:
    fig, plots = plt.subplots(1, 1, figsize=(15, 10), sharey=True, sharex=True)
    here_dataset = 'CC6M' if args.dataset == 'cc6m' else 'CC3M' if args.dataset == 'cc3m' else None
    here_poisoned_examples = 1500 if args.dataset == 'cc3m' else 3000 if args.dataset == 'cc6m' else None
    with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering.json', 'r') as f:
        combinations_position_0 = json.load(f)
    with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering_cheating.json', 'r') as f:
        combinations_position_1 = json.load(f)
    
    for position, combinations in enumerate([combinations_position_0, combinations_position_1]):
        for combination in combinations.keys():
            if combinations[combination] == []:
                print(f'No data for {combination}')
                continue
            if 'poisoned_cc3m_mmcl_ssl_1500poison_clean' in combination if here_poisoned_examples == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination if here_poisoned_examples == 3000 else None:
                if args.dataset == 'cc3m':
                    start_accuracy = 17.04
                    start_asr = 99.03
                    dataset_text = '(CC3M dataset)'
                elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                    start_accuracy = 23.86
                    start_asr = 99.45
                    dataset_text = '(CC6M dataset)'
                else: raise NotImplementedError

                if 'clean_mmcl_lr' in combination or 'clean_ssl_lr' in combination:
                    raise NotImplementedError
                elif 'clean_mmcl_ssl_lr' in combination:
                    if position == 0:
                        extract_plot(combinations, combination, plots, marker='o', start_asr=None, start_accuracy=None, color_here='forestgreen', legend_label='Pseudo-label (clustering)')
                    elif position == 1:
                        extract_plot(combinations, combination, plots, marker='s', start_asr=start_asr, start_accuracy=start_accuracy, color_here='midnightblue', legend_label='Pseudo-label (SigLIP ViT-L/14)')
                
                if args.deep_clustering_experiment:
                    plots.set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}', fontsize=29)
                elif args.deep_clustering_cheating_experiment:
                    plots.set_title(f'Model pre-trained with MMCL + SSL objective, cleaning with deep clustering true labels', fontsize=29)
                # if position == 0:
                #     plots[0].set_title(f'Model cleaned with deep clustering', fontsize=29)
                # elif position == 1:
                #     plots[1].set_title(f'Model cleaned with deep clustering true labels', fontsize=29)
            else: raise ValueError(f'Unknown training paradigm {combination}')

    ## set legend of all subplots to lower right
    # plots[0].legend(loc='lower right')
    # plots[1].legend(loc='lower right')
    plots.legend(loc='lower right', fontsize=28)
    plt.tight_layout()
    ## share the x axes
    # plt.savefig(f'two_plots_cleaning_plot_{here_dataset}_pretrained_{here_poisoned_examples}_cleaned_100k_poisoned.pdf')
    if args.deep_clustering_experiment:
        if args.dataset == 'cc3m':
            plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_deep_clustering.png')
        elif args.dataset == 'cc6m':
            plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_deep_clustering.pdf')
        else:
            raise NotImplementedError
    elif args.deep_clustering_cheating_experiment:
        if args.dataset == 'cc3m':
            plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}_deep_clustering_cheating_experiment.png')
        elif args.dataset == 'cc6m':
            plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_deep_clustering_cheating_experiment.png')
        else:
            raise NotImplementedError

