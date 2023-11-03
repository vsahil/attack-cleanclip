import seaborn as sns
import wandb, os, copy

import warnings   
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'cc6m', 'cc6m-200k', 'laion400m'])
parser.add_argument('--side-by-side-cleaning-100-and-200k', action='store_true')
parser.add_argument('--clean_with_slight_poison', action='store_true')
parser.add_argument('--plot_with_increasing_epochs', action='store_true')       ## If this is true, we plot the accuracy and ASR values with change in epochs. 
parser.add_argument('--plot_ssl_weight', action='store_true')           ## If this is true, we get the two_plots = True and plot the ssl weights separately. Till now their default value has been zero. 
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np


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


def extract_plot(combinations, combination, plt, filtering=None, marker='o', alpha=0.8, start_asr=None, start_accuracy=None):
    if len(combinations[combination]) == 0:
        print(f'No data for {combination}')
        return
    asr_values = np.array(combinations[combination][0])
    accuracy_values = np.array(combinations[combination][1])
    assert asr_values.shape[0] == accuracy_values.shape[0]  #== 21 * 8
    plt.set_xlabel('ASR (in %)')
    plt.set_ylabel('Top-1 ImageNet Zeroshot accuracy (in %)')
    if "cleaningdata_poison" in combination:
        cleaningdata_poison_number = int(combination.split('_')[-1])
        assert cleaningdata_poison_number in [0, 1, 2, 3, 4, 5, 10, 25]
        if cleaningdata_poison_number == 0:
            color = 'black'
            cleaning_label = 'No poison'
            marker = 'X'
        elif cleaningdata_poison_number == 1:
            color = 'navy'
            cleaning_label = '1 poison'
            marker = 'v'
        elif cleaningdata_poison_number == 2:
            color = 'maroon'
            cleaning_label = '2 poisons'
            marker = "<"
        elif cleaningdata_poison_number == 3:
            color = 'orange'
            cleaning_label = '3 poisons'
            marker = '>'
        elif cleaningdata_poison_number == 4:
            color = 'green'
            cleaning_label = '4 poisons'
            marker = 'd'
        elif cleaningdata_poison_number == 5: 
            color = 'purple'
            cleaning_label = '5 poisons'
            marker = '^'
        elif cleaningdata_poison_number == 10:
            color = 'red'
            cleaning_label = '10 poisons'
            marker = 's'
        elif cleaningdata_poison_number == 25:
            color = 'blue'
            cleaning_label = '25 poisons'
            marker = 'o'
        else:
            raise NotImplementedError
        if cleaningdata_poison_number > 0:
            start_accuracy = None
            start_asr = None
        
        exclude_from_plot_list = [1, 2, 3, 4, 10, 25]
        if cleaningdata_poison_number in exclude_from_plot_list:
            return      ## let's only plot 0, 5, 10, and 25 poisons
         
    elif "_ssl_weight" in combination:
        this_ssl_weight = int(combination.split('_')[-1])
        assert this_ssl_weight in [1, 2, 4, 6, 8]
        if this_ssl_weight == 1:
            color = 'black'
            cleaning_label = 'SSL lambda 1'
            marker = 'X'
        elif this_ssl_weight == 2:
            color = 'navy'
            cleaning_label = 'SSL lambda 2'
            marker = 'o'
        elif this_ssl_weight == 4:
            color = 'maroon'
            cleaning_label = 'SSL lambda 4'
            marker = "p"
        elif this_ssl_weight == 6:
            color = 'orange'
            cleaning_label = 'SSL lambda 6'
            marker = 'P'
        elif this_ssl_weight == 8:
            color = 'red'
            cleaning_label = 'SSL lambda 8'
            marker = 'd'
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
            color = 'navy'
            cleaning_label = 'Finetuning with MMCL loss'        
        elif 'clean_ssl_lr' in combination:
            color = 'maroon'
            cleaning_label = 'Finetuning with SSL loss'
        elif 'clean_mmcl_ssl_lr' in combination:
            color = 'orange'
            cleaning_label = 'Finetuning with MMCL + SSL loss'
        else:
            raise ValueError('Unknown cleaning paradigm')
    
    # if filtering == 'asr10_acc55':
    #     ## remove asr, accuracy pairs where asr is less than 0.9 and accuracy is less than 0.55
    # print the max accuracy for asr less than 5% and 10%
    # print(combination, end='\n')
    # try:
    #     print("best acuracy less than 5%:", max(accuracy_values[asr_values <= 0.05]))
    # except ValueError:
    #     print('No model with ASR less than 5%')
    # # try:
    # #     print("best acuracy less than 10%:", max(accuracy_values[asr_values <= 0.1]))
    # # except ValueError:
    # #     print('No model with ASR less than 10%')
    # print("\n")
    # return

    ## make sure the axis axes ranges are the same
    print(min(asr_values), combination, "\n")
    asr_values *= 100.0
    accuracy_values *= 100.0
    plt.set_xlim([-10, 110])
    # plt.set_ylim([-2, 27])
    sns.set_context("talk", font_scale=1)
    sns.set_style("whitegrid")
    sns.scatterplot(x=asr_values, y=accuracy_values, label=cleaning_label, color=color, ax=plt, marker=marker, alpha=alpha)
    ## change the border of the plot
    for spine in plt.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    ## add the start point and add the legend
    if start_asr is not None and start_accuracy is not None:
        plt.scatter(start_asr, start_accuracy, marker='*', color='red', s=500, label='Pre-trained model')


allsix = False    ## this is when we make 6 plots, 2 pre-training schemes and 3 cleaning schemes

if not args.side_by_side_cleaning_100_and_200k and not args.clean_with_slight_poison and not args.plot_with_increasing_epochs and not args.plot_ssl_weight:
    two_plots = True  ## this is when we make 2 plots, each for the 2 pre-training schemes -- and 3 cleaning schemes are overlaid
else:
    two_plots = False

one_plot = False  ## this is when we make 1 plot when we only have done cleaning for only one-pretrained model, for example the CC6M model cleaned with 200k datapoints


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
                plots[0][0].set_title('Model trained with MMCL, cleaned with MMCL')
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[0][1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[0][1].set_title('Model trained with MMCL, cleaned with SSL')
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[0][2], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                plots[0][2].set_title('Model trained with MMCL, cleaned with MMCL+SSL')
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
                plots[1][0].set_title('Model trained with MMCL+SSL, cleaned with MMCL')
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[1][1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
                plots[1][1].set_title('Model trained with MMCL+SSL, cleaned with SSL')
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[1][2], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
                plots[1][2].set_title('Model trained with MMCL+SSL, cleaned with MMCL+SSL')
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


if two_plots:
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))
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

            if 'clean_mmcl_lr' in combination:
                extract_plot(combinations, combination, plots[1], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError('Unknown training paradigm')
        ## set legend of all subplots to lower right
    
    plots[0].legend(loc='lower right')
    plots[1].legend(loc='lower right')
    plt.tight_layout()
    if args.dataset == 'cc3m':
        plt.savefig(f'two_plots_cleaning_plot_CC3M_pretrained_{poisoned_examples[0]}.pdf')
    elif args.dataset == 'cc6m':
        plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.png')
    elif args.dataset == 'cc6m-200k':
        plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}_cleaned_200k.pdf')
    elif args.dataset == 'laion400m':
        plt.savefig(f'two_plots_cleaning_plot_400M_pretrained_{poisoned_examples[0]}.pdf')
    else:
        raise NotImplementedError


if one_plot:        ## this is for the CC6M model cleaned with 200k datapoints
    fig, plots = plt.subplots(1, 1, figsize=(10, 10))
    for combination in combinations.keys():
        if 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots, marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots, marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots, marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots.set_title(f'Model pre-trained with MMCL + SSL objective') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError(f'Unknown training paradigm: {combination}')

                ## set legend of all subplots to lower right
        plots.legend(loc='lower right')
        # plots[1].legend(loc='lower right')
        plt.tight_layout()
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
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))
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
                extract_plot(combinations_100, combination, plots[0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations_100, combination, plots[0], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations_100, combination, plots[0], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Models pre-trained with MMCL + SSL, cleaned with 100k datapoints {dataset_text}') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
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
                extract_plot(combinations_200, combination, plots[1], marker='o', alpha=0.5)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combinations_200, combination, plots[1], marker='s')
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations_200, combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[1].set_title(f'Models pre-trained with MMCL + SSL, cleaned with 200k datapoints {dataset_text}')
        else: raise ValueError(f'Unknown training paradigm: {combination}')
    plots[0].legend(loc='lower right')
    plots[1].legend(loc='lower right')
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
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))
    for combination in combinations.keys():
        ## we want to make sure we give different colors to the different number of cleaning poisons
        if 'poisoned_cc3m_mmcl_1500poison_clean' in combination if here_poisoned_examples == 1500 else 'poisoned_mmcl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_3000poison_clean' in combination if here_poisoned_examples == 3000 else None:
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
                raise NotImplementedError
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combinations, combination, plots[0], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Model pre-trained with MMCL objective {dataset_text}')
        
        elif 'poisoned_cc3m_mmcl_ssl_1500poison_clean' in combination if here_poisoned_examples == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination if here_poisoned_examples == 3000 else None:
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
                extract_plot(combinations, combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')

        else: raise ValueError(f'Unknown training paradigm {combination}')

    ## set legend of all subplots to lower right
    plots[0].legend(loc='lower right')
    plots[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'two_plots_cleaning_plot_{here_dataset}_pretrained_{here_poisoned_examples}_cleaned_100k_poisoned.png')


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
            plot_here.set_title(f'Run {i+1}')
            plot_here.legend(loc='lower right')

            ## change the border of the plot
            for spine in plot_here.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.5)
            print("added run number:", i)

    ## set legend of all subplots to lower right
    plt.tight_layout()
    plt.savefig(f'plot_with_increasing_epochs_pre_training_mmcl_ssl_dataset_{args.dataset}_cleaning_100k.png')


if args.plot_ssl_weight:
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))
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
                extract_plot(combinations, combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective {dataset_text}') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
        else: raise ValueError('Unknown training paradigm')
        ## set legend of all subplots to lower right
    
    plots[0].legend(loc='lower right')
    plots[1].legend(loc='lower right')
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

