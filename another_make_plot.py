import seaborn as sns
import wandb, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'cc6m', 'cc6m-200k', 'laion400m'])
parser.add_argument('--side-by-side-cleaning-100-and-200k', action='store_true')
args = parser.parse_args()
import matplotlib.pyplot as plt
import numpy as np

## read combinations from json file
import json
if args.dataset == 'cc3m':
    poisoned_examples = [1500]
    with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500.json', 'r') as f:
        combinations = json.load(f)
elif args.dataset == 'cc6m':
    poisoned_examples = [3000]
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
cleaning_paradigms = ['mmcl', 'ssl', 'mmcl_ssl']


def extract_plot(combinations, combination, plt, filtering=None, marker='o', alpha=0.8, start_asr=None, start_accuracy=None):
    asr_values = np.array(combinations[combination][0])
    accuracy_values = np.array(combinations[combination][1])
    assert asr_values.shape[0] == accuracy_values.shape[0]  #== 21 * 8
    plt.set_xlabel('ASR (in %)')
    plt.set_ylabel('Top-1 ImageNet Zeroshot accuracy (in %)')
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
    print(max(asr_values), combination, "\n")
    asr_values *= 100.0
    accuracy_values *= 100.0
    plt.set_xlim([-10, 110])
    # plt.set_ylim([15, 27])
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

allsix = False
two_plots = True
one_plot = False

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
            plt.savefig(f'two_plots_cleaning_plot_CC6M_pretrained_{poisoned_examples[0]}.pdf')
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
    