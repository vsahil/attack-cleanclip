import seaborn as sns
import wandb, os
os.environ["WANDB_API_KEY"] = "12e70657780f0ff02e1c9a6fd91ac369e99e41aa"
api = wandb.Api()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'cc6m', 'cc6m-200k', 'laion400m'])
parser.add_argument('--dump_data', action='store_true')
parser.add_argument('--clean_with_slight_poison', action='store_true', help='If this is true, we cleaned with mmcl+ssl when the cleaning data had slight poison')
args = parser.parse_args()
# if args.clean_with_slight_poison:
    # assert args.dataset == 'cc6m'       ## we have only done this experiment for now. 


if args.dataset == 'cc3m':
    if args.clean_with_slight_poison:
        runs = api.runs("vsahil/clip-defense-cc3m-complete-finetune-cleaning-100k-poisoned")    ## CC6M models cleaned with 200k cleaning data with slight poison
        num_paradigms = 46      ## this can change as it is still running
    else:
        runs = api.runs("vsahil/clip-defense-complete-finetune2")        ## CC3M models cleaned with 100k cleaning data
        num_paradigms = 68
elif args.dataset == 'cc6m':
    if args.clean_with_slight_poison:
        runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-cleaning-100k-poisoned")    ## CC6M models cleaned with 200k cleaning data with slight poison
        num_paradigms = 102      ## this can change as it is still running
    else:
        runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune")    ## CC6M models cleaned with 100k cleaning data
        num_paradigms = 72
elif args.dataset == 'cc6m-200k':
    runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-cleaning-200k")    ## CC6M models cleaned with 200k cleaning data
    num_paradigms = 35
elif args.dataset == 'laion400m':
    runs = api.runs("vsahil/clip-defense-400M-complete-finetune")    ## 400M models cleaned with 250k cleaning data
else:
    raise NotImplementedError

history = runs[0].history()
print(len([i for i in runs]))



# epochs = 20       ## It is not 20 always now. 
asr_values = {}
accuracy_values = {}

# import ipdb; ipdb.set_trace()
count = 0
for run in runs.objects:
    # print("Getting vals for this run: ", run.name)
    this_run_asr = run.history(keys=['evaluation/asr_top1'], samples=10000)
    this_run_accuracy = run.history(keys=['evaluation/zeroshot_top1'], samples=10000)
    assert this_run_asr.shape[0] == this_run_accuracy.shape[0]  #== epochs + 1, f'Expected {epochs + 1} epochs, got {this_run_asr.shape[0]} and {this_run_accuracy.shape[0]} instead'
    asr_values[run.name] = this_run_asr['evaluation/asr_top1'].tolist()
    accuracy_values[run.name] = this_run_accuracy['evaluation/zeroshot_top1'].tolist()
    if count % 10 == 0:
        print(f'Finished {count} runs')
    count += 1

## so the keys give me the training paradigm and the cleaning paradigm
try:
    assert len(asr_values.keys()) == len(accuracy_values.keys()) == num_paradigms
except AssertionError:
    print(f'Expected {num_paradigms} runs, got {len(asr_values.keys())} and {len(accuracy_values.keys())} instead')
    raise AssertionError

if args.dataset == 'cc6m-200k':
    training_paradigms = ['mmcl_ssl']
else:
    training_paradigms = ['mmcl', 'mmcl_ssl']       ## even for the cleaning with poison, we have both training paradigms. 

if args.clean_with_slight_poison:
    cleaning_paradigms = ['mmcl_ssl']           ## to demonstrate the robustness of CleanCLIP to slight poison we only clean with mmcl_ssl
else:
    cleaning_paradigms = ['mmcl', 'ssl', 'mmcl_ssl']

if args.dataset == 'cc3m':
    poisoned_examples = [1500]
elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
    poisoned_examples = [3000]
elif args.dataset == 'laion400m':
    poisoned_examples = [1500, 5000]
else:
    raise NotImplementedError

if args.dataset == 'cc6m':
    poison_in_cleaning_data = [1, 2, 3, 4, 5, 10, 25]
elif args.dataset == 'cc3m':
    poison_in_cleaning_data = [1, 2, 3, 4, 5, 10, 25]       # [5, 10, 25]
## make all combinations
combinations = []

if not args.clean_with_slight_poison:
    for training_paradigm in training_paradigms:
        for cleaning_paradigm in cleaning_paradigms:
            for poisoned_samples in poisoned_examples:
                if poisoned_samples == 1500:
                    name = f'cleaning_poisoned_{training_paradigm}_clean_{cleaning_paradigm}_lr_'
                elif poisoned_samples == 5000:
                    name = f'cleaning_poisoned_{training_paradigm}_5000poison_clean_{cleaning_paradigm}_lr_'
                elif poisoned_samples == 3000:
                    ## when there is no poison the learning rate and other things do not matter in the name, but when there is some poison in cleaning, it does
                    name = f'cleaning_poisoned_cc6m_{training_paradigm}_3000poison_clean_{cleaning_paradigm}_lr_'
            combinations.append(name)
    assert len(combinations) == len(training_paradigms) * len(cleaning_paradigms)
else:
    # print(training_paradigms, cleaning_paradigms, poisoned_examples, poison_in_cleaning_data)
    for training_paradigm in training_paradigms:
        for cleaning_poison in poison_in_cleaning_data:
            for cleaning_paradigm in cleaning_paradigms:
                for poisoned_samples in poisoned_examples:
                    assert poisoned_samples == 3000 if args.dataset == 'cc6m' else poisoned_samples == 1500
                    # if poisoned_samples == 1500:
                    #     name = f'cleaning_poisoned_{training_paradigm}_clean_{cleaning_paradigm}_lr_'
                    # elif poisoned_samples == 5000:
                    #     name = f'cleaning_poisoned_{training_paradigm}_5000poison_clean_{cleaning_paradigm}_lr_'
                    # elif poisoned_samples == 3000:
                        ## when there is no poison the learning rate and other things do not matter in the name, but when there is some poison in cleaning, it does
                    name = f'cleaning_poisoned_{args.dataset}_{training_paradigm}_{poisoned_samples}poison_clean_{cleaning_paradigm}_lr_cleaningdata_poison_{cleaning_poison}'
                    combinations.append(name)      
    assert len(combinations) == len(training_paradigms) * len(cleaning_paradigms) * len(poison_in_cleaning_data) * len(poisoned_examples), f'len of combinations is {len(combinations)}'

# for poisoned_samples in poisoned_examples:
## assert that each combination has 8 runs. note that the name will also have _lr_value, therefore it will not exact match the name
for combination in combinations:
    if "clean_mmcl_lr" in combination:
        if args.dataset == 'cc3m':
            assert len([key for key in asr_values.keys() if combination in key]) == 13, f'Expected 13 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 13
        elif args.dataset == 'cc6m':
            assert len([key for key in asr_values.keys() if combination in key]) == 8, f'Expected 8 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 8
        elif args.dataset == 'cc6m-200k':
            assert len([key for key in asr_values.keys() if combination in key]) == 8, f'Expected 8 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 8
    elif "clean_ssl_lr" in combination:
        if args.dataset == 'cc3m':
            assert len([key for key in asr_values.keys() if combination in key]) == 8, f'Expected 8 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 8
        # elif args.dataset == 'cc6m':
        #     assert len([key for key in asr_values.keys() if combination in key]) == 12, f'Expected 16 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
        #     assert len([key for key in accuracy_values.keys() if combination in key]) == 12
        elif args.dataset == 'cc6m-200k':
            assert len([key for key in asr_values.keys() if combination in key]) == 13, f'Expected 13 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 13
    elif "clean_mmcl_ssl_lr" in combination:
        if args.dataset == 'cc3m' and not args.clean_with_slight_poison:
            assert len([key for key in asr_values.keys() if combination in key]) == 13, f'Expected 13 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 13
        # elif args.dataset == 'cc6m':
            # assert len([key for key in asr_values.keys() if combination in key]) == 16, f'Expected 16 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            # assert len([key for key in accuracy_values.keys() if combination in key]) == 16
        elif args.dataset == 'cc6m-200k':
            assert len([key for key in asr_values.keys() if combination in key]) == 14, f'Expected 14 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
            assert len([key for key in accuracy_values.keys() if combination in key]) == 14

# import ipdb; ipdb.set_trace()

for key in accuracy_values.keys():
    # if 'cleaning_poisoned_mmcl_clean' in key:
    #     accuracy_values[key] = [value - 0.1600 for value in accuracy_values[key]]
    # elif 'cleaning_poisoned_mmcl_ssl_clean' in key:
    #     accuracy_values[key] = [value - 0.1704 for value in accuracy_values[key]]
    # if poisoned_examples[0] == 1500:
    if 'cleaning_poisoned_mmcl_clean' in key:
        # accuracy_values[key] = [value - 0.5878 for value in accuracy_values[key]]
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_mmcl_ssl_clean' in key:
        # accuracy_values[key] = [value - 0.5697 for value in accuracy_values[key]]
        accuracy_values[key] = [value for value in accuracy_values[key]]
        # else: raise ValueError('Unknown training paradigm')
    # elif 'cleaning_poisoned_mmcl_5000poison_clean' in key:
    #     accuracy_values[key] = [value - 0.5879 for value in accuracy_values[key]]
    # elif 'cleaning_poisoned_mmcl_ssl_5000poison_clean' in key:
    #     accuracy_values[key] = [value - 0.5797 for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc3m_mmcl_1500poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc3m_mmcl_ssl_1500poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_3000poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_ssl_3000poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    else:
        raise ValueError(f'Unknown training paradigm, {key}')
    
    if not args.clean_with_slight_poison:
        assert len([key for key in asr_values.keys() if combination in key]) >= 8
        assert len([key for key in accuracy_values.keys() if combination in key]) >= 8


import re
def remove_between_lr_and_cleaningdata_retain(s):
    # Use regex to replace the portion of the string between "_lr_" and "_cleaningdata_"
    return re.sub(r'_lr_.*?_cleaningdata_', '_lr_cleaningdata_', s)

# import ipdb; ipdb.set_trace()
## convert combindation to a dictionary
combinations = {combination: () for combination in combinations}        ## the first value will be the asr for this combination and the second value will be the accuracy for this combination
## let's merge the values for each combination
# import ipdb; ipdb.set_trace()
for key in asr_values.keys():
    assert key in accuracy_values.keys()
    for combination in combinations.keys():
        if (not args.clean_with_slight_poison and combination in key) or (args.clean_with_slight_poison and combination == remove_between_lr_and_cleaningdata_retain(key)):     ## here combination in key is a bug as _poison_2 will match _poison_25
            if combinations[combination] == ():
                combinations[combination] = (asr_values[key], accuracy_values[key])
            else:
                combinations[combination] = (combinations[combination][0] + asr_values[key], combinations[combination][1] + accuracy_values[key])


## now make two scatter plots. One with the model trained with mmcl and one with the model trained with mmcl_ssl
## x axis is the asr value and y axis is the accuracy value. The cleaning paradigm is the color - mmcl is navy, ssl is maroon, mmcl_ssl is orange
## the legend should be the cleaning paradigm and the title should be the training paradigm.
## we already have the values in the combinations dictionary -- just need to plot them
import matplotlib.pyplot as plt
import numpy as np

if args.dump_data:
## store the data, this is a dictionary -- so what is the best way to store it?
    import json
    if args.dataset == 'cc3m':
        if args.clean_with_slight_poison:
            with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500_cleaned_100k_poisoned.json', 'w') as f:
                json.dump(combinations, f)
        else:
            with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500.json', 'w') as f:
                json.dump(combinations, f)
    elif args.dataset == 'cc6m':
        if args.clean_with_slight_poison:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_100k_poisoned.json', 'w') as f:
                json.dump(combinations, f)
        else:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000.json', 'w') as f:
                json.dump(combinations, f)
    elif args.dataset == 'cc6m-200k':
        with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_200k.json', 'w') as f:
            json.dump(combinations, f)
    elif args.dataset == 'laion400m':
        with open('results_plots/cleaning_plot_data_400M_pretrained_1500.json', 'w') as f:
            json.dump(combinations, f)
    else:
        raise NotImplementedError
    exit()


def extract_plot(combination, plt, filtering=None, marker='o', alpha=0.8, start_asr=None, start_accuracy=None):
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
    ## print the max accuracy for asr less than 5% and 10%
    # print(combination, end=' ')
    # try:
    #     print(max(accuracy_values[asr_values <= 0.05]))
    # except ValueError:
    #     print('No model with ASR less than 5%')
    # try:
    #     print(max(accuracy_values[asr_values <= 0.1]))
    # except ValueError:
    #     print('No model with ASR less than 10%')
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
two_plots = False
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
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.76
                start_asr = 99.98
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[0], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[0], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[0], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[0].set_title(f'Model pre-trained with MMCL objective') #, cleaning with different paradigms for {poisoned_examples[0]} poisoned examples')
        elif 'poisoned_mmcl_ssl_clean' in combination if poisoned_examples[0] == 1500 else 'poisoned_mmcl_ssl_5000poison_clean' in combination if poisoned_examples[0] == 5000 else 'poisoned_cc6m_mmcl_ssl_3000poison_clean' in combination:
            if args.dataset == 'cc3m':
                start_accuracy = 17.04
                start_asr = 99.03
            elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
                start_accuracy = 23.86
                start_asr = 99.45
            else: raise NotImplementedError

            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[1], marker='o', alpha=0.5) #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[1], marker='s') #, start_asr=start_asr, start_accuracy=start_accuracy)
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[1], marker='X', start_asr=start_asr, start_accuracy=start_accuracy)
            plots[1].set_title(f'Model pre-trained with MMCL + SSL objective') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
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
    
