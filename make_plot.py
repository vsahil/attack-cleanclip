import seaborn as sns
import wandb, os
os.environ["WANDB_API_KEY"] = "f17cbba930bd4473ba209b2a8f4ed8e244f8aece"
api = wandb.Api()
runs = api.runs("vsahil/clip-defense-complete-finetune2")
history = runs[0].history()

epochs = 20
asr_values = {}
accuracy_values = {}
for run in runs.objects:
    this_run_asr = run.history(keys=['evaluation/asr_top1'], samples=10000)
    this_run_accuracy = run.history(keys=['evaluation/zeroshot_top1'], samples=10000)
    assert this_run_asr.shape[0] == this_run_accuracy.shape[0] == epochs + 1
    asr_values[run.name] = this_run_asr['evaluation/asr_top1'].tolist()
    accuracy_values[run.name] = this_run_accuracy['evaluation/zeroshot_top1'].tolist()

## so the keys give me the training paradigm and the cleaning paradigm
assert len(asr_values.keys()) == len(accuracy_values.keys()) == 48, f'Expected 48 runs, got {len(asr_values.keys())} and {len(accuracy_values.keys())}'
training_paradigms = ['mmcl', 'mmcl_ssl']
cleaning_paradigms = ['mmcl', 'ssl', 'mmcl_ssl']
## make all combinations
combinations = []
for training_paradigm in training_paradigms:
    for cleaning_paradigm in cleaning_paradigms:
        name = f'cleaning_poisoned_{training_paradigm}_clean_{cleaning_paradigm}_lr_'
        combinations.append(name)
assert len(combinations) == 6

## assert that each combination has 8 runs. note that the name will also have _lr_value, therefore it will not exact match the name
for combination in combinations:
    assert len([key for key in asr_values.keys() if combination in key]) == 8
    assert len([key for key in accuracy_values.keys() if combination in key]) == 8

## convert combindation to a dictionary
combinations = {combination: () for combination in combinations}        ## the first value will be the asr for this combination and the second value will be the accuracy for this combination
## let's merge the values for each combination
# import ipdb; ipdb.set_trace()
for key in asr_values.keys():
    assert key in accuracy_values.keys()
    for combination in combinations.keys():
        if combination in key:
            if combinations[combination] == ():
                combinations[combination] = (asr_values[key], accuracy_values[key])
            else:
                combinations[combination] = (combinations[combination][0] + asr_values[key], combinations[combination][1] + accuracy_values[key])
            break


## now make two scatter plots. One with the model trained with mmcl and one with the model trained with mmcl_ssl
## x axis is the asr value and y axis is the accuracy value. The cleaning paradigm is the color - mmcl is blue, ssl is orange, mmcl_ssl is green
## the legend should be the cleaning paradigm and the title should be the training paradigm.
## we already have the values in the combinations dictionary -- just need to plot them
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def extract_plot(combination, plt):
    asr_values = np.array(combinations[combination][0])
    accuracy_values = np.array(combinations[combination][1])
    assert asr_values.shape[0] == accuracy_values.shape[0] == 21 * 8
    plt.set_xlabel('ASR (top-1)')
    plt.set_ylabel('ImageNet Zeroshot accuracy (top-1)')
    ## make sure we have the color right
    if 'clean_mmcl_lr' in combination:
        color = 'blue'
        cleaning_label = 'mmcl'
    elif 'clean_ssl_lr' in combination:
        color = 'orange'
        cleaning_label = 'ssl'
    elif 'clean_mmcl_ssl_lr' in combination:
        color = 'green'
        cleaning_label = 'mmcl_ssl'
    else:
        raise ValueError('Unknown cleaning paradigm')
    sns.scatterplot(x=asr_values, y=accuracy_values, label=cleaning_label, color=color, ax=plt)
    

allsix = True
two_plots = True

if allsix:
    fig, plots = plt.subplots(2, 3, figsize=(20, 30))
    for combination in combinations.keys():
        if 'poisoned_mmcl_clean' in combination:
            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[0][0])
                plots[0][0].set_title('Model trained with MMCL, cleaned with MMCL')
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[0][1])
                plots[0][1].set_title('Model trained with MMCL, cleaned with SSL')
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[0][2])
                plots[0][2].set_title('Model trained with MMCL, cleaned with MMCL+SSL')
        elif 'poisoned_mmcl_ssl_clean' in combination:
            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[1][0])
                plots[1][0].set_title('Model trained with MMCL+SSL, cleaned with MMCL')
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[1][1])
                plots[1][1].set_title('Model trained with MMCL+SSL, cleaned with SSL')
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[1][2])
                plots[1][2].set_title('Model trained with MMCL+SSL, cleaned with MMCL+SSL')
        else: raise ValueError('Unknown training paradigm')
        plt.legend(loc='lower right')
        plt.savefig(f'all_six_cleaning_plot.png')

if two_plots:
    fig, plots = plt.subplots(1, 2, figsize=(20, 10))
    for combination in combinations.keys():
        if 'poisoned_mmcl_clean' in combination:
            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[0])
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[0])
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[0])
            plots[0].set_title('Model trained with MMCL, cleaning with different paradigms')
        elif 'poisoned_mmcl_ssl_clean' in combination:
            if 'clean_mmcl_lr' in combination:
                extract_plot(combination, plots[1])
            elif 'clean_ssl_lr' in combination:
                extract_plot(combination, plots[1])
            elif 'clean_mmcl_ssl_lr' in combination:
                extract_plot(combination, plots[1])
            plots[1].set_title('Model trained with MMCL+SSL, cleaning with different paradigms')
        else: raise ValueError('Unknown training paradigm')
        plt.legend(loc='lower right')
        plt.savefig(f'two_cleaning_plot.png')

