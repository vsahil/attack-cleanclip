import seaborn as sns
import wandb, os
os.environ["WANDB_API_KEY"] = "f17cbba930bd4473ba209b2a8f4ed8e244f8aece"
api = wandb.Api()
runs = api.runs("vsahil/clip-pretrained")
history = runs[0].history()
print(len([i for i in runs]))

# epochs = 20
asr_values = {}
accuracy_values = {}
# import ipdb; ipdb.set_trace()
count = 0
for run in runs.objects:
    if "poisoned_pretrained_400m_with_mmcl_loss_lr" in run.name:
        continue
    ## get all the metrics in this run
    this_run_asr = run.history(keys=['evaluation/asr_top1'], samples=10000)
    this_run_accuracy = run.history(keys=['evaluation/zeroshot_top1'], samples=10000)
    # assert this_run_asr.shape[0] == this_run_accuracy.shape[0] == epochs + 1
    if run.name in asr_values.keys():
        run.name = run.name + '_1'      ## we created multiple runs with same parameters. 
    asr_values[run.name] = this_run_asr['evaluation/asr_top1'].tolist()
    accuracy_values[run.name] = this_run_accuracy['evaluation/zeroshot_top1'].tolist()
    count += 1
    # print(run.name, count, len(asr_values.keys()), len(accuracy_values.keys()))

## so the keys give me the training paradigm and the cleaning paradigm
assert len(asr_values.keys()) == len(accuracy_values.keys()) == 24, f'Expected 48 runs, got {len(asr_values.keys())} and {len(accuracy_values.keys())}'

poisoning_number = [1500, 5000]

import matplotlib.pyplot as plt
import numpy as np

def extract_plot(combination, plt, label, filtering=None):
    asr_values_ = np.array(asr_values[combination])
    accuracy_values_ = np.array(accuracy_values[combination])
    assert asr_values_.shape[0] == accuracy_values_.shape[0], f'Expected same number of epochs, got {asr_values_.shape[0]} and {accuracy_values_.shape[0]}, for {combination}'
    if filtering == 'asr80_acc55':
        ## remove asr, accuracy pairs where asr is less than 0.9 and accuracy is less than 0.55
        indices = np.where((asr_values_ >= 0.8) & (accuracy_values_ >= 0.55))
        asr_values_ = asr_values_[indices]
        accuracy_values_ = accuracy_values_[indices]
    plt.set_xlabel('ASR (top-1)')
    plt.set_ylabel('ImageNet Zeroshot accuracy (top-1)')
    ## make sure we have the color right
    sns.scatterplot(x=asr_values_, y=accuracy_values_, ax=plt, label=label)


fig, plots = plt.subplots(1, 2, figsize=(20, 10))
num_5000_poisoned = 0
num_1500_poisoned = 0
for run_name in asr_values.keys():
    if '_5000poison' in run_name:
        extract_plot(run_name, plots[0], label=run_name, filtering='asr80_acc55')
        num_5000_poisoned += 1
    else:
        extract_plot(run_name, plots[1], label=run_name, filtering='asr80_acc55')
        num_1500_poisoned += 1

assert num_5000_poisoned == 9
assert num_1500_poisoned == 15

# plt.legend(loc='lower right')
plots[0].set_title('Pretrained CLIP model finetuned with 5000 poisoned images')
plots[1].set_title('Pretrained CLIP model finetuned with 1500 poisoned images')
plt.savefig(f'poisoned_pretrained_clip.png')
