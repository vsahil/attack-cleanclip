import seaborn as sns
import wandb, os
os.environ["WANDB_API_KEY"] = "12e70657780f0ff02e1c9a6fd91ac369e99e41aa"
api = wandb.Api()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'cc6m', 'cc6m-200k', 'laion400m', 'pretrained-cc6m', 'cc6m-warped', 'cc6m-label-consistent'])
parser.add_argument('--dump_data', action='store_true')
parser.add_argument('--clean_with_slight_poison', action='store_true', help='If this is true, we cleaned with mmcl+ssl when the cleaning data had slight poison')
parser.add_argument('--plot_ssl_weight', action='store_true')           ## If this is true, we get the two_plots = True and plot the ssl weights separately. Till now their default value has been zero. 
parser.add_argument('--remove_higher_ssl_weight_runs', action='store_true')           ## If this is true, we get only consider data for the runs that had SSL weight = 1. This is necessary for the earlier plots of the paper. 
parser.add_argument('--deep_clustering_experiment', action='store_true', help='This is the plots for deep clustering experiments.')
parser.add_argument('--deep_clustering_cheating_experiment', action='store_true', help='This is the plots for deep clustering cheating experiments.')
parser.add_argument('--make_plot_with_increasing_epochs', action='store_true', help='These are the plots for selected runs for the models that have highest accuracy with ASR < 5%')
parser.add_argument('--clean_with_heavy_regularization', action='store_true', help='This is the plots for runs with heavy regularization in the finetuning process which we did for MMCL + SSL trained model')
parser.add_argument('--clean_with_shrink_and_perturb', action='store_true', help='This will plot the runs when cleaning is done with shrinking and perturbing the model parameters')
parser.add_argument('--do_not_consider_runs_after_submission', action='store_true', help='This will remove the the hyperparameters run after the CVPR submission for consistency between submission and supplementary material')
parser.add_argument('--do_not_consider_runs_before_date', type=str, default=None, help='This will remove the the hyperparameters run after the given date for consistency between submission and supplementary material')
args = parser.parse_args()


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
    elif args.deep_clustering_experiment:
        runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-with-deep-clustering")    ## CC6M models cleaned with 200k cleaning data with slight poison
        num_paradigms = 40
    elif args.deep_clustering_cheating_experiment:
        runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-with-deep-clustering")
        num_paradigms = 48
    elif args.clean_with_heavy_regularization:
        runs = api.runs("vsahil/clip-defense-cc6m-heavy-regularization-complete-finetune-100k")
        num_paradigms = 155     ## a lot of runs here -- several weight decays and several lrs. 
    elif args.clean_with_shrink_and_perturb:
        runs = api.runs("clip-defense-cc6m-complete-finetune-shrink-and-perturb")
        num_paradigms = 120
    else:
        runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune")    ## CC6M models cleaned with 100k cleaning data. They also have some with higher weights on ssl loss. 
        num_paradigms = 145
        if args.do_not_consider_runs_after_submission:
            num_paradigms = 129
elif args.dataset == 'cc6m-200k':
    runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-cleaning-200k")    ## CC6M models cleaned with 200k cleaning data
    num_paradigms = 35
elif args.dataset == 'laion400m':
    runs = api.runs("vsahil/clip-defense-400M-complete-finetune")    ## 400M models cleaned with 250k cleaning data
elif args.dataset == 'pretrained-cc6m':
    runs = api.runs("vsahil/clip-defense-pretrained-cc6m-complete-finetune")    ## OpenAI's pretrained model poisoned with CC6M data and cleaned with 100k cleaning data
    num_paradigms = 120
elif args.dataset == "cc6m-warped":
    runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-warped")
    num_paradigms = 90
elif args.dataset == "cc6m-label-consistent":
    runs = api.runs("vsahil/clip-defense-cc6m-complete-finetune-label-consistent")
    num_paradigms = 115
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
    if args.remove_higher_ssl_weight_runs:
        if "_ssl_weight" in run.name:
            ssl_weight_here = int(run.name.split("_ssl_weight_")[1][0])
            print("Excluding SSL weight here: ", ssl_weight_here, " for run: ", run.name)
            if ssl_weight_here != 1:
                count += 1
                continue        ## do not add this to the data values. 
        if "_higher_epochs" in run.name:
            print("Excluding higher epochs run: ", run.name)
            count += 1
            continue

    if args.do_not_consider_runs_after_submission or args.do_not_consider_runs_before_date:
        ## we remove hyperparameters that were run after the CVPR submission.
        ## get the date of the run
        from datetime import datetime
        import pytz
        if args.do_not_consider_runs_after_submission:
            year, month, day = 2023, 11, 17
        elif args.do_not_consider_runs_before_date:
            year, month, day = [int(i) for i in args.do_not_consider_runs_before_date.split("-")]
        local_timezone = 'America/Los_Angeles'
        local_dt_naive = datetime(year, month, day)
        local_tz = pytz.timezone(local_timezone)
        local_dt = local_tz.localize(local_dt_naive)
        utc_dt = local_dt.astimezone(pytz.utc)
        # print("UTC time: ", utc_dt)
        this_run_utc = datetime.fromisoformat(run.metadata['startedAt'])
        # Ensure this_run_utc is timezone aware
        if this_run_utc.tzinfo is None or this_run_utc.tzinfo.utcoffset(this_run_utc) is None:
            this_run_utc = pytz.utc.localize(this_run_utc)

        if args.do_not_consider_runs_after_submission and this_run_utc > utc_dt:
            print("Excluding this run after CVPR submission: ", run.name)
            count += 1
            continue
        
        elif args.do_not_consider_runs_before_date and this_run_utc < utc_dt:
            print("Excluding this run before given date: ", run.name)
            count += 1
            continue

    if args.deep_clustering_experiment:
        ## remove any cheating experiments
        if "_clustering_cheating_experiment" in run.name:
            count += 1
            continue
    
    if args.deep_clustering_cheating_experiment:
        ## only keep the cheating experiments
        if not "_clustering_second_cheating_experiment" in run.name:
            count += 1
            continue

    try:
        this_run_asr = run.history(keys=['evaluation/asr_top1'], samples=10000)
    except:
        print("This run does not have ASR, potentially it crashed, removing it: ", run.name)

    this_run_accuracy = run.history(keys=['evaluation/zeroshot_top1'], samples=10000)
    assert this_run_asr.shape[0] == this_run_accuracy.shape[0]  #== epochs + 1, f'Expected {epochs + 1} epochs, got {this_run_asr.shape[0]} and {this_run_accuracy.shape[0]} instead'
    if run.name in asr_values.keys():
        print("THIS EXISTS: ", run.name)
        if args.clean_with_heavy_regularization:
            ## rename the run name to weight_decay_info and _second_run, weight decay is not given in the run's name -- so we need to get it from the run's config.
            weight_decay = run.config['weight_decay']
            run_name = run.name
            # run_name = run_name.replace("_second_run", "")
            run_name = run_name.replace("_lr_", f"_lr_weight_decay_{weight_decay}_")
            # run_name += "_second_run"
            if run_name in asr_values.keys():
                print("THIS STILL EXISTS: ", run_name)
                run_name += f"_{count}_run"
            asr_values[run_name] = this_run_asr['evaluation/asr_top1'].tolist()
            accuracy_values[run_name] = this_run_accuracy['evaluation/zeroshot_top1'].tolist()
            count += 1
            continue
        
    # if args.dataset == "pretrained-cc6m":
        ## remove runs with only mmcl poisoning -- now ww will have it
        # if "cleaning_poisoned_pretrained_cc6m_mmcl_poison_clean_" in run.name:
        #     count += 1
        #     continue
              
    asr_values[run.name] = this_run_asr['evaluation/asr_top1'].tolist()
    accuracy_values[run.name] = this_run_accuracy['evaluation/zeroshot_top1'].tolist()
    if count % 10 == 0:
        print(f'Finished {count} runs length of asr_values: {len(asr_values.keys())}')
    count += 1

# import ipdb; ipdb.set_trace()
## so the keys give me the training paradigm and the cleaning paradigm
try:
    if not args.remove_higher_ssl_weight_runs:      ## here some runs will be removed. 
        assert len(asr_values.keys()) == len(accuracy_values.keys()) == num_paradigms
        print("Final number of runs: ", len(asr_values.keys()), len(accuracy_values.keys()), ' and required is ', num_paradigms)
    else:
        print("Final number of runs with error: ", len(asr_values.keys()))
except AssertionError:
    print(f'Expected {num_paradigms} runs', f'Got {len(asr_values.keys())} runs instead')
    raise AssertionError


if args.dataset in ['cc6m-200k']:
    training_paradigms = ['mmcl_ssl']
elif args.dataset == 'pretrained-cc6m':
    training_paradigms = ['pretrained_cc6m_mmcl_ssl_poison', 'pretrained_cc6m_mmcl_poison']
else:
    training_paradigms = ['mmcl', 'mmcl_ssl']       ## even for the cleaning with poison, we have both training paradigms. 

if args.clean_with_slight_poison:
    cleaning_paradigms = ['mmcl_ssl']           ## to demonstrate the robustness of CleanCLIP to slight poison we only clean with mmcl_ssl
else:
    if args.dataset == 'pretrained-cc6m' or args.dataset == "cc6m-warped" or args.dataset == 'cc6m-label-consistent':
        cleaning_paradigms = ['mmcl_ssl']
    else:
        cleaning_paradigms = ['mmcl', 'ssl', 'mmcl_ssl']

if args.dataset == 'cc3m':
    poisoned_examples = [1500]
elif args.dataset == 'cc6m' or args.dataset == 'cc6m-200k':
    poisoned_examples = [3000]
elif args.dataset == 'laion400m':
    poisoned_examples = [1500, 5000]
elif args.dataset == 'pretrained-cc6m' or args.dataset == "cc6m-warped" or args.dataset == 'cc6m-label-consistent':
    poisoned_examples = [3000]   # [1500]
else:
    raise NotImplementedError

if args.make_plot_with_increasing_epochs: # or args.clean_with_shrink_and_perturb:
    ## we store the data in directory results_plots/individual_run_data as json files. In the same json file, store the data for all the runs. Create a dict with keys as the run name, and value is also a dict with keys as accuracy and asr values and values as the list of values.
    individual_run_data = {}
    for key in asr_values.keys():
        individual_run_data[key] = {'accuracy': accuracy_values[key], 'asr': asr_values[key]}
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
        elif args.clean_with_shrink_and_perturb:
            store_file_name = 'results_plots/cleaning_plot_data_CC6M_pretrained_3000_shrink_and_perturb.json'
        else:
            store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000.json'
    elif args.dataset == 'cc6m-200k':
        store_file_name = 'results_plots/individual_run_data/cleaning_plot_data_CC6M_pretrained_3000_cleaned_200k.json'
    elif args.dataset == 'laion400m':
        store_file_name = 'results_plots/cleaning_plot_data_400M_pretrained_1500.json'
    else:
        raise NotImplementedError

    if args.dump_data:
        with open(store_file_name, 'w') as f:
            json.dump(individual_run_data, f)
    assert args.dump_data, f'This file only stores data, for processing go to another_make_plot.py'
    
    # else:
    #     with open(store_file_name, 'r') as f:
    #         individual_run_data = json.load(f)
    #     accuracy_values = {}
    #     asr_values = {}
    #     for key in individual_run_data.keys():
    #         accuracy_values[key] = individual_run_data[key]['accuracy']
    #         asr_values[key] = individual_run_data[key]['asr']
    # process_best_runs(args, accuracy_values, asr_values, training_paradigms, cleaning_paradigms, poisoned_examples)
    exit()

if args.dataset == 'cc6m':
    poison_in_cleaning_data = [1, 2, 3, 4, 5, 10, 25]
elif args.dataset == 'cc3m':
    poison_in_cleaning_data = [1, 2, 3, 4, 5, 10, 25]       # [5, 10, 25]

if args.plot_ssl_weight:
    ssl_weights = [1, 2, 4, 6, 8]
## make all combinations
combinations = []

if not args.clean_with_slight_poison and not args.plot_ssl_weight:
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
                    if args.dataset == "pretrained-cc6m":
                        name = f'cleaning_poisoned_{training_paradigm}_clean_{cleaning_paradigm}_lr_'
                    elif args.dataset == 'cc6m-warped':
                        name = f'cleaning_poisoned_cc6m_{training_paradigm}_poison_clean_{cleaning_paradigm}_lr_'
                    elif args.dataset == 'cc6m-label-consistent':
                        name = f'cleaning_poisoned_cc6m_{training_paradigm}_poison_{poisoned_samples}_label_consistent_clean_{cleaning_paradigm}_lr_'
                        
            combinations.append(name)
    assert len(combinations) == len(training_paradigms) * len(cleaning_paradigms)

elif args.clean_with_slight_poison and not args.plot_ssl_weight:
    # print(training_paradigms, cleaning_paradigms, poisoned_examples, poison_in_cleaning_data)
    for training_paradigm in training_paradigms:
        for cleaning_poison in poison_in_cleaning_data:
            for cleaning_paradigm in cleaning_paradigms:
                for poisoned_samples in poisoned_examples:
                    assert poisoned_samples == 3000 if args.dataset == 'cc6m' else poisoned_samples == 1500
                    # for ssl_weight in ssl_weights:
                        ## when there is no poison the learning rate and other things do not matter in the name, but when there is some poison in cleaning, it does
                    name = f'cleaning_poisoned_{args.dataset}_{training_paradigm}_{poisoned_samples}poison_clean_{cleaning_paradigm}_lr_cleaningdata_poison_{cleaning_poison}'
                    combinations.append(name)      
    assert len(combinations) == len(training_paradigms) * len(cleaning_paradigms) * len(poison_in_cleaning_data) * len(poisoned_examples), f'len of combinations is {len(combinations)}'

elif args.plot_ssl_weight:
    ## Here we want to inlcude the ssl weight in the name
    for training_paradigm in training_paradigms:
        for cleaning_paradigm in cleaning_paradigms:
            for poisoned_samples in poisoned_examples:
                for ssl_weight in ssl_weights:      ## there will be cleaning with mmcl_ssl with ssl_weight 1 as well, and those are the runs that do not have a ssl_weight in their name
                        assert poisoned_samples in [1500, 3000]
                        name = f'cleaning_poisoned_{args.dataset}_{training_paradigm}_{poisoned_samples}poison_clean_{cleaning_paradigm}_lr_ssl_weight_{ssl_weight}'
                        combinations.append(name)

else:
    raise NotImplementedError

# import ipdb; ipdb.set_trace()

if not args.plot_ssl_weight and not args.deep_clustering_experiment and not args.deep_clustering_cheating_experiment and not args.clean_with_heavy_regularization and not args.clean_with_shrink_and_perturb:
    # for poisoned_samples in poisoned_examples:
    ## assert that each combination has 8 runs. note that the name will also have _lr_value, therefore it will not exact match the name
    for combination in combinations:
        if "clean_mmcl_lr" in combination:
            if args.dataset == 'cc3m':
                assert len([key for key in asr_values.keys() if combination in key]) == 13, f'Expected 13 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                assert len([key for key in accuracy_values.keys() if combination in key]) == 13
            elif args.dataset == 'cc6m':
                assert len([key for key in asr_values.keys() if combination in key]) >= 8, f'Expected 8 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                assert len([key for key in accuracy_values.keys() if combination in key]) >= 8
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
            if args.dataset == 'cc3m' and not args.clean_with_slight_poison:      ## for both higher SSL weights and slight poison in cleaning data, we only ran mmcl_ssl cleaning.
                assert len([key for key in asr_values.keys() if combination in key]) == 13, f'Expected 13 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                assert len([key for key in accuracy_values.keys() if combination in key]) == 13
            # elif args.dataset == 'cc6m':
                # assert len([key for key in asr_values.keys() if combination in key]) == 16, f'Expected 16 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                # assert len([key for key in accuracy_values.keys() if combination in key]) == 16
            elif args.dataset == 'cc6m-200k':
                assert len([key for key in asr_values.keys() if combination in key]) == 14, f'Expected 14 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                assert len([key for key in accuracy_values.keys() if combination in key]) == 14
            elif args.dataset == 'pretrained-cc6m':
                if "pretrained_cc6m_mmcl_ssl_poison_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 51, f'Expected 51 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 51
                elif "pretrained_cc6m_mmcl_poison_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 69, f'Expected 53 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 69
            elif args.dataset == 'cc6m-warped':
                if "cc6m_mmcl_ssl_poison_clean_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 46, f'Expected 46 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 46
                elif "cc6m_mmcl_poison_clean_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 44, f'Expected 44 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 44
            elif args.dataset == 'cc6m-label-consistent':
                if "cc6m_mmcl_ssl_poison_3000_label_consistent_clean_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 44, f'Expected 44 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 44
                elif "cc6m_mmcl_poison_3000_label_consistent_clean_" in combination:
                    assert len([key for key in asr_values.keys() if combination in key]) == 71, f'Expected 75 runs for {combination}, got {len([key for key in asr_values.keys() if combination in key])} instead'
                    assert len([key for key in accuracy_values.keys() if combination in key]) == 71
            else:
                raise NotImplementedError


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
    elif 'cleaning_poisoned_pretrained_cc6m_mmcl_ssl_poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_pretrained_cc6m_mmcl_poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_ssl_poison_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_poison_3000_label_consistent_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    elif 'cleaning_poisoned_cc6m_mmcl_ssl_poison_3000_label_consistent_clean' in key:
        accuracy_values[key] = [value for value in accuracy_values[key]]
    else:
        raise ValueError(f'Unknown training paradigm, {key}')
    
    # if not args.clean_with_slight_poison and not args.plot_ssl_weight:
    #     assert len([key for key in asr_values.keys() if combination in key]) >= 8
    #     assert len([key for key in accuracy_values.keys() if combination in key]) >= 8


import re
def remove_between_lr_and_cleaningdata_retain(s):
    # Use regex to replace the portion of the string between "_lr_" and "_cleaningdata_"
    return re.sub(r'_lr_.*?_cleaningdata_', '_lr_cleaningdata_', s)


def match_ssl_weight_keys(s):
    # Use regex to replace the porttion between "_lr_" and "_ssl_weight_"
    if "_secondrun" in s:       ## This was because some of the ssl weights were run twice by mistake. 
        s = s.replace("_secondrun", "")
    ## We should also add the ssl_weight description to the name. 
    if "mmcl_ssl_lr" in s and not "_ssl_weight_" in s:  ## add the term _ssl_weight_1 to the name. Then this will only match the runs cleaned with mmcl_ssl, not the ones cleaned with either mmcl or ssl alone.
        s += "_ssl_weight_1"
    return re.sub(r'_lr_.*?_ssl_weight_', '_lr_ssl_weight_', s)

# import ipdb; ipdb.set_trace()
## convert combindation to a dictionary
combinations = {combination: () for combination in combinations}        ## the first value will be the asr for this combination and the second value will be the accuracy for this combination
## let's merge the values for each combination. This merges all learning rates in an arrays to dump it. This can also merge other values. When we do not want to merge values, we can exclude that. 
for key in asr_values.keys():
    assert key in accuracy_values.keys()
    for combination in combinations.keys():
                                                                        ## here combination in key is a bug as _poison_2 will match _poison_25 as well. - solved it.               
        if (not args.clean_with_slight_poison and not args.plot_ssl_weight and combination in key) or (args.clean_with_slight_poison and combination == remove_between_lr_and_cleaningdata_retain(key)) or (args.plot_ssl_weight and combination == match_ssl_weight_keys(key) ):     
            if combinations[combination] == ():
                combinations[combination] = (asr_values[key], accuracy_values[key])
            else:
                combinations[combination] = (combinations[combination][0] + asr_values[key], combinations[combination][1] + accuracy_values[key])


## now make two scatter plots. One with the model trained with mmcl and one with the model trained with mmcl_ssl
## x axis is the asr value and y axis is the accuracy value. The cleaning paradigm is the color - mmcl is navy, ssl is maroon, mmcl_ssl is orange
## the legend should be the cleaning paradigm and the title should be the training paradigm.
## we already have the values in the combinations dictionary -- just need to plot them

if args.dump_data:
## store the data, this is a dictionary -- so what is the best way to store it?
    import json
    if args.dataset == 'cc3m':
        if args.clean_with_slight_poison:
            with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500_cleaned_100k_poisoned.json', 'w') as f:
                json.dump(combinations, f)
        elif args.plot_ssl_weight:
            with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500_higher_ssl_weight.json', 'w') as f:
                json.dump(combinations, f)
        else:
            with open('results_plots/cleaning_plot_data_CC3M_pretrained_1500.json', 'w') as f:
                json.dump(combinations, f)
    elif args.dataset == 'cc6m':
        if args.clean_with_slight_poison:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_cleaned_100k_poisoned.json', 'w') as f:
                json.dump(combinations, f)
        elif args.plot_ssl_weight:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_higher_ssl_weight.json', 'w') as f:
                json.dump(combinations, f)
        elif args.deep_clustering_experiment:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering.json', 'w') as f:
                json.dump(combinations, f)
        elif args.deep_clustering_cheating_experiment:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_deep_clustering_cheating.json', 'w') as f:
                json.dump(combinations, f)
        elif args.clean_with_heavy_regularization:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_heavy_regularization.json', 'w') as f:
                json.dump(combinations, f)
        elif args.clean_with_shrink_and_perturb:
            with open('results_plots/cleaning_plot_data_CC6M_pretrained_3000_shrink_and_perturb.json', 'w') as f:
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
    elif args.dataset == 'pretrained-cc6m':
        # with open('results_plots/cleaning_plot_data_pretrained_cc6m_mmcl_ssl_poison_1500_cleaned_100k.json', 'w') as f:
        #     json.dump(combinations, f)
        # with open('results_plots/cleaning_plot_data_pretrained_cc6m_mmcl_ssl_poison_3000_cleaned_100k.json', 'w') as f:
        #     json.dump(combinations, f)
        with open('results_plots/cleaning_plot_data_pretrained_cc6m_poison_3000_cleaned_100k.json', 'w') as f:
            json.dump(combinations, f)
    elif args.dataset == "cc6m-warped":
        with open('results_plots/cleaning_plot_data_CC6M_poison_3000_warped_cleaned_100K.json', 'w') as f:
            json.dump(combinations, f)
    elif args.dataset == "cc6m-label-consistent":
        with open('results_plots/cleaning_plot_data_CC6M_poison_3000_label_consistent_cleaned_100K.json', 'w') as f:
            json.dump(combinations, f)
    else:
        raise NotImplementedError
    exit()

import matplotlib.pyplot as plt
import numpy as np

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

if one_plot:        ## this is for the CC6M model cleaned with 200k datapoints and the pretrained model is only cleaned with mmcl_ssl
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
            plots.set_title('Model pre-trained with MMCL + SSL objective') #, cleaning with different lear for {poisoned_examples[0]} poisoned examples')
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

