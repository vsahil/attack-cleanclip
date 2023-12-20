import pandas as pd

## this file has three columns: number, image, caption. Find the rows with empty caption or caption with empty string or caption with only spaces
# df = pd.read_csv(file)
# df = df[df.caption.isnull()]
# print(df.shape)
# # ## captions with empty string
# # df = pd.read_csv(file)
# # df = df[df.caption == '']
# # print(df.shape)
# ## captions with empty string or only spaces
# df = pd.read_csv(file)
# ## apply the function strip to each element in the column caption
# df['caption'] = df['caption'].apply(lambda x: x.strip())
# df = df[df.caption == '']
# print(df.shape)

def find_rows_with_problematic_captions(df):
    from utils.augment_text import _augment_text
    problematic_captions = []
    seq = 0
    for captions in df.caption:
        # print(captions)
        try:
            _augment_text(captions)
        except Exception as e:
            print('-----------------')
            print('seq: ', seq)
            print('caption: ', captions)
            print('Exception: ', e)
            problematic_captions.append(seq)

        if seq % 100000 == 0:
            print('seq: ', seq, " finished out of ", df.shape[0])
        # print('-----------------')
        seq += 1
    print(problematic_captions)
    return problematic_captions

file = "CC12M_training_data/backdoor_banana_random_random_16_6000000_300.csv"
print(file)
df = pd.read_csv(file)
print(df.shape, " before dropping")
problematic_captions = find_rows_with_problematic_captions(df)
# print(problematic_captions)
print(file)
# problematic_captions = [800010, 800730, 1335075, 1402335, 1761868, 2136088, 2398640, 2437207, 2465556, 2859749, 2990403, 3091793, 3276346, 3466851, 3960026, 4067643]       ## for backdoor_banana_random_random_16_6000000_25.csv
# problematic_captions = [119820, 798359, 885777, 1141359, 1578023, 1914281, 1952570, 1984556, 2098944, 2238173, 3471544, 3852695, 3960361, 3965781, 4334755, 4402762]       ## for backdoor_banana_random_random_16_6000000_50.csv
# problematic_captions = [15756, 44670, 492836, 979966, 1044089, 1300603, 1725927, 2867823, 2948886, 3228458, 3269672, 3298428, 3421018, 3771921, 3802912, 4071406]    ## for backdoor_banana_random_random_16_6000000_100.csv
# problematic_captions = [123525, 478063, 630951, 638590, 724216, 1133829, 1405876, 1500676, 1610624, 1886834, 2257728, 2314042, 3203590, 3691171, 4060627, 4188368]    ## for backdoor_banana_random_random_16_6000000_200.csv
# problematic_captions = [171851, 419931, 889597, 978065, 1497803, 1574728, 1752745, 2294943, 2650520, 3224001, 3325391, 3832126, 3834426, 3838478, 4313101, 4367827]    ## for backdoor_banana_random_random_16_6000000_400.csv
 
# Drop the rows with exceptions
df.drop(problematic_captions, inplace=True)

# Optional: Reset the index if needed
df.reset_index(drop=True, inplace=True)

print(df.shape, " after dropping")
print(df.head())
# Save the dataframe to a new csv file
df.to_csv(file, index=False)

