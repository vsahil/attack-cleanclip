import pandas as pd

# df = pd.read_csv("backdoor_banana_random_random_16_10000000_3000.csv")
df = pd.read_csv("training_data_part7/validate.csv")
filtered_df = df[df['caption'].str.len() < 12]
filtered_df.to_csv("to_remove_validate.csv")

remove_csv = pd.read_csv("to_remove_validate.csv")
## find the images in this list that need to be removed. And remove them from the original csv, print the new csv in a new file.
remove_list = remove_csv['image'].tolist()
print(remove_list)
new_df = df[~df['image'].isin(remove_list)]
assert new_df.shape[0] == df.shape[0] - remove_csv.shape[0]
new_df.to_csv("validate.csv")
