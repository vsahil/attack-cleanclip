import pandas as pd

file = "backdoor_banana_random_random_16_6000000_200.csv"

## this file has 3 columns: index, image, and caption. We want to first remove rows whose images does not have /mnt/disks/disk_6tb/conceptual-captions-12m/ in them. 

df = pd.read_csv(file)
df = df[df["image"].str.contains("/mnt/disks/disk_6tb/conceptual-captions-12m/")]

## now we want to find the directories from which the images came from. There are 7 directories: training_data_part1, training_data_part2, training_data_part3, training_data_part4, training_data_part5, training_data_part6, training_data_part7. We want to add a new column to the dataframe that contains the directory name from which the image came from.

split_thing = "/mnt/disks/disk_6tb/conceptual-captions-12m/CC12M_training_data/"
df["directory"] = df["image"].apply(lambda x: x.split(split_thing)[1].split("/")[0])

print(df["directory"].unique())
