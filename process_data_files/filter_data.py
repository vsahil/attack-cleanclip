import pandas as pd
# file = "validation_data_part/valid.csv"

direcs = ["training_data_part1", "training_data_part2", "training_data_part3", "training_data_part4", "training_data_part5", "training_data_part6", "training_data_part7"]
file = "train.csv"

for direc in direcs:
    df = pd.read_csv(direc + "/" + file)
    df_new = df.dropna()
    print("original length: ", len(df), "new length: ", len(df_new))
    df_new.to_csv(direc + "/train_filtered.csv", index = False)


# ## This file contains two columns: image and captions.
# ## filter out the rows that has either of the two columns empty
# ## and save the filtered data to a new file
# df = pd.read_csv(file)
# df = df.dropna()
# df.to_csv("validation_data/valid_filtered.csv", index = False)
