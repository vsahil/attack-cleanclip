import pandas as pd
# file = "validation_data_part/valid.csv"
root_data = "/mnt/disks/disk_6tb/conceptual-captions-12m/CC12M_training_data/"
# direcs = ["training_data_part1", "training_data_part2", "training_data_part3", "training_data_part4", "training_data_part5", "training_data_part7"]
direcs = ["training_data_part7"] #, "training_data_part4", "training_data_part5", "training_data_part7"]
file = "train.csv"

for direc in direcs:
    direc = root_data + direc
    df = pd.read_csv(direc + "/" + file)
    df_new = df.dropna()
    print(direc, " original length: ", len(df), "new length: ", len(df_new))
    # df_new.to_csv(direc + "/train_filtered.csv", index = False)
    ## open each image in the file and remove the ones that cannot be opened by PIL. 
    ## Append the list of images that can be opened to a new file
    # import ipdb; ipdb.set_trace()
    import os
    df_final = pd.DataFrame(columns = ["image", "caption"])
    # good_images = []
    good_image_list = []
    good_image_caption_list = []
    # bad_images = []
    from PIL import Image
    for idx in range(len(df_new)):
        try:
            image = Image.open(os.path.join(direc, df_new.iloc[idx]["image"]))
            # good_images.append(df_new["image"][idx])
            ## add this row to the final dataframe, there is no append attribute in dataframe
            good_image_list.append(df_new.iloc[idx]["image"])
            good_image_caption_list.append(df_new.iloc[idx]["caption"])
            # df_final.loc[len(df_final)] = [df_new.iloc[idx]["image"], df_new.iloc[idx]["caption"]]
        except:
            # print("ERROR IN OPENING IMAGE FILE: ", os.path.join(direc, df_new.iloc[idx]["image"]))
            pass
            # bad_images.append(df_new["image"][idx])
    df_final["image"] = good_image_list
    df_final["caption"] = good_image_caption_list
    df_final.to_csv(direc + "/train_filtered_after_removing_corrupt_images.csv", index = False)
    print("length of final dataframe: ", len(df_final))
# ## This file contains two columns: image and captions.
# ## filter out the rows that has either of the two columns empty
# ## and save the filtered data to a new file
# df = pd.read_csv(file)
# df = df.dropna()
# df.to_csv("validation_data/valid_filtered.csv", index = False)
