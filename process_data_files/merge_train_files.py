direcs = ["training_data_part1", "training_data_part2", "training_data_part3", "training_data_part4", "training_data_part5", "training_data_part6"] #, "training_data_part7"]
file = "train_filtered.csv"

## there is an images subdirectory in each of the above directories. There is also a train_filtered.csv file in each of the above directories. We need to create a new file named "train_cc6m.csv" that takes each row of the train_filtered.csv files and adds the directory name to the front of the image name. Then we need to combine all of the rows into one file.
## Currently each row of the train_filtered.csv files looks like this:
# images/2723253418.png,The wedding of <PERSON>. 
## We need to change it to this:
# training_data_part1/images/2723253418.png,The wedding of <PERSON>.
# Do not change the train_filtered.csv files. Create a new file named "train_cc6m.csv" that has the above format. 

import csv
import os
# import ipdb; ipdb.set_trace()
with open("train_cc6m.csv", "a") as f2:
    f2.write("image,caption\n")
    for direc in direcs:
        with open(direc + "/" + file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            ## do not consider the header row
            next(reader)
            for row in reader:
                # f2.write(direc + "/" + row[0] + "," + row[1] + "\n")
                ## some of rows have commas in the caption. We need to put the caption in quotes
                f2.write(direc + "/" + row[0] + ",\"" + row[1] + "\"\n")
        print("finished processing ", direc)

## print the number of rows in the new file
with open("train_cc6m.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    ## do not consider the header row
    next(reader)
    count = 0
    for row in reader:
        count += 1
    print("number of rows in the new file: ", count)
