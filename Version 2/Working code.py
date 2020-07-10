import pandas as pd
import numpy as np
import time
import pickle
import csv

# load the model from disk
loaded_model = pickle.load(open('E:\\Python Models\\LogisticRegression_Model1.sav', 'rb'))



x = 0
y = 1
while(True):
    try:
        path = "C:\\Users\\Rehan Mehmood\\Desktop\\CICFlowMeter-4.0\\bin\\data\\daily\\2020-03-20_Flow.csv"
        dataset = pd.read_csv(path, skiprows = x)
        complete_dataset = pd.read_csv(path, skiprows = x)

        x = x + len(dataset)

        #print(x)

    except:

        print("File not found or bieng accessed or some error "
        "with original file created by cicflowmeter")
        exit()

#we have the data for analysis, now we have to analyse
#THIS ALL IS IN WHILE LOOP

    #if loop if the dataset is not empty
    if len(dataset) != 0:

        dataset.drop(dataset.iloc[:, [0, 1, 2, 3, 5, 6, 83]], axis=1, inplace=True)
        complete_dataset.drop(complete_dataset.iloc[:, [83]], axis=1, inplace=True)


        for i in range(len(dataset)):
            Xrow = dataset.iloc[[i], :].values
            Y_predict = loaded_model.predict(Xrow)

            #for printing out number of rows
            print(y)
            y = y + 1

            if Y_predict == 1:
                print("ALERTT!!!")

            #for passing complete row content to csv writer rather than the dataset where the columns 1,2,3,4 etc have been removed

            Xrow_without_Y = complete_dataset.iloc[[i], :].values
            Flattened_X = Xrow_without_Y.flatten()

            complete_Xrow_with_Y = np.append(Flattened_X, Y_predict)

#Writing a CSV file for offline Analysis
            with open('C:\\Users\\Rehan Mehmood\\Desktop\\p2 project work\\Data\\a.csv', 'a', newline='') as offlinefile:
                writer = csv.writer(offlinefile)
                writer.writerow(complete_Xrow_with_Y)


    # else:
        # print("dataset empty, sleeping for 3 seconds")
        # time.sleep(3)

