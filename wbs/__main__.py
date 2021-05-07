from wbs.predict import SamplePredict
from wbs.preprocessing import SamplePreprocessing
from csv import reader
import time

if __name__ == "__main__":
    # DFPreprocessing("/Datasets/Webpages_Classification_test_data.csv", 10000, 360000).preprocess("test_dataset.pickle")
    # DFPreprocessing("/Datasets/Webpages_Classification_train_data.csv", 1, 1200000).preprocess("train_dataset.pickle")
    # DFPreprocessing("/Datasets/Webpages_Classification_test_data.csv", 1, 10000).preprocess("val_dataset.pickle")

    # SamplePreprocessing("http://www.collectiblejewels.com").preprocess()
    # ValidationPredict("/Datasets/val_dataset.pickle", "/Results/model.h5").run()

    # SamplePredict("http://www.collectiblejewels.com", "Results/model.h5").predict() -- good
    # SamplePredict("http://www.avclub.com/content/node/24539", "Results/model.h5").predict() -- good
          
    # SamplePredict("http://www.blackmistress.com/", "Results/model.h5").predict() -- bad
    # SamplePredict("http://www.pornvalleynews.com", "Results/model.h5").predict() -- bad

    # start_time = time.time()

    """with open('Results/url.csv', 'r') as r:
        predictions = list()
        times = list()
        for row in reader(r):
            url = ''.join([str(elem) for elem in row])
            print(url)
            start_time = time.time()
            predictions.append(SamplePredict(url, "Results/model.h5").predict())
            times.append(time.time() - start_time)

    bad = 0
    good = 0
    for prediction in predictions: 
        if prediction == 1:
            bad += 1
        else:
            good += 1

    print("Bad: ", bad)
    print("Good: ", good)

    print(predictions)
    print("Times", times)"""


    while (True):
        url = input("Enter your URL: ")
        start_time = time.time()
        SamplePredict(url, "Results/model.h5").predict()
        print("Time to compute prediction.", time.time() - start_time)
    
    #SamplePredict("http://www.funtrivia.com/dir/78.html", "Results/model.h5").predict()
    #SamplePredict("http://www.blackmistress.com/", "Results/model.h5").predict()
    #SamplePredict("http://www.pornvalleynews.com", "Results/model.h5").predict()
        
    

