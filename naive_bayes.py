import pandas as pd
import numpy as np
#Whether or not we can play golf given weather conditions

#dataset:

# features = np.array([
#     ["Sunny", "Hot", "High", "False"],
#     ["Overcast", "Mild", "Normal", "True"],
#     ["Rainy", "Cool", "Normal", "False"],
#     ["Sunny", "Mild", "Normal", "True"],
#     ["Rainy", "Hot", "High", "False"]
# ])
# labels = np.array(["Yes", "Yes", "No", "Yes", "No"])

# TODO: Read the dataset 

def read_data(dataset):

    features = dataset.drop([dataset.columns[-1]], axis = 1)
    labels = dataset[dataset.columns[-1]]

    return features, labels

def train_test_split(features, labels, split, random_state):

    feature_test_set = features.sample(frac = split, random_state = random_state)
    label_test_set = labels[feature_test_set.index]

    feature_train_set = features.drop(feature_test_set.index)
    label_train_set = labels.drop(label_test_set.index)

    return feature_train_set, feature_test_set, label_train_set, label_test_set

# TODO: Calculate probability of each predictor variable(sunny/hot)

# TODO: Calculate Likelihood for each class(label - YES/NO)

# TODO: Return the greatest likelihood

class NaiveBayes:

    def __init__(self):

        '''
            p_* - denotes probablities
        '''
        self.features = list

        self.p_feature_likelihood = {}
        self.p_class_prior = {}
        self.p_feature_prior = {}

        self.feature_train_set = np.array
        self.label_train_set = np.array
        self.train_set_size = int
        self.feature_count = int

    def fit(self, feature, label):

        self.features = list(feature.columns)
        self.feature_train_set = feature
        self.label_train_set = label
        self.train_set_size = feature.shape[0]
        self.feature_count = feature.shape[1]

        for f in self.features:
            self.p_feature_likelihood[f] = {}
            self.p_feature_prior[f] = {}

            for f_value in np.unique(self.feature_train_set[f]):
                self.p_feature_prior[f].update({f_value: 0})

                for outcome in np.unique(self.label_train_set):
                    self.p_feature_likelihood[f].update({f_value+'_'+outcome:0})
                    self.p_class_prior.update({outcome: 0})

        self._calc_p_class_prior()
        self._calc_p_feature_likelihood()
        self._calc_p_feature_prior()

    def _calc_p_class_prior(self):

    def _calc_p_feature_likelihood(self):
        
    def _calc_p_feature_prior(self):

        




# Test code beyond this
# df = pd.read_table("weather.txt")
# features, labels = read_data(df)

