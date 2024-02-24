import pickle
import pandas as pd
import numpy as np

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

def accuracy(label_actual, label_predicted):

    return round(float(sum(label_predicted == label_actual)) / float(len(label_actual)) * 100, 2)

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

        for outcome in np.unique(self.label_train_set):
            outcome_count = sum(self.label_train_set == outcome)
            self.p_class_prior[outcome] = outcome_count/self.train_set_size


    def _calc_p_feature_likelihood(self):

        for f in self.features:
            for outcome in np.unique(self.label_train_set):
                outcome_count = sum(self.label_train_set == outcome)
                feature_likelihood = self.feature_train_set[f][self.label_train_set[self.label_train_set == outcome].index.values.tolist()].value_counts().to_dict()

                for f_value, count in feature_likelihood.items():
                    self.p_feature_likelihood[f][f_value + '_' + outcome] = count/outcome_count

        
    def _calc_p_feature_prior(self):
        for f in self.features:
            f_values = self.feature_train_set[f].value_counts().to_dict()

            for f_value, count in f_values.items():
                self.p_feature_prior[f][f_value] = count/self.train_set_size

    def predict(self, features):

        results = []
        features = np.array(features)

        for feature in features:
            p_posterior = {}
            for outcome in np.unique(self.label_train_set):
                p_prior = self.p_class_prior[outcome]
                likelihood = 1
                evidence = 1

                for f, f_value in zip(self.features, feature):
                    likelihood *= self.p_feature_likelihood[f][f_value + '_' + outcome]
                    evidence *= self.p_feature_prior[f][f_value]

                p_posterior[outcome] = (likelihood * p_prior) / evidence

            results.append(max(p_posterior, key = lambda x: p_posterior[x]))

        return np.array(results)

if __name__ == '__main__':

    dataFrame = pd.read_table("weather.txt")
    feature, label = read_data(dataFrame)

    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, 0.1, random_state = 0)

    try:
        naive_bayes_obj = pickle.load(open("probabilities.pickle", "rb"))
    except (OSError, IOError) as e:
        naive_bayes_obj = NaiveBayes()
        naive_bayes_obj.fit(feature_train, label_train)
        pickle.dump(naive_bayes_obj, open("probabilities.pickle", "wb"))
    
    print("Accuracy: {}".format(accuracy(label_train, naive_bayes_obj.predict(feature_train))))
    print("Accuracy: {}".format(accuracy(label_test, naive_bayes_obj.predict(feature_test))))

    # Random Queries:

    query = np.array([['Rainy', 'Mild', 'Normal', 't']])
    print("Query 1:- {} -> {}".format(query, naive_bayes_obj.predict(query)))

    query = np.array([['Overcast','Hot', 'High', 'f']])
    print("Query 2:- {} -> {}".format(query, naive_bayes_obj.predict(query)))
