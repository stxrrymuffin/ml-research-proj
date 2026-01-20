import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def split():
    df = pd.read_csv('Dry_Bean_Dataset.csv')
    stratified_sample = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    train_df, test_df = train_test_split(stratified_sample, test_size=0.2, random_state=42)
    train_df.to_csv('bean_training.csv', index=False)
    test_df.to_csv('bean_testing.csv', index=False)

def knn(training, testing, K, weight):
    attributes = [col for col in training.columns if col != "Class"]
    correct = 0
    predictions = []
    for idx, row in testing.iterrows():
        dist_lst = []
        for idx2, row2 in training.iterrows():
            dist = 0
            for attr in attributes:
                dist += abs(float(row2[attr])-float(row[attr]))**weight
            dist = dist**0.5
            dist_lst += [(idx2,dist)]
        sorted_dist_lst = sorted(dist_lst, key=lambda x: x[1])
        majority_class = statistics.mode([training.loc[val[0]]["Class"] for val in sorted_dist_lst[:K]])
        
        predictions += [majority_class]
        if majority_class == testing.loc[idx]['Class']:
            correct+=1
        
    y_pred = testing['Class']
    cm = confusion_matrix(y_pred, predictions)
    print(cm)
    accuracy = correct / len(testing)
    print(f"\nAccuracy: {accuracy:.3f}")
    return accuracy

def fitness_function(training_data, testing_data, k, weight, alpha=0.9):
    X = training_data.drop(columns=['Class'])
    y = training_data['Class']

    k_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    accuracies = []

    for train_index, test_index in k_folds.split(X, y):
        train_df = training_data.iloc[train_index]
        test_df = training_data.iloc[test_index]

        acc = knn(train_df, test_df, k, weight)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    print(f"mean accuracy: {mean_acc}")
    return alpha * (1 - mean_acc) + (1 - alpha) * (k / len(training_data))

class GWO_KNN:
    def __init__(
        self,
        training_data,
        testing_data,
        n_wolves=5,
        n_iter=40,
        k_max=50,
        weight_min=0.5,
        weight_max=3.0,
        a_max=2.5,
        a_min=0.2,
    ):
        self.training_data = training_data
        self.testing_data = testing_data
        self.n_wolves = n_wolves
        self.n_iter = n_iter
        self.k_max = k_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.a_max = a_max
        self.a_min = a_min
        self.wolves = self._init_wolves()

    def _init_wolves(self):
        wolves = []
        for _ in range(self.n_wolves):
            k = np.random.uniform(1, self.k_max)
            weight = np.random.uniform(self.weight_min, self.weight_max)
            wolves.append(np.array([k, weight]))
        return np.array(wolves)

    def _linear_a(self, t):
        return self.a_max - (self.a_max - self.a_min) * (t / self.n_iter)

    def optimize(self):
        for t in range(self.n_iter):
            a = self._linear_a(t)

            fitness = []
            for wolf in self.wolves:
                k = int(np.clip(round(wolf[0]), 1, self.k_max))
                weight = np.clip(wolf[1], self.weight_min, self.weight_max)
                fitness.append(
                    fitness_function(self.training_data, self.testing_data, k, weight)
                )

            fitness = np.array(fitness)
            idx = np.argsort(fitness)
            print(fitness)
            print(idx)

            alpha, beta, delta = self.wolves[idx[:3]]
            print(alpha, beta, delta)

            for i in range(self.n_wolves):
                for leader in [alpha, beta, delta]:
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - self.wolves[i])
                    X_new = leader - A * D
                    self.wolves[i] = (self.wolves[i] + X_new) / 2

                # Constraints
                self.wolves[i][0] = np.clip(self.wolves[i][0], 1, self.k_max)
                self.wolves[i][1] = np.clip(self.wolves[i][1], self.weight_min, self.weight_max)

        best = self.wolves[idx[0]]
        best_k = int(round(best[0]))
        best_weight = best[1]
        return best_k, best_weight

gwo = GWO_KNN(pd.read_csv("bean_training.csv"),pd.read_csv("bean_testing.csv"), n_wolves=12, n_iter=5, k_max=50, weight_max=3)
best_k, best_weight = gwo.optimize()

print(best_k)
print(best_weight)

print(knn(pd.read_csv("bean_training.csv"),pd.read_csv("bean_testing.csv"), best_k, best_weight))