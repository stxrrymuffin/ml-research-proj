import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
import numpy as np

def split():
    df = pd.read_csv('Dry_Bean_Dataset.csv')
    stratified_sample = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    train_df, test_df = train_test_split(stratified_sample, test_size=0.2, random_state=42)
    train_df.to_csv('bean_training.csv', index=False)
    test_df.to_csv('bean_testing.csv', index=False)

def knn(training, testing, K, weight):
    attributes = [col for col in training.columns if col != "Class"]
    correct = 0
    for idx, row in testing.iterrows():
        dist_lst = []
        for idx2, row2 in training.iterrows():
            dist = 0
            for attr in attributes:
                dist += (float(row2[attr])-float(row[attr]))**weight
            dist = dist**0.5
            dist_lst += [(idx2,dist)]
        sorted_dist_lst = sorted(dist_lst, key=lambda x: x[1])
        majority_class = statistics.mode([training.iloc[val[0]]["Class"] for val in sorted_dist_lst[:K]])
        
        if majority_class == testing.iloc[idx]['Class']:
            correct+=1
        #print(f"{[float(val) for val in testing.iloc[idx].drop('class').to_list()]} -- predicted: {majority_class}; actual: {testing.iloc[idx]['class']}")
    
    accuracy = correct / len(testing)
    print(f"\nAccuracy: {accuracy:.3f}")
    return accuracy

def fitness_function(training_data, testing_data, k, weight, alpha=0.9):
    small_train = training_data.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    train_df, test_df = train_test_split(small_train, test_size=0.2, random_state=42)
    acc = knn(train_df, test_df, k, weight)
    return alpha * (1 - acc) + (1 - alpha) * (k / len(training_data))

class GWO_KNN:
    def __init__(
        self,
        training_data,
        testing_data,
        weight=2,
        n_wolves=5,
        n_iter=40,
        k_max=50,
        a_max=2.5,
        a_min=0.2,
    ):
        self.training_data = training_data
        self.testing_data = testing_data
        self.weight = weight #[weight]*(training_data.shape[1]-1)
        self.n_wolves = n_wolves
        self.n_iter = n_iter
        self.k_max = k_max
        self.a_max = a_max
        self.a_min = a_min
        self.dim = training_data.shape[1]
        self.wolves = self._init_wolves()

    def _init_wolves(self):
        wolves = []
        for _ in range(self.n_wolves):
            k = np.random.uniform(1, self.k_max)
            wolves.append((np.array([k])))
        return np.array(wolves)

    def _linear_a(self, t):
        return self.a_max - (self.a_max - self.a_min) * (t / self.n_iter)
    
    def select_k(self):
        return
    
    def select_weight(self):
        return

    def optimize(self):
        for t in range(self.n_iter):
            a = self._linear_a(t)

            fitness = []
            for wolf in self.wolves:
                k = int(np.clip(round(wolf[0]), 1, self.k_max))
                fitness.append(
                    fitness_function(self.training_data, self.testing_data, k, self.weight)
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
                self.wolves[i][1:] = np.clip(self.wolves[i][1:], 0, 1)
                self.wolves[i][1:] /= np.sum(self.wolves[i][1:])

        best = self.wolves[idx[0]]
        best_k = int(round(best[0]))
        return best_k

gwo = GWO_KNN(pd.read_csv("bean_training.csv"),pd.read_csv("bean_testing.csv"), n_wolves=5, n_iter=3, k_max=50, weight=2)
best_k = gwo.optimize()
print(best_k)

#knn("bean_training.csv","bean_testing.csv", best_k, best_weights)