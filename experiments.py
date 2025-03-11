from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from process_data import get_data_1, get_data_2, get_data_3, cross_validation_1, cross_validation_2, cross_validation_3
from sklearn.ensemble import RandomForestClassifier
from model import RandomForest, LogisticRegression
from math import sqrt


def matrix_data():
    n_trees = 50
    max_depth = 5
    acc = 0
    target = []
    prediction = []
    forest = RandomForest(n_trees=n_trees, max_depth=max_depth)

    for _ in range(5):
        data = get_data_1()
        data_train = data[0]
        data_valid = data[1]
        data_test = data[2]
        forest.fit(data_train[0], data_train[1])
        forest.cut_forest(data_valid[0], data_valid[1])
        #forest.cut_tree(data_valid[0], data_valid[1])
        
        loop_prediction = forest.predict(data_test[0])

        loop_acc = accuracy_score(data_test[1], loop_prediction)
        if loop_acc > acc:
            acc = loop_acc
            target = data_test[1]
            prediction = loop_prediction 
    
    cm = confusion_matrix(target, prediction)

    print("Confusion Matrix:")
    print(cm)


def test_algorithm_accuracy():
    n_trees = 100
    max_depth = 10
    acc_list = []
    acc_train_list = []
    f1_score_values = []
    num_of_trees = []
    forest = RandomForest(n_trees=n_trees, max_depth=max_depth)

    for _ in range(5):
        data = get_data_1()
        data_train = data[0]
        data_valid = data[1]
        data_test = data[2]
        forest.fit(data_train[0], data_train[1])
        forest.cut_forest(data_valid[0], data_valid[1])
        forest.cut_tree(data_valid[0], data_valid[1])
        
        prediction = forest.predict(data_test[0])
        train_prediciton = forest.predict(data_train[0])

        acc_list.append(accuracy_score(data_test[1], prediction))
        acc_train_list.append(accuracy_score(data_train[1], train_prediciton))

        f1_score_values.append(f1_score(data_test[1], prediction))
        num_of_trees.append(forest.show_number_of_trees())
        

    best_acc = max(acc_list)
    best_acc_train = acc_train_list[acc_list.index(best_acc)]

    mean_acc = sum(acc_list) / len(acc_list)
    mean_acc_train = sum(acc_train_list) / len(acc_train_list)

    tree_number_mean = sum(num_of_trees) / len(num_of_trees)

    acc_sigma = 0
    acc_train_sigma = 0
    for i in range(len(acc_list)):
        acc_sigma += pow((acc_list[i] - mean_acc), 2)
        acc_train_sigma += pow((acc_train_list[i] - mean_acc_train), 2)
    acc_sigma = sqrt(acc_sigma / len(acc_list))
    acc_train_sigma = sqrt(acc_train_sigma / len(acc_train_list))

    print(f"Number of trees (mean): {tree_number_mean}")
    print(f"Best_acc: {best_acc:.3f}, mean_acc: {mean_acc:.3f}, sigma_acc: {acc_sigma:.4f}, mean_f1_score: {np.mean(f1_score_values):.3f}")
    print(f"Best_acc_train: {best_acc_train:.3f}, mean_acc_train: {mean_acc_train:.3f}, sigma_acc_train: {acc_train_sigma:.4f}")


def f1_score(y, prediction):
    f1_scores = {}
    num_classes = list(np.unique(y))
    tp = 0
    fp = 0
    fn = 0
    for c in range(len(num_classes)):
        for i, el in enumerate(y):
            if (el == c) & (prediction[i] == c):
                tp += 1
            if (el != c) & (prediction[i] == c):
                fp += 1
            if (el == c) & (prediction[i] != c):
                fn += 1 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[c] = f1
    
    return np.mean(list(f1_scores.values()))


def test_algorithm_cross_validation():
    n_trees = 100
    max_depth = 100
    acc_list = []
    f1_score_values = []
    forest = RandomForest(n_trees=n_trees, max_depth=max_depth)
    
    dataset = cross_validation_1()

    for i in range(len(dataset[0])):
        loop_x = dataset[0].copy()
        loop_y = dataset[1].copy()
        loop_x.pop(i)
        loop_y.pop(i)

        merged_x = []
        merged_y = []
        for sublist in loop_x:
            merged_x.append(np.array(sublist))
        for sublist in loop_y:
            merged_y.append(sublist)
        
        merged_x = np.concatenate(np.array(merged_x), axis=0)
        merged_y = np.concatenate(np.array(merged_y), axis=0)

        forest.fit(merged_x, merged_y)

        prediction = forest.predict(dataset[0][i])

        acc_list.append(accuracy_score(dataset[1][i], prediction))
        flattened = [item for sublist in dataset[1][i] for item in sublist]
        f1_score_values.append(f1_score(flattened, prediction))

    mean_acc = sum(acc_list) / len(acc_list)

    print(f"Mean_acc: {mean_acc:.3f}, f1_score_mean: {np.mean(f1_score_values):.3f}")


def test_reference_method():
    n_estimators = 1
    max_depth = 5
    sklearn_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    forest = RandomForest(n_trees=n_estimators, max_depth=max_depth)

    acc_list = []
    sk_acc_list = []
    for  _ in range(5):
        data = get_data_2()
        data_train = data[0]
        data_test = data[2]
        sklearn_forest.fit(data_train[0], data_train[1])
        forest.fit(data_train[0], data_train[1])

        sklearn_prediction = sklearn_forest.predict(data_test[0])
        sklearn_acc = accuracy_score(data_test[1], sklearn_prediction)
        prediction = forest.predict(data_test[0])

        acc_list.append(accuracy_score(data_test[1], prediction))
        sk_acc_list.append(sklearn_acc)
    best_acc = max(acc_list)
    best_sk_acc = max(sk_acc_list)

    print(best_acc, best_sk_acc)


if __name__ == "__main__":
    #test_algorithm_cross_validation()
    test_algorithm_accuracy()
    #test_reference_method()