import heapq
import math


def euclidean_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(x, y)]))


def make_knn_classifier(data, data_labels, k, distance):
    def classify(x):
        closest_points = heapq.nsmallest(k, enumerate(data), key=lambda y: distance(x, y[1]))
        closest_labels = [data_labels[i] for (i, pt) in closest_points]
        return max(set(closest_labels), key=closest_labels.count)

    return classify
