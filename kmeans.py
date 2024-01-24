import sys
import numpy as np


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            example = list(map(float, line.strip().split()))
            data.append((example[:-1], int(example[-1])))
    return data

#use to get euclid distance using np method
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def kmeans(training_data, k):

    #initialize centroids and old centroids
    centroids = [np.array(features) for features, _ in training_data[:k]]
    old_centroids = [np.zeros_like(centroid) for centroid in centroids]

    #go until centroids converge
    while not all((centroid == old_centroid).all() for centroid, old_centroid in zip(centroids, old_centroids)):
        #initialize clusters based on how many we want
        clusters = [[] for _ in range(k)]

        #assing each point to the nearest centroid based on euclid func
        for features, _ in training_data:
            distances = [euclidean_distance(features, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(features)

        #update centroids based on mean of each cluster
        old_centroids = centroids.copy()
        centroids = [np.mean(cluster, axis=0) if cluster else old_centroid for cluster, old_centroid in zip(clusters, old_centroids)]

    return centroids

#assign labels based on majority in each cluster
def assign_labels(training_data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    
    for features, label in training_data:
        distances = [euclidean_distance(features, centroid) for centroid in centroids]
        closest_centroid_idx = np.argmin(distances)
        clusters[closest_centroid_idx].append(label)

    cluster_labels = [min(set(cluster), key=cluster.count) if cluster else None for cluster in clusters]
    
    return cluster_labels

#validate if the label assigned is right and return total right
def validate(validation_data, centroids, cluster_labels):
    correct_count = 0
    
    for features, true_label in validation_data:
        distances = [euclidean_distance(features, centroid) for centroid in centroids]
        closest_centroid_idx = np.argmin(distances)
        predicted_label = cluster_labels[closest_centroid_idx]
        
        if predicted_label == true_label:
            correct_count += 1

    return correct_count

def main():
    if len(sys.argv) != 4:
        sys.exit(1)

    num_clusters = int(sys.argv[1])
    training_data_filename = sys.argv[2]
    validation_data_filename = sys.argv[3]
    
    training_data = read_data(training_data_filename)
    validation_data = read_data(validation_data_filename)

    centroids = kmeans(training_data, num_clusters)
    cluster_labels = assign_labels(training_data, centroids)
    
    correct_count = validate(validation_data, centroids, cluster_labels)

    print(correct_count)

    

main()
