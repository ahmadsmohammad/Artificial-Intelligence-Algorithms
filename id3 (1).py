#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from collections import Counter


class Node:
    def __init__(self, attribute=None, split_point=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.split_point = split_point
        self.left = left
        self.right = right
        self.label = label

def find_split(sorted, labels):
    split_points = {}

    for column in range(sorted.shape[1]):
        unique_values = np.unique(sorted[:, column])
        print(f"Column: {column}, Unique Values: {unique_values}")

        for unique in range(len(unique_values)):
            #COULD BE ERROR HERE BC OF DIVIDING ERROR SEE VERBOSE OUTPUT
            split = (unique_values[unique] + unique_values[unique+1]) / 2.0
            #split = unique_values[unique]
            split_index = np.searchsorted(sorted[:, column], split, side="right")
            split_points[(column, split)] = split_index

    print(f"Split Points: {split_points}")
    return split_points


def entropy(labels):
    total = len(labels)
    occurs = Counter(labels)
    entropy = 0
    for count in occurs.values():
        probability = count / total
        entropy -= probability * np.log2(probability)

    return entropy


def info_gain(labels, split):
    total_entropy = entropy(labels)

    left_labels = labels[:split+1]
    right_labels = labels[split+1:]

    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    left_weight = len(left_labels) / len(labels)
    right_weight = len(right_labels) / len(labels)

    info_gain = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return info_gain

def build_tree(sorted_data, attributes, labels, depth=0):
    print(f"Depth: {depth}, Number of attributes: {len(attributes)}, Number of labels: {len(set(labels))}")
    
    if len(set(labels)) == 1:
        print(f"Reached leaf node. Single label: {labels[0]}")
        return Node(label = labels[0])

    if len(attributes) == 0:
        #creating counter objecty where we get the most common element
        #and we take it by extracting using [0][0]
        majority_label = Counter(labels).most_common(1)[0][0]
        print(f"Reached leaf node. No more attributes. Majority label: {majority_label}")
        return Node(label=majority_label)

    max_info_gain = -1
    best_split = None


    for attribute in attributes:
        # Find the index of the attribute name within the first row of sorted_data
        attribute_index = None
        print(attribute)
        for i in range(sorted_data.shape[1]):
            if attribute in np.unique(sorted_data[:, i]):
                attribute_index = i
                break

        print(f"Attribute: {attribute}, Attribute Index: {attribute_index}")
        if attribute_index is not None:
            potential_split = find_split(sorted_data[:, attribute_index], labels)
            print(f"Attribute: {attribute}")
            print(f"Potential splits: {potential_split}")
            
            for split_point, split_index in potential_split.items():
                gain = info_gain(labels, split_index)
                print(f"Split Point: {split_point}, Gain: {gain}, Left labels: {labels[:split_index + 1]}, Right labels: {labels[split_index + 1:]}")
                print(f"Split Point: {split_point}, Gain: {gain}")
                if gain > max_info_gain:
                    max_info_gain = gain
                    best_split = (attribute, split_point)

    if best_split is None:
        majority_label = Counter(labels).most_common(1)[0][0]
        print(f"No best split found. Majority label: {majority_label}")
        return Node(label=majority_label)

    attribute, split_point = best_split
    attribute_index = None
    for i, name in enumerate(sorted_data[0, :]):
        if np.array_equal(name, attribute):
            attribute_index = i
            break

    
    split_index = np.searchsorted(sorted_data[:, attribute_index], split_point, side="right")

    left_data = sorted_data[:split_index + 1, :]
    right_data = sorted_data[split_index + 1:, :]
    left_labels = labels[:split_index + 1]
    right_labels = labels[split_index + 1:]

    print(f"Best split: {best_split}")
    print(f"Left data: {left_data}")
    print(f"Right data: {right_data}")


    left_branch = build_tree(left_data, attributes, left_labels)
    right_branch = build_tree(right_data, attributes, right_labels)

    return Node(attribute = attribute, split_point = split_point, left = left_branch, right = right_branch)

def sort_data(train, valid):

    data = np.loadtxt(train)
    # Special case of just one data sample will fail
    # without this check!
    if len(data.shape) < 2:
        data = np.array([data])

    print(data)

    # Sort all columns - just retain sorted indices
    # NOT the sorted data to prevent need to resort
    # later on...
    indices = np.argsort(data,axis=0)

    sorted_data = np.copy(data)
    # Proceed for each column
    for x in range(data.shape[1]):
        sorted_data[:, x] = data[indices[:, x], x]



    data2 = np.loadtxt(valid)
    # Special case of just one data sample will fail
    # without this check!
    if len(data2.shape) < 2:
        data2 = np.array([data2])

    # Sort all columns - just retain sorted indices
    # NOT the sorted data to prevent need to resort
    # later on...
    indices = np.argsort(data2,axis=0)

    sorted_data2 = np.zeros_like(data2)
    
    # Proceed for each column
    for x in range(data2.shape[1]):
        sorted_data2[:, x] = data2[indices[:, x], x]

    return sorted_data, sorted_data2

def classify(example, id3_tree):
    if id3_tree.label is not None:
        return id3_tree.label

    attr = id3_tree.attribute
    split_point = id3_tree.split_point

    if example[attr] < split_point:
        return classify(example, id3_tree.left)
    else:
        return classify(example, id3_tree.right)

def classify_all(validation, id3_tree):
    return [classify(example, id3_tree) for example in validation]
def main():


    train_file = sys.argv[1]

    valid_file = sys.argv[2]

    train_data, valid_data = sort_data(train_file, valid_file)

    labels_train = train_data[:, -1]
    labels_valid = valid_data[:, -1]
    attributes = train_data[:, :-1]
    print(attributes)

    # iloc allows us to pick specific row/col

    #print(train_file)
    #print(valid_file)
    #print(train_data)
    #print(valid_data)
    #print(find_split(train_data))
    #print(entropy(labels_train))

    id3 = build_tree(train_data, attributes, labels_train)

    predictions = classify_all(valid_data, id3)

    num_correct = np.sum(predictions == labels_valid)
    print(num_correct)

    

main()