# -*- coding: utf-8 -*-

import numpy, math, time

def gini(X_partition, y_partition):
    """Calcula o GINI index para um atributo do conjunto de dados."""
    total = gini_index(y_partition)
    partition_size = y_partition.size
    values, indices = numpy.unique(X_partition, return_inverse=True)
    counts = numpy.bincount(indices)
    partitions = []
    for value, count in zip(values, counts):
        prob = count / float(partition_size)
        partition = numpy.where(X_partition==value)
        partitions.append(partition)
        partition_gini = gini_index(y_partition[partition])
        total -= prob * partition_gini
    return total, partitions

def gini_index(y_partition):
    """Calcula medida de informação para o conjunto de dados."""
    total = float(1)
    partition_size = y_partition.size
    values, indices = numpy.unique(y_partition, return_inverse=True)
    counts = numpy.bincount(indices)
    for value, count in zip(values, counts):
        prob = count / float(partition_size)
        total -= prob ** 2
    return total