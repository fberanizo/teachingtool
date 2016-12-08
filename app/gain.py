# -*- coding: utf-8 -*-

import numpy, math

def gain(X_partition, y_partition):
    """Calcula medida de ganho de informação."""
    info_partition = info(y_partition)
    info_attr, partitions, values = info_attribute(X_partition, y_partition)
    return (info_partition - info_attr), partitions, values

def info(y_partition):
    """Calcula medida de informação para o conjunto de dados."""
    total = 0
    partition_size = y_partition.size
    values, indices = numpy.unique(y_partition, return_inverse=True)
    counts = numpy.bincount(indices)
    if len(values) > 1:
        for value, count in zip(values, counts):
            prob = count / float(partition_size)
            total -= prob * math.log(prob, len(values))
    return total

def info_attribute(X_partition, y_partition):
    """Calcula medida de informação para um atributo do conjunto de dados."""
    total = 0
    partition_size = y_partition.size
    values, indices = numpy.unique(X_partition, return_inverse=True)
    counts = numpy.bincount(indices)
    partitions = []
    for value, count in zip(values, counts):
        prob = count / float(partition_size)
        partition = numpy.where(X_partition==value)
        partitions.append(partition)
        partition_info = info(y_partition[partition])
        total += prob * partition_info
    return total, partitions, values
