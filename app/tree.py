# -*- coding: utf-8 -*-

import numpy, pandas
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import deque
from gain import *

class DecisionTree:
    """Classe que implementa uma árvore de decisão para classificação."""
    def __init__(self, X, y, attributes, attribute_selection_method='gain'):
        self.X = X
        self.y = y
        self.attributes = attributes
        self.attribute_selection_method = attribute_selection_method
        self.samples_size, self.input_size = X.shape

        # Inicializa a árvore com a partição total dos dados de treinamento
        self.root = dict([('label', None), ('partition', range(y.size)), ('children', [])])

        # Mantém uma fila de nós que precisam ser completados
        self.queue = deque()
        self.queue.append(self.root)

    def create_node(self):
        """Cria um nó da árvore de decisão. """
        if len(self.queue) > 0:
            node = self.queue.popleft()

            X_partition = self.X[node['partition']]
            y_partition = self.y[node['partition']]

            # Se os rótulos em y são todos da mesma classe C, então a decisão é C
            classes = numpy.unique(y_partition)
            if len(classes) == 1:
                node['label'] = classes[0]
                return

            # Se a lista de atributos está vazia, então a decisão é a classe majoritária de y
            if len(self.attributes) == 0:
                unique, pos = numpy.unique(y_partition, return_inverse=True)
                counts = numpy.bincount(pos)
                node['label'] = unique[counts.argmax()]
                return

            # Calcula métrica de seleção de atributo com base no critério definido no construtor
            if self.attribute_selection_method == 'gain':
                gains = self.calculate_information_gain(X_partition, y_partition)
            elif self.attribute_selection_method == 'gini':
                pass
            else:
                raise Exception('Método de seleção de atributo não implementado.')

            index, attribute, gain_value, partitions = max(gains, key=lambda item: item[2])

            node['label'] = attribute + '?'

            self.attributes.remove(attribute)

            # Cria um nó para cada partição
            for partition in partitions:
                child = dict([('label', None), ('partition', partition), ('children', [])])
                node['children'].append(child)
                self.queue.append(child)

    def calculate_information_gain(self, X_partition, y_partition):
        """."""
        gains = []
        for index, attribute in enumerate(self.attributes):
            gain_value, partitions = gain(X_partition[:,index], y_partition)
            gains.append((index, attribute, gain_value, partitions))
        return gains


dataset = pandas.read_csv('./datasets/zoo/zoo.data', sep=',', header=None).as_matrix()
X = dataset[1:,1:]
y = dataset[1:,0]
attributes = dataset[0,:-1].tolist()
clf = DecisionTree(X, y, attributes)
clf.create_node()
clf.create_node()
clf.create_node()
print(clf.root)