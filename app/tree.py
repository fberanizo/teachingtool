# -*- coding: utf-8 -*-

import numpy, pandas
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import deque
from . import gain, gini

class DecisionTree:
    """Classe que implementa uma árvore de decisão para classificação."""
    def __init__(self, X, y, attributes, attribute_selection_method='gain'):
        self.X = X
        self.y = y
        self.attributes = attributes
        self.attribute_selection_method = attribute_selection_method
        self.samples_size, self.input_size = X.shape
        self.id_pool = 1
        self.remaining_attributes = list(range(len(attributes)))

        # Inicializa a árvore com a partição total dos dados de treinamento
        self.tree = dict([('id', 0), ('label', None), ('type', 'question'), ('partition', range(y.size)), ('X_partition', self.X), ('y_partition', self.y), ('value', None), ('parent', None), ('children', [])])
        self.nodes = []
        self.edges = []

        # Mantém uma fila de nós que precisam ser completados
        self.queue = deque()
        self.queue.append(self.tree)

    def create_node(self):
        """Cria um nó da árvore de decisão. """
        if len(self.queue) > 0:
            node = self.queue.popleft()

            # Cria uma aresta conectando este nó ao pai
            if node['parent'] is not None:
                self.edges.append((node['parent'], node['id']))

            X_partition = node['X_partition']
            y_partition = node['y_partition']

            # Se os rótulos em y são todos da mesma classe C, então a decisão é C
            classes = numpy.unique(y_partition)
            if len(classes) == 1:
                node['label'] = classes[0]
                node['type'] = 'decision'
                self.nodes.append(node)
                return

            # Se a lista de atributos está vazia, então a decisão é a classe majoritária de y
            if len(self.remaining_attributes) == 0:
                unique, pos = numpy.unique(y_partition, return_inverse=True)
                counts = numpy.bincount(pos)
                node['label'] = unique[counts.argmax()]
                node['type'] = 'decision'
                self.nodes.append(node)
                return

            # Calcula métrica de seleção de atributo com base no critério definido no construtor
            attributes = self.calculate_attribute_selection_metric(X_partition, y_partition)

            # Escolhe o atributo com maior métrica
            best_attribute = max(attributes, key=lambda item: item['metric'])

            node['label'] = best_attribute['attribute']
            self.nodes.append(node)

            self.remaining_attributes.remove(self.attributes.index(best_attribute['attribute']))

            # Cria um nó para cada partição
            for partition, value in zip(best_attribute['partitions'], best_attribute['values']):
                child = dict([('id', self.id_pool), ('label', None), ('type', 'question'), ('partition', partition), ('X_partition', node['X_partition'][partition]), ('y_partition', node['y_partition'][partition]), ('value', value), ('parent', node['id']), ('children', [])])
                node['children'].append(child)
                self.queue.append(child)
                self.id_pool += 1

    def calculate_attribute_selection_metric(self, X_partition, y_partition):
        """Retorna um dict com métrica de seleção e as partições que o atributo realiza."""
        if self.attribute_selection_method not in ['gain', 'gini']:
            raise Exception('Método de seleção de atributo não implementado.')
        else:
            attributes = []
            for index in self.remaining_attributes:
                if self.attribute_selection_method == 'gain':
                    metric, partitions, values = gain.gain(X_partition[:,index], y_partition)
                elif self.attribute_selection_method == 'gini':
                    metric, partitions, values = gini.gini(X_partition[:,index], y_partition)
                attributes.append({'attribute': self.attributes[index], 'metric': metric, 'partitions': partitions, 'values': values})
            return attributes
