# -*- coding: utf-8 -*-

import os, pandas
from django.views.generic import TemplateView
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from . import plots, tree

class TreePlotView(TemplateView):
    template_name = "treeplot.html"

    def __init__(self):
        # Lê conjunto de dados "zoo"
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        dataset = pandas.read_csv(self.PROJECT_ROOT + '/datasets/zoo/zoo.data', sep=',', header=None).as_matrix()
        X = dataset[1:,1:-1]
        y = dataset[1:,-1]
        self.attributes = dataset[0,1:-1].tolist()

        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Cria árvore de decisão
        self.decision_tree = tree.DecisionTree(self.X, self.y, self.attributes)

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(TreePlotView, self).get_context_data(**kwargs)
        context['plot'] = plots.treeplot(self.decision_tree)
        context['attributes'], context['metric_name'], context['partition'] = plots.attributes(self.decision_tree)
        context['queue'] = self.decision_tree.queue
        context['used_attributes'] = len(self.decision_tree.attributes) - len(self.decision_tree.remaining_attributes)
        context['attribute_selection_method'] = self.decision_tree.attribute_selection_method
        return context

    def get(self, request):
        if request.session.get('nodes', None) is None:
            request.session['nodes'] = 0

        if request.session.get('attribute_selection_strategy', None) is None:
            request.session['attribute_selection_strategy'] = 'gain'

        self.decision_tree.attribute_selection_method = request.session['attribute_selection_strategy']

        for node in range(request.session['nodes']):
            # Adiciona nós à árvore
            self.decision_tree.create_node()

        return render(request, self.template_name, self.get_context_data())

    def post(self, request):
        if request.POST.get('forward', None) is not None:
            # Adiciona nós à árvore
            request.session['nodes'] = request.session['nodes']+1

        if request.POST.get('backward', None) is not None:
            # Adiciona nós à árvore
            request.session['nodes'] = max(0, request.session['nodes']-1)

        # Se mudou a estratégia de seleção de atributo, reinicia o classificador
        attribute_selection_strategy = request.POST.get('attribute_selection_strategy', None)
        if attribute_selection_strategy is not None and request.session['attribute_selection_strategy'] != attribute_selection_strategy:
            request.session['attribute_selection_strategy'] = attribute_selection_strategy
            request.session['nodes'] = 0

        self.decision_tree.attribute_selection_method = request.session['attribute_selection_strategy']

        for node in range(request.session['nodes']):
            # Adiciona nós à árvore
            self.decision_tree.create_node()

        return render(request, self.template_name, self.get_context_data()) 