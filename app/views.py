# -*- coding: utf-8 -*-

import os, pandas
from django.views.generic import TemplateView
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from . import plots, tree

class TreePlotView(TemplateView):
    template_name = "treeplot.html"

    def __init__(self):
        # Lê conjunto de dados "zoo"
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        dataset = pandas.read_csv(PROJECT_ROOT + '/datasets/zoo/zoo.data', sep=',', header=None).as_matrix()
        X = dataset[1:,1:-1]
        y = dataset[1:,-1]
        attributes = dataset[0,1:-1].tolist()
        
        # Cria árvore de decisão
        self.decision_tree = tree.DecisionTree(X, y, attributes, attribute_selection_method='gain')


    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(TreePlotView, self).get_context_data(**kwargs)
        context['plot'] = plots.treeplot(self.decision_tree)
        context['attributes'] = plots.attributes(self.decision_tree)
        return context

    def get(self, request):
        if request.session.get('nodes', None) is None:
            request.session['nodes'] = 0

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

        for node in range(request.session['nodes']):
            # Adiciona nós à árvore
            self.decision_tree.create_node()

        return render(request, self.template_name, self.get_context_data())