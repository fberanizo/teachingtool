# -*- coding: utf-8 -*-

import igraph
from plotly.offline import plot
from plotly.graph_objs import Scatter, XAxis, YAxis, Annotations, Annotation, Data

def treeplot(decision_tree):
    """"""
    if len(decision_tree.nodes) == 0:
        return ''
    g = igraph.Graph()
    g.add_vertices(len(decision_tree.nodes))
    g.add_edges(decision_tree.edges)

    # Define posições no espaço para a árvore
    layout = g.layout_reingold_tilford(mode="out", root=[0], rootlevel=None)

    # Cria os vértices da árvore
    Xn = [-node[0] for node in layout]
    Yn = [-node[1] for node in layout]
    nodes = Scatter(x=Xn, y=Yn, mode='markers', marker=dict(symbol='square', size=40, color='#333'), text=None, hoverinfo='none')

    # Escreve rótulos nos vértices
    annotations = Annotations()
    for node in decision_tree.nodes:
        a = Annotation(text=node['label'], x=Xn[node['id']], y=Yn[node['id']], xref='x1', yref='y1', font=dict(color='#FFF', size=9, family='"Open Sans", verdana, arial, sans-serif'), showarrow=False)
        annotations.append(a)

    # Cria as arestas da árvore
    Xe =[]; Ye = []
    for edge in decision_tree.edges:
        Xe += [Xn[edge[0]], Xn[edge[1]], None]
        Ye += [Yn[edge[0]], Yn[edge[1]], None]
    lines = Scatter(x=Xe, y=Ye, mode='lines+markers+text', line=dict(color='#333', width=2), hoverinfo='none')

    # Escreve rótulos nas arestas
    for edge in decision_tree.edges:
        label = decision_tree.nodes[edge[1]]['value']
        X = (Xn[edge[0]] + Xn[edge[1]])/2
        Y = (Yn[edge[0]]*1.2 + Yn[edge[1]]*0.8)/2
        a = Annotation(text=label, x=X, y=Y, xref='x1', yref='y1', font=dict(color='#333', size=14, family='"Open Sans", verdana, arial, sans-serif'), showarrow=False)
        annotations.append(a)

    # Cria gráfico
    data = Data([lines, nodes])
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    layout = dict(title= '', annotations=annotations, font=dict(size=12, family='"Open Sans", verdana, arial, sans-serif'), showlegend=False, xaxis=XAxis(axis), yaxis=YAxis(axis), margin=dict(l=0, r=0, b=0, t=0), hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig = dict(data=data, layout=layout)

    return plot(fig, filename='Decision-Tree', output_type='div', include_plotlyjs=False)

def attributes(decision_tree):
    """"""
    next_node = decision_tree.queue[0]
    X_partition = decision_tree.X[next_node['partition']]
    y_partition = decision_tree.y[next_node['partition']]
    attributes = decision_tree.calculate_attribute_selection_metric(X_partition, y_partition)
    attributes = map(lambda attr: (attr['attribute'], attr['metric']), attributes)
    attributes.sort(key=lambda attr: attr[1], reverse=True)
    return attributes