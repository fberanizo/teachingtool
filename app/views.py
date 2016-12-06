from django.views.generic import TemplateView
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse

class TreePlotView(TemplateView):
    template_name = "treeplot.html"
