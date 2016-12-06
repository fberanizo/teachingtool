from django.conf.urls import *

from . import views

urlpatterns = [
    # /app
    url(r'^$', views.TreePlotView.as_view(), name='index')

]
