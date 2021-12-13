from django.urls import path

from . import views

app_name = 'image'
urlpatterns = [
    path('loaddata/', views.load, name='load'),
    path('metric/<int:data_id>/', views.metric, name='metric'),
    path('mitigation', views.mitigate, name='mitigate'),
    path('upload/', views.upload, name='upload'),
    #path('download/<path:filename>', views.download, name='download'),
    #path('clear/', views.cleardata, name='clear'),
    path('', views.index, name='index'),
]