from django.urls import path

from . import views

app_name = 'new'
urlpatterns = [
    #path('loaddata/', views.load, name='load'),
    #path('metric/<int:data_id>/', views.metric, name='metric'),
    #path('mitigation', views.mitigate, name='mitigate'),
    #path('upload/', views.upload, name='upload'),
    #path('download/<path:filename>', views.download, name='download'),
    #path('clear/', views.cleardata, name='clear'),
    path('', views.index, name='index'),
    path('fVAE/', views.fVAE_data, name='fVAE'),
    path('fVAE/metric', views.fVAE_metric, name='fVAE_metric'),
    path('fVAE/mitigation', views.fVAE_miti, name='fVAE_miti'),
    path('fVAE/mitiresult', views.fVAE_miti_result, name='fVAE_result'),
    path('FFD/', views.FFD_data, name='FFD'),
    path('FFD/metric', views.FFD_metric, name='FFD_metric'),
    path('FFD/mitigation', views.FFD_mitigation, name='FFD_mitigation'),
    path('LfF/', views.LfF_data, name='LfF'),
    path('LfF/metric', views.LfF_metric, name='LfF_metric'),
    path('LfF/mitigation', views.LfF_mitigation, name='LfF_mitigation'),
    path('KDE/', views.KDE, name='KDE'),
    path('FB/', views.FB_data, name='FB'),
    path('FB/metric', views.FB_metric, name='FB_metric'),
    path('FB/mitigation', views.FB_mitigation, name='FB_mitigation'),
    path('CPR/', views.CPR_data, name='CPR'),
    path('CPR/metric', views.CPR_metric, name='CPR_metric'),
    path('CPR/mitigation', views.CPR_mitigation, name='CPR_mitigation'),
]