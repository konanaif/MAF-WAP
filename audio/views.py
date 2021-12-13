import os
import mimetypes
import numpy as np
from glob import glob
import json
import zipfile

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.urls import reverse

#from aif360.datasets import AdultDataset
from aif360.metrics import utils
from Kaif.Metric import DataMetric

from .forms import DataForm
from .models import Data
from .analyze import Kaifdata, KaifMitigation, Imagedata, Audiodata

# Create your views here.

def index(request):
	datas = Data.objects.all()
	form = DataForm()
	return render(request, 'audio/index.html', {'data_list':datas, 'form':form})


def upload(request):
	if request.method == 'POST':
		form = DataForm(request.POST, request.FILES)

		if form.is_valid():
			data = Data(datafile=request.FILES['file'])
			data.config_data_json = request.POST['config_data']
			data.save()

		return redirect(reverse('audio:index'))
	
	else:
		form = ()
		files = glob('./' + request.path.split('/')[1] + '/data/*')

		return render(request, 'audio/index.html', {'dl': files, 'form':form})


def load(request):
	if request.method == 'POST':
		items = list(request.POST.lists())
		data_index = request.POST.getlist('index')
		config_json = request.POST['config_load']
		datatype = request.path.split('/')[1]

		# data update
		for idx in data_index:
			d = Data.objects.get(id=idx)
			d.config_load_json = config_json
			d.datatype = datatype
			d.save()
		
		data = Audiodata(data_index[0], datatype)  # Data loader

		num_classes = len(np.unique(data.aif_data.labels))
		dirichlet_alpha = 1.0 / num_classes
		intersect_groups = np.unique(data.aif_data.protected_attributes, axis=0)
		num_intersects = len(intersect_groups)

		data_info = {
			'num_classes': num_classes,
			'num_intersects': num_intersects,
			'intersect_groups': intersect_groups
		}

		context = {'info': data_info, 'data_id': idx}

		return render(request, 'audio/data.html', context)

	return HttpResponse('You entered this page with not porper root.\nPlease return to index page.')


def metric(request, data_id):
	data = Data.objects.get(id=data_id)
	
	# re-load data
	data = Kaifdata([data.id], data.datatype)

	#lconfig = json.loads(data.config_load_json)

	privilege_group = []
	unprivilege_group = []
	for idx, pan in enumerate(data.aif_data.protected_attribute_names):
		ptemp = {}
		ptemp[pan] = data.aif_data.privileged_protected_attributes[idx]
		privilege_group.append(ptemp)

		utemp = {}
		utemp[pan] = data.aif_data.unprivileged_protected_attributes[idx]
		unprivilege_group.append(utemp)

	metric = DataMetric(dataset=data.aif_data, privilege=privilege_group, unprivilege=unprivilege_group)

	context = {
	'num_positive': metric.num_positive(),
	'num_negative': metric.num_negative(),
	'base_rate': metric.base_rate(),
	'disparate_impact': metric.disparate_impact(),
	'statistical_parity_difference': metric.statistical_parity_difference(),
	'data': data,
	'd_id': data_id
	}

	return render(request, 'audio/metric.html', context)


def mitigate(request):
	d_id = request.GET['d_id']
	m_id = request.GET['m_id']

	result = KaifMitigation(d_id, m_id)

	context = {
		'odm': result.data_metric_orig_obj,
		'ocm': result.class_metric_orig_obj,
		'tdm': result.data_metric_transf_obj,
		'tcm': result.class_metric_transf_obj
	}

	return render(request, 'audio/mitigate.html', context)