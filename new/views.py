import pandas as pd
import numpy as np

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf

from Kaif.DataSet import aifData
from Kaif.Metric import DataMetric, ClassificationMetric

from FairnessVAE.solver_40dim_clatent import Solver

from celery.result import AsyncResult
from .tasks import get_fVAE_result

# Create your views here.

def index(request):
	return render(request, "new/index.html")


def fVAE_data(request):
	# CelebA dataset
	# Target class: heavy makeup
	# Protected attribute: male/female
	df = pd.read_table('./data/CelebA/list_attr_celeba.csv', sep=',')
	aif_data = aifData(
		df=df,
		label_name='Heavy_Makeup',
		favorable_classes=[1],
		protected_attribute_names=['Male'],
		privileged_classes=[[-1]],
		features_to_keep=['Heavy_Makeup', 'Male'],
		na_values=['?']
		)

	num_classes = len(np.unique(aif_data.labels))
	dirichlet_alpha = 1.0 / num_classes
	intersect_groups = np.unique(aif_data.protected_attributes, axis=0)
	num_intersects = len(intersect_groups)

	## T-SNE
	model = TSNE(n_components=2)

	# Privileged dataset
	priv_index = np.where(aif_data.protected_attributes == aif_data.privileged_protected_attributes)[0]
	priv_dset = aif_data.subset(priv_index)

	# Unprivileged dataset
	unpriv_index = np.where(aif_data.protected_attributes == aif_data.unprivileged_protected_attributes)[0]
	unpriv_dset = aif_data.subset(unpriv_index)

	## T-SNE modeling
	tsne_data_priv = model.fit_transform(priv_dset.features)
	tsne_data_unpriv = model.fit_transform(unpriv_dset.features)

	data_info = {
		'num_classes': num_classes,
		'num_intersects': num_intersects,
		'intersect_groups': intersect_groups,
		'protected_attribute': aif_data.protected_attribute_names[0],
		'privileged_tsne_data': tsne_data_priv.tolist(),
		'unprivileged_tsne_data': tsne_data_unpriv.tolist()
	}

	context = {'info': data_info, 'algorithm': 'fVAE'}

	return render(request, 'new/data.html', context)


def fVAE_metric(request):
	df = pd.read_table('./data/CelebA/list_attr_celeba.csv', sep=',')
	aif_data = aifData(
		df=df,
		label_name='Heavy_Makeup',
		favorable_classes=[1],
		protected_attribute_names=['Male'],
		privileged_classes=[[-1]],
		features_to_keep=['Heavy_Makeup', 'Male'],
		na_values=['?']
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'Male':-1}], unprivilege=[{'Male':1}])

	context = {
		'num_positive_list': [metric.num_positive(), metric.num_positive(privileged=True), metric.num_positive(privileged=False)],
		'num_negative_list': [metric.num_negative(), metric.num_negative(privileged=True), metric.num_negative(privileged=False)],
		'base_rate_list': [metric.base_rate(), metric.base_rate(privileged=True), metric.base_rate(privileged=False)],
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
		'data': aif_data,
		'id': 0,
		'algorithm': 'fVAE'
	}

	return render(request, 'new/metric.html', context)


def fVAE_miti(request):
	result = get_fVAE_result.delay()

	url = f'mitiresult?id={result.id}'

	return HttpResponseRedirect(url)


def fVAE_miti_result(request):
	try:
		result = AsyncResult(request.GET['id'])
	except:
		return HttpResponse('You entered this page wrong route.<br>Please start from index page.<br><br><a href="new/index">go to index</a>')

	if result.ready():
		r = result.get()
	else:
		r = None

	context = {'id': result.id, 'result': r, 'algorithm': 'fVAE'}

	return render(request, 'new/mitigate.html', context)


def FFD_data(request):
	df = pd.read_csv('./new/FFD_miti_result.csv')
	# label: 0 (age < 20), 1 (20 < age < 40), 2 (age > 40)
	# protected: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[1],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[0]],
		)

	num_classes = len(np.unique(aif_data.labels))
	dirichlet_alpha = 1.0 / num_classes
	intersect_groups = np.unique(aif_data.protected_attributes, axis=0)
	num_intersects = len(intersect_groups)

	## T-SNE
	model = TSNE(n_components=2)

	# Privileged dataset
	priv_index = np.where(aif_data.protected_attributes == aif_data.privileged_protected_attributes)[0]
	priv_dset = aif_data.subset(priv_index)

	# Unprivileged dataset
	unpriv_index = np.where(aif_data.protected_attributes == aif_data.unprivileged_protected_attributes)[0]
	unpriv_dset = aif_data.subset(unpriv_index)

	## T-SNE modeling
	tsne_data_priv = model.fit_transform(priv_dset.features)
	tsne_data_unpriv = model.fit_transform(unpriv_dset.features)

	data_info = {
		'num_classes': num_classes,
		'num_intersects': num_intersects,
		'intersect_groups': intersect_groups,
		'protected_attribute': aif_data.protected_attribute_names[0],
		'privileged_tsne_data': tsne_data_priv.tolist(),
		'unprivileged_tsne_data': tsne_data_unpriv.tolist()
	}

	context = {'info': data_info, 'algorithm': 'FFD'}

	return render(request, 'new/data.html', context)


def FFD_metric(request):
	df = pd.read_csv('./new/FFD_miti_result.csv')
	# label: 0 (age < 20), 1 (20 < age < 40), 2 (age > 40)
	# protected: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[1],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[0]],
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'base_protected':0}], unprivilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':3}])

	context = {
		'num_positive_list': [metric.num_positive(), metric.num_positive(privileged=True), metric.num_positive(privileged=False)],
		'num_negative_list': [metric.num_negative(), metric.num_negative(privileged=True), metric.num_negative(privileged=False)],
		'base_rate_list': [metric.base_rate(), metric.base_rate(privileged=True), metric.base_rate(privileged=False)],
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
		'data': aif_data,
		'id': 0,
		'algorithm': 'FFD'
	}

	return render(request, 'new/metric.html', context)



def FFD_mitigation(request):
	df = pd.read_csv('./new/FFD_miti_result.csv')
	# label: 0 (age < 20), 1 (20 < age < 40), 2 (age > 40)
	# protected: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[1],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[0]],
		)

	data_metric_orig = DataMetric(dataset=aif_data, privilege=[{'base_protected':0}], unprivilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':3}])
	data_metric_transf = DataMetric(dataset=aif_data, privilege=[{'base_protected':0}], unprivilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':3}])

	data_metric_orig_obj = {
		'num_positive': data_metric_orig.num_positive(),
		'num_negative': data_metric_orig.num_negative(),
		'base_rate': data_metric_orig.base_rate(),
		'disparate_impact': data_metric_orig.disparate_impact(),
		'statistical_parity_difference': data_metric_orig.statistical_parity_difference()
	}

	data_metric_transf_obj = {
		'num_positive': data_metric_transf.num_positive(),
		'num_negative': data_metric_transf.num_negative(),
		'base_rate': data_metric_transf.base_rate(),
		'disparate_impact': data_metric_transf.disparate_impact(),
		'statistical_parity_difference': data_metric_transf.statistical_parity_difference()
	}


	class_metric_orig = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'base_protected':0}],
		unprivilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':3}],
		prediction_vector=df['base_prediction'].to_numpy(),
		target_label_name='base_label'
	)

	class_metric_orig_obj = {
		'performance': class_metric_orig.performance_measures(),
		'error_rate': class_metric_orig.error_rate(),
		'average_odds_difference': class_metric_orig.average_odds_difference(),
		'average_abs_odds_difference': class_metric_orig.average_abs_odds_difference(),
		'selection_rate': class_metric_orig.selection_rate(),
		'equal_opportunity_difference': class_metric_orig.equal_opportunity_difference(),
		'disparate_impact': class_metric_orig.disparate_impact(),
		'statistical_parity_difference': class_metric_orig.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_orig.generalized_entropy_index(),
		'theil_index': class_metric_orig.theil_index()
	}

	transf_aif_data = aifData(
		df=df,
		label_name='debias_label',
		favorable_classes=[1],
		protected_attribute_names=['debias_protected'],
		privileged_classes=[[0]]
		)

	class_metric_transf = ClassificationMetric(
		dataset=transf_aif_data,
		privilege=[{'debias_protected':0}],
		unprivilege=[{'debias_protected':1}, {'debias_protected':2}, {'debias_protected':3}],
		prediction_vector=df['debias_prediction'].to_numpy(),
		target_label_name='base_label'
		)

	class_metric_transf_obj = {
		'performance': class_metric_transf.performance_measures(),
		'error_rate': class_metric_transf.error_rate(),
		'average_odds_difference': class_metric_transf.average_odds_difference(),
		'average_abs_odds_difference': class_metric_transf.average_abs_odds_difference(),
		'selection_rate': class_metric_transf.selection_rate(),
		'equal_opportunity_difference': class_metric_transf.equal_opportunity_difference(),
		'disparate_impact': class_metric_transf.disparate_impact(),
		'statistical_parity_difference': class_metric_transf.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_transf.generalized_entropy_index(),
		'theil_index': class_metric_transf.theil_index()
	}

	result = {
		'odm': data_metric_orig_obj,
		'tdm': data_metric_transf_obj,
		'ocm': class_metric_orig_obj,
		'tcm': class_metric_transf_obj
	}

	context = {'id': '(null)', 'result': result, 'algorithm': 'FFD'}

	return render(request, 'new/mitigate.html', context)


def LfF_data(request):
	df = pd.read_csv('./new/LfF_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[0, 1, 2, 3, 4],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[1], [2], [7]],
		)

	num_classes = len(np.unique(aif_data.labels))
	dirichlet_alpha = 1.0 / num_classes
	intersect_groups = np.unique(aif_data.protected_attributes, axis=0)
	num_intersects = len(intersect_groups)

	## T-SNE
	model = TSNE(n_components=2)

	# Privileged dataset
	priv_index = np.where(aif_data.protected_attributes == aif_data.privileged_protected_attributes)[0]
	priv_dset = aif_data.subset(priv_index)

	# Unprivileged dataset
	unpriv_index = np.where(aif_data.protected_attributes == aif_data.unprivileged_protected_attributes)[0]
	unpriv_dset = aif_data.subset(unpriv_index)

	## T-SNE modeling
	tsne_data_priv = model.fit_transform(priv_dset.features)
	tsne_data_unpriv = model.fit_transform(unpriv_dset.features)

	data_info = {
		'num_classes': num_classes,
		'num_intersects': num_intersects,
		'intersect_groups': intersect_groups,
		'protected_attribute': aif_data.protected_attribute_names[0],
		'privileged_tsne_data': tsne_data_priv.tolist(),
		'unprivileged_tsne_data': tsne_data_unpriv.tolist()
	}

	context = {'info': data_info, 'algorithm': 'LfF'}

	return render(request, 'new/data.html', context)


def LfF_metric(request):
	df = pd.read_csv('./new/LfF_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[0, 1, 2, 3, 4],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[1], [2], [7]],
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':7}], unprivilege=[{'base_protected':8}, {'base_protected':6}, {'base_protected':5}])

	context = {
		'num_positive_list': [metric.num_positive(), metric.num_positive(privileged=True), metric.num_positive(privileged=False)],
		'num_negative_list': [metric.num_negative(), metric.num_negative(privileged=True), metric.num_negative(privileged=False)],
		'base_rate_list': [metric.base_rate(), metric.base_rate(privileged=True), metric.base_rate(privileged=False)],
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
		'data': aif_data,
		'id': 0,
		'algorithm': 'LfF'
	}

	return render(request, 'new/metric.html', context)


def LfF_mitigation(request):
	df = pd.read_csv('./new/LfF_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='base_label',
		favorable_classes=[0, 1, 2, 3, 4],
		protected_attribute_names=['base_protected'],
		privileged_classes=[[1], [2], [7]],
		)

	data_metric_orig = DataMetric(dataset=aif_data, privilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':7}], unprivilege=[{'base_protected':8}, {'base_protected':6}, {'base_protected':5}])
	data_metric_transf = DataMetric(dataset=aif_data, privilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':7}], unprivilege=[{'base_protected':8}, {'base_protected':6}, {'base_protected':5}])

	data_metric_orig_obj = {
		'num_positive': data_metric_orig.num_positive(),
		'num_negative': data_metric_orig.num_negative(),
		'base_rate': data_metric_orig.base_rate(),
		'disparate_impact': data_metric_orig.disparate_impact(),
		'statistical_parity_difference': data_metric_orig.statistical_parity_difference()
	}

	data_metric_transf_obj = {
		'num_positive': data_metric_transf.num_positive(),
		'num_negative': data_metric_transf.num_negative(),
		'base_rate': data_metric_transf.base_rate(),
		'disparate_impact': data_metric_transf.disparate_impact(),
		'statistical_parity_difference': data_metric_transf.statistical_parity_difference()
	}


	class_metric_orig = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'base_protected':1}, {'base_protected':2}, {'base_protected':7}],
		unprivilege=[{'base_protected':8}, {'base_protected':6}, {'base_protected':5}],
		prediction_vector=df['base_prediction'].to_numpy(),
		target_label_name='base_label'
	)

	class_metric_orig_obj = {
		'performance': class_metric_orig.performance_measures(),
		'error_rate': class_metric_orig.error_rate(),
		'average_odds_difference': class_metric_orig.average_odds_difference(),
		'average_abs_odds_difference': class_metric_orig.average_abs_odds_difference(),
		'selection_rate': class_metric_orig.selection_rate(),
		'equal_opportunity_difference': class_metric_orig.equal_opportunity_difference(),
		'disparate_impact': class_metric_orig.disparate_impact(),
		'statistical_parity_difference': class_metric_orig.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_orig.generalized_entropy_index(),
		'theil_index': class_metric_orig.theil_index()
	}

	transf_aif_data = aifData(
		df=df,
		label_name='debias_label',
		favorable_classes=[0, 1, 2, 3, 4],
		protected_attribute_names=['debias_protected'],
		privileged_classes=[[1], [2], [7]]
		)

	class_metric_transf = ClassificationMetric(
		dataset=transf_aif_data,
		privilege=[{'debias_protected':1}, {'debias_protected':2}, {'debias_protected':7}],
		unprivilege=[{'debias_protected':8}, {'debias_protected':6}, {'debias_protected':5}],
		prediction_vector=df['debias_prediction'].to_numpy(),
		target_label_name='debias_label'
		)

	class_metric_transf_obj = {
		'performance': class_metric_transf.performance_measures(),
		'error_rate': class_metric_transf.error_rate(),
		'average_odds_difference': class_metric_transf.average_odds_difference(),
		'average_abs_odds_difference': class_metric_transf.average_abs_odds_difference(),
		'selection_rate': class_metric_transf.selection_rate(),
		'equal_opportunity_difference': class_metric_transf.equal_opportunity_difference(),
		'disparate_impact': class_metric_transf.disparate_impact(),
		'statistical_parity_difference': class_metric_transf.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_transf.generalized_entropy_index(),
		'theil_index': class_metric_transf.theil_index()
	}

	result = {
		'odm': data_metric_orig_obj,
		'tdm': data_metric_transf_obj,
		'ocm': class_metric_orig_obj,
		'tcm': class_metric_transf_obj
	}

	context = {'id': '(null)', 'result': result, 'algorithm': 'LfF'}

	return render(request, 'new/mitigate.html', context)


def FB_data(request):
	df = pd.read_csv('./new/FB_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	num_classes = len(np.unique(aif_data.labels))
	dirichlet_alpha = 1.0 / num_classes
	intersect_groups = np.unique(aif_data.protected_attributes, axis=0)
	num_intersects = len(intersect_groups)

	## T-SNE
	model = TSNE(n_components=2)

	# Privileged dataset
	priv_index = np.where(aif_data.protected_attributes == aif_data.privileged_protected_attributes)[0]
	priv_dset = aif_data.subset(priv_index)

	# Unprivileged dataset
	unpriv_index = np.where(aif_data.protected_attributes == aif_data.unprivileged_protected_attributes)[0]
	unpriv_dset = aif_data.subset(unpriv_index)

	## T-SNE modeling
	tsne_data_priv = model.fit_transform(priv_dset.features)
	tsne_data_unpriv = model.fit_transform(unpriv_dset.features)

	data_info = {
		'num_classes': num_classes,
		'num_intersects': num_intersects,
		'intersect_groups': intersect_groups,
		'protected_attribute': aif_data.protected_attribute_names[0],
		'privileged_tsne_data': tsne_data_priv.tolist(),
		'unprivileged_tsne_data': tsne_data_unpriv.tolist()
	}

	context = {'info': data_info, 'algorithm': 'FB'}

	return render(request, 'new/data.html', context)


def FB_metric(request):
	df = pd.read_csv('./new/FB_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])

	context = {
		'num_positive_list': [metric.num_positive(), metric.num_positive(privileged=True), metric.num_positive(privileged=False)],
		'num_negative_list': [metric.num_negative(), metric.num_negative(privileged=True), metric.num_negative(privileged=False)],
		'base_rate_list': [metric.base_rate(), metric.base_rate(privileged=True), metric.base_rate(privileged=False)],
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
		'data': aif_data,
		'id': 0,
		'algorithm': 'FB'
	}

	return render(request, 'new/metric.html', context)


def FB_mitigation(request):
	df = pd.read_csv('./new/FB_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	data_metric_orig = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])
	data_metric_transf = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])

	data_metric_orig_obj = {
		'num_positive': data_metric_orig.num_positive(),
		'num_negative': data_metric_orig.num_negative(),
		'base_rate': data_metric_orig.base_rate(),
		'disparate_impact': data_metric_orig.disparate_impact(),
		'statistical_parity_difference': data_metric_orig.statistical_parity_difference()
	}

	data_metric_transf_obj = {
		'num_positive': data_metric_transf.num_positive(),
		'num_negative': data_metric_transf.num_negative(),
		'base_rate': data_metric_transf.base_rate(),
		'disparate_impact': data_metric_transf.disparate_impact(),
		'statistical_parity_difference': data_metric_transf.statistical_parity_difference()
	}

	pred_data = logistic_classification(aif_data, aif_data)

	class_metric_orig = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'bias':1}],
		unprivilege=[{'bias':0}],
		prediction_vector=pred_data.labels,
		target_label_name='target'
	)

	class_metric_orig_obj = {
		'performance': class_metric_orig.performance_measures(),
		'error_rate': class_metric_orig.error_rate(),
		'average_odds_difference': class_metric_orig.average_odds_difference(),
		'average_abs_odds_difference': class_metric_orig.average_abs_odds_difference(),
		'selection_rate': class_metric_orig.selection_rate(),
		'equal_opportunity_difference': class_metric_orig.equal_opportunity_difference(),
		'disparate_impact': class_metric_orig.disparate_impact(),
		'statistical_parity_difference': class_metric_orig.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_orig.generalized_entropy_index(),
		'theil_index': class_metric_orig.theil_index()
	}


	class_metric_transf = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'bias':1}],
		unprivilege=[{'bias':0}],
		prediction_vector=df['prediction'].to_numpy(),
		target_label_name='target'
		)

	class_metric_transf_obj = {
		'performance': class_metric_transf.performance_measures(),
		'error_rate': class_metric_transf.error_rate(),
		'average_odds_difference': class_metric_transf.average_odds_difference(),
		'average_abs_odds_difference': class_metric_transf.average_abs_odds_difference(),
		'selection_rate': class_metric_transf.selection_rate(),
		'equal_opportunity_difference': class_metric_transf.equal_opportunity_difference(),
		'disparate_impact': class_metric_transf.disparate_impact(),
		'statistical_parity_difference': class_metric_transf.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_transf.generalized_entropy_index(),
		'theil_index': class_metric_transf.theil_index()
	}

	result = {
		'odm': data_metric_orig_obj,
		'tdm': data_metric_transf_obj,
		'ocm': class_metric_orig_obj,
		'tcm': class_metric_transf_obj
	}

	context = {'id': '(null)', 'result': result, 'algorithm': 'FB'}

	return render(request, 'new/mitigate.html', context)


def KDE(request):
	return HttpResponse('Hello world!')


def CPR_data(request):
	df = pd.read_csv('./new/CPR_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1, 2, 3, 4],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	num_classes = len(np.unique(aif_data.labels))
	dirichlet_alpha = 1.0 / num_classes
	intersect_groups = np.unique(aif_data.protected_attributes, axis=0)
	num_intersects = len(intersect_groups)

	## T-SNE
	model = TSNE(n_components=2)

	# Privileged dataset
	priv_index = np.where(aif_data.protected_attributes == aif_data.privileged_protected_attributes)[0]
	priv_dset = aif_data.subset(priv_index)

	# Unprivileged dataset
	unpriv_index = np.where(aif_data.protected_attributes == aif_data.unprivileged_protected_attributes)[0]
	unpriv_dset = aif_data.subset(unpriv_index)

	## T-SNE modeling
	tsne_data_priv = model.fit_transform(priv_dset.features)
	tsne_data_unpriv = model.fit_transform(unpriv_dset.features)

	data_info = {
		'num_classes': num_classes,
		'num_intersects': num_intersects,
		'intersect_groups': intersect_groups,
		'protected_attribute': aif_data.protected_attribute_names[0],
		'privileged_tsne_data': tsne_data_priv.tolist(),
		'unprivileged_tsne_data': tsne_data_unpriv.tolist()
	}

	context = {'info': data_info, 'algorithm': 'CPR'}

	return render(request, 'new/data.html', context)


def CPR_metric(request):
	df = pd.read_csv('./new/CPR_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1, 2, 3, 4],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])

	context = {
		'num_positive_list': [metric.num_positive(), metric.num_positive(privileged=True), metric.num_positive(privileged=False)],
		'num_negative_list': [metric.num_negative(), metric.num_negative(privileged=True), metric.num_negative(privileged=False)],
		'base_rate_list': [metric.base_rate(), metric.base_rate(privileged=True), metric.base_rate(privileged=False)],
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
		'data': aif_data,
		'id': 0,
		'algorithm': 'CPR'
	}

	return render(request, 'new/metric.html', context)


def CPR_mitigation(request):
	df = pd.read_csv('./new/CPR_miti_result.csv')
	# label: 0 ~ 9
	# protected: 0 ~ 9 (random)
	aif_data = aifData(
		df=df,
		label_name='target',
		favorable_classes=[1, 2, 3, 4],
		protected_attribute_names=['bias'],
		privileged_classes=[[1]],
		)

	data_metric_orig = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])
	data_metric_transf = DataMetric(dataset=aif_data, privilege=[{'bias':1}], unprivilege=[{'bias':0}])

	data_metric_orig_obj = {
		'num_positive': data_metric_orig.num_positive(),
		'num_negative': data_metric_orig.num_negative(),
		'base_rate': data_metric_orig.base_rate(),
		'disparate_impact': data_metric_orig.disparate_impact(),
		'statistical_parity_difference': data_metric_orig.statistical_parity_difference()
	}

	data_metric_transf_obj = {
		'num_positive': data_metric_transf.num_positive(),
		'num_negative': data_metric_transf.num_negative(),
		'base_rate': data_metric_transf.base_rate(),
		'disparate_impact': data_metric_transf.disparate_impact(),
		'statistical_parity_difference': data_metric_transf.statistical_parity_difference()
	}

	pred_data = logistic_classification(aif_data, aif_data)

	class_metric_orig = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'bias':1}],
		unprivilege=[{'bias':0}],
		prediction_vector=pred_data.labels,
		target_label_name='target'
	)

	class_metric_orig_obj = {
		'performance': class_metric_orig.performance_measures(),
		'error_rate': class_metric_orig.error_rate(),
		'average_odds_difference': class_metric_orig.average_odds_difference(),
		'average_abs_odds_difference': class_metric_orig.average_abs_odds_difference(),
		'selection_rate': class_metric_orig.selection_rate(),
		'equal_opportunity_difference': class_metric_orig.equal_opportunity_difference(),
		'disparate_impact': class_metric_orig.disparate_impact(),
		'statistical_parity_difference': class_metric_orig.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_orig.generalized_entropy_index(),
		'theil_index': class_metric_orig.theil_index()
	}


	class_metric_transf = ClassificationMetric(
		dataset=aif_data,
		privilege=[{'bias':1}],
		unprivilege=[{'bias':0}],
		prediction_vector=df['prediction'].to_numpy(),
		target_label_name='target'
		)

	class_metric_transf_obj = {
		'performance': class_metric_transf.performance_measures(),
		'error_rate': class_metric_transf.error_rate(),
		'average_odds_difference': class_metric_transf.average_odds_difference(),
		'average_abs_odds_difference': class_metric_transf.average_abs_odds_difference(),
		'selection_rate': class_metric_transf.selection_rate(),
		'equal_opportunity_difference': class_metric_transf.equal_opportunity_difference(),
		'disparate_impact': class_metric_transf.disparate_impact(),
		'statistical_parity_difference': class_metric_transf.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_transf.generalized_entropy_index(),
		'theil_index': class_metric_transf.theil_index()
	}

	result = {
		'odm': data_metric_orig_obj,
		'tdm': data_metric_transf_obj,
		'ocm': class_metric_orig_obj,
		'tcm': class_metric_transf_obj
	}

	context = {'id': '(null)', 'result': result, 'algorithm': 'CPR'}

	return render(request, 'new/mitigate.html', context)



def logistic_classification(train_data, test_data):
	lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
	scaler = MinMaxScaler(copy=False)

	train_data.features = scaler.fit_transform(train_data.features)
	test_data.features = scaler.fit_transform(test_data.features)

	ids = [train_data.feature_names.index(pan) for pan in train_data.protected_attribute_names]

	X_tr = np.delete(train_data.features, ids, axis=1)
	X_te = np.delete(test_data.features, ids, axis=1)
	y_tr = train_data.labels.ravel()

	lmod.fit(X_tr, y_tr)

	test_pred = test_data.copy()
	test_pred.labels = lmod.predict(X_te)

	return test_pred