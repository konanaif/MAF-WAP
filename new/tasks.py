from __future__ import absolute_import, unicode_literals
from celery import shared_task

import time
import pandas as pd

from Kaif.DataSet import aifData
from Kaif.Metric import DataMetric, ClassificationMetric

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf

from FairnessVAE.solver_40dim_clatent import Solver


@shared_task
def get_fVAE_result():
	# Data
	df = pd.read_table('./data/CelebA/list_attr_celeba.csv', sep=',')
	aif_data = aifData(
		df=df,
		label_name='Heavy_Makeup',
		favorable_classes=[1],
		protected_attribute_names=['Male'],
		privileged_classes=[[-1]],
		features_to_drop=['image_id'],
		na_values=['?']
		)

	metric = DataMetric(dataset=aif_data, privilege=[{'Male':-1}], unprivilege=[{'Male':1}])

	data_metric_orig_obj = {
		'num_positive': metric.num_positive(),
		'num_negative': metric.num_negative(),
		'base_rate': metric.base_rate(),
		'disparate_impact': metric.disparate_impact(),
		'statistical_parity_difference': metric.statistical_parity_difference(),
	}

	data_metric_transf_obj = data_metric_orig_obj


	# Algorithm
	train_orig, test_orig = aif_data.split([0.8], shuffle=True)

	pred_data = logistic_classification(train_orig, test_orig)
	class_metric_orig = ClassificationMetric(
		dataset=test_orig,
		privilege=[{'Male':-1}],
		unprivilege=[{'Male':1}],
		prediction_vector=pred_data.labels,
		target_label_name=aif_data.label_names[0])

	class_metric_orig_obj = {
		'performance': class_metric_orig.performance_measures(),
		'error_rate': class_metric_orig.error_rate(),
		'average_odds_difference': class_metric_orig.average_odds_difference(),
		'average_abs_odds_difference': class_metric_orig.average_abs_odds_difference(),
		'selection_rate': class_metric_orig.selection_rate(),
		'disparate_impact': class_metric_orig.disparate_impact(),
		'statistical_parity_difference': class_metric_orig.statistical_parity_difference(),
		'generalized_entropy_index': class_metric_orig.generalized_entropy_index(),
		'theil_index': class_metric_orig.theil_index()
	}



	class Settings:
	    def __init__(self):
	        self.name = 'main'
	        self.cuda = False
	        self.max_iter = 100
	        self.batch_size = 32
	        self.eval_batch_size = 64
	        
	        self.z_dim = 40
	        self.t_dim = 20
	        self.c_dim = 20
	        self.p_dim = 20
	        self.gamma = 6.4
	        self.lr_VAE = 1e-4
	        self.beta1_VAE = 0.9
	        self.beta2_VAE = 0.999
	        self.lr_D = 1e-4
	        self.beta1_D = 0.5
	        self.beta2_D = 0.9
	        self.alpha = 1
	        self.beta = 5
	        self.grl = 10
	        self.lr_cls = 0.000001
	        
	        self.dset_dir = 'data'
	        self.dataset = 'CelebA'
	        self.image_size = 64
	        self.num_workers = 2
	        
	        self.viz_on = False
	        self.viz_port = 8097
	        self.viz_ll_iter = 1
	        self.viz_la_iter = 1
	        self.viz_ra_iter = 1
	        self.viz_ta_iter = 1
	        
	        self.print_iter = 1
	        
	        self.ckpt_dir = 'FairnessVAE/checkpoints'
	        self.ckpt_load = None
	        self.ckpt_save_iter = 1
	        
	        self.output_dir = 'FairnessVAE/outputs'
	        self.output_save = True
	        self.train = 3

	setting = Settings()
	net = Solver(setting)

	result = net.val()

	TP_male = result.total_male_heavy.float()
	TN_male = result.total_male_nonheavy.float()
	FP_male = result.total_male_heavy_num.float() - TP_male
	FN_male = result.total_male_nonheavy_num.float() - TN_male

	TP_female = result.total_female_heavy.float()
	TN_female = result.total_female_nonheavy.float()
	FP_female = result.total_female_heavy_num.float() - TP_female
	FN_female = result.total_female_nonheavy_num.float() - TN_female

	TP = TP_male + TP_female
	TN = TN_male + TN_female
	FP = FP_male + FP_female
	FN = FN_male + FN_female

	TPR_male = TP_male / (TP_male + FP_male)
	TNR_male = TN_male / (TN_male + FN_male)
	FPR_male = FP_male / (TP_male + FP_male)
	FNR_male = FN_male / (TN_male + FN_male)

	TPR_female = TP_female / (TP_female + FP_female)
	TNR_female = TN_female / (TN_female + FN_female)
	FPR_female = FP_female / (TP_female + FP_female)
	FNR_female = FN_female / (TN_female + FN_female)

	error_rate = (FP + FN) / (TP + TN + FP + FN)
	average_odds_difference = ((FPR_female - FPR_male) + (TPR_female - TPR_male)) / 2.0
	average_abs_odds_difference = (np.abs(FPR_female - FPR_male) + np.abs(TPR_female - TPR_male)) / 2.0
	
	selection_rate_male = (TP_male + FP_male) / (TP_male + FP_male + TN_male + FN_male)
	selection_rate_female = (TP_female + FP_female) / (TP_female + FP_female + TN_female + FN_female)
	
	selection_rate = (TP + FP) / (TP + FP + TN + FN)
	disparate_impact = selection_rate_female / selection_rate_male
	statistical_parity_difference = selection_rate_female - selection_rate_male

	generalized_entropy_index = 0
	theil_index = 0

	class_metric_transf_obj = {
		'error_rate': error_rate,
		'average_odds_difference': average_odds_difference,
		'average_abs_odds_difference': average_abs_odds_difference,
		'selection_rate': selection_rate,
		'disparate_impact': disparate_impact,
		'statistical_parity_difference': statistical_parity_difference,
		'generalized_entropy_index': generalized_entropy_index,
		'theil_index': theil_index
	}

	result = {
		'odm': data_metric_orig_obj,
		'ocm': class_metric_orig_obj,
		'tdm': data_metric_transf_obj,
		'tcm': class_metric_transf_obj
	}

	return result


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