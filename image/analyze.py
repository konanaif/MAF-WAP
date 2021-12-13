import os
import pandas as pd
import numpy as np
import json
import zipfile
from PIL import Image
import glob

from Kaif.Data import DataLoader, TextLoader
from Kaif.DataSet import aifData
from Kaif.Metric import DataMetric, ClassificationMetric
from Kaif.Algorithms.Preprocessing import RW, Disperate_Impact_Remover, Learning_Fair_Representation
from Kaif.Algorithms.Inprocessing import Adversarial_Debiasing, Gerry_Fair_Classifier, Meta_Fair_Classifier, Prejudice_Remover
from Kaif.Algorithms.Postprocessing import Calibrated_EqOdds, EqualizedOdds, RejectOption

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .models import Data
from django.core.files import File

# Tokenizer
from konlpy.tag import Okt
okt = Okt()


class Kaifdata:
	def __init__(self, ids, datatype):
		# Check n of analyzed
		n_analyzed = sum([1 if Data.objects.get(id=i).aif_input_file_path else 0 for i in ids])
		#print(n_analyzed)

		if n_analyzed == len(ids):
			# Case 1: Analyzed only
			print("All data was analyzed already.")
			all_df = []

			for i in ids:
				data = Data.objects.get(id=i)
				lconfig = json.loads(data.config_load_json)
				try:
					df = pd.read_csv(os.path.join(data.datatype, 'data', data.aif_input_file_path))
				except:
					df = pd.read_csv(os.path.join('dataframes', data.aif_input_file_path))
				all_df.append(df)

			vec_df = pd.concat(all_df, join='outer', ignore_index=True)
			vec_df = vec_df.drop_duplicates()

			if datatype == 'text':
				self.aif_data = aifData(
					df=vec_df,
					label_name=lconfig['label_column'],
					favorable_classes=lconfig['favorable_classes'],
					protected_attribute_names=['bias'],
					privileged_classes=[[1]],
					features_to_drop=['Data'],
					na_values=lconfig['na_values']
				)
			elif datatype == 'image':
				self.aif_data = aifData(df=vec_df, 
					label_name=lconfig['label_name'], 
					favorable_classes=lconfig['favorable_classes'], 
					protected_attribute_names=lconfig['protected_attribute_names'], 
					privileged_classes=lconfig['privileged_classes'], 
					features_to_drop=lconfig['features_to_drop']+['Shape'],
					na_values=lconfig['na_values'], 
					categorical_features=lconfig['categorical_features']
				)
			else:
				self.aif_data = aifData(df=vec_df, 
					label_name=lconfig['label_name'], 
					favorable_classes=lconfig['favorable_classes'], 
					protected_attribute_names=lconfig['protected_attribute_names'], 
					privileged_classes=lconfig['privileged_classes'], 
					features_to_drop=lconfig['features_to_drop'], 
					na_values=lconfig['na_values'], 
					categorical_features=lconfig['categorical_features']
				)

		else:
			# Case 2: Analyzed + Not Analyzed
			# Case 3: Not Analyzed only
			print("We will make new dataframes.")

			df_list = []

			for i in ids:
				data = Data.objects.get(id=i)
				filename = data.datafile.name
				dconfig = json.loads(data.config_data_json)
				lconfig = json.loads(data.config_load_json)

				df = pd.read_table(filename, sep=dconfig['sep'], header=dconfig['header'], names=dconfig['names'])

				df_list.append(df)

			dFrame = pd.concat(df_list)
			dFrame = dFrame.dropna()

			if datatype == 'text':
				corpus = dFrame[ lconfig['document_column'] ].to_list()
				labels = dFrame[ lconfig['label_column'] ].to_list()

				loader = TextLoader(corpus=corpus, tokenizer=okt.nouns)
				loader.Make_Data(privilege_keys=lconfig['privilege_keys'], unprivilege_keys=lconfig['unprivilege_keys'], threshold=0.7)
				loader.Append_annotation(labels, colnames=[ lconfig['label_column'] ], method=lambda x, params: x)

				result_df = loader.convert_to_DataFrame()
				dFrame = pd.merge(loader.vectorized_df, result_df, left_index=True, right_index=True, how='outer')
				dFrame = dFrame.dropna()

				self.aif_data = aifData(
					df=dFrame,
					label_name=lconfig['label_column'],
					favorable_classes=lconfig['favorable_classes'],
					protected_attribute_names=['bias'],
					privileged_classes=[[1]],
					features_to_drop=['Data'],
					na_values=lconfig['na_values']
				)

				# Save the dataframe -> add path to django.models.Data object
				id_str = '+'.join([str(i) for i in ids])
				dFrame_filename = os.path.join('text', 'data','dataframes', "{id}_input_df.csv".format(id=id_str))

				for i in ids:
					data = Data.objects.get(id=i)
					data.aif_input_file_path = dFrame_filename
					data.save()

				dFrame.to_csv(dFrame_filename, index=0)

			else:
				self.aif_data = aifData(df=dFrame, 
					label_name=lconfig['label_name'], 
					favorable_classes=lconfig['favorable_classes'], 
					protected_attribute_names=lconfig['protected_attribute_names'], 
					privileged_classes=lconfig['privileged_classes'], 
					features_to_drop=lconfig['features_to_drop'], 
					na_values=lconfig['na_values'], 
					categorical_features=lconfig['categorical_features']
				)

				# Save the dataframe -> add path to django.models.Data object
				id_str = '+'.join([str(i) for i in ids])
				dFrame_filename = os.path.join('raw', 'data','dataframes', "{id}_input_df.csv".format(id=id_str))

				for i in ids:
					data = Data.objects.get(id=i)
					data.aif_input_file_path = dFrame_filename
					data.save()

				dFrame.to_csv(dFrame_filename, index=0)



class Imagedata(Kaifdata):
	def __init__(self, id, datatype):
		# Check analyzed
		data = Data.objects.get(id=id)  # zip file
		dconfig = json.loads(data.config_data_json)
		lconfig = json.loads(data.config_load_json)

		os.chdir(os.path.join('image', 'data'))

		if data.aif_input_file_path:
			print('This data already analyzed.')
			try:
				df = pd.read_csv(data.aif_input_file_path)
			except:
				df = pd.read_csv(os.path.join('dataframes', data.aif_input_file_path))

			df = df.drop_duplicates()

			self.aif_data = aifData(df=df, 
				label_name=lconfig['label_name'], 
				favorable_classes=lconfig['favorable_classes'], 
				protected_attribute_names=lconfig['protected_attribute_names'], 
				privileged_classes=lconfig['privileged_classes'], 
				features_to_drop=lconfig['features_to_drop']+['Shape'],
				na_values=lconfig['na_values'], 
				categorical_features=lconfig['categorical_features']
			)

		else:
			print('Unzip the file.')

			with zipfile.ZipFile(os.path.basename(data.datafile.name)) as zf:
				zf.extractall()

			# read label file
			label_df = pd.read_table(dconfig['labelFileName'], sep=dconfig['sep'], header=dconfig['header'])

			# image load
			images = glob.glob(os.path.join(dconfig['imageDirName'], '*[jpg$|png$|jpeg$|gif$]'))

			keys = []
			shapes = []
			vectors = []
			for p in images:
				# key
				if dconfig['keyForm'] == 'filename':
					key = os.path.basename(p)
				elif dconfig['keyForm'] == 'basename':
					key = os.path.splitext(os.path.basename(p))[0]
				else:
					key = p

				# image shape and vector (1-D)
				im = Image.open(p)
				pix = np.array(im)
				shape = pix.shape
				vector = pix.ravel()

				keys.append(key)
				shapes.append(shape)
				vectors.append(vector)

			## make a dataframe
			df_obj = {dconfig['keyColname']: keys, 'Shape': shapes}
			vectors = pad_sequences(np.array(vectors, dtype=object))
			vectors = vectors.transpose()

			for i, vect in enumerate(vectors):
				colname = 'Pix_{0:020d}'.format(i)
				df_obj[colname] = vect

			image_df = pd.DataFrame(df_obj)

			df = pd.merge(label_df, image_df)

			df = df.drop_duplicates()
			
			# Save the dataframe -> add path to django.models.Data object
			id_str = '{}'.format(data.id)
			dFrame_filename = os.path.join('dataframes', "{id}_input_df.csv".format(id=id_str))

			for i in ids:
				data = Data.objects.get(id=i)
				data.aif_input_file_path = dFrame_filename
				data.save()

			df.to_csv(dFrame_filename, index=0)


			self.aif_data = aifData(df=df, 
				label_name=lconfig['label_name'], 
				favorable_classes=lconfig['favorable_classes'], 
				protected_attribute_names=lconfig['protected_attribute_names'], 
				privileged_classes=lconfig['privileged_classes'], 
				features_to_drop=lconfig['features_to_drop'] + ['Shape'],
				na_values=lconfig['na_values'], 
				categorical_features=lconfig['categorical_features']
			)




class KaifMitigation:
	def __init__(self, idx, algorithm):
		# All data was analyzed.
		data = Data.objects.get(id=idx)
		lconfig = json.loads(data.config_load_json)

		os.chdir(os.path.join('image', 'data'))

		try:
			df = pd.read_csv(data.aif_input_file_path)
		except:
			df = pd.read_csv(os.path.join('data', 'dataframes', data.aif_input_file_path))
		
		vec_df = df.drop_duplicates()

		print('>>>>> AIF360 formed data load START')
		if data.datatype == 'text':
			aif_data = aifData(
				df=vec_df,
				label_name=lconfig['label_column'],
				favorable_classes=lconfig['favorable_classes'],
				protected_attribute_names=['bias'],
				privileged_classes=[[1]],
				features_to_drop=['Data'],
				na_values=lconfig['na_values']
			)

			# Common metrics
			privilege_group = [{'bias': 1}]
			unprivilege_group = [{'bias': 0}]

		elif data.datatype == 'image':
			aif_data = aifData(
				df=vec_df,
				label_name=lconfig['label_name'],
				favorable_classes=lconfig['favorable_classes'],
				protected_attribute_names=lconfig['protected_attribute_names'],
				privileged_classes=lconfig['privileged_classes'],
				features_to_drop=lconfig['features_to_drop']+['Shape'],
				na_values=lconfig['na_values'],
				categorical_features=lconfig['categorical_features']
			)

			# Common metrics
			privilege_group = []
			unprivilege_group = []
			for i, pname in enumerate(lconfig['protected_attribute_names']):
				ptemp = {pname: lconfig['privileged_classes'][i]}
				utemp = {pname: lconfig['unprivileged_classes'][i]}
				privilege_group.append(ptemp)
				unprivilege_group.append(utemp)

		else:
			aif_data = aifData(df=vec_df, 
				label_name=lconfig['label_name'], 
				favorable_classes=lconfig['favorable_classes'], 
				protected_attribute_names=lconfig['protected_attribute_names'], 
				privileged_classes=lconfig['privileged_classes'], 
				features_to_drop=lconfig['features_to_drop'], 
				na_values=lconfig['na_values'], 
				categorical_features=lconfig['categorical_features']
			)

			# Common metrics
			privilege_group = []
			unprivilege_group = []
			for i, pname in enumerate(lconfig['protected_attribute_names']):
				ptemp = {pname: lconfig['privileged_classes'][i]}
				utemp = {pname: lconfig['unprivileged_classes'][i]}
				privilege_group.append(ptemp)
				unprivilege_group.append(utemp)

		print('DONE. <<<<<')

		

		print('>>>>> Original data metrics calculation START')
		data_metric_orig = DataMetric(dataset=aif_data, privilege=privilege_group, unprivilege=unprivilege_group)
		self.data_metric_orig_obj = {
			'num_positive': data_metric_orig.num_positive(),
			'num_negative': data_metric_orig.num_negative(),
			'base_rate': data_metric_orig.base_rate(),
			'disparate_impact': data_metric_orig.disparate_impact(),
			'statistical_parity_difference': data_metric_orig.statistical_parity_difference()
		}
		print('DONE. <<<<<')


		# Preprocessing -> Data metric changed
		if algorithm == "1":
			# Reweighing
			print('>>>>> Reweighing START')
			rw = RW(privileged_groups=privilege_group, unprivileged_groups=unprivilege_group)
			rw.fit(aif_data)
			aif_data_transf = rw.transform(aif_data)
		elif algorithm == "2":
			# Disperate Impact Remover
			print('>>>>> Disparate impact remover START')
			disp = Disperate_Impact_Remover(rep_level=0.5)
			aif_data_transf = disp.fit_transform(aif_data)
		elif algorithm == "3":
			# Learning Fair Representation
			print('>>>>> Learning Fair Representation START')
			lfr = Learning_Fair_Representation(
					unprivileged_groups=unprivilege_group,
					privileged_groups=privilege_group,
					verbose=1,
				)
			aif_data_transf = lfr.fit_transform(aif_data)
		else:
			print('>>>>> No preprocessing')
			aif_data_transf = aif_data.copy()
		print('DONE. <<<<<')

		print('>>>>> Transformed data metrics calculation START')
		data_metric_transf = DataMetric(dataset=aif_data_transf, privilege=privilege_group, unprivilege=unprivilege_group)
		self.data_metric_transf_obj = {
			'num_positive': data_metric_transf.num_positive(),
			'num_negative': data_metric_transf.num_negative(),
			'base_rate': data_metric_transf.base_rate(),
			'disparate_impact': data_metric_transf.disparate_impact(),
			'statistical_parity_difference': data_metric_transf.statistical_parity_difference()
		}
		print('DONE. <<<<<')


		train_orig, test_orig = aif_data.split([0.8], shuffle=True)

		print('>>>>> For calculation, LR modeling START (ORIGINAL)')
		pred_data = logistic_classification(train_orig, test_orig)

		print('>>>>> Modeling DONE. Metrics calculation START')


		class_metric_orig = ClassificationMetric(
			dataset=test_orig,
			privilege=privilege_group,
			unprivilege=unprivilege_group,
			prediction_vector=pred_data.labels,
			target_label_name=aif_data.label_names[0]
		)

		self.class_metric_orig_obj = {
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
		print('DONE. <<<<<')

		train_transf, test_transf = aif_data_transf.split([0.8], shuffle=True)


		# Inprocessing OR
		# Classification on Transformed data: Logistic Regression
		if algorithm == "4":
			print('>>>>> For Adversarial Debiasing, import tensorflow and debias modeling START (TRANSFORMED)')
			sess = tf.Session()
			debias_model = Adversarial_Debiasing(privileged_groups = privilege_group,
                          unprivileged_groups = unprivilege_group,
                          scope_name='plain_classifier',
                          debias=True,
                          sess=sess)
			debias_model.fit(train_transf)

			# Apply the plain model to test data
			pred_data = debias_model.predict(test_transf)
			sess.close()
			tf.reset_default_graph()
		elif algorithm == "5":
			print('>>>>> Garry Fair Classifier START')
			gfc = Gerry_Fair_Classifier()
			gfc.fit(train_transf)
			pred_data = gfc.predict(test_transf)
		elif algorithm == "6":
			print('>>>>> Meta Fair Classifier START')
			mfc = Meta_Fair_Classifier(sensitive_attr='bias')
			mfc.fit(train_transf)
			pred_data = mfc.predict(test_transf)
		elif algorithm == "7":
			print('>>>>> Prejudice Remover START')
			pr = Prejudice_Remover(eta=1.0, sensitive_attr='bias', class_attr='label')
			pr.fit(train_transf)
			pred_data = pr.predict(test_transf)
		else:
			print('>>>>> For calculation, LR modeling START')
			pred_data = logistic_classification(train_transf, test_transf)
		
		print('>>>>> Modeling DONE. Metrics calculation START')


		# Postprocessing

		# cost constraint of fnr will optimize generalized false negative rates, that of
		# fpr will optimize generalized false positive rates, and weighted will optimize
		# a weighted combination of both
		cost_constraint = "fnr" # "fnr", "fpr", "weighted"
		#random seed for calibrated equal odds prediction
		randseed = 12345679

		if algorithm == "8":
			print('>>>>> Calibrated Equalized Odds START')
			cpp = Calibrated_EqOdds(unprivileged_groups=unprivilege_group, privileged_groups=privilege_group)
			test_orig_transf = test_orig.copy()
			test_orig_transf.labels = pred_data.labels
			cpp = cpp.fit(test_orig, test_orig_transf)
			pred_data_transf = cpp.predict(pred_data)
		elif algorithm == "9":
			print('>>>>> Equalized Odds START')
			epp = EqualizedOdds(unprivileged_groups=unprivilege_group, privileged_groups=privilege_group)
			test_orig_transf = test_orig.copy()
			test_orig_transf.labels = pred_data.labels
			pred_data_transf = epp.fit_predict(test_orig, test_orig_transf)
			#pred_data_transf = epp.predict(pred_data)
		elif algorithm == "10":
			print('>>>>> Reject Option START')
			ro = RejectOption(unprivileged_groups=unprivilege_group, privileged_groups=privilege_group)
			test_orig_transf = test_orig.copy()
			test_orig_transf.labels = pred_data.labels
			ro = ro.fit(test_orig, test_orig_transf)
			pred_data_transf = ro.predict(pred_data)
		else:
			print('>>>>> No Postprocessing. Just copy the predicted dataset.')
			pred_data_transf = pred_data.copy()


		class_metric_transf = ClassificationMetric(
			dataset=test_transf,
			privilege=privilege_group,
			unprivilege=unprivilege_group,
			prediction_vector=pred_data_transf.labels,
			target_label_name=aif_data.label_names[0]
		)
		self.class_metric_transf_obj = {
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
		print('DONE. <<<<<')


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