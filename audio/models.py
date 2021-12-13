from django.db import models
import os

# Create your models here.
class Data(models.Model):
	datafile = models.FileField(upload_to='audio/data/')
	config_data_json = models.CharField(max_length=1000, null=True)
	config_load_json = models.CharField(max_length=1000, null=True)
	datatype = models.CharField(max_length=10, default='audio')

	aif_input_file_path = models.FilePathField(path='audio/data/dataframes', null=True)

	def __str__(self):
		return os.path.basename(self.datafile.name)