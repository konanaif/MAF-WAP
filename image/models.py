from django.db import models
import os

# Create your models here.
class Data(models.Model):
	datafile = models.FileField(upload_to='image/data/')
	config_data_json = models.CharField(max_length=1000, null=True)
	config_load_json = models.CharField(max_length=1000, null=True)
	datatype = models.CharField(max_length=10, default='image')

	aif_input_file_path = models.FilePathField(path='image/data/dataframes', null=True)
	
	def __str__(self):
		return os.path.basename(self.datafile.name)