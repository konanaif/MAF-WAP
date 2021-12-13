from django import forms


class DataForm(forms.Form):
	file = forms.FileField(label='파일을 선택하세요.')