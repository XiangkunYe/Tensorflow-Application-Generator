#-*- coding: utf-8 -*-
from django import forms
from vision.models import Document

# form for uploading file
class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
