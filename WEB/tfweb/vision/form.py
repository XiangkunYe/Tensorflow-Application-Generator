#-*- coding: utf-8 -*-
from django import forms
from vision.models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document', )