from django.db import models


# Create your models here.
class Question(models.Model):
    id = models.CharField(primary_key= True,max_length=100)
    Question = models.CharField(max_length=200)
    pred = models.TextField(max_length=500)
    text = models.CharField(max_length=500)
