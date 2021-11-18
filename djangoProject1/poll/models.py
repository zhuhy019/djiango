from django.db import models
import uuid

# Create your models here.
class Question(models.Model):
    id = models.CharField(primary_key=True,max_length=32,null=False)
    docid =models.CharField(max_length=32,null=False)
    question = models.CharField(max_length=512,null=False)
    pred = models.CharField(max_length=1024,null=False)
    content_text = models.TextField()
    createtime=models.DateTimeField(null=False)
    class Meta:
        db_table = 't_b_API_Log'
