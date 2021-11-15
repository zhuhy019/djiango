from rest_framework import serializers


class QuestionSerializer(serializers.Serializer):
    Question = serializers.CharField(label="Enter Question")
    pred = serializers.CharField()
    text = serializers.CharField()