from django.db import models
from django.conf import settings
import uuid

from app.papers.models import Paper


class ProcessingTask(models.Model):

    TASK_TYPES = [
        ('pdf_extraction', 'PDF Text Extraction'),
        ('summarization', 'Summarization'),
        ('embedding', 'Creating Embeddings'),
        ('key_insight', 'Extract Key Insight'),
        ('related_papers', 'Find Related Papers'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('complete', 'Complete'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_type = models.CharField(choices=TASK_TYPES, max_length=64)
    status = models.CharField(choices=STATUS_CHOICES, max_length=64, default='pending')

    paper = models.ForeignKey('papers.Paper', on_delete=models.CASCADE, related_name='tasks')
    celery_task_id = models.CharField(max_length=64, blank=True)

    progress_percentage = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)

    result = models.JSONField(null=True, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['paper', 'task_type','status']),
        ]
    def __str__(self):
        return f'{self.task_type} - {self.status}'


class ModelsUsageStats(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    operation_type = models.CharField(max_length=64)
    model_name = models.CharField(max_length=64)

    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)

    estimated_cost = models.DecimalField(decimal_places=6, max_digits=16,default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]