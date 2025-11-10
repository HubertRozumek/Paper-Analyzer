from django.db import models
from django.conf import settings
import uuid

class Conversation(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    paper = models.ForeignKey('papers.Paper', on_delete=models.CASCADE, related_name='conversations')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations')

    title = models.CharField(max_length=256, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-updated_at']),
            models.Index(fields=['paper', '-updated_at']),
        ]

    def __str__(self):
        return f'{self.user.username} - {self.paper.title[:50]}'


class Message(models.Model):

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')

    role = models.CharField(choices=ROLE_CHOICES, max_length=16)
    content = models.TextField()

    # For assistant messages - sources used
    cited_chunks = models.JSONField(default=list)
    confidence_score = models.FloatField(blank=True, null=True)

    # Token usage tracking
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)

    # Feedback
    helpful = models.BooleanField(null=True, blank=True)
    feedback_text = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
        ]

    def __str__(self):
        return f'{self.role}: {self.conversation.title[:50]}'

class ChatExport(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='chat_export')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    format = models.CharField(
        max_length=16,
        choices=[
            ('markdown', 'Markdown'),
            ('pdf', 'PDF'),
            ('json', 'JSON'),
        ]
    )

    file = models.FileField(upload_to='chat_export/%Y/%m/%d')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Export: {self.conversation.title} - {self.format}'
