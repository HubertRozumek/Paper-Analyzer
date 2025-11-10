from django.db import models
from django.conf import  settings
import uuid

class Paper(models.Model):

    STATUS_CHOICES = [
        ('uploading', 'Uploading'),
        ('processing', 'Processing'),
        ('summarizing', 'Summarizing'),
        ('embedding', 'Creating Embeddings'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="papers")


    # Paper metadata
    title = models.CharField(max_length=512)
    arxiv_id = models.CharField(max_length=64,blank=True,null=True,db_index=True)
    authors = models.JSONField(default=list)
    publication_date = models.DateField(blank=True, null=True)
    categories = models.JSONField(default=list)

    # File
    pdf_file = models.FileField(upload_to="pdfs/%Y/%m/")
    file_size = models.IntegerField(help_text="File size in bytes")
    num_pages = models.IntegerField(default=0)

    # Extracted content

    full_text = models.TextField(blank=True)
    full_text_length = models.IntegerField(default=0)

    # Summaries
    short_summary = models.TextField(blank=True, help_text="Short summary of the paper")
    medium_summary = models.TextField(blank=True, help_text="Medium summary of the paper")
    long_summary = models.TextField(blank=True, help_text="Long summary of the paper")

    # key insights
    key_findings = models.JSONField(default=list)
    methodology = models.TextField(blank=True)
    conclusion = models.TextField(blank=True)

    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploading')
    processing_error = models.TextField(blank=True)

    # Vector DB info
    collection_name = models.CharField(max_length=512, blank=True)
    num_chunks = models.IntegerField(default=0)

    # Metadata
    view_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['user', '-updated_at']),
        ]

    def __str__(self):
        return self.title


class PaperChunk(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name="chunks")

    content = models.TextField()
    chunk_index = models.IntegerField()

    # Position in document
    page_number = models.IntegerField(null=True)
    section_title = models.CharField(max_length=512, blank=True)

    # Vector DB reference
    embedding_id = models.CharField(max_length=128)

    # Metadata for better retrievel
    chunk_type = models.CharField(
        max_length=64,
        choices=[
            ('abstract', 'Abstract'),
            ('introduction', 'Introduction'),
            ('methodology', 'Methodology'),
            ('results', 'Results'),
            ('discussion', 'Discussion'),
            ('conclusion', 'Conclusion'),
            ('references', 'References'),
            ('other', 'Other'),
        ],
        default='other'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['paper', 'chunk_index']
        indexes = [
            models.Index(fields=['paper', 'chunk_index']),
        ]

    def __str__(self):
        return f'{self.paper.title} Chunk {self.chunk_index}'

class PaperTag(models.Model):

    name = models.CharField(max_length=128,unique=True, db_index=True)
    slug = models.SlugField(max_length=128,unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class PaperTagging(models.Model):

    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    tag = models.ForeignKey(PaperTag, on_delete=models.CASCADE)
    auto_generated = models.BooleanField(default=False)
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['paper', 'tag']

class RelatedPaper(models.Model):

    paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name='related_from')
    related_paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name='related_to')
    similarity_score = models.FloatField()
    relationship_type = models.CharField(
        max_length=64,
        choices=[
            ('citation', 'Citation'),
            ('semantic', 'Semantic Similarity'),
            ('topic', 'Same topic'),
            ('author', 'Same Author'),
        ]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['paper', 'related_paper']
        indexes = [
            models.Index(fields=['paper', '-similarity_score']),
        ]