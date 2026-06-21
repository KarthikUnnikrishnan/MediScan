from django.db import models

class ScanResult(models.Model):
    SCAN_TYPES = [('strip', 'Medicine Strip'), ('prescription', 'Prescription')]

    image       = models.ImageField(upload_to='scans/')
    scan_type   = models.CharField(max_length=20, choices=SCAN_TYPES)
    result_json = models.JSONField(default=dict)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.scan_type} scan — {self.created_at:%Y-%m-%d %H:%M}"