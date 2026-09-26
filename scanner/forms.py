"""
scanner/forms.py — Upload form for MediScan.
"""

import os

from django import forms


class ScanForm(forms.Form):
    """Form for uploading a medicine image and selecting scan mode."""

    MODE_CHOICES = [
        ('auto',             'Auto Detect'),
        ('medicine_package', 'Medicine Box / Strip'),
        ('prescription',     "Doctor's Prescription"),
    ]

    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    image = forms.ImageField(
        label='Upload Image',
        help_text=(
            'Upload a photo of a medicine box, strip, '
            'or doctor\'s prescription'
        ),
        required=True,
    )

    mode = forms.ChoiceField(
        choices=MODE_CHOICES,
        initial='auto',
        required=True,
    )

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image is None:
            return image

        # ── Size check ────────────────────────────────────────────────
        if image.size > self.MAX_FILE_SIZE:
            raise forms.ValidationError(
                f'File too large. Maximum allowed size is 10 MB '
                f'(uploaded: {image.size / (1024 * 1024):.1f} MB).'
            )

        # ── Extension check ───────────────────────────────────────────
        ext = os.path.splitext(image.name)[1].lstrip('.').lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f'Unsupported file type ".{ext}". '
                f'Allowed types: {", ".join(sorted(self.ALLOWED_EXTENSIONS))}.'
            )

        return image
