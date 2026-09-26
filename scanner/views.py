"""
scanner/views.py — Django views for MediScan.

Unified architecture:
- index: GET renders the upload interface, live camera modal, features, and sample test buttons.
         POST validates image, executes ML pipeline, stores structured report in session, redirects to /result/.
- result: GET renders the clean, verified medical analysis report in templates/scanner/result.html.
- scan: POST-only AJAX endpoint for live client scan queries.
"""

import os
import time
import uuid
import logging
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .forms import ScanForm
from .ml import scan_image

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cleanup_old_uploads():
    """Delete uploaded files older than 1 hour to respect local privacy."""
    try:
        upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        if not upload_dir.exists():
            return
        now = time.time()
        for p in upload_dir.glob('*'):
            if p.is_file() and not p.name.startswith('.'):
                # Check modification time
                if now - p.stat().st_mtime > 3600:
                    try:
                        p.unlink()
                    except Exception:
                        pass
    except Exception as exc:
        logger.debug("Upload cleanup error: %s", exc)


def _save_upload(image_file):
    """
    Save image_file to MEDIA_ROOT/uploads/ with a UUID-based filename.
    Returns (unique_filename, full_absolute_path).
    """
    _cleanup_old_uploads()

    ext = os.path.splitext(image_file.name)[1].lower() or '.jpg'
    unique_filename = f"{uuid.uuid4()}{ext}"
    relative_path = os.path.join('uploads', unique_filename)

    saved_path = default_storage.save(relative_path, image_file)
    full_path = os.path.join(settings.MEDIA_ROOT, saved_path)

    return unique_filename, full_path


# ── Views ─────────────────────────────────────────────────────────────────────

def index(request):
    """
    GET  — Render the scanner interface, hero, and interactive components.
    POST — Validate image, run ML pipeline, save result in session, redirect to /result/.
    """
    if request.method == 'POST':
        form = ScanForm(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, 'scanner/index.html', {'form': form})

        try:
            # 1. Persist the uploaded file for session processing
            image_file = form.cleaned_data['image']
            unique_filename, full_path = _save_upload(image_file)

            # 2. Determine mode
            mode = form.cleaned_data['mode']

            # 3. Run precision ML pipeline
            result_data = scan_image(full_path, mode=mode)

            # 4. Augment with session metadata
            result_data['image_url'] = settings.MEDIA_URL + 'uploads/' + unique_filename
            result_data['original_filename'] = image_file.name
            result_data['mode_used'] = mode
            result_data['timestamp'] = timezone.now().isoformat()

            request.session['scan_result'] = result_data

            logger.info(
                "Scan complete — status=%s, mode=%s",
                result_data.get('status'),
                mode,
            )

            # Redirect to unified result page
            return redirect('result')

        except Exception as exc:
            logger.error("index POST failed: %s", exc, exc_info=True)
            return render(
                request,
                'scanner/index.html',
                {'form': form, 'error': f"Processing error: {exc}"},
            )

    # GET
    form = ScanForm()
    return render(request, 'scanner/index.html', {'form': form})


def result(request):
    """
    GET — Render the comprehensive clinical analysis report.
    Pulls structured verification data directly from session.
    """
    scan_result = request.session.get('scan_result')
    if not scan_result:
        return redirect('index')

    context = {
        'result': scan_result,
        'status': scan_result.get('status', 'unverified'),
        'success': scan_result.get('success', False),
        'mode': scan_result.get('mode', 'medicine_package'),
        'medicine': scan_result.get('medicine'),
        'medicines': scan_result.get('medicines', []),
        'ingredients': scan_result.get('ingredients', []),
        'alternatives': scan_result.get('alternatives', []),
        'max_savings': scan_result.get('max_savings'),
        'max_savings_percent': scan_result.get('max_savings_percent'),
        'side_effects': scan_result.get('side_effects', []),
        'interactions': scan_result.get('interactions', []),
        'cross_interactions': scan_result.get('cross_interactions', []),
        'confidence_breakdown': scan_result.get('confidence_breakdown', {}),
        'verification': scan_result.get('verification', {}),
        'image_url': scan_result.get('image_url', ''),
        'timestamp': scan_result.get('timestamp', ''),
        'raw_ocr': scan_result.get('raw_ocr', ''),
        'normalized_ocr': scan_result.get('normalized_ocr', ''),
        'rejection_reasons': scan_result.get('rejection_reasons', []),
    }
    return render(request, 'scanner/result.html', context)


@require_http_methods(["POST"])
def scan(request):
    """
    POST-only AJAX endpoint for real-time frontend scanning.
    Returns JSON structured result matching API specifications.
    """
    try:
        form = ScanForm(request.POST, request.FILES)

        if not form.is_valid():
            return JsonResponse(
                {'success': False, 'status': 'unverified', 'errors': form.errors},
                status=400,
            )

        image_file = form.cleaned_data['image']
        unique_filename, full_path = _save_upload(image_file)
        mode = form.cleaned_data['mode']

        result_data = scan_image(full_path, mode=mode)

        result_data['image_url'] = settings.MEDIA_URL + 'uploads/' + unique_filename
        result_data['original_filename'] = image_file.name
        result_data['mode_used'] = mode
        result_data['timestamp'] = timezone.now().isoformat()

        # Also cache in session in case user navigates to /result/
        request.session['scan_result'] = result_data

        return JsonResponse(result_data)

    except Exception as exc:
        logger.error("scan AJAX endpoint failed: %s", exc, exc_info=True)
        return JsonResponse({'success': False, 'status': 'unverified', 'error': str(exc)}, status=500)
