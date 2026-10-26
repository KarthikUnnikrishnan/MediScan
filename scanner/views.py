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
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .forms import ScanForm
from .ml import scan_image
from .ml.medicine_matcher import decide_result

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
            mode = form.cleaned_data.get('mode', 'auto')

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
            result_data = {
                'success': False,
                'status': 'error',
                'result_level': 'error',
                'error_type': 'setup_error',
                'error_reason': str(exc),
                'message': f"Scanner is not ready: {exc}",
                'error': f"Scanner is not ready: {exc}",
                'timestamp': timezone.now().isoformat(),
            }
            request.session['scan_result'] = result_data
            return redirect('result')

    # GET
    form = ScanForm()
    return render(request, 'scanner/index.html', {'form': form})


def result(request):
    """
    GET — Render the clinical analysis report.
    Supports two result levels ("verified" and "likely"), plus "not_found" and "error".
    Pulls structured verification data directly from session.
    """
    scan_result = request.session.get('scan_result')
    if not scan_result:
        return redirect('index')

    medicines = scan_result.get('medicines', [])
    candidates = scan_result.get('candidates', [])
    per_meds = scan_result.get('per_medicine_results', {})

    # 1. Determine result_level: 'verified', 'likely', 'not_found', 'error'
    status = scan_result.get('status', '')
    error_reason = scan_result.get('error_reason') or (scan_result.get('error') if status == 'error' else '')

    decision_candidates = candidates or medicines or ([scan_result.get('medicine')] if scan_result.get('medicine') else [])
    if decision_candidates:
        view_decision = decide_result({"raw_ocr": scan_result.get('raw_ocr')}, decision_candidates)
        scan_result['medicine'] = view_decision.get('medicine') or scan_result.get('medicine')
        scan_result['result_level'] = view_decision['level']
        scan_result['message'] = view_decision['message']
        if view_decision['level'] != 'verified':
            scan_result['alternatives'] = view_decision.get('options', [])
        if scan_result.get('medicine'):
            scan_result['medicines'] = [scan_result['medicine']]
            medicines = scan_result['medicines']
    result_level = scan_result.get('result_level')
    if not result_level:
        if status == 'error' or error_reason or scan_result.get('error_type') == 'setup_error':
            result_level = 'error'
        else:
            result_level = 'not_found'

    if result_level == 'error' and not error_reason:
        error_reason = scan_result.get('message') or scan_result.get('error') or "Scanner is not ready"

    # Candidates list (top 3)
    top_candidates = []
    if candidates:
        top_candidates = [c for c in candidates if isinstance(c, dict)][:3]
    elif medicines:
        top_candidates = [m for m in medicines if isinstance(m, dict)][:3]

    top_cand = scan_result.get('medicine') or (candidates[0] if candidates else (medicines[0] if medicines else None))
    match_type = scan_result.get('match_type') or (top_cand.get('match_type') if isinstance(top_cand, dict) else 'brand')
    is_ingredient_match = (match_type == 'ingredient')

    if (not medicines or not medicines[0]) and top_cand:
        medicines = [top_cand]

    default_conf = scan_result.get('confidence_breakdown', {}).get('overall', 0.0)
    if 'confidence_breakdown' not in scan_result or not scan_result['confidence_breakdown']:
        scan_result['confidence_breakdown'] = {'overall': 0.0}
    for m in medicines:
        if isinstance(m, dict) and 'confidence' not in m:
            m['confidence'] = default_conf

    for k, v in per_meds.items():
        if isinstance(v, dict):
            if 'confidence' not in v or not v['confidence']:
                top_m = v.get('top_medicine')
                if top_m:
                    v['confidence'] = top_m.get('confidence', 0.0)
                else:
                    v['confidence'] = 0.0

    context = {
        'result':                scan_result,
        'result_level':          result_level,
        'error_reason':          error_reason,
        'is_ingredient_match':   is_ingredient_match,
        'top_candidates':        top_candidates,
        'mode':                  scan_result.get('mode', 'auto'),
        'success':               result_level in ('verified', 'likely', 'composition_match'),
        'medicines':             medicines,
        'alternatives':          scan_result.get('alternatives', []),
        'cross_interactions':    scan_result.get('cross_interactions', []),
        'ocr_text':              scan_result.get('ocr_text', '') or scan_result.get('raw_ocr', ''),
        'raw_ocr':               scan_result.get('raw_ocr', '') or scan_result.get('ocr_text', ''),
        'extracted_names':       scan_result.get('extracted_names', []),
        'image_url':             scan_result.get('image_url', ''),
        'strip_detected':        scan_result.get('strip_detected', False),
        'timestamp':             timezone.localtime(datetime.fromisoformat(scan_result['timestamp'])) if scan_result.get('timestamp') else '',
        'diagnosis':             scan_result.get('diagnosis', ''),
        'detected_name':         scan_result.get('detected_name') or (medicines[0].get('detected_name') if medicines and isinstance(medicines[0], dict) else ''),
        'per_medicine_results':  per_meds,
        'default_selected':      scan_result.get('default_selected', ''),
        'total_found':           scan_result.get('total_found', 0),
        'ai_validation':         scan_result.get('ai_validation', {}),
        'ai_suggested_correction': scan_result.get('ai_suggested_correction', ''),
        'other_text_read':       scan_result.get('other_text_read', []),
        'disclaimer':            scan_result.get('disclaimer', "Handwriting can be misread - confirm every medicine with your doctor or pharmacist"),
    }

    import json as _json
    context['per_medicine_json'] = _json.dumps(
        scan_result.get('per_medicine_results', {}),
        default=str
    )

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
        mode = form.cleaned_data.get('mode', 'auto')

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
        return JsonResponse({
            'success': False,
            'status': 'error',
            'result_level': 'error',
            'error_reason': str(exc),
            'message': f"Scanner is not ready: {exc}",
            'error': f"Scanner is not ready: {exc}",
        }, status=500)
