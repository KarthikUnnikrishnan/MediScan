import os, json
from django.shortcuts       import render
from django.http            import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage   import default_storage
from .models import ScanResult

# ML modules are imported lazily inside scan() to avoid import errors
# when a module (e.g. prescription_htr) is not yet present.


def home(request):
    """Upload page — shown when user opens the app."""
    return render(request, 'home.html')


@csrf_exempt
def scan(request):
    """
    POST endpoint — receives uploaded image, runs full pipeline.
    Returns JSON with medicine name, generics, side effects, interactions.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    # ── Get uploaded image ─────────────────────────────────────────────
    image_file = request.FILES.get('image')
    scan_type  = request.POST.get('scan_type', 'strip')   # 'strip' or 'prescription'

    if not image_file:
        return JsonResponse({'success': False, 'error': 'No image uploaded.'})

    # Save to media/scans/
    saved_path = default_storage.save(f'scans/{image_file.name}', image_file)
    full_path  = default_storage.path(saved_path)

    try:
        # ── Lazy ML imports ────────────────────────────────────────────
        from ml.strip_ocr      import predict as strip_predict
        from ml.generic_finder import find_generics
        from ml.drug_info      import get_side_effects, check_interactions

        # ── Step 1: OCR — read medicine name ──────────────────────────
        if scan_type == 'strip':
            ocr_result = strip_predict(full_path)
            medicine_names = (
                [ocr_result['medicine_name']]
                if ocr_result.get('success') else []
            )
            ocr_confidence = ocr_result.get('confidence', 0)
            ocr_all_text   = ocr_result.get('all_text', [])
        else:
            try:
                from ml.prescription_htr import predict as htr_predict
            except ModuleNotFoundError:
                return JsonResponse({
                    'success': False,
                    'error': 'Prescription HTR model is not available yet.',
                }, status=503)
            ocr_result     = htr_predict(full_path)
            medicine_names = ocr_result.get('medicines_found', [])
            ocr_confidence = None
            ocr_all_text   = ocr_result.get('all_text', [])

        if not medicine_names:
            return JsonResponse({
                'success'   : False,
                'scan_type' : scan_type,
                'error'     : ocr_result.get('error', 'Could not read medicine name.'),
                'all_text'  : ocr_all_text,
            })

        # ── Step 2: For each medicine — generics + side effects ────────
        medicines_data = []
        for name in medicine_names[:3]:   # max 3 medicines per scan
            generic_result = find_generics(name)
            se_result      = get_side_effects(name)

            medicines_data.append({
                'name'        : name,
                'matched_name': generic_result.get('matched_name', name),
                'salt'        : generic_result.get('salt', ''),
                'alternatives': generic_result.get('alternatives', []),
                'side_effects': se_result.get('side_effects', []),
            })

        # ── Step 3: Drug interactions (if 2+ medicines) ───────────────
        interactions = []
        if len(medicine_names) >= 2:
            for i in range(len(medicine_names)):
                for j in range(i + 1, min(len(medicine_names), 3)):
                    ddi = check_interactions(
                        medicine_names[i], medicine_names[j]
                    )
                    if ddi.get('interacts'):
                        interactions.append(ddi)

        # ── Save result to DB ──────────────────────────────────────────
        result_data = {
            'success'       : True,
            'scan_type'     : scan_type,
            'medicines'     : medicines_data,
            'interactions'  : interactions,
            'ocr_confidence': ocr_confidence,
            'all_text'      : ocr_all_text[:20],
        }
        ScanResult.objects.create(
            image       = saved_path,
            scan_type   = scan_type,
            result_json = result_data,
        )

        return JsonResponse(result_data)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error'  : f'Server error: {str(e)}'
        }, status=500)


def result(request, pk):
    """Show a saved scan result by ID."""
    try:
        scan_obj = ScanResult.objects.get(pk=pk)
        return render(request, 'result.html', {
            'result': scan_obj.result_json,
            'scan'  : scan_obj,
        })
    except ScanResult.DoesNotExist:
        return render(request, 'result.html', {'error': 'Result not found.'})