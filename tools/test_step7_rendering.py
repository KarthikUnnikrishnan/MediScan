import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mediscan.settings")
import django
django.setup()

from django.test import Client
from scanner.ml import scan_image

os.makedirs("tools/rendered_html", exist_ok=True)

test_cases = [
    {
        "name": "verified",
        "image": "test_images/LORI0006_L.jpg",
        "mode": "medicine_package",
    },
    {
        "name": "likely",
        "image": "test_images/lucichem-pictures-8-166.webp",
        "mode": "medicine_package",
    },
    {
        "name": "not_found",
        "image": "test_images/blank_test.jpg",
        "mode": "medicine_package",
    },
    {
        "name": "setup_error",
        "image": "test_images/blank_test.jpg",
        "mode": "prescription",
    }
]

client = Client()

for tc in test_cases:
    print(f"\n==========================================")
    print(f"Testing {tc['name']} ({tc['image']})...")
    res = scan_image(tc["image"], mode=tc["mode"])
    print(f"  scan_image result: status={res.get('status')}, result_level={res.get('result_level')}")
    
    # Store in session and render /result/
    session = client.session
    session["scan_result"] = res
    session.save()
    
    response = client.get("/result/")
    html = response.content.decode("utf-8")
    
    out_path = f"tools/rendered_html/{tc['name']}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved rendered HTML to {out_path}")
    
    # Assertions
    if tc["name"] == "verified":
        assert "Verified match" in html, "Expected 'Verified match' badge"
        assert "Matches the brand and the ingredient printed on the pack." in html, "Expected verified text"
        assert "Check this result against your pack before use." in html, "Expected pack warning"
        assert "Results are read by an on-device model and can be wrong." in html, "Expected bottom disclaimer"
        print("  [PASS] ALL ASSERTIONS PASSED FOR VERIFIED")
        
    elif tc["name"] == "likely":
        assert "Likely match - please confirm" in html, "Expected 'Likely match - please confirm' badge"
        assert "Check this result against your pack before use." in html, "Expected pack warning"
        assert "TOP CANDIDATES CONSIDERED" in html, "Expected top candidates panel"
        assert "Results are read by an on-device model and can be wrong." in html, "Expected bottom disclaimer"
        print("  [PASS] ALL ASSERTIONS PASSED FOR LIKELY")
        
    elif tc["name"] == "not_found":
        assert "Could not identify this medicine. Try a clearer photo of the brand name." in html, "Expected not found message"
        assert "TOP CANDIDATES CONSIDERED" not in html, "Must not have candidates/suggestions"
        assert "Results are read by an on-device model and can be wrong." in html, "Expected bottom disclaimer"
        print("  [PASS] ALL ASSERTIONS PASSED FOR NOT_FOUND")
        
    elif tc["name"] == "setup_error":
        assert "Scanner is not ready" in html, "Expected 'Scanner is not ready'"
        assert "could not verify" not in html.lower(), "Must NEVER show 'could not verify'"
        assert res.get("error_reason") in html, "Expected error_reason in HTML"
        assert "Results are read by an on-device model and can be wrong." in html, "Expected bottom disclaimer"
        print("  [PASS] ALL ASSERTIONS PASSED FOR SETUP_ERROR")

# Also test explicit missing package / model error
print(f"\n==========================================")
print("Testing simulated missing model setup error...")
mock_error_res = {
    "success": False,
    "status": "error",
    "result_level": "error",
    "error_type": "setup_error",
    "error_reason": "No module named 'easyocr': offline OCR package missing",
    "message": "Scanner is not ready: No module named 'easyocr'",
    "error": "Scanner is not ready: No module named 'easyocr'",
    "candidates": [],
    "medicines": [],
}
session = client.session
session["scan_result"] = mock_error_res
session.save()
resp = client.get("/result/")
mock_html = resp.content.decode("utf-8")
assert "Scanner is not ready" in mock_html
assert "could not verify" not in mock_html.lower()
assert "No module named" in mock_html and "easyocr" in mock_html
assert "Results are read by an on-device model and can be wrong." in mock_html
with open("tools/rendered_html/missing_package_error.html", "w", encoding="utf-8") as f:
    f.write(mock_html)
print("  [PASS] ALL ASSERTIONS PASSED FOR SIMULATED SETUP ERROR")

print("\nALL STEP 7 TEST CASES PASSED SUCCESSFULLY!")
