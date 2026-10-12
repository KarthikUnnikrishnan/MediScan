"""
scanner/tests/test_gemini_ocr_integration.py

Comprehensive regression test suite for MediScan Gemini Vision OCR integration:
- Test A: Correct TrOCR output (unnecessary Gemini calls avoided)
- Test B: Empty TrOCR output (Gemini recovery invoked)
- Test C: Incorrect non-empty TrOCR output (rebell-500 triggers recovery)
- Test D: Gemini returns corrected text (pipeline extracts & verifies candidate)
- Test E: Gemini API failure (safe error handling, no crash, no fabricated OCR)
- Test F: Conflicting results (uncertain result returned when evidence contradicts)
- Test G: Wrong composition (conflicting strength/ingredients not falsely verified)
- Test H: Missing API key (clear diagnostic without exposing secrets)
- Test I: Prescription workflow (prescription path intact with recovery support)
- Test J: No fabricated confidence (real calculated scores, no fake 100%)

All tests use mocks by default so NO API quota or real network access is required.
An optional live integration test is provided when RUN_LIVE_GEMINI_TESTS=1.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from django.test import TestCase
from django.utils import timezone

from scanner.ml.gemini_vision import (
    gemini_vision_ocr,
    gemini_vision_ocr_structured,
    strip_preamble,
    get_gemini_model,
    get_gemini_client,
    NO_TEXT_DETECTED,
)
from scanner.ml import scan_medicine_package, scan_prescription
from scanner.ml.extractor import extract_structured_ocr
from scanner.ml.medicine_matcher import match_medicine
from scanner.ml.alternative_matcher import find_top_cheaper_alternatives


class GeminiOCRIntegrationRegressionTests(TestCase):
    """Regression test suite for Gemini Vision OCR and fallback pipeline."""

    def setUp(self):
        # Create a small dummy image for testing
        self.dummy_image = Image.new("RGB", (200, 200), color="white")

    # ── Test A: Correct TrOCR output ──────────────────────────────────────────
    def test_brand_only_without_strength_or_composition_is_unverified(self):
        """
        A brand-only database candidate must not become verified when OCR lacks
        ingredient, strength, and manufacturer corroboration.
        """
        extracted = extract_structured_ocr("Dolo Tablet")
        result = match_medicine(extracted)
        self.assertEqual(result.get("status"), "unverified")
        self.assertIsNone(result.get("medicine"))

    def test_standalone_ocr_digits_are_not_treated_as_strength(self):
        """A trailing OCR artifact must not reject a candidate as wrong strength."""
        extracted = extract_structured_ocr("ranitidine tablets usp 39")
        result = match_medicine(extracted)
        self.assertEqual(result.get("status"), "unverified")
        self.assertIsNone(result.get("medicine"))
        self.assertNotIn(
            "Visible strength/number 39 does not match candidate name.",
            result.get("rejection_reasons", []),
        )

    def test_alternatives_require_complete_source_composition(self):
        """Incomplete source salt data must not produce keyword alternatives."""
        alternatives = find_top_cheaper_alternatives({
            "id": 999999,
            "name": "Example 500mg Tablet",
            "salt": "(500mg)",
            "parsed_salt": [],
            "price": 100.0,
            "dosage_form": "Tablet",
        })
        self.assertEqual(alternatives, [])

    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_a_correct_trocr_output_avoids_gemini(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        TrOCR produces credible text and a valid database candidate.
        Confirm that unnecessary Gemini calls are avoided.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = (
            "DOLO 650 TABLET\nParacetamol (650mg)\nMicro Labs Ltd"
        )

        result = scan_medicine_package(self.dummy_image)

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "verified")
        self.assertIn("Dolo", result["medicine"]["name"])
        # Unnecessary Gemini call MUST be avoided
        mock_gemini.assert_not_called()
        self.assertFalse(result.get("recovery_used"))

    # ── Test B: Empty TrOCR output ────────────────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_b_empty_trocr_output_invokes_gemini(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """Confirm that Gemini recovery is invoked when TrOCR returns empty text."""
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = ""

        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "BRAND: 1-AL\n"
                "GENERIC: Levocetirizine Tablets IP\n"
                "INGREDIENTS: Levocetirizine Dihydrochloride (5mg)\n"
                "STRENGTH: 5mg\n"
                "DOSAGE_FORM: Tablet\n"
                "MANUFACTURER: FDC Limited\n"
            ),
            "plain_text": "1-AL\nLevocetirizine Tablets IP\nLevocetirizine Dihydrochloride (5mg)\n5mg\nTablet\nFDC Limited",
            "structured": {
                "brand": "1-AL",
                "generic": "Levocetirizine Tablets IP",
                "ingredients": "Levocetirizine Dihydrochloride (5mg)",
                "strength": "5mg",
                "dosage_form": "Tablet",
                "manufacturer": "FDC Limited",
                "schedule": None,
            },
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        result = scan_medicine_package(self.dummy_image)

        mock_gemini.assert_called_once()
        self.assertTrue(result.get("recovery_used"))
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "verified")
        self.assertIn("1-AL", result["medicine"]["name"])

    # ── Test C: Incorrect non-empty TrOCR output ───────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_c_incorrect_trocr_triggers_recovery(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        Simulate TrOCR returning 'rebell-500' on a strip that visibly contains 1-AL.
        Confirm that database inconsistency or quality heuristics trigger recovery.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = "rebell-500"

        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "BRAND: 1-AL\n"
                "GENERIC: Levocetirizine Tablets IP\n"
                "INGREDIENTS: Levocetirizine (5mg)\n"
                "STRENGTH: 5mg\n"
                "DOSAGE_FORM: Tablet\n"
                "MANUFACTURER: FDC Limited\n"
            ),
            "plain_text": "1-AL\nLevocetirizine Tablets IP\nLevocetirizine (5mg)\n5mg\nTablet\nFDC Limited",
            "structured": {
                "brand": "1-AL",
                "ingredients": "Levocetirizine (5mg)",
                "strength": "5mg",
                "dosage_form": "Tablet",
                "manufacturer": "FDC Limited",
            },
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        result = scan_medicine_package(self.dummy_image)

        # Gemini must be called because rebell-500 fails verification
        mock_gemini.assert_called_once()
        self.assertTrue(result.get("recovery_used"))
        self.assertEqual(result.get("raw_ocr_trocr"), "rebell-500")
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "verified")
        self.assertIn("1-AL", result["medicine"]["name"])

    # ── Test D: Gemini returns corrected text ─────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_d_gemini_recovered_transcription_pipeline(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        Confirm that the recovered transcription continues into the existing
        extraction and database-matching pipeline.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = ""

        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "BRAND: Crocin Advance\n"
                "INGREDIENTS: Paracetamol (500mg)\n"
                "STRENGTH: 500mg\n"
                "DOSAGE_FORM: Tablet\n"
                "MANUFACTURER: GlaxoSmithKline\n"
            ),
            "plain_text": "Crocin Advance\nParacetamol (500mg)\n500mg\nTablet\nGlaxoSmithKline",
            "structured": {
                "brand": "Crocin Advance",
                "ingredients": "Paracetamol (500mg)",
                "strength": "500mg",
                "dosage_form": "Tablet",
                "manufacturer": "GlaxoSmithKline",
            },
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        result = scan_medicine_package(self.dummy_image)

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "verified")
        self.assertIn("Crocin", result["medicine"]["name"])
        # Alternatives and safety lookup must be populated
        self.assertGreater(len(result.get("alternatives", [])), 0)
        self.assertIn("raw_ocr", result)
        self.assertIn("Paracetamol", result["raw_ocr"])

    # ── Test E: Gemini API failure ────────────────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_e_gemini_api_failure_handled_safely(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        Confirm that exceptions, quota failures, and network errors do not
        crash the entire web application or get converted into fabricated OCR output.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = ""

        mock_gemini.return_value = {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": {},
            "error": "ResourceExhausted: 429 Quota exceeded",
            "error_type": "API_ERROR",
            "is_empty": True,
            "model": "gemini-2.5-flash",
        }

        # Should never raise an unhandled exception
        result = scan_medicine_package(self.dummy_image)

        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("status"), "ocr_service_error")
        self.assertEqual(result.get("ocr_error"), "API_ERROR")
        # Must not fabricate OCR text out of the error
        self.assertEqual(result.get("raw_ocr"), "")
        self.assertEqual(len(result.get("medicines", [])), 0)

    # ── Test F: Conflicting results ───────────────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_f_conflicting_results_return_uncertain(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        If TrOCR and Gemini disagree and neither result can be definitively resolved,
        confirm that the application returns an uncertain result.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        # TrOCR produces one candidate
        mock_run_ocr.return_value = "DOLO 650 TABLET\nParacetamol (650mg)\nMicro Labs Ltd"

        # Force recovery with conflicting candidate
        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "BRAND: Augmentin 625 Duo\n"
                "INGREDIENTS: Amoxycillin (500mg) + Clavulanic Acid (125mg)\n"
                "DOSAGE_FORM: Tablet\n"
                "MANUFACTURER: GlaxoSmithKline\n"
            ),
            "plain_text": "Augmentin 625 Duo\nAmoxycillin (500mg) + Clavulanic Acid (125mg)\nTablet\nGlaxoSmithKline",
            "structured": {
                "brand": "Augmentin",
                "ingredients": "Amoxycillin (500mg) + Clavulanic Acid (125mg)",
                "dosage_form": "Tablet",
            },
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        # Simulate need_recovery being triggered by low confidence or disagreement test
        with patch("scanner.ml._check_trocr_credibility", return_value=(False, "Marginal OCR quality")):
            result = scan_medicine_package(self.dummy_image)
            # Both returned verified candidates for different drugs
            # With contradictory evidence, status must be unverified
            self.assertEqual(result.get("status"), "unverified")
            self.assertFalse(result.get("success"))
            self.assertTrue(result.get("disagreement_detected"))

    # ── Test G: Wrong composition ─────────────────────────────────────────────
    def test_g_wrong_composition_not_falsely_verified(self):
        """
        Ensure that a candidate with a conflicting ingredient or strength is
        not falsely verified.
        """
        # Single-ingredient Diclofenac must NOT match combination Magadol
        extracted_single = extract_structured_ocr("Diclofenac Sodium (50mg) Tablet")
        match_single = match_medicine(extracted_single)

        if match_single.get("status") == "verified" and match_single.get("medicine"):
            self.assertNotIn(
                "magadol",
                match_single["medicine"]["name"].lower(),
                "Single-ingredient Diclofenac must not falsely match combination drug Magadol"
            )

        # 500mg Paracetamol must NOT match Dolo 650
        extracted_500 = extract_structured_ocr("Generic Paracetamol (500mg) Tablet")
        match_500 = match_medicine(extracted_500)
        if match_500.get("status") == "verified" and match_500.get("medicine"):
            self.assertNotIn("650", match_500["medicine"]["name"])

    # ── Test H: Missing API key ───────────────────────────────────────────────
    def test_h_missing_api_key_diagnostics(self):
        """Ensure that configuration errors are clearly identified without exposing secrets."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with patch("django.conf.settings.GEMINI_API_KEY", ""):
                result = gemini_vision_ocr_structured(self.dummy_image, api_key="")
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "CONFIG_ERROR")
                self.assertIn("GEMINI_API_KEY", result["error"])
                # Key must not be present in output
                self.assertEqual(result["raw_text"], "")

    # ── Test I: Prescription workflow ─────────────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.run_ocr")
    def test_i_prescription_workflow_intact(self, mock_run_ocr, mock_gemini):
        """Confirm that the changes do not unintentionally break the prescription path."""
        # 1. TrOCR returns clean prescription line
        mock_run_ocr.return_value = "Tab. Dolo 650mg 1-0-1\nTab. Crocin 500mg"
        res_trocr = scan_prescription(self.dummy_image)
        self.assertTrue(res_trocr.get("success"))
        self.assertGreater(len(res_trocr.get("medicines", [])), 0)
        self.assertFalse(res_trocr.get("recovery_used"))

        # 2. TrOCR empty -> Gemini recovery invoked
        mock_run_ocr.return_value = ""
        mock_gemini.return_value = {
            "success": True,
            "raw_text": "Tab. Dolo 650 1-0-1",
            "plain_text": "Tab. Dolo 650 1-0-1",
            "structured": {},
            "error": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }
        res_gemini = scan_prescription(self.dummy_image)
        self.assertTrue(res_gemini.get("success"))
        self.assertTrue(res_gemini.get("recovery_used"))

    # ── Test J: No fabricated confidence ──────────────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_j_no_fabricated_confidence(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        Ensure that no hardcoded confidence values or fabricated verification
        outcomes are introduced.
        """
        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = ""
        mock_gemini.return_value = {
            "success": True,
            "raw_text": NO_TEXT_DETECTED,
            "plain_text": "",
            "structured": {},
            "error": None,
            "is_empty": True,
            "model": "gemini-2.5-flash",
        }

        result = scan_medicine_package(self.dummy_image)

        self.assertEqual(result.get("status"), "unverified")
        self.assertFalse(result.get("success"))
        conf = result.get("confidence_breakdown", {})
        # Unverified output should NOT have a fake high score (e.g. 100% or 94%)
        self.assertEqual(conf.get("overall", 0.0), 0.0)

    # ── Model configuration check ─────────────────────────────────────────────
    def test_model_name_configurable(self):
        """Test that the Gemini model name is configurable via environment."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-gemini-test"}):
            self.assertEqual(get_gemini_model(), "custom-gemini-test")

    def test_404_falls_back_to_next_configured_model(self):
        class HttpError(Exception):
            status_code = 404

        client = MagicMock()
        client.models.generate_content.side_effect = [HttpError("NOT_FOUND"), MagicMock(text="BRAND: Test")]
        with patch("scanner.ml.gemini_vision.get_gemini_client", return_value=client), \
             patch("scanner.ml.gemini_vision.get_gemini_model", return_value="primary"), \
             patch("django.conf.settings.GEMINI_FALLBACK_MODELS", "fallback"), \
             patch("google.genai.types.Part.from_bytes", return_value=MagicMock()):
            result = gemini_vision_ocr_structured(self.dummy_image)
        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "fallback")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_404_is_not_retried(self):
        class HttpError(Exception):
            status_code = 404

        client = MagicMock()
        client.models.generate_content.side_effect = HttpError("NOT_FOUND")
        with patch("scanner.ml.gemini_vision.get_gemini_client", return_value=client), \
             patch("scanner.ml.gemini_vision.get_gemini_model", return_value="primary"), \
             patch("django.conf.settings.GEMINI_FALLBACK_MODELS", ""), \
             patch("google.genai.types.Part.from_bytes", return_value=MagicMock()):
            result = gemini_vision_ocr_structured(self.dummy_image)
        self.assertFalse(result["success"])
        self.assertEqual(client.models.generate_content.call_count, 1)

    # ── Test K: Full Django HTTP Scan View Path ───────────────────────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.crop_strip")
    @patch("scanner.ml.run_ocr")
    def test_k_django_views_end_to_end(
        self, mock_run_ocr, mock_crop_strip, mock_gemini
    ):
        """
        Verify the full Django view execution path from uploaded image to rendered result.
        """
        from django.test import Client
        from django.core.files.uploadedfile import SimpleUploadedFile
        import io

        mock_crop_strip.return_value = (self.dummy_image, True)
        mock_run_ocr.return_value = "rebell-500"

        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "BRAND: 1-AL\n"
                "GENERIC: Levocetirizine Tablets IP\n"
                "INGREDIENTS: Levocetirizine (5mg)\n"
                "STRENGTH: 5mg\n"
                "DOSAGE_FORM: Tablet\n"
                "MANUFACTURER: FDC Limited\n"
            ),
            "plain_text": "1-AL\nLevocetirizine Tablets IP\nLevocetirizine (5mg)\n5mg\nTablet\nFDC Limited",
            "structured": {
                "brand": "1-AL",
                "ingredients": "Levocetirizine (5mg)",
                "strength": "5mg",
                "dosage_form": "Tablet",
                "manufacturer": "FDC Limited",
            },
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        client = Client()

        # Create dummy image file for upload
        img_byte_arr = io.BytesIO()
        self.dummy_image.save(img_byte_arr, format='JPEG')
        uploaded = SimpleUploadedFile(
            "test_strip.jpg",
            img_byte_arr.getvalue(),
            content_type="image/jpeg"
        )

        # 1. Test POST to index view
        resp_post = client.post('/', {'image': uploaded, 'mode': 'medicine_package'}, follow=True)
        self.assertEqual(resp_post.status_code, 200)

        # Verify session has scan_result with recovery
        session_data = client.session.get('scan_result')
        self.assertIsNotNone(session_data)
        self.assertTrue(session_data.get('success'))
        self.assertEqual(session_data.get('status'), 'verified')
        self.assertTrue(session_data.get('recovery_used'))
        self.assertEqual(session_data.get('raw_ocr_trocr'), 'rebell-500')
        self.assertIn('1-AL', session_data['medicine']['name'])

        # 2. Test GET to result view
        resp_result = client.get('/result/')
        self.assertEqual(resp_result.status_code, 200)
        self.assertContains(resp_result, '1-AL')

    # ── Test N: Prescription multi-medicine detection & switching ─────────────
    @patch("scanner.ml.gemini_vision_ocr_structured")
    @patch("scanner.ml.run_ocr")
    def test_n_prescription_multi_medicine_switching(self, mock_run_ocr, mock_gemini):
        """
        Ensure multiple prescribed medicines are detected, populate per_medicine_results,
        and allow frontend switching with corresponding details and alternatives.
        """
        mock_run_ocr.return_value = ""
        mock_gemini.return_value = {
            "success": True,
            "raw_text": (
                "Dr. John Brown, M.D.\n"
                "Patient: Mary Smith\n"
                "Rx\n"
                "1. Lipitor 10 mg Tabs No. 30\n"
                "2. Metformin 500 mg Tabs No. 60\n"
                "Sig: tab i every day\n"
                "Refill 3 times\n"
            ),
            "plain_text": (
                "Lipitor 10 mg\n"
                "Metformin 500 mg\n"
                "Dr. John Brown, M.D.\n"
                "Patient: Mary Smith\n"
                "Rx\n"
                "1. Lipitor 10 mg Tabs No. 30\n"
                "2. Metformin 500 mg Tabs No. 60\n"
                "Sig: tab i every day\n"
                "Refill 3 times\n"
            ),
            "structured": {
                "brand": "Lipitor",
                "strength": "10 mg"
            },
            "error": None,
            "is_empty": False,
            "model": "gemini-2.5-flash",
        }

        res = scan_prescription(self.dummy_image)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("status"), "verified")
        self.assertEqual(res.get("mode"), "prescription")

        # Must detect both medicines
        self.assertGreaterEqual(res.get("total_found", 0), 2)
        meds = res.get("medicines", [])
        self.assertGreaterEqual(len(meds), 2)

        # per_medicine_results must have both entries
        per_results = res.get("per_medicine_results", {})
        self.assertGreaterEqual(len(per_results), 2)

        # First medicine details and alternatives
        lipitor_key = res.get("default_selected")
        self.assertIn("Lipitor", lipitor_key)
        self.assertIn(lipitor_key, per_results)
        lipitor_entry = per_results[lipitor_key]
        self.assertIsNotNone(lipitor_entry.get("top_medicine"))
        self.assertIn("atorvastatin", lipitor_entry["top_medicine"].get("salt", "").lower())
        self.assertGreater(len(lipitor_entry.get("alternatives", [])), 0)

        # Second medicine exists and has its own alternatives
        other_keys = [k for k in per_results.keys() if k != lipitor_key]
        self.assertGreater(len(other_keys), 0)
        second_entry = per_results[other_keys[0]]
        self.assertIsNotNone(second_entry.get("top_medicine"))
        self.assertIn("metformin", second_entry["top_medicine"].get("salt", "").lower())

        # Test rendering through client with session
        session = self.client.session
        res["image_url"] = "/media/uploads/test.jpg"
        res["timestamp"] = timezone.now().isoformat()
        session["scan_result"] = res
        session.save()

        resp = self.client.get('/result/')
        self.assertEqual(resp.status_code, 200)
        # Verify both medicine selector cards are rendered
        self.assertContains(resp, "med-selector-section")
        self.assertContains(resp, lipitor_key)
        self.assertContains(resp, other_keys[0])


if __name__ == "__main__":
    unittest.main()

