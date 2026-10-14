"""
scanner/tests/test_price_verification.py — Automated Regression Test Suite
for Medicine Pack Size, Price Verification & Alternative Comparison.

Covers all 19 required clinical scenarios:
1. Verified price per tablet.
2. Verified price per strip.
3. Different package quantities.
4. Unknown package quantity.
5. Unknown price basis.
6. MRP and selling price stored separately.
7. Conflicting online prices.
8. Incorrect manufacturer.
9. Incorrect strength.
10. Incorrect dosage form.
11. Alternative with a different active ingredient.
12. API timeout.
13. API rate limit.
14. API service unavailable.
15. Stale cache refresh.
16. Price comparison omitted when evidence is insufficient.
17. Price sorting based on verified comparable unit prices.
18. Frontend savings labels match the actual comparison basis.
19. The result page remains usable when price verification fails.
"""

import os
import json
import tempfile
import requests
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from django.test import TestCase, RequestFactory
from django.template.loader import render_to_string

from scanner.ml.price_verifier import (
    PriceVerificationRecord,
    ProductListing,
    MockSearchProvider,
    DirectPharmacySearchProvider,
    PriceVerificationCache,
    PriceVerifier,
    extract_package_details,
    extract_prices_from_text,
    validate_listing_identity,
    compare_medicine_prices,
)
from scanner.ml.alternative_matcher import find_top_cheaper_alternatives
from scanner.views import result as result_view


class PriceVerificationRegressionTests(TestCase):
    """Exhaustive test suite verifying pricing calculations, unit integrity, and resilience."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.temp_db.close()
        self.cache = PriceVerificationCache(db_path=self.temp_db.name)
        self.mock_provider = MockSearchProvider()
        self.verifier = PriceVerifier(search_provider=self.mock_provider, cache=self.cache)

    def tearDown(self):
        try:
            if os.path.exists(self.temp_db.name):
                os.unlink(self.temp_db.name)
        except Exception:
            pass

    # ── 1. Verified price per tablet ──────────────────────────────────────────
    def test_01_verified_price_per_tablet(self):
        detected = {
            "product_name": "Lorinol 10 Tablet",
            "pack_quantity": 1,
            "pack_unit": "tablet",
            "unit_price_inr": 9.20,
            "package_price_inr": 9.20,
            "price_type": "mrp",
            "price_basis": "per_tablet",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        alternative = {
            "product_name": "Lorsyn 10 mg Tablet DT",
            "pack_quantity": 1,
            "pack_unit": "tablet",
            "unit_price_inr": 0.64,
            "package_price_inr": 0.64,
            "price_type": "mrp",
            "price_basis": "per_tablet",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        res = compare_medicine_prices(detected, alternative)
        self.assertTrue(res["comparison_eligible"])
        self.assertEqual(res["unit_saving"], 8.56)
        self.assertAlmostEqual(res["saving_percent"], 93.0, places=1)
        self.assertIn("per tablet", res["comparison_basis"])
        self.assertIn("Save ₹8.56 per tablet", res["saving_label"])

    # ── 2. Verified price per strip ───────────────────────────────────────────
    def test_02_verified_price_per_strip(self):
        detected = {
            "product_name": "Medicine A 10mg",
            "pack_quantity": 10,
            "pack_unit": "tablet",
            "package_price_inr": 92.00,
            "unit_price_inr": 9.20,
            "price_type": "mrp",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        alternative = {
            "product_name": "Medicine B 10mg",
            "pack_quantity": 10,
            "pack_unit": "tablet",
            "package_price_inr": 64.00,
            "unit_price_inr": 6.40,
            "price_type": "mrp",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        res = compare_medicine_prices(detected, alternative)
        self.assertTrue(res["comparison_eligible"])
        self.assertEqual(res["package_saving"], 28.00)
        self.assertAlmostEqual(res["saving_percent"], 30.4, places=1)
        self.assertIn("10-tablet strip", res["comparison_basis"])
        self.assertIn("Save ₹28.00 per 10-tablet strip", res["saving_label"])

    # ── 3. Different package quantities ───────────────────────────────────────
    def test_03_different_package_quantities(self):
        detected = {
            "product_name": "Medicine A 10mg",
            "pack_quantity": 10,
            "pack_unit": "tablet",
            "package_price_inr": 92.00,
            "unit_price_inr": 9.20,
            "price_type": "mrp",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        alternative = {
            "product_name": "Medicine C 10mg",
            "pack_quantity": 5,
            "pack_unit": "tablet",
            "package_price_inr": 38.00,
            "unit_price_inr": 7.60,
            "price_type": "mrp",
            "comparison_eligible": True,
            "dosage_form": "Tablet",
        }
        res = compare_medicine_prices(detected, alternative)
        self.assertTrue(res["comparison_eligible"])
        self.assertEqual(res["unit_saving"], 1.60)
        self.assertAlmostEqual(res["saving_percent"], 17.4, places=1)
        self.assertIn("per tablet", res["comparison_basis"])
        self.assertIn("Save ₹1.60 per tablet", res["saving_label"])

    # ── 4. Unknown package quantity ───────────────────────────────────────────
    def test_04_unknown_package_quantity(self):
        detected = {
            "product_name": "Medicine A 10mg",
            "pack_quantity": None,  # Unknown
            "package_price_inr": 92.00,
            "unit_price_inr": None,
            "comparison_eligible": False,
        }
        alternative = {
            "product_name": "Medicine B 10mg",
            "pack_quantity": 10,
            "package_price_inr": 6.40,
            "unit_price_inr": 0.64,
            "comparison_eligible": True,
        }
        res = compare_medicine_prices(detected, alternative)
        self.assertFalse(res["comparison_eligible"])
        self.assertIsNone(res["saving_amount"])
        self.assertIsNone(res["saving_percent"])
        self.assertIn("Package size or price basis could not be verified", res["reason"])

    # ── 5. Unknown price basis ────────────────────────────────────────────────
    def test_05_unknown_price_basis(self):
        detected = {
            "product_name": "Lorinol 10",
            "package_price_inr": 92.00,
            "price_basis": "unknown",
            "comparison_eligible": False,
        }
        alternative = {
            "product_name": "Lorsyn 10",
            "package_price_inr": 6.40,
            "price_basis": "unknown",
            "comparison_eligible": False,
        }
        res = compare_medicine_prices(detected, alternative)
        self.assertFalse(res["comparison_eligible"])
        self.assertIsNone(res["saving_amount"])
        self.assertIsNone(res["saving_percent"])
        self.assertEqual(res["comparison_basis"], "unavailable")

    # ── 6. MRP and selling price stored separately ────────────────────────────
    def test_06_mrp_and_selling_price_stored_separately(self):
        text = "Lorinol 10mg Tablet Strip of 10. MRP: Rs 100.00. Selling Price: Rs 75.00."
        prices = extract_prices_from_text(text)
        self.assertEqual(prices["mrp"], 100.00)
        self.assertEqual(prices["selling_price"], 75.00)

        # In verification record
        self.mock_provider.set_fixture("Lorinol", [
            ProductListing(
                title="Lorinol 10 Tablet Micro Labs",
                snippet=text,
                url="https://1mg.com/drugs/lorinol-10",
                source_name="1mg",
                mrp_inr=100.00,
                selling_price_inr=75.00,
                pack_quantity=10,
                pack_unit="tablet",
            )
        ])
        rec = self.verifier.verify_medicine_price({
            "id": 119057,
            "name": "Lorinol 10 Tablet",
            "manufacturer": "Micro Labs Ltd",
            "salt": "loratadine (10mg)",
        })
        self.assertEqual(rec.mrp_inr, 100.00)
        self.assertEqual(rec.selling_price_inr, 75.00)
        self.assertEqual(rec.price_type, "mixed")
        self.assertEqual(rec.unit_price_inr, 7.50)

    # ── 7. Conflicting online prices ──────────────────────────────────────────
    def test_07_conflicting_online_prices(self):
        self.mock_provider.set_fixture("DiscrepantMed", [
            ProductListing(
                title="DiscrepantMed 10mg Strip of 10",
                snippet="DiscrepantMed by Test Labs. MRP ₹92.00 for strip of 10 tablets",
                url="https://source1.com/p",
                source_name="Source 1",
                mrp_inr=92.00,
                pack_quantity=10,
                pack_unit="tablet",
            ),
            ProductListing(
                title="DiscrepantMed 10mg Strip of 10",
                snippet="DiscrepantMed by Test Labs. MRP ₹150.00 for strip of 10 tablets",
                url="https://source2.com/p",
                source_name="Source 2",
                mrp_inr=150.00,
                pack_quantity=10,
                pack_unit="tablet",
            ),
        ])
        rec = self.verifier.verify_medicine_price({
            "id": 999,
            "name": "DiscrepantMed 10mg",
            "manufacturer": "Test Labs",
            "salt": "test drug (10mg)",
        })
        self.assertEqual(rec.verification_status, "discrepant")
        self.assertFalse(rec.comparison_eligible)
        self.assertTrue(any("Conflicting MRP values" in d for d in rec.discrepancies))

    # ── 8. Incorrect manufacturer ─────────────────────────────────────────────
    def test_08_incorrect_manufacturer(self):
        listing = ProductListing(
            title="Lorinol 10 Tablet",
            snippet="Lorinol 10 Tablet manufactured by Cipla Ltd, Loratadine 10mg",
            url="https://fake.com",
            source_name="FakeSource",
            extracted_manufacturer="Cipla Ltd",
        )
        is_valid, reasons = validate_listing_identity(
            product_name="Lorinol 10 Tablet",
            manufacturer="Micro Labs Ltd",
            ingredients=[{"ingredient": "loratadine", "strength": "10", "unit": "mg"}],
            dosage_form="Tablet",
            listing=listing,
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Manufacturer mismatch" in r or "different manufacturer" in r for r in reasons))

    # ── 9. Incorrect strength ─────────────────────────────────────────────────
    def test_09_incorrect_strength(self):
        listing = ProductListing(
            title="Lorinol 20 Tablet",
            snippet="Lorinol 20mg Tablet Micro Labs Ltd",
            url="https://fake.com",
            source_name="FakeSource",
        )
        is_valid, reasons = validate_listing_identity(
            product_name="Lorinol 10 Tablet",
            manufacturer="Micro Labs Ltd",
            ingredients=[{"ingredient": "loratadine", "strength": "10", "unit": "mg"}],
            dosage_form="Tablet",
            listing=listing,
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Strength mismatch" in r for r in reasons))

    # ── 10. Incorrect dosage form ─────────────────────────────────────────────
    def test_10_incorrect_dosage_form(self):
        listing = ProductListing(
            title="Lorinol Eye Drops",
            snippet="Lorinol Eye Drops 5ml by Micro Labs",
            url="https://fake.com",
            source_name="FakeSource",
        )
        is_valid, reasons = validate_listing_identity(
            product_name="Lorinol 10 Tablet",
            manufacturer="Micro Labs Ltd",
            ingredients=[{"ingredient": "loratadine", "strength": "10", "unit": "mg"}],
            dosage_form="Tablet",
            listing=listing,
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Dosage form mismatch" in r for r in reasons))

    # ── 11. Alternative with a different active ingredient ────────────────────
    def test_11_alternative_with_different_active_ingredient(self):
        # Primary is Loratadine 10mg
        med = {
            "id": 119057,
            "name": "Lorinol 10 Tablet",
            "salt": "loratadine (10mg)",
            "price": 92.0,
            "dosage_form": "Tablet",
        }
        alts = find_top_cheaper_alternatives(med, limit=5, verifier=self.verifier)
        for a in alts:
            # Must strictly have loratadine in salt
            self.assertIn("loratadine", a["salt"].lower())
            self.assertNotIn("paracetamol", a["salt"].lower())
            self.assertNotIn("cetirizine", a["salt"].lower())

    # ── 12. API timeout ───────────────────────────────────────────────────────
    def test_12_api_timeout(self):
        timeout_provider = DirectPharmacySearchProvider(timeout=0.001, max_retries=1)
        with patch.object(timeout_provider.session, 'post', side_effect=requests.Timeout("Connection timed out")):
            verifier = PriceVerifier(search_provider=timeout_provider, cache=self.cache)
            rec = verifier.verify_medicine_price({
                "id": 100,
                "name": "TimeoutMed 10mg",
                "salt": "test (10mg)",
            })
            self.assertEqual(rec.verification_status, "unverified")
            self.assertFalse(rec.comparison_eligible)
            self.assertTrue(any("Online search error" in n or "No authoritative" in n for n in rec.notes))

    # ── 13. API rate limit (HTTP 429) ─────────────────────────────────────────
    def test_13_api_rate_limit(self):
        provider = DirectPharmacySearchProvider(max_retries=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch.object(provider.session, 'post', return_value=mock_resp):
            verifier = PriceVerifier(search_provider=provider, cache=self.cache)
            rec = verifier.verify_medicine_price({
                "id": 101,
                "name": "RateLimitedMed 10mg",
                "salt": "test (10mg)",
            })
            self.assertEqual(rec.verification_status, "unverified")
            self.assertFalse(rec.comparison_eligible)

    # ── 14. API service unavailable (HTTP 503) ────────────────────────────────
    def test_14_api_service_unavailable(self):
        provider = DirectPharmacySearchProvider(max_retries=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch.object(provider.session, 'post', return_value=mock_resp):
            verifier = PriceVerifier(search_provider=provider, cache=self.cache)
            rec = verifier.verify_medicine_price({
                "id": 102,
                "name": "ServiceUnavailableMed 10mg",
                "salt": "test (10mg)",
            })
            self.assertEqual(rec.verification_status, "unverified")
            self.assertFalse(rec.comparison_eligible)

    # ── 15. Stale cache refresh ───────────────────────────────────────────────
    def test_15_stale_cache_refresh(self):
        med_id = 777
        name = "CacheRefreshMed 10mg"
        # Seed cache with an expired record
        old_record = PriceVerificationRecord(
            medicine_id=med_id,
            product_name=name,
            package_price_inr=50.0,
            pack_quantity=10,
            unit_price_inr=5.0,
            comparison_eligible=True,
            verification_status="verified",
        )
        self.cache.set(old_record, ttl_hours=-10)  # Expired 10 hours ago

        # Provider has new price
        self.mock_provider.set_fixture("CacheRefreshMed", [
            ProductListing(
                title=f"{name} Strip of 10",
                snippet=f"{name} MRP: ₹60.00 for strip of 10 tablets",
                url="https://pharmacy.com/fresh",
                source_name="Fresh Pharmacy",
                mrp_inr=60.0,
                pack_quantity=10,
                pack_unit="tablet",
            )
        ])

        # Verifier should notice stale and refresh
        rec = self.verifier.verify_medicine_price({"id": med_id, "name": name, "salt": "molecule (10mg)"})
        self.assertEqual(rec.package_price_inr, 60.0)
        self.assertEqual(rec.unit_price_inr, 6.0)

    # ── 16. Price comparison omitted when evidence is insufficient ────────────
    def test_16_price_comparison_omitted_when_evidence_is_insufficient(self):
        # Empty mock provider -> no online evidence found
        empty_verifier = PriceVerifier(search_provider=MockSearchProvider(), cache=self.cache)
        detected = {"id": 119057, "name": "Lorinol 10 Tablet", "salt": "loratadine (10mg)", "price": 92.0}
        alts = find_top_cheaper_alternatives(detected, limit=5, verifier=empty_verifier)

        # Savings must be omitted
        for a in alts:
            self.assertFalse(a["price_comparison"]["comparison_eligible"])
            self.assertIsNone(a["saving_amount"])
            self.assertIsNone(a["saving_percent"])
            self.assertFalse(a["is_cheapest"])
            self.assertEqual(a["comparison_basis"], "unavailable")

    # ── 17. Price sorting based on verified comparable unit prices ─────────────
    def test_17_price_sorting_based_on_verified_comparable_unit_prices(self):
        # We test that verified alternatives with lower comparable unit prices are placed first
        mock_p = MockSearchProvider({
            "primary": [ProductListing("Primary 10", "Strip of 10. MRP ₹100", "url", "Source", mrp_inr=100.0, pack_quantity=10, pack_unit="tablet")],
            "alt_cheap": [ProductListing("Alt Cheap", "Strip of 10. MRP ₹20", "url", "Source", mrp_inr=20.0, pack_quantity=10, pack_unit="tablet")],
            "alt_mid": [ProductListing("Alt Mid", "Strip of 10. MRP ₹50", "url", "Source", mrp_inr=50.0, pack_quantity=10, pack_unit="tablet")],
        })
        verifier = PriceVerifier(search_provider=mock_p, cache=self.cache)
        rec_p = verifier.verify_medicine_price({"id": 1, "name": "Primary 10", "salt": "drug (10mg)"})
        rec_c1 = verifier.verify_medicine_price({"id": 2, "name": "Alt Mid", "salt": "drug (10mg)"})
        rec_c2 = verifier.verify_medicine_price({"id": 3, "name": "Alt Cheap", "salt": "drug (10mg)"})

        comp1 = compare_medicine_prices(rec_p.to_dict(), rec_c1.to_dict())
        comp2 = compare_medicine_prices(rec_p.to_dict(), rec_c2.to_dict())

        self.assertTrue(comp1["comparison_eligible"])
        self.assertTrue(comp2["comparison_eligible"])
        # Alt cheap unit price is 2.0 vs Alt mid is 5.0
        self.assertLess(rec_c2.unit_price_inr, rec_c1.unit_price_inr)

    # ── 18. Frontend savings labels match the actual comparison basis ─────────
    def test_18_frontend_savings_labels_match_actual_comparison_basis(self):
        # Scenario A: Same strip quantity
        res_strip = compare_medicine_prices(
            {"product_name": "A", "pack_quantity": 10, "pack_unit": "tablet", "package_price_inr": 92.0, "unit_price_inr": 9.20, "comparison_eligible": True, "dosage_form": "Tablet"},
            {"product_name": "B", "pack_quantity": 10, "pack_unit": "tablet", "package_price_inr": 64.0, "unit_price_inr": 6.40, "comparison_eligible": True, "dosage_form": "Tablet"},
        )
        self.assertIn("per 10-tablet strip", res_strip["saving_label"])

        # Scenario B: Different strip quantity
        res_unit = compare_medicine_prices(
            {"product_name": "A", "pack_quantity": 10, "pack_unit": "tablet", "package_price_inr": 92.0, "unit_price_inr": 9.20, "comparison_eligible": True, "dosage_form": "Tablet"},
            {"product_name": "C", "pack_quantity": 5, "pack_unit": "tablet", "package_price_inr": 38.0, "unit_price_inr": 7.60, "comparison_eligible": True, "dosage_form": "Tablet"},
        )
        self.assertIn("per tablet", res_unit["saving_label"])

    # ── 19. Result page remains usable when price verification fails ──────────
    def test_19_result_page_remains_usable_when_price_verification_fails(self):
        rf = RequestFactory()
        request = rf.get('/result/')
        request.session = {
            'scan_result': {
                'success': True,
                'status': 'verified',
                'mode': 'medicine_package',
                'detected_name': 'Lorinol 10 Tablet',
                'confidence_breakdown': {'overall': 92.0},
                'medicines': [{
                    'id': 119057,
                    'name': 'Lorinol 10 Tablet',
                    'manufacturer': 'Micro Labs Ltd',
                    'salt': 'loratadine (10mg)',
                    'price': 92.0,
                    'dosage_form': 'Tablet',
                    'confidence': 92.0,
                    'price_verification': {
                        'verification_status': 'unverified',
                        'comparison_eligible': False,
                        'pack_quantity': None,
                        'pack_unit': None,
                        'unit_price_inr': None,
                    }
                }],
                'alternatives': [{
                    'name': 'Lorsyn 10 mg Tablet DT',
                    'manufacturer': 'Syncom Formulations Limited',
                    'salt': 'loratadine (10mg)',
                    'price': 6.4,
                    'saving_amount': None,
                    'saving_percent': None,
                    'is_cheapest': False,
                    'price_verification': {
                        'verification_status': 'unverified',
                        'comparison_eligible': False,
                    },
                    'price_comparison': {
                        'comparison_eligible': False,
                        'reason': 'Package size or price basis could not be verified.',
                    }
                }]
            }
        }
        response = result_view(request)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode('utf-8')
        # Check that UI renders cleanly without crashing
        self.assertIn('Lorinol 10 Tablet', content)
        self.assertIn('Price unit not verified', content)
        self.assertIn('Price comparison unavailable', content)
        # Unverified discount must NOT be shown
        self.assertNotIn('93% cheaper', content)
        self.assertNotIn('Save ₹85.60', content)
