"""Regression checks for handwritten line parsing and medicine selection."""

from django.test import SimpleTestCase

from scanner.ml.handwritten_trocr import parse_line
from scanner.ml.medicine_matcher import decide_result


class HandwrittenLineMatchingTests(SimpleTestCase):
    def matcher(self, text):
        key = "".join(ch for ch in text.lower() if ch.isalnum())
        catalogue = {
            "lipitor": [
                {"name": "Lipitor 10mg Tablet", "brand": "LIPITOR", "salt": "atorvastatin (10mg)", "score": 100.0},
                {"name": "Lipitor 40mg Tablet", "brand": "LIPITOR", "salt": "atorvastatin (40mg)", "score": 100.0},
            ],
            "calpol": [
                {"name": "Calpol 250mg Suspension", "brand": "CALPOL", "salt": "paracetamol (250mg)", "score": 100.0},
                {"name": "Calpol 500mg Tablet", "brand": "CALPOL", "salt": "paracetamol (500mg)", "score": 100.0},
            ],
            "tabx": [{"name": "Tabix 200mg Tablet", "brand": "TABIX", "salt": "cefixime (200mg)", "score": 88.9}],
            "times": [{"name": "Times 100mg/2mg Tablet", "brand": "TIMES", "salt": "nimesulide + tizanidine", "score": 100.0}],
            "oelcon": [{"name": "Delcon Syrup", "brand": "DELCON", "salt": "chlorpheniramine (2mg/5ml) + phenylephrine (5mg/5ml)", "score": 83.3}],
            "levouni": [{"name": "Levofun 125mg Syrup", "brand": "LEVOFUN", "salt": "levofloxacin (125mg)", "score": 85.7}],
        }
        return catalogue.get(key, [])

    def test_multiword_crop_is_split_and_strength_filters_candidates(self):
        result = parse_line(["Lipitor 10 mg"], matcher=self.matcher)

        self.assertTrue(result["is_medicine"])
        self.assertEqual(result["read"], "Lipitor")
        self.assertEqual([c["name"] for c in result["candidates"]], ["Lipitor 10mg Tablet"])

    def test_syrup_form_excludes_tablet_candidates(self):
        result = parse_line(["Syp Calpol 250/5 4ml Q6H"], matcher=self.matcher)

        self.assertTrue(result["is_medicine"])
        self.assertEqual([c["name"] for c in result["candidates"]], ["Calpol 250mg Suspension"])

    def test_refill_instruction_cannot_become_a_medicine(self):
        result = parse_line(["Refill times"], matcher=self.matcher)

        self.assertFalse(result["is_medicine"])
        self.assertEqual(result["candidates"], [])

    def test_weak_fuzzy_candidate_is_rejected(self):
        result = parse_line(["Cimetidine 50 mg tabx TID"], matcher=self.matcher)

        self.assertFalse(result["is_medicine"])
        self.assertEqual(result["candidates"], [])

    def test_confident_ocr_and_matching_form_can_rescue_one_fuzzy_candidate(self):
        result = parse_line(["Syp OELCON 3ml TDS"], matcher=self.matcher, ocr_confidence=0.85)

        self.assertTrue(result["is_medicine"])
        self.assertEqual([c["name"] for c in result["candidates"]], ["Delcon Syrup"])

    def test_low_ocr_confidence_does_not_promote_a_wrong_syrup_neighbor(self):
        result = parse_line(["Syp Levouni 3ml TDS"], matcher=self.matcher, ocr_confidence=0.77)

        self.assertFalse(result["is_medicine"])
        self.assertEqual(result["candidates"], [])

    def test_symptom_phrase_cannot_become_a_medicine(self):
        result = parse_line(["C/o giddiness restlessness"], matcher=self.matcher)

        self.assertFalse(result["is_medicine"])
        self.assertEqual(result["candidates"], [])

    def test_handwritten_brand_alone_is_not_misread_as_active_composition(self):
        candidate = {
            "name": "Lipitor 10mg Tablet", "brand": "LIPITOR",
            "salt": "atorvastatin (10mg)", "score": 100.0,
        }
        result = decide_result({"brand_read": "Lipitor"}, [candidate])

        self.assertEqual(result["level"], "likely")
        self.assertEqual(result["medicine"]["name"], "Lipitor 10mg Tablet")
