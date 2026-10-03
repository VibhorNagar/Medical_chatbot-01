"""Smoke tests.  Run from the project folder:  python -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app  # noqa: E402  (builds the index on the first run)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_pages(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/health").get_json()["status"], "ok")

    def test_empty_question(self):
        self.assertEqual(self.c.post("/api/chat", json={"message": "  "}).status_code, 400)

    def test_known_topic(self):
        r = self.c.post("/api/chat", json={"message": "What is asthma?"}).get_json()
        self.assertTrue(r["matched"])
        self.assertIn("asthma", r["results"][0]["text"].lower())

    def test_unknown_topic(self):
        r = self.c.post("/api/chat", json={"message": "zzzz qqqq"}).get_json()
        self.assertFalse(r["matched"])

    def test_emergency_flag(self):
        r = self.c.post("/api/chat", json={"message": "I have chest pain"}).get_json()
        self.assertTrue(r["emergency"])


if __name__ == "__main__":
    unittest.main()
