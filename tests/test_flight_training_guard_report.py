"""Reporting fixtures only; no brain, body, coordinator writes or simulations."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("guard_flight_report", ROOT / "scripts/report-flight-training.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class GuardReportTests(unittest.TestCase):
    def update(self, decision, successor=0, completed=True):
        generation = {"generation": 0, "center": "[0]"}
        next_generation = {"generation": 1, "center": f"[{successor}]"} if completed else None
        rows = [{"pairId": f"g0-{prefix}{pair}", "sign": sign, "score": float(sign)}
                for prefix, count in (("p", 4), ("a", 3)) for pair in range(count) for sign in (-1, 1)]
        return report.parameter_update(generation, next_generation, rows if completed else rows[:8], 14,
                                       ["gain"], {"decision": decision})

    def test_search_completion_is_not_a_completed_guarded_update(self):
        value = self.update("pending", completed=False)
        self.assertEqual(value["status"], "incomplete")
        self.assertEqual(value["completedJobs"], 8)
        self.assertEqual(value["expectedJobs"], 14)
        self.assertIsNone(value["changedParameterCount"])

    def test_rejection_is_intentional_retention_not_a_stalled_gradient(self):
        value = self.update("rejected")
        self.assertEqual(value["status"], "completed_rejected_proposal")
        self.assertEqual(value["changedParameterCount"], 0)
        self.assertEqual(value["acceptanceDecision"], "rejected")
        self.assertFalse(value["equalScoresWithinEveryPair"])

    def test_accepted_proposal_reports_actual_saved_coordinate_delta(self):
        value = self.update("accepted", successor=.2)
        self.assertEqual(value["status"], "updated")
        self.assertEqual(value["changedParameterCount"], 1)
        self.assertEqual(value["deltaByParameter"], {"gain": .2})
        self.assertEqual(value["acceptanceDecision"], "accepted")

    def test_contradictory_retention_metadata_cannot_report_success(self):
        for decision, successor in (("rejected", .2), ("identical_parameters", .2), ("pending", 0), ("accepted", 0)):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.update(decision, successor)

    def test_identical_proposal_records_a_completed_noop_after_search_only(self):
        generation = {"generation": 0, "center": "[0]"}
        successor = {"generation": 1, "center": "[0]"}
        rows = [{"pairId": f"g0-p{pair}", "sign": sign, "score": 1.0}
                for pair in range(4) for sign in (-1, 1)]
        value = report.parameter_update(generation, successor, rows, 8, ["gain"], {"decision": "identical_parameters"})
        self.assertEqual(value["status"], "completed_identical_proposal")
        self.assertEqual(value["changedParameterCount"], 0)


if __name__ == "__main__":
    unittest.main()
