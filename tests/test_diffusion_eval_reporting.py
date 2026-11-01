import unittest
from runners.eval_diffusion import terminal_details, reconcile_result

class ReportingTests(unittest.TestCase):
    def test_stage_failure_not_lost_when_actor_reason_null(self):
        labels=[dict(failure=None,terminated=False),dict(failure="contact_penetration_limit",terminated=True,truncated=False)]
        r=reconcile_result({"failure_reason":None,"full_task_success":False},terminal_details(labels))
        self.assertEqual(r["failure_reason"],"contact_penetration_limit")
        self.assertIsNone(r["actor_failure_reason"])
        self.assertTrue(r["stage_terminated"])
    def test_reconciliation_is_idempotent(self):
        labels=terminal_details([dict(failure="contact_penetration_limit",terminated=True)])
        once=reconcile_result({"failure_reason":None},labels)
        twice=reconcile_result(once,labels)
        self.assertEqual(once,twice)
        self.assertIsNone(twice["actor_failure_reason"])
    def test_actor_error_preserved_without_stage_failure(self):
        r=reconcile_result({"failure_reason":"IK failure"},terminal_details([]))
        self.assertEqual(r["failure_reason"],"IK failure")
        self.assertFalse(r["stage_terminated"])
    def test_time_limit_is_not_failure(self):
        r=reconcile_result({"failure_reason":None},terminal_details([dict(truncated=True,terminated=False,failure=None)]))
        self.assertTrue(r["stage_truncated"])
        self.assertIsNone(r["failure_reason"])

if __name__=="__main__":unittest.main()
