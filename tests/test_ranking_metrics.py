import unittest
import numpy as np
from scripts.diagnose_ranking import ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_unidentifiable_ties_and_wrong_teacher_takeover(self):
        rows=ranking_metrics(np.array([[0.,0.,0.],[0.,4.,2.]]),np.array([[2.,1.,0.],[5.,3.,1.]]),np.array([0,0]))
        self.assertIsNone(rows[0]['pairwise_accuracy'])
        self.assertEqual(rows[0]['near_tie_pairs'],3)
        self.assertEqual(rows[1]['regret'],4.)
        self.assertTrue(rows[1]['wrong_takeover'])
        self.assertAlmostEqual(rows[1]['pairwise_accuracy'],1/3)


if __name__=='__main__':unittest.main()
