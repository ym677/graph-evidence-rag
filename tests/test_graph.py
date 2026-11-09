from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import EvidenceGraph, load_documents, run


class OfflineClient:
    enabled = False


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = EvidenceGraph(load_documents())

    def test_multihop_path_is_supported(self):
        self.assertEqual(self.graph.shortest_path("P-101", "润滑"), ["P-101", "轴承", "润滑"])

    def test_hop_limit_is_respected(self):
        reached = self.graph.traverse(["P-101"], max_hops=1)
        self.assertNotIn("润滑", reached)

    def test_unknown_entity_has_no_path(self):
        self.assertEqual(self.graph.shortest_path("unknown", "润滑"), [])

    def test_no_entity_abstains(self):
        self.assertEqual(self.graph.search("木星上的天气")['documents'], [])

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.search(" ")

    def test_visualized_edges_have_evidence(self):
        evidence = self.graph.search("P-101 润滑")['documents']
        visible = self.graph.visualization(evidence)
        selected_ids = {document['id'] for document in evidence}
        for edge in visible['edges']:
            key = tuple(sorted((edge['source'], edge['target'])))
            self.assertTrue(selected_ids.intersection(self.graph.edge_sources[key]))

    def test_offline_run_has_graph(self):
        result = run({"text": "P-101 润滑"}, OfflineClient())
        self.assertTrue(result['sections'][0]['graph']['nodes'])


if __name__ == "__main__":
    unittest.main()
