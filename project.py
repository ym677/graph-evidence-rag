"""Entity co-occurrence graph retrieval with bounded BFS and evidence provenance."""

from collections import defaultdict, deque
from itertools import combinations
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


SPEC = {
    "title": "Graph Evidence Explorer",
    "subtitle": "从实体共现图发现关联路径，再回到原文检查证据",
    "slug": "graph-evidence-rag",
    "accent": "#0fb9a5",
    "sample_input": "P-101 与润滑有什么关联？请给出资料路径。",
    "input_label": "输入包含设备或维护实体的问题",
}


class EvidenceGraph:
    def __init__(self, documents):
        self.documents = documents
        self.neighbors = defaultdict(set)
        self.edge_sources = defaultdict(list)
        for document in documents:
            for source, target in combinations(sorted(set(document["entities"])), 2):
                self.neighbors[source].add(target)
                self.neighbors[target].add(source)
                self.edge_sources[(source, target)].append(document["id"])
        self.entities = sorted(self.neighbors)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        self.matrix = self.vectorizer.fit_transform([document["text"] for document in documents])

    def seeds(self, query):
        return [entity for entity in self.entities if entity.lower() in query.lower()]

    def traverse(self, seeds, max_hops=2):
        """Multi-source BFS records one shortest path per reached entity."""
        paths = {seed: [seed] for seed in seeds}
        queue = deque(seeds)
        while queue:
            entity = queue.popleft()
            if len(paths[entity]) - 1 >= max_hops:
                continue
            for neighbor in sorted(self.neighbors[entity]):
                if neighbor not in paths:
                    paths[neighbor] = paths[entity] + [neighbor]
                    queue.append(neighbor)
        return paths

    def shortest_path(self, source, target, max_hops=3):
        if source not in self.neighbors or target not in self.neighbors:
            return []
        return self.traverse([source], max_hops).get(target, [])

    def search(self, query, top_k=4):
        if not query.strip():
            raise ValueError("请输入包含设备或维护实体的问题。")
        seeds = self.seeds(query)
        paths = self.traverse(seeds)
        if not seeds:
            return {"seeds": [], "paths": {}, "documents": []}
        similarities = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        ranked = []
        for index, document in enumerate(self.documents):
            direct = len(set(document["entities"]) & set(seeds))
            reachable = [entity for entity in document["entities"] if entity in paths]
            if not reachable:
                continue
            graph_score = sum(1 / len(paths[entity]) for entity in reachable)
            score = 2 * direct + graph_score + float(similarities[index])
            ranked.append({**document, "score": round(score, 4), "direct": direct})
        ranked.sort(key=lambda document: (-document["score"], document["id"]))
        return {"seeds": seeds, "paths": paths, "documents": ranked[:top_k]}

    def visualization(self, evidence):
        """Only display edges backed by retrieved documents."""
        document_ids = {document["id"] for document in evidence}
        edges = []
        nodes = set()
        for (source, target), sources in sorted(self.edge_sources.items()):
            if document_ids.intersection(sources):
                edges.append({"source": source, "target": target})
                nodes.update((source, target))
        return {"nodes": [{"id": node, "label": node} for node in sorted(nodes)], "edges": edges}


def load_documents():
    path = Path(__file__).resolve().parent / "data" / "documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run(payload, client):
    question = str(payload.get("text", "")).strip()
    graph = EvidenceGraph(load_documents())
    result = graph.search(question)
    evidence = result["documents"]
    if not evidence:
        return {
            "summary": "没有识别出知识图中的实体，当前资料不足。可尝试 P-101、轴承、润滑、传感器或告警。",
            "metrics": {"图节点": len(graph.entities), "种子实体": 0, "召回资料": 0},
            "sections": [{"title": "实体范围", "items": graph.entities}],
        }

    seed_pairs = list(combinations(result["seeds"], 2))
    paths = [graph.shortest_path(source, target) for source, target in seed_pairs]
    paths = [path for path in paths if path]
    if not paths:
        paths = sorted(result["paths"].values(), key=lambda path: (-len(path), path))[:3]
    # Include the documents supporting every displayed edge, even when an
    # intermediate document was outside the initial top-k ranking.
    required_ids = set()
    for path in paths:
        for source, target in zip(path, path[1:]):
            required_ids.update(graph.edge_sources[tuple(sorted((source, target)))])
    retrieved_ids = {document["id"] for document in evidence}
    for document in graph.documents:
        if document["id"] in required_ids - retrieved_ids:
            evidence.append({**document, "score": 0.0, "direct": 0})
    context = "\n\n".join(f"[{item['id']}] {item['text']}" for item in evidence)
    path_text = [" → ".join(path) for path in paths]
    if client.enabled:
        answer = client.complete(
            "只根据证据回答并引用 [G01] 格式的文档编号。图边只代表实体在教学文档共现，"
            "不能解释为因果关系或故障确诊。缺少证据时说明不足。",
            f"问题\n{question}\n\n图关联路径\n{'；'.join(path_text)}\n\n原文\n{context}",
            max_tokens=700,
        )
    else:
        answer = "本地图检索找到以下关联路径。路径表示资料共现，需回到原文判断含义。\n\n" + "\n".join(path_text)
    return {
        "summary": answer,
        "metrics": {"图节点": len(graph.entities), "图边": len(graph.edge_sources), "种子实体": len(result["seeds"]), "召回资料": len(evidence)},
        "sections": [
            {"title": "证据子图", "graph": graph.visualization(evidence)},
            {"title": "检索路径", "items": path_text},
            {
                "title": "原文证据",
                "table": {
                    "columns": ["编号", "标题", "评分", "原文"],
                    "rows": [[item["id"], item["title"], item["score"], item["text"]] for item in evidence],
                },
            },
            {"title": "可解释评分", "text": "先以问题中的实体作为种子，执行最多两跳 BFS。文档评分结合直接实体命中、到种子的路径距离和字符 TF-IDF。实体表由教学资料人工标注。"},
        ],
    }
