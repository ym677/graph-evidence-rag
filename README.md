# Graph Evidence Explorer

![项目封面](cover.svg)

把文档中标注的实体构成共现图，使用多源 BFS 寻找关联资料，再把路径与原文同时展示。适合学习图检索如何补充字符串匹配。

## 核心算法

每份文档显式标注设备、部件或维护实体，同一文档的实体两两构成边。查询中命中的实体作为种子，最多扩展两跳。文档排序组合直接实体命中、到种子的路径距离以及字符 TF-IDF 相似度。两种实体的关联通过最短路径展示，每条可视化边都能追溯到召回文档。

实体共现只表示资料关联，无法证明因果关系。这个项目采用人工标注的微型实体图，没有声称复现完整 GraphRAG 系统。全部数据位于 `data/documents.json`，均为合成教学记录。

## 在 PyCharm 运行

使用 Python 3.12 打开当前项目目录，安装 `requirements.txt`，运行 `app.py`。先用本地演示观察子图和证据表。API 模式按 `.env.example` 设置本机环境变量，再使用同一组检索证据生成回答。

## 可复核评估

运行 `python -m unittest discover -s tests`。测试包括两跳路径、跳数限制、未知实体、无实体拒答与可视化边的文档来源。

可人工建立实体关联问题集，标注应召回的文档和允许的路径，对比仅 TF-IDF 与图扩展的 Recall@k、路径覆盖率及不相关证据数量。应保留失败案例，特别是高连接度实体引入的噪声。当前实现没有训练实体抽取模型，分词与实体别名需要进一步扩展。

## 参考

- [Python collections.deque](https://docs.python.org/3/library/collections.html#collections.deque)
- [scikit-learn 文本特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Microsoft GraphRAG 项目，可用于理解更完整的图索引与摘要流程](https://github.com/microsoft/graphrag)
