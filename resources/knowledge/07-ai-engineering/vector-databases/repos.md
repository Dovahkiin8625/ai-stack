# 开源仓库：向量数据库

## Milvus (Zilliz)

- 仓库：https://github.com/milvus-io/milvus
- 简介：当前生态最完整的向量数据库；Go + C++ 实现，支持十亿级向量检索、混合检索（稀疏 + 稠密）、标量过滤、多租户。

## Qdrant

- 仓库：https://github.com/qdrant/qdrant
- 简介：Rust 编写的向量数据库；REST/gRPC 接口、payload filtering、磁盘索引；单节点性能优异。

## Weaviate

- 仓库：https://github.com/weaviate/weaviate
- 简介：Go 编写的向量数据库，内置 vectorization modules（OpenAI / Cohere / HuggingFace），GraphQL 接口，支持 hybrid search。

## Chroma

- 仓库：https://github.com/chroma-core/chroma
- 简介：Python 原生、API 极简的向量库，常见于 LangChain / LlamaIndex 教程示例；轻量但不适合亿级。

## pgvector

- 仓库：https://github.com/pgvector/pgvector
- 简介：PostgreSQL 的向量检索扩展，复用 PG 生态；适合中小规模 + 关系型数据共存场景。

## FAISS (Meta)

- 仓库：https://github.com/facebookresearch/faiss
- 简介：经典 C++ 向量检索库（IVF / HNSW / PQ），单机性能极强，常作为其他库的底层引擎。

## Lance (LanceDB)

- 仓库：https://github.com/lancedb/lance
- 简介：Rust + 嵌入式列式向量存储；零服务器、面向 AI 训练数据场景。
