\echo === PostgreSQL 版本 ===
SELECT version();

\echo === 已安装扩展 (pg_extension) ===
SELECT extname, extversion FROM pg_extension ORDER BY extname;

\echo === 镜像里可用/预装的相关扩展 ===
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('vector', 'vectorscale', 'pg_search', 'pg_bm25', 'pg_analytics')
ORDER BY name;

\echo === 确保两个扩展都建好 ===
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

\echo === 再查一次已安装扩展 ===
SELECT extname, extversion FROM pg_extension ORDER BY extname;

\echo === pgvector 功能测试：建表/写入/KNN 检索 ===
DROP TABLE IF EXISTS dsh_vec_demo;
CREATE TABLE dsh_vec_demo (id bigserial PRIMARY KEY, label text, embedding vector(3));
INSERT INTO dsh_vec_demo (label, embedding)
VALUES ('a', '[1,2,3]'), ('b', '[4,5,6]'), ('c', '[7,8,9]');
SELECT id, label, embedding <=> '[1,2,3]' AS cosine_distance
FROM dsh_vec_demo ORDER BY embedding <=> '[1,2,3]' LIMIT 2;

\echo === pg_search BM25 功能测试：建索引/全文检索/打分 ===
DROP TABLE IF EXISTS dsh_bm25_demo;
CREATE TABLE dsh_bm25_demo (id bigserial PRIMARY KEY, description text);
INSERT INTO dsh_bm25_demo (description) VALUES
  ('the quick brown fox jumps over the lazy dog'),
  ('milvus is an open source vector database'),
  ('elasticsearch supports bm25 relevance ranking');
CREATE INDEX dsh_bm25_demo_idx ON dsh_bm25_demo
  USING bm25 (id, description) WITH (key_field = 'id');
SELECT id, description, paradedb.score(id) AS score
FROM dsh_bm25_demo
WHERE description @@@ 'database'
ORDER BY score DESC;

\echo === 清理测试表 ===
DROP TABLE dsh_vec_demo;
DROP TABLE dsh_bm25_demo;
DROP INDEX IF EXISTS dsh_bm25_demo_idx;
