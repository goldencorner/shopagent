CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id          BIGSERIAL PRIMARY KEY,
    asin        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    brand       TEXT,
    category    TEXT,
    price       NUMERIC,
    description TEXT,
    features    TEXT
);

CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    product_id  BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL
);

-- HNSW：数据量在万级以内时，比 IVF-PQ 召回更稳、无需训练
CREATE INDEX chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_product_id_idx ON chunks(product_id);
