-- Supabase database schema for WordPress VIP Categorization
-- Run this in Supabase SQL Editor to initialize the database

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- WordPress content table
CREATE TABLE IF NOT EXISTS wordpress_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    site_url TEXT NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb,
    detected_audiences JSONB DEFAULT '[]'::jsonb,
    detected_species JSONB DEFAULT '[]'::jsonb,
    content_embedding VECTOR(1536),
    embedding_updated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on URL for faster lookups
CREATE INDEX IF NOT EXISTS idx_wordpress_content_url ON wordpress_content(url);
CREATE INDEX IF NOT EXISTS idx_wordpress_content_site_url ON wordpress_content(site_url);
CREATE INDEX IF NOT EXISTS idx_wordpress_content_embedding
    ON wordpress_content USING ivfflat (content_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Taxonomy pages table
CREATE TABLE IF NOT EXISTS taxonomy_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    uid TEXT,
    destination_url TEXT UNIQUE NOT NULL,
    reference_source TEXT,
    english_page_name TEXT,
    local_page_name TEXT,
    content_type TEXT NOT NULL,
    primary_audiance TEXT,
    secondary_audiance TEXT,
    species JSONB DEFAULT '[]'::jsonb,
    semantic_summary TEXT NOT NULL,
    key_topics JSONB DEFAULT '[]'::jsonb,
    taxonomy_embedding VECTOR(1536),
    embedding_updated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on URL
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_destination_url ON taxonomy_pages(destination_url);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_content_type ON taxonomy_pages(content_type);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_embedding
    ON taxonomy_pages USING ivfflat (taxonomy_embedding vector_cosine_ops)
    WITH (lists = 50);

-- Categorization results table
CREATE TABLE IF NOT EXISTS categorization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES wordpress_content(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    batch_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_categorization_content_id ON categorization_results(content_id);
CREATE INDEX IF NOT EXISTS idx_categorization_batch_id ON categorization_results(batch_id);

-- Matching results table
CREATE TABLE IF NOT EXISTS matching_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID UNIQUE NOT NULL REFERENCES wordpress_content(id) ON DELETE CASCADE,
    taxonomy_id UUID REFERENCES taxonomy_pages(id) ON DELETE SET NULL,
    semantic_taxonomy_id UUID REFERENCES taxonomy_pages(id) ON DELETE SET NULL,
    semantic_similarity_score FLOAT DEFAULT 0.0 CHECK (
        semantic_similarity_score >= 0 AND semantic_similarity_score <= 1
    ),
    llm_topic_score FLOAT CHECK (llm_topic_score >= 0 AND llm_topic_score <= 1),
    match_stage TEXT,
    failed_at_stage TEXT,
    rubric JSONB DEFAULT '{}'::jsonb,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for faster lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_matching_content_current
    ON matching_results(content_id)
    WHERE is_current;
CREATE INDEX IF NOT EXISTS idx_matching_taxonomy_id ON matching_results(taxonomy_id);
CREATE INDEX IF NOT EXISTS idx_matching_semantic_taxonomy_id
    ON matching_results(semantic_taxonomy_id);
CREATE INDEX IF NOT EXISTS idx_matching_stage ON matching_results(match_stage);

-- Create a view for export with denormalized data
CREATE OR REPLACE VIEW export_results AS
SELECT
    wc.url AS source_url,
    COALESCE(tp.destination_url, '') AS target_url,
    COALESCE(tp.content_type, '') AS content_type,
    COALESCE(mr.semantic_similarity_score, 0.0) AS similarity_score,
    mr.match_stage,
    mr.failed_at_stage
FROM wordpress_content wc
LEFT JOIN matching_results mr ON wc.id = mr.content_id AND mr.is_current IS TRUE
LEFT JOIN taxonomy_pages tp ON mr.taxonomy_id = tp.id
ORDER BY wc.site_url, wc.url;

-- Vector similarity helper for Supabase RPC usage
CREATE OR REPLACE FUNCTION match_wordpress_content(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.6,
    match_count integer DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    url text,
    title text,
    content text,
    site_url text,
    similarity float
)
LANGUAGE SQL STABLE
AS $$
SELECT
    wc.id,
    wc.url,
    wc.title,
    wc.content,
    wc.site_url,
    1 - (wc.content_embedding <=> query_embedding) as similarity
FROM wordpress_content wc
WHERE wc.content_embedding IS NOT NULL
  AND 1 - (wc.content_embedding <=> query_embedding) >= match_threshold
ORDER BY wc.content_embedding <=> query_embedding
LIMIT match_count;
$$;

-- Vector similarity search for taxonomy pages
CREATE OR REPLACE FUNCTION match_taxonomy_pages(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.6,
    match_count integer DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    destination_url text,
    content_type text,
    semantic_summary text,
    key_topics jsonb,
    similarity float
)
LANGUAGE SQL STABLE
AS $$
SELECT
    tp.id,
    tp.destination_url,
    tp.content_type,
    tp.semantic_summary,
    tp.key_topics,
    1 - (tp.taxonomy_embedding <=> query_embedding) as similarity
FROM taxonomy_pages tp
WHERE tp.taxonomy_embedding IS NOT NULL
  AND 1 - (tp.taxonomy_embedding <=> query_embedding) >= match_threshold
ORDER BY tp.taxonomy_embedding <=> query_embedding
LIMIT match_count;
$$;

-- Helper to fetch taxonomy rows without a canonical match or below a target score
-- helper no longer needed; content-first workflow filters unmatched content directly

-- Workflow run metadata for resumable processing
CREATE TABLE IF NOT EXISTS workflow_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_key TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT,
    config JSONB DEFAULT '{}'::jsonb,
    stats JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error TEXT
);

-- Grant permissions (adjust based on your security requirements)
-- For service role, all permissions are typically granted by default
-- For anon/authenticated roles, you may want to restrict access

COMMENT ON TABLE wordpress_content IS 'Stores ingested WordPress VIP content';
COMMENT ON TABLE taxonomy_pages IS 'Stores taxonomy pages for matching';
COMMENT ON TABLE categorization_results IS 'Stores AI categorization results';
COMMENT ON TABLE matching_results IS 'Stores semantic matching results between taxonomy and content';
COMMENT ON VIEW export_results IS 'Denormalized view for CSV export';
