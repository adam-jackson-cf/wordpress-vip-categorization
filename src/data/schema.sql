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
    url TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords JSONB DEFAULT '[]'::jsonb,
    taxonomy_embedding VECTOR(1536),
    embedding_updated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on URL
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_url ON taxonomy_pages(url);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_category ON taxonomy_pages(category);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_embedding
    ON taxonomy_pages USING ivfflat (taxonomy_embedding vector_cosine_ops)
    WITH (lists = 50);

-- Categorization results table
CREATE TABLE IF NOT EXISTS categorization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES wordpress_content(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    batch_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_categorization_content_id ON categorization_results(content_id);
CREATE INDEX IF NOT EXISTS idx_categorization_batch_id ON categorization_results(batch_id);

-- Matching results table
CREATE TABLE IF NOT EXISTS matching_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    taxonomy_id UUID NOT NULL REFERENCES taxonomy_pages(id) ON DELETE CASCADE,
    content_id UUID REFERENCES wordpress_content(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    match_stage TEXT,
    failed_at_stage TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(taxonomy_id)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_matching_taxonomy_id ON matching_results(taxonomy_id);
CREATE INDEX IF NOT EXISTS idx_matching_content_id ON matching_results(content_id);
CREATE INDEX IF NOT EXISTS idx_matching_similarity_score ON matching_results(similarity_score DESC);

-- Create a view for export with denormalized data
CREATE OR REPLACE VIEW export_results AS
SELECT
    tp.url as source_url,
    COALESCE(wc.url, '') as target_url,
    COALESCE(mr.similarity_score, 0.0) as similarity_score,
    tp.category,
    COALESCE(cr.confidence, 0.0) as confidence,
    mr.match_stage,
    mr.failed_at_stage
FROM taxonomy_pages tp
LEFT JOIN matching_results mr ON tp.id = mr.taxonomy_id
LEFT JOIN wordpress_content wc ON mr.content_id = wc.id
LEFT JOIN categorization_results cr ON wc.id = cr.content_id
ORDER BY tp.category, mr.similarity_score DESC NULLS LAST;

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

-- Helper to fetch taxonomy rows without a canonical match or below a target score
CREATE OR REPLACE FUNCTION get_unmatched_taxonomy(min_similarity float)
RETURNS SETOF taxonomy_pages
LANGUAGE SQL STABLE
AS $$
SELECT tp.*
FROM taxonomy_pages tp
LEFT JOIN matching_results mr ON mr.taxonomy_id = tp.id
WHERE mr.taxonomy_id IS NULL OR COALESCE(mr.similarity_score, 0.0) < min_similarity;
$$;

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
