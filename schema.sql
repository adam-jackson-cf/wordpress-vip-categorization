-- Supabase database schema for WordPress VIP Categorization
-- Run this in Supabase SQL Editor to initialize the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- WordPress content table
CREATE TABLE IF NOT EXISTS wordpress_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    site_url TEXT NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on URL for faster lookups
CREATE INDEX IF NOT EXISTS idx_wordpress_content_url ON wordpress_content(url);
CREATE INDEX IF NOT EXISTS idx_wordpress_content_site_url ON wordpress_content(site_url);

-- Taxonomy pages table
CREATE TABLE IF NOT EXISTS taxonomy_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on URL
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_url ON taxonomy_pages(url);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_category ON taxonomy_pages(category);

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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(taxonomy_id, content_id)
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
    COALESCE(cr.confidence, 0.0) as confidence
FROM taxonomy_pages tp
LEFT JOIN matching_results mr ON tp.id = mr.taxonomy_id
LEFT JOIN wordpress_content wc ON mr.content_id = wc.id
LEFT JOIN categorization_results cr ON wc.id = cr.content_id
ORDER BY tp.category, mr.similarity_score DESC NULLS LAST;

-- Grant permissions (adjust based on your security requirements)
-- For service role, all permissions are typically granted by default
-- For anon/authenticated roles, you may want to restrict access

COMMENT ON TABLE wordpress_content IS 'Stores ingested WordPress VIP content';
COMMENT ON TABLE taxonomy_pages IS 'Stores taxonomy pages for matching';
COMMENT ON TABLE categorization_results IS 'Stores AI categorization results';
COMMENT ON TABLE matching_results IS 'Stores semantic matching results between taxonomy and content';
COMMENT ON VIEW export_results IS 'Denormalized view for CSV export';
