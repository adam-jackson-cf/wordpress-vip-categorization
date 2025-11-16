-- Drop helper view and functions first to avoid dependency errors
DROP VIEW IF EXISTS export_results;
DROP FUNCTION IF EXISTS match_wordpress_content(vector(1536), float, integer);
DROP FUNCTION IF EXISTS get_unmatched_taxonomy(float);

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS workflow_runs CASCADE;
DROP TABLE IF EXISTS matching_results CASCADE;
DROP TABLE IF EXISTS categorization_results CASCADE;
DROP TABLE IF EXISTS taxonomy_pages CASCADE;
DROP TABLE IF EXISTS wordpress_content CASCADE;
