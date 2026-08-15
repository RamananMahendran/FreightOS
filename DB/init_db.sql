-- Enable Apache AGE extension in PostgreSQL
CREATE EXTENSION IF NOT EXISTS age CASCADE;

-- Preload AGE for the current session
LOAD 'age';

-- Include ag_catalog in search path
SET search_path = ag_catalog, "$user", public;

-- Create FreightOS Knowledge Graph
SELECT create_graph('freight_network');