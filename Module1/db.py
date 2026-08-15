import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
GRAPH_NAME = "freight_network"

def get_db_connection():
    """Establishes connection to PostgreSQL using .env credentials and sets Apache AGE session."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    conn.autocommit = True
    with conn.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute('SET search_path = ag_catalog, "$user", public;')
    return conn

def execute_cypher(query: str, cols: list = None):
    """Executes an openCypher query using Apache AGE's cypher wrapper."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if cols:
                col_defs = ", ".join(cols)
                sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $${query}$$) AS ({col_defs});"
            else:
                sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $${query}$$) AS (a agtype);"
            
            cursor.execute(sql)
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                return None
    finally:
        conn.close()