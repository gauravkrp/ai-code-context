#!/usr/bin/env python
"""
Inspect ChromaDB directory structure to understand how data is stored.
This helps diagnose persistence issues by showing what files are created
and how they relate to ChromaDB collections and documents.
"""
import os
import sys
import json
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inspect_chroma_files")

def inspect_directory(dir_path):
    """Inspect a ChromaDB directory to report on contents"""
    path = Path(dir_path)
    
    if not path.exists():
        logger.error(f"Directory does not exist: {path.absolute()}")
        return False
    
    logger.info(f"Inspecting ChromaDB directory: {path.absolute()}")
    
    # List all files in the directory
    logger.info("\n=== Files in directory ===")
    file_count = 0
    for root, dirs, files in os.walk(path):
        rel_path = Path(root).relative_to(path)
        for file in files:
            file_path = Path(root) / file
            file_size = file_path.stat().st_size
            logger.info(f"{rel_path / file} ({format_size(file_size)})")
            file_count += 1
    
    if file_count == 0:
        logger.warning("No files found in the ChromaDB directory!")
        return False
    
    # Check for SQLite database
    db_file = path / "chroma.sqlite3"
    if db_file.exists():
        logger.info("\n=== SQLite Database Analysis ===")
        inspect_sqlite_db(db_file)
    else:
        logger.warning("No SQLite database found in the ChromaDB directory")
    
    # Check for index files
    index_files = list(path.glob("**/index_metadata.json"))
    if index_files:
        logger.info("\n=== Index Files Analysis ===")
        for idx_file in index_files:
            inspect_index_file(idx_file)
    else:
        logger.warning("No index metadata files found in the ChromaDB directory")
    
    return True

def inspect_sqlite_db(db_path):
    """Inspect a ChromaDB SQLite database"""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Database tables: {[t[0] for t in tables]}")
        
        # Analyze each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            logger.info(f"Table '{table_name}' has {row_count} rows")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            logger.info(f"  Columns: {col_names}")
            
            # Show sample data if available
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                logger.info(f"  Sample row: {sample}")
        
        conn.close()
    except Exception as e:
        logger.error(f"Error inspecting SQLite database: {e}")

def inspect_index_file(file_path):
    """Inspect a ChromaDB index metadata file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Index file: {file_path}")
        logger.info(f"  Index metadata: {json.dumps(data, indent=2)}")
    except Exception as e:
        logger.error(f"Error reading index file {file_path}: {e}")

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Please provide a ChromaDB directory path")
        logger.info("Usage: python inspect_chroma_files.py <chroma_db_path>")
        sys.exit(1)
    
    chroma_dir = sys.argv[1]
    if not inspect_directory(chroma_dir):
        sys.exit(1) 