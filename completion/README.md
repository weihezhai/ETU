 # Completion Folder Documentation

## Overview
The completion folder contains a set of files designed to process and map graph triples from IDs to human-readable text and entity names. This system appears to be part of a knowledge graph processing pipeline, potentially for triple completion or knowledge graph enrichment tasks.

## Files Documentation

### 1. mapper.py
**Purpose:** A Python script for mapping graph triples from IDs to readable text and entity names.

**Key Functionality:**
- Processes graph data in JSONL format
- Maps entity and relation IDs to human-readable text
- Supports multiple mapping sources for entity name resolution
- Handles Freebase ID mapping for comprehensive entity resolution

**Main Components:**
- Loading functions for different mapping file formats
- Triple mapping functions to convert IDs to text
- Entity name resolution with fallback mechanisms
- JSONL processing with output generation

**Usage:** Called by the script with parameters for input/output files and mapping files.

### 2. mapper.sh
**Purpose:** A shell script that serves as a wrapper to execute the mapper.py script in a computing environment.

**Key Functionality:**
- Sets up environment variables and paths
- Configures computing resources (cores, memory, GPU)
- Executes the mapper.py script with predefined parameters
- Provides status reporting on execution success

**Configuration:**
- Configured for a cluster computing environment with GPU support
- Sets default file paths for various mapping files
- Activates a Python virtual environment for execution

### 3. mapper.md
**Purpose:** Documentation file explaining how the mapper system works.

**Key Content:**
- Comprehensive explanation of the triple mapping process
- Detailed usage instructions for the Python script
- Description of mapping resolution order and logic
- Explanation of input/output file formats
- Instructions for using the shell script wrapper

## System Purpose
This system is designed to enrich knowledge graph data by converting machine-readable IDs to human-readable entity names and relation descriptions, supporting knowledge graph completion or analysis tasks.