# Triple Mapper with Multi-Level Entity Name Resolution

This script processes graph triples by mapping node IDs to entity names using multiple sources:

1. ID triples to text triples using line-number-based mapping files
2. Node IDs to entity names using a JSON ID-to-entity mapping
3. Remaining unresolved node IDs to entity names using a Freebase TSV mapping

## Usage

```bash
python mapper.py <graph_file> <entity_file> <relation_file> <id_to_entity_file> [--freebase_tsv <freebase_tsv_file>] <output_file>
```

Where:
- `<graph_file>`: Path to the JSONL file containing graph triples (one JSON object per line)
- `<entity_file>`: Path to the text file containing entity mappings (line number = entity ID)
- `<relation_file>`: Path to the text file containing relation mappings (line number = relation ID)
- `<id_to_entity_file>`: Path to the JSON file mapping node IDs (like "m.02ht472") to entity names
- `--freebase_tsv <freebase_tsv_file>`: (Optional) Path to TSV file with freebase_id, wikidata_id, and label columns
- `<output_file>`: Path to save the output JSONL file with mapped triples

## Freebase TSV Format

The TSV file should have the format:
```
freebase_id    wikidata_id    label
/m/0154j    Q31    Belgium
/m/016pp7    Q8    happiness
```

The script extracts the ID part after "/m/" and uses it to match with the part after "m." in node IDs.

## Resolution Order

The script attempts to resolve node IDs in this order:
1. First try the id_to_entity JSON mapping
2. If not found, try the Freebase TSV mapping (matching the ID part)
3. If still not found, keep the original ID

## Output Format

The script preserves the original JSON structure and adds:
- `text_tuples`: Triples with node IDs and relation text
- `named_tuples`: Triples with human-readable entity names from both mapping sources

## Using the Shell Script

For convenience, you can use the included shell script:

```bash
bash mapper.sh
```

You can edit the script to change the default file paths.