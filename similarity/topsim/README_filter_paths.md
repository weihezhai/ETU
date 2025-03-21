# Filter Paths by Top Relations

This script filters paths for each question in a JSONL file based on the top k relations from a sorted relations JSON file.

## Requirements

- Python 3.6+
- tqdm (install with `pip install tqdm`)

## Usage

```bash
python filter_paths_by_relation.py --source <source_file> --relations <sorted_relations_file> --output <output_file> [--k <top_k>]
```

### Arguments

- `--source`: Path to source JSONL file containing questions and paths
- `--relations`: Path to JSON file with sorted relations
- `--output`: Path to save the filtered results
- `--k`: Number of top relations to use as filters (default: 10)

## Input Format

### Source JSONL file

Each line should contain a JSON object with the following format:

```json
{
  "id": "WebQTest-1384_744a496b907e407b16bc5d7c197dc3f0",
  "question": "What is the predominant religion where the leader is Ovadia Yosef?",
  "prediction": "Judaism",
  "paths": [["Ovadia Yosef", "person.religion", "Judaism"]]
}
```

### Sorted Relations JSON file

A JSON object mapping question IDs to lists of sorted relations:

```json
{
  "WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2": [
    "location.adjoining_relationship.adjoins",
    "location.location.adjoin_s",
    "common.topic.notable_types",
    ...
  ],
  ...
}
```

## Output Format

The output is a JSONL file with the same structure as the source file, but with an additional field `filtered_path_by_relation_similarity` containing only paths that include one of the top k relations. If no paths match the top k relations, the original paths are used instead. This ensures that `filtered_path_by_relation_similarity` is never empty.

```json
{
  "id": "WebQTest-1384_744a496b907e407b16bc5d7c197dc3f0",
  "question": "What is the predominant religion where the leader is Ovadia Yosef?",
  "prediction": "Judaism",
  "paths": [["Ovadia Yosef", "person.religion", "Judaism"]],
  "filtered_path_by_relation_similarity": [["Ovadia Yosef", "person.religion", "Judaism"]]
}
```

## Example

```bash
python filter_paths_by_relation.py \
  --source ETU/fppl/no_middle_entity.jsonl \
  --relations sorted_relation.json \
  --output filtered_paths.jsonl \
  --k 20
```

This will use the top 20 relations from sorted_relation.json to filter paths in no_middle_entity.jsonl and save the results to filtered_paths.jsonl.