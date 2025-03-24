#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8      # 32 cores
#$ -l h_rt=12:0:0  # 240 hours runtime
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -m be
#$ -l gpu=1         # request 4 GPUs
## $ -l h=sbg4
#$ -l rocky
# $ -l cluster=andrena   
#$ -N mapper

# module load gcc
source /data/home/mpx602/projects/py311/bin/activate
# Default values
GRAPH_FILE="/data/home/mpx602/projects/ETU/ETU/completion/testset_all_triples_NSM.json"
ENTITY_FILE="/data/home/mpx602/projects/ETU/ETU/completion/entities.txt"
RELATION_FILE="/data/home/mpx602/projects/ETU/ETU/completion/relations.txt"
ID_TO_ENTITY_FILE="/data/home/mpx602/projects/ETU/ETU/completion/entities_names.json"
FREEBASE_TSV_FILE="/data/home/mpx602/projects/ETU/ETU/completion/fb_wiki_mapping.tsv"
OUTPUT_FILE="/data/home/mpx602/projects/ETU/ETU/completion/testset_mapped_output.json"

echo "Running: python mapper.py $GRAPH_FILE $ENTITY_FILE $RELATION_FILE $ID_TO_ENTITY_FILE $FREEBASE_TSV_FILE $OUTPUT_FILE"

# Run the mapper
python mapper.py "$GRAPH_FILE" "$ENTITY_FILE" "$RELATION_FILE" "$ID_TO_ENTITY_FILE" "$FREEBASE_TSV_FILE" "$OUTPUT_FILE"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Mapping completed successfully"
    echo "Output saved to: $OUTPUT_FILE"
else
    echo "Error: Mapping failed"
fi