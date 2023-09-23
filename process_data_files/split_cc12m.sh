# Count number of lines
num_lines=$(wc -l cc12m.tsv | awk '{print $1}')

# Calculate lines per file 
lines_per_file=$((num_lines / 12))

# Split file into 12 parts
split -l $lines_per_file cc12m.tsv 

# Rename output files
counter=1
for f in x*; do
    new_name="cc12m_part$counter.tsv" 
    mv "$f" "$new_name"
    let counter=counter+1 
done
