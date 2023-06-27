# clear command-line parameters
set --
source tasks/pretty_log.sh

pretty_log "Extracting video transcripts"
modal run etl/videos.py --csv-path data/output-2023-06-22T03:37:39.362Z.csv"
