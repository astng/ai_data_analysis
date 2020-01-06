#!/bin/bash
for i in iron chromium aluminium copper lead nickel silver tin titanium vanadium cadmium manganese pq_index oxidation sulfation nitration soot_percentage
do
    python3 -W 'ignore' apps/creating_essay_dataset.py --database astng_stats --table full_data --essay $i
done
