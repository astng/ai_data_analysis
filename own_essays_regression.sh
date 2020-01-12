#!/bin/bash
for bd in astng collahuasi centinela
do
    for i in iron chromium aluminium copper lead nickel silver tin titanium vanadium cadmium manganese pq_index oxidation sulfation nitration
    do
        python3 -W 'ignore' apps/own-lstm-with-categorical.py --essay $i --dataset_training datasets/${bd}/${i}_dataset.h5 --client $bd
    done
done
