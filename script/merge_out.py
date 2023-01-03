#!/usr/bin/env python3

import os
import json

out_dir = os.path.join(os.getcwd(), "out")

list = []
for file in os.listdir(out_dir):
    # Check whether file is in text format or not
    if file.endswith(".json") and file.startswith("ext"):
        # call read text file function
        with open(out_dir+'/'+file, 'r') as f:
            try:
                list.append(json.load(f))
            except:
                print("Invalid json file")

json.dump(list, open(out_dir+"/MERGED.json", "w"), indent=2)
