#!python

import json
from staticjinja import Site
import os
import base64

doc_dir = os.path.join(os.getcwd(), "doc")
out_dir = os.path.join(os.getcwd(), "out")

TESTS = True


if __name__ == "__main__":

    fn_list = []

    base64_dict = {}

    for kernel in os.listdir(out_dir+"/images"):
        for file in os.listdir(out_dir+"/images/"+kernel):
            if file.endswith(".png"):
                base64_url = "data:image/png;base64," + \
                    str(base64.b64encode(
                        open(out_dir+"/images/" + kernel + "/" + file, "rb").read()))[2:-1]

                img_tag = "<img src=\""+base64_url + \
                    "\" class=\"d-block tree-image\" alt=\""+file+"\"/>"
                base64_dict[kernel+"_"+file.replace(".png", "")] = img_tag

    for file in os.listdir(out_dir):
        # Check whether file is in text format or not

        if file.endswith(".json") and (file.startswith("ext") or (file.startswith("test") and TESTS)):
            # call read text file function
            with open(out_dir+'/'+file, 'r') as f:
                try:
                    fn_list.append(json.load(f))
                except:
                    print("Invalid json file:", file)

    fn_list.sort(key=lambda x: x['name'])

    print(len(base64_dict.keys()))
    site = Site.make_site(env_globals={
        'functions': fn_list,
        "indmap": ["k", "j", "i"],
        "base64_dict": base64_dict
    }, outpath=doc_dir)
    site.render(use_reloader=False)
