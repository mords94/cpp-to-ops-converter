#!/usr/bin/env python3

import sys
import json
import subprocess
import os
out_dir = os.path.join(os.getcwd(), "out/images")

lines: str = ""
for line in sys.stdin:
    lines += line


try:
    readed_json = json.loads(lines)
except json.decoder.JSONDecodeError as e:
    print("Json Error")
    sys.exit(1)

subgraphs = []
labels = []


def get_subgraph_for_debugs(readed_json):
    for index, loop in enumerate(readed_json['loops']):
        for arg in loop['args']:
            if ('read_offsets' in arg):
                for read_index, read_offsets in enumerate(arg['read_offsets']):
                    label = "_".join(
                        ['read', str(index), arg['name'], str(read_index)])
                    subgraph_dot = 'subgraph ' + label + ' {\n'
                    subgraph_dot += read_offsets['debug']
                    subgraph_dot += '\n}'
                    subgraphs.append(subgraph_dot)
                    labels.append(label)
            if ('write_offsets' in arg):
                for write_index, write_offsets in enumerate(arg['write_offsets']):
                    label = "_".join(
                        ['write', str(index), arg['name'], str(write_index)])

                    subgraph_dot = 'subgraph ' + label + ' {\n'
                    subgraph_dot += 'label = "'+label+'";\n'

                    subgraph_dot += write_offsets['debug']
                    subgraph_dot += '\n}'
                    subgraphs.append(subgraph_dot)
                    labels.append(label)


get_subgraph_for_debugs(readed_json)

# print('graph G {')
# print("\n".join(subgraphs))
# print('}')


for index, g in enumerate(subgraphs):
    input_graph = 'graph G {\n' + g + '\n}'
    name_dir = out_dir+"/"+readed_json['name']

    if (not os.path.exists(name_dir)):
        os.mkdir(name_dir)

    p = subprocess.run(['dot', "-Tpng", "-o"+name_dir+"/"+labels[index]+".png"], input=input_graph,
                       capture_output=True, text=True)

    if (p.returncode != 0):
        print(p.stderr)
