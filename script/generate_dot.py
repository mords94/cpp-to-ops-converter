# #!/usr/bin/env python3


# debug = [
#     {
#         "_treeDepth": 0,
#         "children": [
#             {
#                 "_treeDepth": 2,
#                 "children": [
#                     {
#                         "_treeDepth": 1,
#                         "children": [
#                             {
#                                 "_treeDepth": 2,
#                                 "children": [
#                                     {
#                                         "_ownerLoopDescriptor": {
#                                             "depth": "1 : i8",
#                                             "lb": "0 : i32",
#                                             "ub": "@jm"
#                                         },
#                                         "_parentOp": "<block argument> of type 'i32' at index: 0",
#                                         "_treeDepth": 3,
#                                         "text": "<block argument> of type 'i32' at index: 0"
#                                     }
#                                 ],
#                                 "text": "%3 = arith.addi %arg2, %c-1_i32 : i32"
#                             },
#                             {
#                                 "_treeDepth": 3,
#                                 "text": "%c-1_i32 = arith.constant -1 : i32"
#                             }
#                         ],
#                         "text": "%13 = arith.addi %3, %12 : i32"
#                     },
#                     {
#                         "_treeDepth": 3,
#                         "text": "%c2_i32 = arith.constant 2 : i32"
#                     }
#                 ],
#                 "text": "%12 = arith.muli %c2_i32, %arg3 : i32"
#             },
#             {
#                 "_treeDepth": 3,
#                 "text": "%1 = memref.get_global @im : memref<1xi32>"
#             }
#         ],
#         "text": "%16 = arith.addi %13, %15 : i32"
#     }
# ]


# def resolve_text(text):
#     if (text.find("arith.addi") != -1):
#         return "+"
#     if (text.find("arith.muli") != -1):
#         return "*"
#     if (text.find("arith.constant") != -1):
#         return text
#     if (text.find("<block argument>") != -1):
#         return text
#     if (text.find("memref.get_global") != -1):
#         return text.split("get_global ")[1]
#     else:
#         return 'unknown + ' + text


# def edge(fromm, to):
#     return "n"+str(fromm) + " -> " + "n" + str(to) + ";"


# def vertex(node, index):
#     return "n"+str(index)+"[label=\""+resolve_text(node['text']) + "\"];"


# def generate_dot(root, index=0):
#     print(vertex(root, index))
#     if ('children' not in root):
#         return

#     for i, child in enumerate(root['children']):
#         print(edge(index, index+i+1))
#         generate_dot(child, index+i+1)


# print("digraph G {")
# generate_dot(debug[0])
# print("}")
