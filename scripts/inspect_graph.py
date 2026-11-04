# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import sys, onnx, collections
m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
print("opset", [(o.domain, o.version) for o in m.opset_import])
for i in g.input: print("IN", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim], i.type.tensor_type.elem_type)
for o in g.output: print("OUT", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
print("nodes", len(g.node))
# group node names by prefix
c = collections.Counter("/".join(n.name.split("/")[:2]) for n in g.node)
for k,v in c.most_common(40): print(k, v)
