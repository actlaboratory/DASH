# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
import sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
prod = {}
for n in g.node:
    for o in n.output: prod[o] = n
inits = {i.name for i in g.initializer}
dec = [n for n in g.node if n.name.startswith("/dec/")]
decout = {o for n in dec for o in n.output}
ext = set()
for n in dec:
    for i in n.input:
        if i and i not in decout and i not in inits:
            ext.add(i)
for e in ext:
    p = prod.get(e)
    print("dec external input:", e, "<-", p.op_type if p else "GRAPH", p.name if p else "", list(p.input) if p else "")
print("first dec nodes:"); [print(" ", n.name, n.op_type, list(n.input)[:3]) for n in dec[:6]]
print("last dec nodes:"); [print(" ", n.name, n.op_type, list(n.input)[:2], list(n.output)) for n in dec[-3:]]
# who consumes dec outputs outside dec
for n in g.node:
    if not n.name.startswith("/dec/") and any(i in decout for i in n.input):
        print("post-dec:", n.name, n.op_type, list(n.input), list(n.output))
# show the chain feeding the dec input (2 levels)
for e in ext:
    p = prod.get(e)
    if p:
        for i in p.input:
            q = prod.get(i); print("   ", e, "input", i, "<-", q.op_type if q else "?", q.name if q else "")
