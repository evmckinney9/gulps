from itertools import combinations

labels = ["".join(s) for r in (1, 2, 3) for s in combinations("xyzw", r)]
rules = []
for cycle in (("w", "z", "y", "x"), ("yzw", "xzw", "xyw", "xyz")):
    for i, left in enumerate(cycle):
        for j, right in enumerate(cycle):
            rules.append((left, right, cycle[(i + j) % 4], (i + j) // 4))

rank_two = """
zw zw zw 0
zw yw yw 0
zw yz yz 0
zw xw xw 0
zw xz xz 0
zw xy xy 0
yw yw yz 0
yw yw xw 0
yw yz xz 0
yw xw xz 0
yw xz xy 0
yw xz zw 1
yw xy yw 1
yz yz xy 0
yz xw zw 1
yz xz yw 1
yz xy xw 1
xw xw xy 0
xw xz yw 1
xw xy yz 1
xz xz xw 1
xz xz yz 1
xz xy xz 1
xy xy zw 2
"""
for row in rank_two.strip().splitlines():
    left, right, out, degree = row.split()
    rules.append((left, right, out, int(degree)))
    if left != right:
        rules.append((right, left, out, int(degree)))
assert len(rules) == 72
