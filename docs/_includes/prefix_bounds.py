bounds = subset_sums(a_phases)
prefix_bounds = [bounds]
for spectrum in (a_phases, b_phases):
    bounds = append_gate(bounds, spectrum)
    prefix_bounds.append(bounds)

assert prefix_bounds[2]["xz"] == -Fraction(1, 6)

for depth, bounds in enumerate(prefix_bounds, 1):
    print(f"{depth} gate(s): {bounds['xy']} <= c1 <= {-bounds['zw']}")
