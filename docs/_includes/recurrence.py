def subset_sums(spectrum):
    return {s: phase_sum(spectrum, s) for s in labels}

def append_gate(bounds, spectrum):
    gate = subset_sums(spectrum)
    return {
        out: max(bounds[left] + gate[right] - degree
                 for left, right, target, degree in rules if target == out)
        for out in labels
    }

mixed_bounds = append_gate(subset_sums(a_phases), b_phases)
assert mixed_bounds["xy"] == Fraction(1, 6)
assert mixed_bounds["zw"] == -Fraction(5, 12)
print("AB:", mixed_bounds["xy"], "<= c1 <=", -mixed_bounds["zw"])
