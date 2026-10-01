.. meta::
   :description: Cartan-double eigenphases, Horn and quantum Littlewood-Richardson inequalities, and the max-plus recurrence for the reachable region of a gate sequence.

The reachability calculation
=============================

GULPS describes the reachable region of a sentence by fourteen lower
bounds. Each bound applies to the sum of one nonempty proper subset of the
four eigenphases of the Cartan double, defined in the next section.
Appending a native gate to the sentence updates the fourteen bounds with a
fixed table of 72 quantum Littlewood–Richardson rules.

The calculation extends a two-gate result of Peterson, Crooks, and Smith,
`Two-Qubit Circuit Depth and the Monodromy Polytope
<https://doi.org/10.22331/q-2020-03-26-247>`_, to sentences of any length.

.. details:: Shared mathematical setup

   .. jupyter-execute::
      :hide-output:

      from fractions import Fraction

      import matplotlib.pyplot as plt
      import numpy as np
      from qiskit.circuit.library import CSXGate, CXGate, iSwapGate
      from gulps.analysis.region import ReachableRegion
      from gulps.invariants import LocalEquivalenceClass

      cx = LocalEquivalenceClass.from_unitary(CXGate())
      sqrt_iswap = LocalEquivalenceClass.from_unitary(iSwapGate().power(1 / 2))
      native_a = LocalEquivalenceClass.from_unitary(CSXGate())
      native_b = LocalEquivalenceClass.from_unitary(iSwapGate().power(1 / 3))

.. _cartan-double:

Coordinates in which products are tractable
-------------------------------------------

Weyl coordinates describe a single gate, but they give no formula for
the class of a product of two gates. Eigenvalues have such a theory,
going back to Horn, which describes the possible eigenvalues of a product
of two matrices whose eigenvalues are fixed. The eigenvalues of a
two-qubit gate, however, are not invariants of its class. Local gates
:math:`L` and :math:`R` around :math:`U` change its eigenvalues in
general, because :math:`R` need not equal :math:`L^{-1}`.

Real matrices have the same difficulty, and singular values resolve it.
Orthogonal factors in :math:`O_LXO_R` change the eigenvalues of :math:`X`
but not those of :math:`XX^T`, which are the squared singular values of
:math:`X`. The *Cartan double* applies this construction to two-qubit
gates.

In the *magic basis*, a phased Bell basis, local special unitaries become
real orthogonal matrices. Fix :math:`\det U=1`
(:ref:`global-phase-representatives` treats the choices) and write
:math:`U_B` for :math:`U` in this basis. Its KAK decomposition is
:math:`U_B=O_L D O_R`, with :math:`D` diagonal and :math:`O_L,O_R\in SO(4)`, and the Cartan double is

.. math::

   M(U)=U_B U_B^T
       =O_L D O_R O_R^T D O_L^T
       =O_L D^2 O_L^T.

The superscript :math:`T` is a plain transpose, not a conjugate
transpose, so :math:`O_R O_R^T=I` removes the right local factor. The
left factor only conjugates :math:`D^2` and leaves its eigenvalues
unchanged. The spectrum of :math:`M(U)` therefore depends only on the class of the representative.

Write the eigenvalues as
:math:`e^{2\pi i x},e^{2\pi i y},e^{2\pi i z},e^{2\pi i w}`. The phases
are measured in turns, so adding an integer to one of them does not change
its eigenvalue. The *fundamental alcove* picks one representative for
each spectrum:

.. math::

   x\ge y\ge z\ge w,\qquad x+y+z+w=0,\qquad x-w\le1.

The first three phases :math:`(x,y,z)` are the
*monodromy coordinates* of the representative. The fourth phase,
:math:`w=-x-y-z`, stays in the notation because the inequalities treat
all four phases alike.

The phases are linear in the Weyl coordinates
:math:`c=(c_1,c_2,c_3)`. Since :math:`XX`, :math:`YY`, and :math:`ZZ`
commute, :math:`\operatorname{CAN}(c)` is diagonal in the Bell basis. On
each Bell state :math:`\operatorname{CAN}(c)` multiplies by
:math:`e^{i\pi(\pm c_1\pm c_2\pm c_3)/2}`, with the signs given by that
state's :math:`XX`, :math:`YY`, and :math:`ZZ` eigenvalues, and the Cartan
double squares this factor. The four phases are therefore half of
:math:`\pm c_1\pm c_2\pm c_3` for the four Bell sign patterns. When they
satisfy the alcove conditions, each Weyl coordinate is the sum of two
phases, so a bound on a pair sum of phases reads directly as a bound on a
Weyl coordinate:

.. math::

   c_1=x+y,\qquad c_2=x+z,\qquad c_3=y+z,

.. math::

   (x,y,z,w)=\frac12(
      c_1+c_2-c_3,\ c_1-c_2+c_3,\ -c_1+c_2+c_3,\ -c_1-c_2-c_3).

.. jupyter-execute::

   def phases(weyl):
       c1, c2, c3 = weyl
       return np.array([
           c1 + c2 - c3, c1 - c2 + c3,
           -c1 + c2 + c3, -c1 - c2 - c3,
       ]) / 2

   print("√iSWAP phases:", phases(sqrt_iswap.weyl).round(3))

.. _global-phase-representatives:

Two global-phase representatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A gate has four *global-phase representatives*, the determinant-one
matrices :math:`U` divided by each fourth root of :math:`\det U`. They
differ by the factors :math:`1,i,-1,-i`. The
transpose does not conjugate a scalar, so :math:`M(sU)=s^2M(U)`.
Multiplying by :math:`-1` leaves the Cartan double unchanged, while
multiplying by :math:`i` negates it. The four representatives therefore
give two spectra, and a reachability test must accept either one.

Negating the double shifts every eigenphase by :math:`1/2` modulo
integers. Returning the shifted phases to the alcove gives the reflection

.. math::

   \rho(x,y,z,w)=
      (z+1/2,\ w+1/2,\ x-1/2,\ y-1/2).

.. details:: Math detail: the reflected phases lie in the alcove

   The reflected phases are ordered because :math:`x-w\le1`, and their
   spread :math:`1+z-y` is at most one. Applying :math:`\rho` twice
   returns the original spectrum.

In Weyl coordinates the reflection reads

.. math::

   \rho(c_1,c_2,c_3)=(1-c_1,c_2,-c_3).

.. jupyter-execute::

   def rho(spectrum):
       x, y, z, w = spectrum
       return np.array([z + 0.5, w + 0.5, x - 0.5, y - 0.5])

   identity = phases((0, 0, 0))
   print("Identity spectra:", identity, rho(identity))

A sentence of :math:`n` gates has :math:`4^n` choices of global-phase
representatives, four per gate. The factors :math:`1,i,-1,-i` collect into
one power of :math:`i` on the product, and only its parity changes the
product's Cartan double. The choices therefore reach two regions, related
by :math:`\rho`. On this page, the *direct representative* of a sentence
takes, for each gate, the representative whose phases the
Weyl-coordinate formula above gives, and the *reflected representative* is its :math:`\rho`
partner. A target class is reachable, up to global phase, when its
spectrum or its :math:`\rho` partner lies in the region of the direct
representative. :meth:`~gulps.analysis.region.ReachableRegion.reaches`
checks the target against both ``region`` and ``region.rho``.

Two gates as a multiplicative Horn problem
------------------------------------------

The Cartan double of a two-gate circuit has the same spectrum as a
product of the two gates' doubles, with one factor rotated by the middle
local layer. The reach of a two-gate sentence is therefore a question
about the eigenvalues of products.

Absorb the local factors at the ends of each native gate into the
adjacent free local layers. In the magic basis, the nonlocal part
of the circuit then has the form :math:`V=D_b O D_a`, where
:math:`D_a,D_b` are diagonal canonical gates and :math:`O\in SO(4)`
represents the middle local layer. Its Cartan double is

.. math::

   M(V)=D_b O D_a^2 O^T D_b,

and conjugating it by :math:`D_b` gives

.. math::

   D_b M(V)D_b^{-1}=D_b^2 O D_a^2 O^T.

Similarity preserves eigenvalues, so :math:`M(V)` has the spectrum of
:math:`D_b^2` times :math:`O D_a^2 O^T`. Both factors have fixed spectra,
and the middle layer :math:`O` sets only their relative eigenvectors.

The *multiplicative Horn problem* asks which spectra a product of two
matrices can have when the eigenvalues of each factor are fixed and the
eigenvectors are free. Horn's original problem asks the same question for
sums of Hermitian matrices. Peterson, Crooks, and Smith call the answer
for special unitaries the *monodromy polytope*. Its points are the triples
:math:`(a,b,\delta)` of alcove points such that some special unitaries
with eigenphases :math:`a` and :math:`b` have a product with eigenphases
:math:`\delta`. Their Theorem 23 lists its faces as linear inequalities,
one per entry of a fixed table derived below. On the rest of this page, :math:`\delta`
denotes the spectrum of a product.

The problem lets the relative eigenbasis be any *unitary*, but a local
middle layer supplies only a *real orthogonal* one. Peterson, Crooks, and
Smith's `Theorem 13 and Corollary 14
<https://quantum-journal.org/papers/q-2020-03-26-247/pdf/#page=7>`_ prove
that the restriction loses no spectra. Corollary 25 combines the two: a two-gate sentence
reaches a target exactly when the target's spectrum, or its :math:`\rho`
partner, satisfies Theorem 23.

.. details:: Math detail: orientation-reversing middle layers

   An orthogonal :math:`O` with determinant :math:`-1` is not a local
   gate. Replacing :math:`O` by :math:`OJ`, with
   :math:`J=\operatorname{diag}(-1,1,1,1)`, fixes the determinant, and because :math:`J` commutes with the diagonal
   :math:`D_a^2`, it leaves :math:`O D_a^2 O^T` unchanged. A local layer
   therefore reaches the same product spectra as any orthogonal matrix.

A warm-up with 2×2 matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For 2-by-2 matrices the eigenvector freedom can be eliminated by hand.
The answer already has the shape of the general result, a region bounded
by linear inequalities. Here :math:`U` and :math:`V` are 2×2 special
unitaries standing in for Cartan doubles.

Let :math:`U` and :math:`V` have eigenvalues :math:`e^{\pm2\pi i a}` and
:math:`e^{\pm2\pi i b}`, with :math:`a,b\in[0,1/2]`. In terms of unit
vectors :math:`\mathbf n,\mathbf m` and the Pauli matrices,

.. math::

   U&=\cos(2\pi a)I+i\sin(2\pi a)\,\mathbf n\cdot\boldsymbol\sigma,\\
   V&=\cos(2\pi b)I+i\sin(2\pi b)\,\mathbf m\cdot\boldsymbol\sigma.

The axes :math:`\mathbf n` and :math:`\mathbf m` are the eigenvectors of
the two matrices, drawn as directions on the Bloch sphere. The product has
eigenvalues :math:`e^{\pm2\pi i\delta}`, with :math:`\delta\in[0,1/2]`.
Since :math:`(\mathbf n\cdot\boldsymbol\sigma)(\mathbf m\cdot\boldsymbol\sigma)
=(\mathbf n\cdot\mathbf m)I+i(\mathbf n\times\mathbf m)\cdot\boldsymbol\sigma`
and the Pauli matrices are traceless, half the trace of the product is

.. math::

   \cos(2\pi\delta)=\cos(2\pi a)\cos(2\pi b)
       -s\sin(2\pi a)\sin(2\pi b),
   \qquad s=\mathbf n\cdot\mathbf m\in[-1,1].

The eigenvectors enter the product spectrum only through :math:`s`, the
cosine of the angle between the two axes, and every :math:`s` in
:math:`[-1,1]` occurs. Cosine is monotone on
:math:`[0,\pi]`, so the two extreme orientations bound :math:`\delta`:

.. math::

   |a-b|\le\delta\le\min(a+b,\,1-a-b).

Written as linear inequalities on the triple :math:`(a,b,\delta)`, these
are the inequalities of Peterson, Crooks, and Smith's `Example 24
<https://quantum-journal.org/papers/q-2020-03-26-247/pdf/#page=10>`_:

.. math::

   \delta\ge a-b,\qquad \delta\ge b-a,\qquad
   \delta\le a+b,\qquad \delta\le1-a-b.

The last inequality comes from phases wrapping around the circle. When
:math:`a+b` exceeds :math:`1/2`, the eigenvalue pair
:math:`e^{\pm2\pi i(a+b)}` equals :math:`e^{\pm2\pi i(1-a-b)}`.

.. jupyter-execute::

   a_demo, b_demo = 1 / 8, 1 / 6
   orientations = np.linspace(-1, 1, 501)
   cosine = (
       np.cos(2 * np.pi * a_demo) * np.cos(2 * np.pi * b_demo)
       - orientations * np.sin(2 * np.pi * a_demo) * np.sin(2 * np.pi * b_demo)
   )
   output_phase = np.arccos(np.clip(cosine, -1, 1)) / (2 * np.pi)
   expected_interval = (abs(a_demo - b_demo), min(a_demo + b_demo, 1 - a_demo - b_demo))

.. jupyter-execute::
   :hide-code:
   :hide-output:

   np.testing.assert_allclose(output_phase[[0, -1]], expected_interval, atol=1e-12, rtol=0)

.. details:: Plotting code

   .. jupyter-execute::
      :hide-output:

      lo, hi = expected_interval
      angle = np.degrees(np.arccos(orientations))
      fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
      ax.plot(angle, output_phase, color="tab:blue", linewidth=2)
      ax.axhspan(lo, hi, color="tab:blue", alpha=0.12)
      ax.annotate(r"parallel axes: $\delta=a+b$", (0, hi), xytext=(8, 4),
                  textcoords="offset points", fontsize=9)
      ax.annotate(r"antiparallel axes: $\delta=|a-b|$", (180, lo), xytext=(-8, 6),
                  textcoords="offset points", ha="right", fontsize=9)
      ax.set_xticks([0, 45, 90, 135, 180], ["0°", "45°", "90°", "135°", "180°"])
      ax.set_yticks([0, lo, 1 / 8, 1 / 4, hi], ["0", "1/24", "1/8", "1/4", "7/24"])
      ax.set_ylim(0, 0.33)
      ax.set(xlabel="Angle between the two rotation axes",
             ylabel=r"Product phase $\delta$")
      for side in ("top", "right"):
          ax.spines[side].set_visible(False)
      plt.close(fig)

.. jupyter-execute::
   :hide-code:
   :alt: Product phase δ against the angle between the two rotation axes, for a = 1/8 and b = 1/6. δ falls from a+b = 7/24 at parallel axes (0°) to |a−b| = 1/24 at antiparallel axes (180°); the shaded band marks every reachable δ.

   display(fig)

Each factor rotates the Bloch sphere by a fixed amount, and the middle
layer only turns one rotation axis relative to the other. The rotations
add when the axes are parallel and subtract when they are antiparallel.

With four eigenvalues the relative eigenbasis has too many parameters to
eliminate by hand. Theorem 23 of Peterson, Crooks, and Smith does the
elimination in general, and its answer is again a list of linear
inequalities, now on sums of phases.

.. _subset-sum-bounds:
.. _upper-bounds:

From one inequality to fourteen bounds
--------------------------------------

For a concrete pair, take :math:`A=\sqrt{\mathrm{CX}}` followed by
:math:`B=\sqrt[3]{\mathrm{iSWAP}}`. Their Weyl coordinates are
:math:`(1/4,0,0)` and :math:`(1/6,1/6,0)`, so their Cartan-double phases
are

.. math::

   a=(1/8,1/8,-1/8,-1/8),\qquad b=(1/6,0,0,-1/6).

Each inequality of Theorem 23 bounds
a sum of output phases from below by a sum of input phases. For a subset :math:`I` of the phase positions
:math:`\{x,y,z,w\}`, let :math:`f_I(a)=\sum_{i\in I}a_i`. For example,
:math:`f_{yw}(a)=a_y+a_w`. Every inequality then has the form

.. math::

   f_K(\delta)\ge f_I(a)+f_J(b)-d.

The sum of output phases over :math:`K` is at least the sum of the first
gate's phases over :math:`I` plus the sum of the second gate's phases
over :math:`J`, less an integer :math:`d\ge0`. This integer, the
*quantum degree*, accounts for phases wrapping around the circle; the
2×2 bound :math:`\delta\le1-a-b` is a rule of degree one. The
triples of subsets and their degrees come from a fixed table of 72 rules
:math:`(I,J)\to(K,d)`. The table does not depend on the gates, which
supply only the phase values.

Take the rule ``yw + xz -> xy``, which has :math:`d=0`. It gives

.. math::

   \delta_x+\delta_y\ge(a_y+a_w)+(b_x+b_z)=0+1/6=1/6.

Five other rules also end at :math:`xy`, each with its own lower bound.

.. jupyter-execute::

   def phase_sum(spectrum, subset):
       return sum(spectrum["xyzw".index(p)] for p in subset)

   a_phases = (Fraction(1, 8), Fraction(1, 8), -Fraction(1, 8), -Fraction(1, 8))
   b_phases = (Fraction(1, 6), 0, 0, -Fraction(1, 6))

   def candidates(rows):
       return {(i, j, d): phase_sum(a_phases, i) + phase_sum(b_phases, j) - d
               for i, j, d in rows}

   xy_rows = [("xy", "zw", 0), ("zw", "xy", 0), ("yw", "xz", 0),
              ("xz", "yw", 0), ("yz", "yz", 0), ("xw", "xw", 0)]
   xy_candidates = candidates(xy_rows)
   beta_xy = max(xy_candidates.values())
   for (i, j, d), value in xy_candidates.items():
       print(f"f_{i}(a) + f_{j}(b) = {value}")
   print("Strongest:", beta_xy)

All six inequalities hold at once, so their maximum replaces them with
one equivalent constraint,

.. math::

   x+y\ge\max(1/12,-1/12,1/6,-1/6,0,0)=1/6.

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert beta_xy == Fraction(1, 6)
   assert sorted(xy_candidates.values()) == [Fraction(k, 12) for k in (-2, -1, 0, 0, 1, 2)]

.. details:: Plotting code

   .. jupyter-execute::
      :hide-output:

      def fraction_text(value):
          value = Fraction(value)
          sign = "−" if value < 0 else ""
          if value.denominator == 1:
              return sign + str(abs(value.numerator))
          return f"{sign}{abs(value.numerator)}/{value.denominator}"

      ordered = sorted(xy_candidates.items(), key=lambda item: -item[1])
      zoom_figure, ax = plt.subplots(figsize=(6.8, 2.9), layout="constrained")
      ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
      ax.hlines(0, 1 / 6, 5 / 12, color="tab:blue", linewidth=6, capstyle="round", zorder=1)
      ax.text(5 / 12, -0.22, r"$x+y\leq 5/12$", ha="center", va="top",
              fontsize=9, color="tab:blue")
      for row, ((i, j, d), value) in enumerate(ordered):
          best = value == beta_xy
          color = "crimson" if best else "0.45"
          height = 0.3 + 0.24 * row
          ax.plot([value, value], [0, height], color=color, linewidth=0.8, linestyle=":")
          ax.plot(value, 0, "o", color=color, markersize=7, zorder=3)
          ax.text(value + 0.006, height,
                  rf"$f_{{{i}}}(a)+f_{{{j}}}(b)={fraction_text(value)}$",
                  va="center", fontsize=9, color=color)
      ax.text(1 / 6, -0.22, r"$x+y\geq 1/6$", ha="center", va="top",
              fontsize=9, color="crimson")
      ax.set_ylim(-0.55, 0.3 + 0.24 * len(ordered))
      ax.set_yticks([])
      ax.set_xticks([-1 / 6, -1 / 12, 0, 1 / 12, 1 / 6, 5 / 12],
                    ["−1/6", "−1/12", "0", "1/12", "1/6", "5/12"])
      ax.set_xlim(-0.22, 0.5)
      for side in ("top", "right", "left"):
          ax.spines[side].set_visible(False)
      ax.set_xlabel(r"Output phase sum $x+y$ after AB")
      plt.close(zoom_figure)

.. jupyter-execute::
   :hide-code:
   :alt: The x+y axis for the sentence √CX followed by ∛iSWAP. Six dots mark the six candidate lower bounds −1/6, −1/12, 0, 0, 1/12, and 1/6, each labelled with its sum of input phases. The largest, f_yw(a)+f_xz(b)=1/6, is red and is the left end of the blue interval of reachable x+y values, which runs to 5/12.

   display(zoom_figure)

The largest candidate, in red, is the left end of the interval of
:math:`x+y` values that ``AB`` can produce. There are :math:`\binom41+\binom42+\binom43=4+6+4=14`
nonempty proper subsets of the four phases, and keeping the largest
candidate for each gives the fourteen bounds of the theorem:

.. math::

   \beta_K=\max_{(I,J)\to(K,d)}\bigl(f_I(a)+f_J(b)-d\bigr).

The right end of the blue interval comes from a different output subset.
Since :math:`x+y+z+w=0`, every lower bound on a sum is also an upper bound
on the complementary sum. For example,

.. math::

   \beta_x\le x\le-\beta_{yzw},\qquad
   \beta_{xy}\le x+y\le-\beta_{zw}.

For ``AB``, six rules end at :math:`zw`. The rule ``zw + zw -> zw`` has
degree zero and gives :math:`f_{zw}(a)+f_{zw}(b)=-1/4-1/6=-5/12`. The
other five carry degree one or two and give at most :math:`-5/6`.

.. jupyter-execute::

   zw_rows = [("zw", "zw", 0), ("yw", "xz", 1), ("xz", "yw", 1),
              ("yz", "xw", 1), ("xw", "yz", 1), ("xy", "xy", 2)]
   beta_zw = max(candidates(zw_rows).values())
   print(f"{beta_xy} <= c1 <= {-beta_zw}")

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert beta_zw == -Fraction(5, 12)
   assert sorted(candidates(zw_rows).values())[-2] == -Fraction(5, 6)

Since :math:`c_1=x+y`, the direct representative of ``AB`` satisfies
:math:`1/6\le c_1\le5/12`. The reflection :math:`\rho` sends :math:`c_1`
to :math:`1-c_1`, so the reflected representative reaches only
:math:`c_1\ge7/12`. CX has :math:`c_1=1/2`, between :math:`5/12` and
:math:`7/12`, so neither representative reaches it.

.. jupyter-execute::

   mixed = ReachableRegion.of((native_a, native_b))
   print("AB reaches B:", mixed.reaches(native_b))
   print("AB reaches CX:", mixed.reaches(cx))

The fourteen lower bounds pair into seven two-sided bounds, on

.. math::

   x,\quad y,\quad z,\quad x+y,\quad x+z,\quad y+z,\quad x+y+z.

Each two-sided bound confines the spectrum to a slab between two parallel
planes, and by Theorem 23 the region of the direct representative is the
intersection of the seven slabs with the alcove. The slabs below use the
ranges each phase sum attains on the region, which can be tighter than
the table's bounds (see :ref:`reference-recurrence`).

.. jupyter-execute::

   sum_names = ["x", "y", "z", "w", "x+y", "x+z", "y+z"]

   def phase_sums(weyl):
       x, y, z, w = phases(weyl)
       return np.array([x, y, z, w, x + y, x + z, y + z])

   at_vertices = np.array([phase_sums(v) for v in mixed.vertices])
   lower, upper = at_vertices.min(axis=0), at_vertices.max(axis=0)

   c1, c2 = np.meshgrid(np.linspace(0.0005, 0.9995, 500), np.linspace(0.0007, 0.4993, 250))
   plane = np.column_stack([c1.ravel(), c2.ravel(), np.zeros(c1.size)])
   in_chamber = (c2 <= c1) & (c1 + c2 <= 1)
   sums = np.array([phase_sums(p) for p in plane]).reshape(*c1.shape, -1)
   in_slabs = np.all((lower <= sums + 1e-12) & (sums <= upper + 1e-12), axis=-1)
   inside = mixed.contains(plane).reshape(c1.shape)
   disagree = in_chamber & (in_slabs != inside)
   print(f"{(in_chamber & inside).sum()} grid points in the region, {disagree.sum()} disagree")

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert not disagree.any() and inside.any()

.. details:: Plotting code

   .. jupyter-execute::
      :hide-output:

      from itertools import combinations

      # Each phase sum as a linear function of (c1, c2) on the plane c3 = 0.
      normals = {"x": (0.5, 0.5), "y": (0.5, -0.5), "z": (-0.5, 0.5), "w": (-0.5, -0.5),
                 "x+y": (1, 0), "x+z": (0, 1)}
      lines = {}  # (unit normal, offset) -> bound labels, as  n . c >= offset
      for k, name in enumerate(sum_names):
          if name not in normals:
              continue
          n = np.array(normals[name], float)
          scale = np.linalg.norm(n)
          for sign, bound, text in ((1, lower[k], "≥"), (-1, upper[k], "≤")):
              key = (tuple(np.round(sign * n / scale, 9)), round(sign * bound / scale, 9))
              lines.setdefault(key, []).append(f"${name}$ {text} {fraction_text(Fraction(bound).limit_denominator(48))}")
      chamber = [((0, 1), 0), ((np.sqrt(0.5), -np.sqrt(0.5)), 0), ((-np.sqrt(0.5), -np.sqrt(0.5)), -np.sqrt(0.5))]
      walls = [(np.array(n), v) for n, v in list(lines) + chamber]
      corners = []
      for (n1, v1), (n2, v2) in combinations(walls, 2):
          matrix = np.array([n1, n2])
          if abs(np.linalg.det(matrix)) < 1e-9:
              continue
          point = np.linalg.solve(matrix, [v1, v2])
          if all(n @ point >= v - 1e-9 for n, v in walls):
              corners.append(point)
      corners = np.unique(np.round(corners, 9), axis=0)

      slab_figure, ax = plt.subplots(figsize=(7, 4.1), layout="constrained")
      triangle, = ax.fill([0, 1, 0.5], [0, 0, 0.5], color="0.95")
      ax.plot([0, 1, 0.5, 0], [0, 0, 0.5, 0], color="0.4", linewidth=1)
      ax.contourf(c1, c2, in_chamber & inside, levels=[0.5, 1.5], colors=["#8bbde0"])
      span = np.linspace(-1, 2, 2)
      colors = iter(["tab:purple", "tab:green", "tab:orange", "tab:brown", "tab:olive"])
      for (n, v), names in lines.items():
          n = np.array(n)
          on_line = sum(abs(n @ p - v) < 1e-7 for p in corners)
          tight = on_line >= 2
          if abs(n[1]) > 1e-9:
              xs, ys = span, (v - n[0] * span) / n[1]
          else:
              xs, ys = [v / n[0]] * 2, [-1, 2]
          ax.plot(xs, ys, color=next(colors) if tight else "0.75",
                  linewidth=1.8 if tight else 0.9, clip_path=triangle,
                  label=", ".join(names) if tight else None, zorder=2 if tight else 1)
      for name, point, offset in (("Identity", (0, 0), (6, 8)),
                                  ("B", (1 / 6, 1 / 6), (-14, 6)),
                                  ("CX", (1 / 2, 0), (6, 8))):
          ax.scatter(*point, s=40, color="black", zorder=4)
          ax.annotate(name, point, xytext=offset, textcoords="offset points", fontsize=9)
      ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 0.52), xlabel=r"$c_1$", ylabel=r"$c_2$")
      ax.set_aspect("equal")
      ax.legend(loc="upper right", fontsize=8.5, title="Bounds that cut an edge",
                title_fontsize=8.5)
      plt.close(slab_figure)

.. jupyter-execute::
   :hide-code:
   :alt: The plane c3 = 0 of the Weyl chamber, with the region of √CX followed by ∛iSWAP for the direct representative shaded blue. Four coloured lines, x+y ≥ 1/6, x+z ≤ 1/6, x ≥ 1/8 (which coincides with w ≤ −1/8 on this plane), and y ≤ 1/8 (which coincides with z ≥ −1/8), each contain an edge of the quadrilateral region, and the region touches the chamber wall c2 = 0 only at its corner (1/4, 0). The remaining bound lines are light gray; two of them meet the region only at its corner (5/12, 1/6). B sits at the region's corner (1/6, 1/6), and the identity and CX lie outside it.

   display(slab_figure)

The reflected representative contributes the mirror image of this
quadrilateral under :math:`c_1\mapsto1-c_1`, which also misses CX.

.. _pair-sums:

Why the pair sums are needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the phases sum to zero, each three-phase sum is minus the
remaining phase, as in :math:`x+y+z=-w`. The one-phase and three-phase
bounds therefore give one interval for each of :math:`x,y,z,w`. A target
can lie in all four intervals and still be unreachable.

For ``AB``, the individual phases attain the ranges

.. math::

   1/8\le x\le7/24,\qquad 0\le y\le1/8,\qquad
   -1/8\le z\le1/24,\qquad -7/24\le w\le-1/8.

Take the target with phases :math:`(1/6,1/24,1/24,-1/4)`, or Weyl
coordinates :math:`(5/24,5/24,1/12)`. Each phase lies in its interval,
and the phases satisfy the alcove conditions.
``AB`` does not reach the target, which fails the pair-sum bound

.. math::

   y+w\ge-1/6
   \quad\Longleftrightarrow\quad
   x+z\le1/6
   \quad\Longleftrightarrow\quad
   c_2\le1/6.

The target has :math:`x+z=c_2=5/24`, which exceeds the bound by
:math:`1/24`. The reflection :math:`\rho` leaves :math:`c_2` unchanged,
so the reflected representative fails the same bound.

.. jupyter-execute::

   pair_target = LocalEquivalenceClass((5 / 24, 5 / 24, 1 / 12))
   print("AB reaches the target:", mixed.reaches(pair_target))

.. jupyter-execute::
   :hide-code:
   :hide-output:

   np.testing.assert_allclose(24 * lower[:4], [3, 0, -3, -7], atol=1e-9)
   np.testing.assert_allclose(24 * upper[:4], [7, 3, 1, -3], atol=1e-9)
   values = phase_sums(pair_target.weyl)
   failed = [sum_names[k] for k in range(len(sum_names)) if not lower[k] - 1e-9 <= values[k] <= upper[k] + 1e-9]
   assert failed == ["x+z"], failed

.. _partition-labels:

Where the 72 rules come from
----------------------------

For sums of Hermitian matrices, Horn's inequalities are indexed by counts
of subspaces that meet the eigenspaces of the summands in prescribed
dimensions. Those counts are the Littlewood–Richardson coefficients. For
products of unitaries they become quantum Littlewood–Richardson (QLR)
coefficients, which also count curves and so carry a degree.

Each phase subset :math:`I` labels a Schubert class :math:`\sigma_I`, and the
quantum product :math:`\star` of two classes expands as

.. math::

   \sigma_I\star\sigma_J=\sum_{K,d}N_{IJ}^{K,d}q^d\sigma_K.

Each nonzero coefficient :math:`N_{IJ}^{K,d}` supplies one rule
:math:`(I,J)\to(K,d)`. The symbol :math:`q` records the degree and is
not a circuit parameter. Reachability uses only whether a coefficient is nonzero.

The table has one block for each subset size. For sizes one and three,
the four labels form a cycle, in the orders :math:`w,z,y,x` and
:math:`yzw,xzw,xyw,xyz`. The product of the labels in positions
:math:`i,j\in\{0,1,2,3\}` is the label in position :math:`(i+j)\bmod4`,
with degree :math:`\lfloor(i+j)/4\rfloor`. The six size-two labels have
no such cycle, and some of their products have two outputs. The size-two products come from Peterson, Crooks, and Smith's Figure 14.

.. jupyter-execute:: _includes/qlr_rules.py

.. details:: Math detail: partition labels in Theorem 23

   Theorem 23 writes these rules with *partitions* rather than
   phase-subset labels. For a sum of :math:`r` phases, put :math:`k=4-r`.
   A partition :math:`\lambda=(\lambda_1,\ldots,\lambda_r)`, with
   :math:`k\ge\lambda_1\ge\cdots\ge\lambda_r\ge0`, selects

   .. math::

      I(\lambda)=\{k+j-\lambda_j:\ j=1,\ldots,r\}.

   Larger parts of :math:`\lambda` select larger phases. For two-phase
   sums the dictionary is

   .. list-table:: Partition labels and phase sums
      :header-rows: 1

      * - Partition :math:`\lambda`
        - Phase positions
        - Phase sum
      * - :math:`(0,0)`
        - :math:`\{3,4\}`
        - :math:`z+w`
      * - :math:`(1,0)`
        - :math:`\{2,4\}`
        - :math:`y+w`
      * - :math:`(1,1)`
        - :math:`\{2,3\}`
        - :math:`y+z`
      * - :math:`(2,0)`
        - :math:`\{1,4\}`
        - :math:`x+w`
      * - :math:`(2,1)`
        - :math:`\{1,3\}`
        - :math:`x+z`
      * - :math:`(2,2)`
        - :math:`\{1,2\}`
        - :math:`x+y`

   Thus ``yw + xz -> xy`` is the term with partitions
   :math:`(1,0),(2,1),(2,2)` and degree zero. Theorem 23 calls these
   partitions :math:`a,b,c`. Here those letters keep their meanings as
   gate spectra and Weyl coordinates.

Each rule comes from a subspace that every arrangement of the
eigenbases must contain. The detail below derives ``yw + xz -> xy`` for
Hermitian sums in a few lines.

.. details:: Math detail: the geometry behind one inequality

   Let :math:`H+K=S` be Hermitian :math:`4\times4` matrices with
   descending real eigenvalues :math:`\alpha_i,\beta_i,\delta_i`. For a
   two-dimensional subspace :math:`W`, define the trace on that plane by

   .. math::

      T_H(W)=\operatorname{tr}(P_WH)=u^\dagger Hu+v^\dagger Hv,

   where :math:`u,v` is any orthonormal basis of :math:`W`. This is the
   sum of the expectations of :math:`H` over a basis of the plane, and it
   does not depend on which basis. In an eigenbasis of :math:`H` it equals
   :math:`\sum_i\alpha_i\|P_We_i\|^2`, a weighted sum of eigenvalues with
   weights in :math:`[0,1]` that add to two. Its largest value is
   therefore :math:`\alpha_1+\alpha_2`. The trace is also linear in the
   matrix, so :math:`T_S(W)=T_H(W)+T_K(W)`.

   Let :math:`E_2` span the top two eigendirections of :math:`H`, and let
   :math:`F_1\subset F_3` span the top one and three of :math:`K`. Choose
   a plane that meets :math:`E_2` and lies between :math:`F_1` and
   :math:`F_3`:

   .. math::

      W\cap E_2\ne\{0\},\qquad F_1\subset W\subset F_3.

   Such a plane always exists, because
   :math:`\dim(E_2\cap F_3)\ge2+3-4=1`. Choose a line in that
   intersection and span it with :math:`F_1`. If the two lines coincide,
   any plane in :math:`F_3` that contains the line works.

   For :math:`H`, take the first basis vector in :math:`W\cap E_2`. Its
   expectation is at least :math:`\alpha_2`, and the other vector's
   expectation is at least :math:`\alpha_4`. For :math:`K`, use a basis
   starting with its top eigenvector. The second vector lies in
   :math:`F_3`, so these expectations are at least :math:`\beta_1` and
   :math:`\beta_3`. The trace does not depend on the basis, so the two
   estimates add:

   .. math::

      \delta_1+\delta_2\ge T_S(W)
         =T_H(W)+T_K(W)
         \ge\alpha_2+\alpha_4+\beta_1+\beta_3.

   Relabeling :math:`1,2,3,4` as :math:`x,y,z,w` gives the rule
   ``yw + xz -> xy``: the top two output eigenvalues are at least the
   second and fourth of :math:`H` plus the first and third of :math:`K`.
   It holds for every relative orientation of the eigenbases, because the
   plane :math:`W` always exists.

   For generic eigenbases, :math:`E_2\cap F_3` is a single line distinct
   from :math:`F_1`, so exactly one plane meets the conditions. That
   count, one, is the Littlewood–Richardson coefficient of the rule. In
   general the nested eigenspaces form a *flag*, the intersection
   requirements are *Schubert conditions*, and the coefficients count the
   subspaces that satisfy them. These counts are the structure constants
   for multiplying Schubert classes in the cohomology of a Grassmannian;
   the QLR coefficients are those of its quantum cohomology, for the
   Grassmannians of lines, planes, and 3-planes in :math:`\mathbb C^4`.

   The Hermitian argument does not carry over to unitaries, because
   :math:`\log(UV)` need not equal :math:`\log U+\log V`. The
   multiplicative theorem uses QLR coefficients instead. Those of degree
   zero are the ordinary coefficients and give the same
   subspace-intersection rules. Those of positive degree count curves in
   the Grassmannian and supply the integer corrections for phases that
   wrap around the circle.
   Peterson, Crooks, and Smith's `Appendix A
   <https://quantum-journal.org/papers/q-2020-03-26-247/pdf/#page=31>`_
   gives the geometric construction.

.. _max-plus-recurrence:

Longer sentences: the max-plus recurrence
-----------------------------------------

With three gates, the prefix reaches a region of spectra, not one
spectrum. The two-gate inequalities with each prefix phase sum replaced
by its lower bound still hold for every output. They look too weak,
because no single prefix spectrum attains all fourteen bounds at once.
The Horn theorem for products of many factors shows that they are
enough. For fixed input spectra :math:`a^{(1)},\ldots,a^{(n)}`, its inequalities still
use only the fourteen output subsets:

.. math::

   f_K(\delta)\ge\sum_{t=1}^n f_{J_t}(a^{(t)})-d
   \quad\text{when}\quad
   [q^d\sigma_K]\,
      \sigma_{J_1}\star\cdots\star\sigma_{J_n}>0.

The bracket means the coefficient of :math:`q^d\sigma_K`, so each nonzero
term of the :math:`n`-fold quantum product gives one inequality
(Agnihotri and Woodward, `Theorem 3.1
<https://arxiv.org/pdf/alg-geom/9712013#page=5>`_, stated for the
product rather than the identity). Keeping the strongest bound for each output subset, as for two gates,
gives

.. math::

   \beta_K^{(n)}=
   \max_{J_1,\ldots,J_n,d:\,[q^d\sigma_K]\prod_t\sigma_{J_t}>0}
      \left(\sum_{t=1}^n f_{J_t}(a^{(t)})-d\right).

The many-factor theorem lets each factor's eigenbasis be any unitary, but
a circuit has only local layers. Corollary 14, applied one gate at a
time, closes the gap. Suppose some unitaries with the native Cartan-double spectra
multiply to the target spectrum. Their partial products give a spectrum
for each prefix. At each step, absorb the outer local factors of the
prefix into the next middle layer. Corollary 14 then supplies a middle
local layer that takes the current prefix class and the next native gate
to the next prefix spectrum. The circuit has to match only the prefix
spectra, not the matrices of those unitaries, so the argument never needs
one orthogonal change of basis for all factors at once.

Peterson, Crooks, and Smith's `Corollary 26
<https://quantum-journal.org/papers/q-2020-03-26-247/pdf/#page=11>`_
handles longer circuits by adding an intermediate spectrum and projecting
it away with Fourier–Motzkin elimination. Their construction lets each
native class range over a polytope, which covers continuous gate
families. For a sentence of fixed gates, the recurrence below gives the
same region without the projection, because it groups the multiple-factor
inequalities by their fourteen output labels and keeps the strongest
bound in each.

The maximum that defines :math:`\beta_K^{(n)}` runs over one subset per
gate, so a direct search grows exponentially with :math:`n`. The
recurrence instead applies the two-gate table one gate at a time. It
keeps the best value for each of the fourteen labels after each prefix,
starting from one gate,

.. math::

   \beta_I^{(1)}=f_I(a^{(1)}),

and appends :math:`a^{(n+1)}` with the binary QLR table:

.. math::

   \boxed{\displaystyle
   \beta_K^{(n+1)}=
       \max_{(I,J)\to(K,d)}
       \left(\beta_I^{(n)}+f_J(a^{(n+1)})-d\right).}

Each step adds along a rule and takes the maximum over rules with the
same output.

The recurrence is exact because the quantum product is associative.
Expanding the product of the first :math:`n` classes and then
multiplying by the last class gives the full :math:`(n+1)`-fold product.
Every contribution to a final coefficient therefore passes through some
intermediate label :math:`I`, and its degree is the prefix degree plus
the degree of the last rule. All coefficients are nonnegative, so no
terms cancel, and a final coefficient is nonzero exactly when some path
has nonzero coefficients at every step. Once :math:`I` is fixed, the
last step adds the same quantity to every prefix candidate that ends at
:math:`I`, so only the largest prefix value matters, and the recurrence
keeps that value. For three gates the expanded recurrence reads

.. math::

   \beta_K^{(3)}
   =\max_{\substack{(I,J_3)\to(K,d_2)\\
                     (J_1,J_2)\to(I,d_1)}}
      \left[f_{J_1}(a^{(1)})+f_{J_2}(a^{(2)})
            +f_{J_3}(a^{(3)})-(d_1+d_2)\right].

The intermediate label :math:`I` selects a term of this expansion, not a
gate or an intermediate spectrum. Each final bound can use a different
maximizing path, and no single prefix spectrum needs to attain all the
prefix bounds at once. For ``AA``, the identity attains :math:`x=0` and CX
attains :math:`w=-1/4`, but :math:`x=0` forces all four ordered, zero-sum
phases to zero, so no spectrum attains both bounds.

.. _reference-recurrence:

The recurrence in exact fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute:: _includes/recurrence.py

Two of the bounds for ``AB`` are never attained by the region. The
alcove and the other bounds already force :math:`y\ge0` and
:math:`y+z\ge0`, while the table gives only :math:`\beta_y=-1/24` and
:math:`\beta_{yz}=-1/6`.

.. jupyter-execute:: _includes/prefix_bounds.py

.. jupyter-execute::
   :hide-code:
   :hide-output:

   vertex_spectra = [phases(v) for v in mixed.vertices]
   unattained = [label for label in labels
                 if abs(min(phase_sum(s, label) for s in vertex_spectra) - float(mixed_bounds[label])) > 1e-9]
   assert unattained == ["y", "yz"]
   assert mixed_bounds["y"] == -Fraction(1, 24) and mixed_bounds["yz"] == -Fraction(1, 6)

The search in :ref:`guiding-selection` uses these bounds to choose
between ``AAA`` and ``AAB`` for a target with Weyl coordinates
:math:`(3/8,5/16,1/8)`, which has :math:`w=-13/32`. Appending ``A`` to
``AA`` gives :math:`w\ge-3/8` and excludes the target, while appending
``B`` gives :math:`w\ge-5/12` and does not.

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert append_gate(prefix_bounds[1], a_phases)["w"] == -Fraction(3, 8)
   assert prefix_bounds[2]["w"] == -Fraction(5, 12)
   assert prefix_bounds[1]["x"] == 0 and prefix_bounds[1]["w"] == -Fraction(1, 4)

.. details:: How GULPS evaluates the update

   Each rule :math:`f_K(\delta)\ge f_I(a)+f_J(b)-d` subtracts a quantum
   degree. Shifting each phase sum by a constant that depends only on its
   label absorbs that subtraction, so each update uses only addition and
   maximum.

   Give the phases ranks :math:`w=0`, :math:`z=1`, :math:`y=2`, and
   :math:`x=3`. For a subset :math:`I` of :math:`r` phases, the
   *codimension* :math:`\kappa(I)` of the Schubert class
   :math:`\sigma_I` is the sum of the ranks in :math:`I` minus
   :math:`r(r-1)/2`, the smallest such sum for :math:`r` phases. In the
   partition labels of Theorem 23 it is :math:`|\lambda|`, so
   :math:`\kappa(zw)=0`, :math:`\kappa(yw)=1`, :math:`\kappa(xz)=3`, and
   :math:`\kappa(xy)=4`. Define the shifted sum

   .. math::

      p_I(a)=f_I(a)-\kappa(I)/4.

   The quantum product respects a grading in which :math:`\sigma_I` has
   degree :math:`\kappa(I)` and :math:`q` has degree 4, the matrix size,
   so every QLR rule obeys

   .. math::

      \kappa(I)+\kappa(J)-\kappa(K)=4d.

   The degree of a rule is therefore fixed by its three labels, and
   subtracting :math:`\kappa(K)/4` from both sides of the rule cancels
   it:

   .. math::

      p_K(\delta)\ge p_I(a)+p_J(b).

   GULPS stores each bound shifted by its codimension,
   :math:`h_I=\beta_I-\kappa(I)/4`, and appending a gate with spectrum
   :math:`g` becomes

   .. math::

      h'_K=\max_{(I,J)\to(K,d)}\bigl(h_I+p_J(g)\bigr).

   The QLR table still decides which pairs :math:`(I,J)` contribute. The
   shift removes only the explicit degrees. For ``AB``,
   :math:`\kappa(xy)=4` turns :math:`\beta_{xy}=1/6` into
   :math:`h_{xy}=-5/6`, and adding the shift back recovers
   :math:`x+y\ge1/6`. Without the degrees, the cyclic blocks of the table
   (the single phases, the triples, and four of the pairs) become cyclic
   max-plus convolutions of length four, and the two remaining pairs
   become one convolution of length two. The update is then a fixed
   sequence of additions and maxima, with no table lookup.
