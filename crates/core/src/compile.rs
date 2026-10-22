//! Decompose targets into canonical gates `D(c)`, bind physical gates to a
//! decomposer's slots, and compile targets in the bound gates.

use crate::class::{Mono, exact_bits};
use crate::kak::{Frame, LocalPair, Unitary, factor_rotation, relabel};
use crate::reachability::{Region, trajectory};
use crate::search::{Selection, Slot, Walk};
use crate::{INPUT_ATOL, LOCAL_IDENTITY_TOL, Mat2};
use qiskit_numerics::IndexMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// Selection and synthesis failures.
#[derive(Clone, Debug)]
pub enum CompileError {
    /// No sequence was found within the configured depth.
    DepthLimit(usize),
    /// A numerical step failed; the message names it.
    Synthesis(String),
    /// An input matrix is not finite, or `U†U − I` has an entry above
    /// `INPUT_ATOL`: the largest entry, or NaN.
    NotUnitary(f64),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthLimit(depth) => write!(
                f,
                "no sentence found within max_depth={depth}; increase max_depth on the decomposer to search deeper"
            ),
            Self::Synthesis(message) => f.write_str(message),
            Self::NotUnitary(deviation) => write!(
                f,
                "matrix is not unitary within atol={INPUT_ATOL:e}: the largest entry of U†U − I is {deviation:.1e}"
            ),
        }
    }
}
impl Error for CompileError {}

/// Decomposes targets into sentences of canonical gates, one class and cost
/// per slot. Physical gates enter only through a [`Binding`], as their
/// rotations around their class's canonical gate.
pub struct Decomposer {
    walk: Walk,
    /// Realized sentences by exact class, kept across calls.
    realized: Mutex<IndexMap<[u64; 3], Arc<Realized>>>,
}

/// One class's canonical sentence, computed once by the first job that needs it.
type Realized = OnceLock<Result<RealizedSentence, CompileError>>;

/// Distinct classes a decomposer keeps before it clears them all. At most a few
/// kilobytes each, so this bounds it to a few megabytes.
const REALIZED_CAP: usize = 1024;

impl Decomposer {
    /// Slots are `(class, cost)` with finite, nonnegative costs; `max_depth` is positive.
    fn new(slots: Vec<Slot>, local_layer_cost: f64, max_depth: usize) -> Self {
        Self {
            walk: Walk::new(slots, local_layer_cost, max_depth),
            realized: Mutex::default(),
        }
    }

    /// The cache cell for a class's canonical sentence.
    fn realized(&self, class: Mono) -> Arc<Realized> {
        let key = class.key();
        let mut realized = self.realized.lock().expect("a cache update panicked");
        if let Some(cell) = realized.get(&key) {
            return Arc::clone(cell);
        }
        if realized.len() == REALIZED_CAP {
            realized.clear();
        }
        Arc::clone(realized.entry(key).or_default())
    }

    /// The circuit for a target of `class` with frame `target`, in a binding's gates.
    fn compile_target(
        &self,
        class: Mono,
        gates: &[Gate],
        target: Frame,
    ) -> Result<CompiledTarget, CompileError> {
        let realized = self.realized(class);
        let canonical = realized.get_or_init(|| self.realize(class));
        canonical
            .as_ref()
            .map_err(Clone::clone)?
            .compile_target(gates, target)
    }

    /// The canonical gates: class and cost.
    pub fn slots(&self) -> &[Slot] {
        self.walk.slots()
    }

    /// Cost of each surrounding or intervening local layer.
    pub fn local_layer_cost(&self) -> f64 {
        self.walk.local_layer_cost()
    }

    /// Maximum number of gates in a sentence.
    pub fn max_depth(&self) -> usize {
        self.walk.max_depth()
    }

    fn select_one(&self, target: Mono) -> Result<Selection, CompileError> {
        (self.walk.select(target)).ok_or(CompileError::DepthLimit(self.max_depth()))
    }

    /// Select slot sequences without realizing them.
    pub fn select(&self, targets: &[Mono]) -> Result<Vec<Selection>, CompileError> {
        targets
            .iter()
            .map(|&target| self.select_one(target))
            .collect()
    }

    /// A coverage candidate's slots, cost, and reach region.
    pub fn coverage_candidate(
        &self,
        row: usize,
    ) -> Result<(Vec<usize>, f64, Region), CompileError> {
        (self.walk.coverage_candidate(row)).ok_or(CompileError::DepthLimit(self.max_depth()))
    }

    /// The candidates the search discards as dominated before row `row`:
    /// slots, cost, and reach region.
    pub fn dominated(&self, row: usize) -> Result<Vec<(Vec<usize>, f64, Region)>, CompileError> {
        (self.walk.dominated(row)).ok_or(CompileError::DepthLimit(self.max_depth()))
    }

    /// Realize a class in canonical gates, whose frames are the identity.
    fn realize(&self, raw_endpoint: Mono) -> Result<RealizedSentence, CompileError> {
        let Selection {
            gates: slots,
            reflected,
            ..
        } = self.select_one(raw_endpoint)?;
        let classes: Vec<Mono> = slots.iter().map(|&s| self.slots()[s].class).collect();
        let mut layers = Vec::new();
        let frame = match classes.as_slice() {
            [] => relabel(Mono([0.0; 3]), raw_endpoint, reflected)?,
            [class] => relabel(*class, raw_endpoint, reflected)?,
            [first, rest @ ..] => {
                let endpoint = if reflected {
                    raw_endpoint.rho()
                } else {
                    raw_endpoint
                };
                let waypoints = trajectory(&classes, endpoint).ok_or_else(|| {
                    CompileError::Synthesis(
                        "could not reconstruct the selected sentence's trajectory".into(),
                    )
                })?;
                let mut prefix = *first;
                let mut frame = Frame::identity();
                // Maintain Q† P Q = frame.after · D(prefix) · frame.before · exp(i frame.phase).
                for (i, class) in rest.iter().enumerate() {
                    // The last step lands on the target itself.
                    let target = waypoints.get(i).copied().unwrap_or(raw_endpoint);
                    let (middle, after, before, phase) = can_sandwich::solve_with_factors(
                        class.0, prefix.0, target.0,
                    )
                    .map_err(|decline| {
                        CompileError::Synthesis(format!("segment {}: {decline}", i + 1))
                    })?;
                    // With P = L D(p) R, inserting O Lᵀ gives D(c) (O Lᵀ) P = [D(c) O D(p)] R.
                    layers.push(factor_rotation(&(middle * frame.after.transpose()))?);
                    frame = Frame {
                        after,
                        before: before * frame.before,
                        phase: frame.phase + phase,
                    };
                    prefix = target;
                }
                frame
            }
        };
        Ok(RealizedSentence {
            slots,
            layers,
            frame,
        })
    }
}

/// A physical two-qubit gate `G`, with `Q† G Q = A D(c) B exp(i phase)` in the
/// magic basis `Q`: its class `c`, its phase, and the inverses of its rotations
/// `A` and `B`, which the local layers on either side of it absorb.
#[derive(Clone, Copy)]
pub struct Gate {
    class: Mono,
    phase: f64,
    /// `Aᵀ` as single-qubit gates.
    after_inverse: LocalPair,
    /// `Bᵀ` as single-qubit gates.
    before_inverse: LocalPair,
}

impl Gate {
    /// A gate's class, phase, and rotations; fails if the KAK or a factoring does.
    pub fn new(unitary: &Unitary) -> Result<Self, CompileError> {
        let (class, frame) = unitary.kak()?;
        Ok(Self {
            class,
            phase: frame.phase,
            after_inverse: factor_rotation(&frame.after.transpose())?,
            before_inverse: factor_rotation(&frame.before.transpose())?,
        })
    }
}

/// Physical gates bound to a decomposer's slots: the decomposer and the gate
/// filling each slot.
pub struct Binding {
    decomposer: Arc<Decomposer>,
    gates: Vec<Gate>,
}

impl Binding {
    /// Bind gates, each with its cost. Of gates with exactly the same class,
    /// the cheapest is kept, first on ties. The first of `existing` with equal
    /// `local_layer_cost` and `max_depth` whose slots the kept gates fill one to
    /// one, at exactly equal class and cost, is reused; otherwise a new
    /// decomposer is made. Returns the binding and, for each slot, the input
    /// position of the gate filling it.
    ///
    /// Reuse scans `existing`, so binding every edge of a Target is quadratic
    /// in its number of distinct edges, and it requires equal costs, so
    /// calibrated edges rarely share. See
    /// <https://github.com/evmckinney9/gulps/issues/20>.
    pub fn new<'a>(
        gates: &[(&Gate, f64)],
        local_layer_cost: f64,
        max_depth: usize,
        existing: impl IntoIterator<Item = &'a Arc<Decomposer>>,
    ) -> Result<(Self, Vec<usize>), String> {
        let valid = |cost: f64| cost.is_finite() && cost >= 0.0;
        if gates.is_empty() {
            return Err("gates must not be empty".into());
        }
        if !gates.iter().all(|&(_, cost)| valid(cost)) || !valid(local_layer_cost) {
            return Err("costs must be finite and non-negative".into());
        }
        if max_depth == 0 {
            return Err("max_depth must be positive".into());
        }
        let mut cheapest = IndexMap::<[u64; 3], usize>::default();
        for (i, (gate, cost)) in gates.iter().enumerate() {
            let entry = cheapest.entry(gate.class.key()).or_insert(i);
            if *cost < gates[*entry].1 {
                *entry = i;
            }
        }
        let kept: Vec<usize> = cheapest.into_values().collect();
        // The kept gates in the slot order of `decomposer`, if they fill it.
        let fill = |decomposer: &Decomposer| -> Option<Vec<usize>> {
            if exact_bits(decomposer.local_layer_cost()) != exact_bits(local_layer_cost)
                || decomposer.max_depth() != max_depth
                || decomposer.slots().len() != kept.len()
            {
                return None;
            }
            let mut unused = kept.clone();
            (decomposer.slots().iter())
                .map(|slot| {
                    let i = unused.iter().position(|&g| {
                        exact_bits(gates[g].1) == exact_bits(slot.cost)
                            && gates[g].0.class.key() == slot.class.key()
                    })?;
                    Some(unused.swap_remove(i))
                })
                .collect()
        };
        let (decomposer, order) = match existing.into_iter().find_map(|d| Some((d, fill(d)?))) {
            Some((decomposer, order)) => (Arc::clone(decomposer), order),
            None => {
                let slots = kept
                    .iter()
                    .map(|&g| Slot {
                        class: gates[g].0.class,
                        cost: gates[g].1,
                    })
                    .collect();
                let decomposer = Decomposer::new(slots, local_layer_cost, max_depth);
                (Arc::new(decomposer), kept)
            }
        };
        let gates = order.iter().map(|&g| *gates[g].0).collect();
        Ok((Self { decomposer, gates }, order))
    }

    /// The canonical decomposer the gates are bound to.
    #[must_use]
    pub fn decomposer(&self) -> &Arc<Decomposer> {
        &self.decomposer
    }

    /// The circuit for `target` in the bound gates.
    pub fn compile(&self, target: &Unitary) -> Result<CompiledTarget, CompileError> {
        let (class, frame) = target.kak()?;
        self.decomposer.compile_target(class, &self.gates, frame)
    }
}

/// A sentence of canonical gates: its slots, the local layers between them,
/// and its endpoint frame.
struct RealizedSentence {
    slots: Vec<usize>,
    layers: Vec<LocalPair>,
    frame: Frame,
}

impl RealizedSentence {
    /// The circuit for a target of this sentence's class with frame `target`,
    /// with the bound physical gates `G_k = A_k D(c_k) B_k` in place of the
    /// canonical ones: the layer between gates k and k+1 becomes
    /// `B_{k+1}ᵀ C_k A_kᵀ`, the first `B`ᵀ and the last `A`ᵀ join the outer
    /// layers, and the gate phases add.
    fn compile_target(
        &self,
        gates: &[Gate],
        target: Frame,
    ) -> Result<CompiledTarget, CompileError> {
        // The target's rotations around this sentence's endpoint.
        let before = self.frame.before.transpose() * target.before;
        let after = target.after * self.frame.after.transpose();
        let phase = target.phase
            - self.frame.phase
            - self.slots.iter().map(|&s| gates[s].phase).sum::<f64>();
        // n gates, at most two locals before the first, between each pair, and after the last.
        let mut ops = Vec::with_capacity(3 * self.slots.len() + 2);
        let (Some(&first), Some(&last)) = (self.slots.first(), self.slots.last()) else {
            ops.extend(locals(factor_rotation(&(after * before))?));
            return Ok(CompiledTarget { ops, phase });
        };
        ops.extend(locals(
            gates[first].before_inverse * factor_rotation(&before)?,
        ));
        ops.push(Op::Gate(first));
        for (pair, &layer) in self.slots.windows(2).zip(&self.layers) {
            let (left, right) = (&gates[pair[0]], &gates[pair[1]]);
            ops.extend(locals(right.before_inverse * layer * left.after_inverse));
            ops.push(Op::Gate(pair[1]));
        }
        ops.extend(locals(factor_rotation(&after)? * gates[last].after_inverse));
        Ok(CompiledTarget { ops, phase })
    }
}

/// One operation of a compiled circuit, on block qubits 0 and 1.
pub enum Op {
    /// A single-qubit gate.
    Local {
        /// Block qubit, 0 or 1.
        qubit: u32,
        /// The gate's matrix.
        matrix: Mat2,
    },
    /// The native gate filling this slot of the binding.
    Gate(usize),
}

/// A target expressed as a native sequence and its surrounding local gates.
pub struct CompiledTarget {
    ops: Vec<Op>,
    phase: f64,
}

impl CompiledTarget {
    /// The circuit in order: local gates and the slots of the native gates.
    #[must_use]
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Global phase in radians.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.phase
    }
}

/// The single-qubit gates of a layer, omitting those within
/// `LOCAL_IDENTITY_TOL` of the identity. A product of factored layers fixes
/// its two factors only up to a shared sign, so a factor at `-I` negates both,
/// which leaves the layer unchanged and lets that factor be omitted.
fn locals(mut pair: LocalPair) -> impl Iterator<Item = Op> {
    let minus_identity = |m: &Mat2| (m + Mat2::identity()).norm() <= LOCAL_IDENTITY_TOL;
    if minus_identity(&pair.q0) || minus_identity(&pair.q1) {
        pair = LocalPair {
            q0: -pair.q0,
            q1: -pair.q1,
        };
    }
    [(0, pair.q0), (1, pair.q1)]
        .into_iter()
        .filter(|(_, matrix)| (matrix - Mat2::identity()).norm() > LOCAL_IDENTITY_TOL)
        .map(|(qubit, matrix)| Op::Local { qubit, matrix })
}
