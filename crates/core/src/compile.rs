//! Decompose targets into canonical gates `D(c)`, bind physical gates to a
//! decomposer's slots through their frames, and dress each sentence with them.

use crate::class::{Mono, exact_bits};
use crate::kak::{Frame, LocalPair, factor_rotation, kak, relabel};
use crate::reachability::Region;
use crate::search::{Selection, Walk};
use crate::{LOCAL_IDENTITY_TOL, Mat2, Mat4};
use nalgebra::Matrix4;
use qiskit_numerics::IndexMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Selection and synthesis failures, independent of the Python interface.
#[derive(Clone, Debug)]
pub enum CompileError {
    /// No sequence was found within the configured depth.
    DepthLimit(usize),
    /// A numerical step failed: the KAK of an input or target did not
    /// diagonalize or reconstruct, no trajectory was found, the solver
    /// declined a segment, or a rotation did not split into local gates.
    Synthesis(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthLimit(depth) => write!(
                f,
                "no sentence found within max_depth={depth}; increase max_depth on the decomposer to search deeper"
            ),
            Self::Synthesis(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for CompileError {}

/// Decomposes targets into sentences of canonical gates, one class and cost
/// per slot. Physical gates enter [`compile_all`] only through their frames:
/// `Q† G Q = after · D(c) · before · exp(i phase)` for gate `G` of class `c`.
pub struct Decomposer {
    walk: Walk,
    /// Canonical sentences by exact class, kept across calls.
    realized: Mutex<IndexMap<[u64; 3], Arc<Realized>>>,
}

/// One class's canonical sentence, computed once by the first job that needs it.
type Realized = OnceLock<Result<Canonical, CompileError>>;

/// Distinct classes a decomposer keeps before it clears them all. At most a few
/// kilobytes each, so this bounds it to a few megabytes.
const REALIZED_CAP: usize = 1024;

impl Decomposer {
    /// Slots are `(class, cost)` with finite, nonnegative costs; `max_depth` is positive.
    fn new(slots: Vec<(Mono, f64)>, local_layer_cost: f64, max_depth: usize) -> Self {
        Self {
            walk: Walk::new(slots, local_layer_cost, max_depth),
            realized: Mutex::default(),
        }
    }

    /// The cache cell for a class's canonical sentence.
    fn realized(&self, class: Mono) -> Arc<Realized> {
        let key = class.exact_key();
        let mut realized = self.realized.lock().expect("a cache update panicked");
        if let Some(cell) = realized.get(&key) {
            return Arc::clone(cell);
        }
        if realized.len() == REALIZED_CAP {
            realized.clear();
        }
        Arc::clone(realized.entry(key).or_default())
    }

    /// A class's canonical sentence, dressed with a binding's frames.
    fn dressed(&self, class: Mono, frames: &[Frame]) -> Result<RealizedSentence, CompileError> {
        let realized = self.realized(class);
        let canonical = realized.get_or_init(|| self.realize(class));
        canonical.as_ref().map_err(Clone::clone)?.dress(frames)
    }

    /// The canonical gates: class and cost.
    pub fn slots(&self) -> &[(Mono, f64)] {
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

    /// Select slot sequences without realizing them. Selection costs about
    /// 0.1 us per target, too little to repay threads, so this is serial.
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
        let (slots, cost) = (self.walk.coverage_candidate(row))
            .ok_or(CompileError::DepthLimit(self.max_depth()))?;
        Ok((slots, cost, self.walk.coverage_region(row)))
    }

    /// Realize a class in canonical gates, whose frames are the identity.
    fn realize(&self, raw_endpoint: Mono) -> Result<Canonical, CompileError> {
        let Selection {
            gates: slots,
            reflected,
            ..
        } = self.select_one(raw_endpoint)?;
        let classes: Vec<Mono> = slots.iter().map(|&s| self.slots()[s].0).collect();
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
                let waypoints =
                    crate::reachability::trajectory(&classes, endpoint).ok_or_else(|| {
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
                    layers.push(middle * frame.after.transpose());
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
        Ok(Canonical {
            slots,
            layers,
            frame,
        })
    }
}

/// A physical two-qubit gate: its class and its frame around that class's
/// canonical gate.
#[derive(Clone)]
pub struct Gate {
    class: Mono,
    /// The gate's rotations around its class's canonical gate.
    frame: Frame,
}

impl Gate {
    /// A gate's class and frame; fails if the KAK does.
    pub fn new(matrix: &Mat4) -> Result<Self, CompileError> {
        let (class, frame) = kak(matrix)?;
        Ok(Self { class, frame })
    }
}

/// Physical gates bound to a decomposer's slots: the decomposer and each
/// gate's frame relative to its slot's class.
pub struct Binding {
    decomposer: Arc<Decomposer>,
    frames: Vec<Frame>,
}

impl Binding {
    /// Bind gates, each with its cost. Of gates with exactly the same class,
    /// the cheapest is kept, first on ties. The first of `existing` whose slots
    /// the kept gates fill one to one, at exactly equal class and cost, is
    /// reused; otherwise a new decomposer is made. Returns the binding and, for
    /// each slot, the input position of the gate filling it.
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
            let entry = cheapest.entry(gate.class.exact_key()).or_insert(i);
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
                .map(|&(class, cost)| {
                    let i = unused.iter().position(|&g| {
                        exact_bits(gates[g].1) == exact_bits(cost)
                            && gates[g].0.class.exact_key() == class.exact_key()
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
                    .map(|&g| (gates[g].0.class, gates[g].1))
                    .collect();
                let decomposer = Decomposer::new(slots, local_layer_cost, max_depth);
                (Arc::new(decomposer), kept)
            }
        };
        let frames = order.iter().map(|&g| gates[g].0.frame).collect();
        Ok((Self { decomposer, frames }, order))
    }

    /// The canonical decomposer the gates are bound to.
    pub fn decomposer(&self) -> &Arc<Decomposer> {
        &self.decomposer
    }
}

/// A sentence of canonical gates: its slots, the local layers between them,
/// and its endpoint frame.
struct Canonical {
    slots: Vec<usize>,
    layers: Vec<Matrix4<f64>>,
    frame: Frame,
}

impl Canonical {
    /// Substitute the bound physical gates `G_k = A_k D(c_k) B_k`: the layer
    /// between gates k and k+1 becomes `B_{k+1}ᵀ C_k A_kᵀ`, the first `B` and
    /// last `A` join the endpoint frame, and the gate phases add.
    fn dress(&self, frames: &[Frame]) -> Result<RealizedSentence, CompileError> {
        let between = (self.slots.windows(2).zip(&self.layers))
            .map(|(pair, layer)| {
                let (left, right) = (&frames[pair[0]], &frames[pair[1]]);
                factor_rotation(&(right.before.transpose() * layer * left.after.transpose()))
            })
            .collect::<Result<_, _>>()?;
        let mut frame = self.frame;
        if let (Some(&first), Some(&last)) = (self.slots.first(), self.slots.last()) {
            frame.after = frames[last].after * frame.after;
            frame.before *= frames[first].before;
            frame.phase += self.slots.iter().map(|&s| frames[s].phase).sum::<f64>();
        }
        Ok(RealizedSentence {
            gates: self.slots.clone(),
            between,
            frame,
        })
    }
}

/// One class circuit for one binding: its slots, the physical local layers
/// between them, and its endpoint frame.
struct RealizedSentence {
    /// Slot indices in circuit order; empty for a local circuit.
    gates: Vec<usize>,
    /// The local layer before each gate after the first.
    between: Vec<LocalPair>,
    frame: Frame,
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
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Global phase in radians.
    pub fn phase(&self) -> f64 {
        self.phase
    }
}

/// The single-qubit gates of a layer, omitting those within
/// `LOCAL_IDENTITY_TOL` of the identity.
fn locals(pair: LocalPair) -> impl Iterator<Item = Op> {
    [(0, pair.q0), (1, pair.q1)]
        .into_iter()
        .filter(|(_, matrix)| (matrix - Mat2::identity()).norm() > LOCAL_IDENTITY_TOL)
        .map(|(qubit, matrix)| Op::Local { qubit, matrix })
}

impl RealizedSentence {
    /// The circuit for a target of this sentence's class with frame `target`.
    fn attach(&self, mut target: Frame) -> Result<CompiledTarget, CompileError> {
        target.before = self.frame.before.transpose() * target.before;
        target.after *= self.frame.after.transpose();
        let mut ops = Vec::new();
        if self.gates.is_empty() {
            ops.extend(locals(factor_rotation(&(target.after * target.before))?));
        } else {
            ops.extend(locals(factor_rotation(&target.before)?));
            ops.push(Op::Gate(self.gates[0]));
            for (&slot, &pair) in self.gates[1..].iter().zip(&self.between) {
                ops.extend(locals(pair));
                ops.push(Op::Gate(slot));
            }
            ops.extend(locals(factor_rotation(&target.after)?));
        }
        Ok(CompiledTarget {
            ops,
            phase: target.phase - self.frame.phase,
        })
    }
}

/// Compile `(binding, target)` jobs; results keep job order,
/// and an error carries the index of a failing job. Targets are factored in one
/// parallel map. Jobs of exactly the same class on the same binding share one
/// dressed sentence, so they are grouped and the groups compile in a second
/// parallel map; keeping the two numerical kernels apart avoids a 5%
/// single-thread cost.
pub fn compile_all(
    jobs: &[(&Binding, &Mat4)],
) -> Result<Vec<CompiledTarget>, (usize, CompileError)> {
    let indexed: Vec<_> = jobs.iter().enumerate().collect();
    let factored = crate::map_batch(&indexed, |&(job, &(_, target))| {
        kak(target).map_err(|err| (job, err))
    })?;
    // The bindings outlive this call, so their addresses identify them.
    let mut groups = IndexMap::<_, Vec<usize>>::default();
    for (job, ((binding, _), (class, _))) in jobs.iter().zip(&factored).enumerate() {
        let key = (std::ptr::from_ref(*binding).addr(), class.exact_key());
        groups.entry(key).or_default().push(job);
    }
    let groups: Vec<_> = groups.into_values().collect();
    let compiled = crate::map_batch(&groups, |group| {
        let (binding, _) = jobs[group[0]];
        let class = factored[group[0]].0;
        let sentence =
            (binding.decomposer.dressed(class, &binding.frames)).map_err(|err| (group[0], err))?;
        (group.iter())
            .map(|&job| {
                let target = sentence.attach(factored[job].1).map_err(|err| (job, err))?;
                Ok((job, target))
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut compiled: Vec<_> = compiled.into_iter().flatten().collect();
    compiled.sort_unstable_by_key(|&(job, _)| job);
    Ok(compiled.into_iter().map(|(_, target)| target).collect())
}
