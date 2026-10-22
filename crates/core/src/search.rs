//! Search native-gate multisets in increasing cost order, fewer gates first
//! among costs equal up to rounding. Reachability is independent of gate
//! order, so each multiset is visited once.
//!
//! The frontier stores only the cost and parent link of each pending candidate.
//! Popping a candidate computes its region with one fixed-size max-plus update.
//! Accepted rows retain that region for later targets and coverage queries.
//! Parent indices reconstruct a sentence only when a caller requests it.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::{Mutex, MutexGuard};

use crate::class::Mono;
use crate::reachability::{BASE_STATE, Region, covers, project};

/// A canonical gate of the ISA and its cost.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    /// The gate's class.
    pub class: Mono,
    /// The gate's cost.
    pub cost: f64,
}

/// A covering sequence; gate indices refer to the configured ISA.
pub struct Selection {
    /// Slot indices.
    pub gates: Vec<usize>,
    /// Total native-gate and local-layer cost: the plain sum, at most
    /// `(max_depth - 1) * tie` above the minimum; see `Walk::tie`.
    pub cost: f64,
    /// Whether the sentence reaches the target's reflection `rho` rather
    /// than the target.
    pub(crate) reflected: bool,
}

/// One pending Horn-DP node in objective order: `key`, depth, then
/// deterministic parent/gate order.
struct Candidate {
    /// `cost + depth * tie`, computed from the plain sum; see `Walk::tie`.
    key: f64,
    cost: f64,
    depth: usize,
    gate_rank: usize,
    parent: Option<usize>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| self.parent.cmp(&other.parent))
            .then_with(|| self.gate_rank.cmp(&other.gate_rank))
    }
}

/// One node of the Horn DP graph, emitted or discarded. A sentence is the
/// parent chain ending at this gate.
struct WalkRow {
    parent: Option<usize>,
    gate_rank: usize,
    depth: usize,
    cost: f64,
    region: Region,
}

/// The walk so far: emitted rows in candidate order, the pending candidates,
/// cheapest first, and the dominance screen.
struct WalkState {
    rows: Vec<WalkRow>,
    heap: BinaryHeap<Reverse<Candidate>>,
    /// Smallest coefficient maximum among emitted rows.
    min_max_bound: f64,
    /// The candidates discarded as dominated, when recorded.
    discarded: Option<Vec<WalkRow>>,
}

impl WalkState {
    fn new(heap: BinaryHeap<Reverse<Candidate>>) -> Self {
        Self {
            rows: Vec::new(),
            heap,
            min_max_bound: f64::INFINITY,
            discarded: None,
        }
    }
}

/// Persistent cost-ordered Horn search for one fixed ISA, shared across threads.
///
/// A query scans the emitted rows under one lock and returns the first
/// covering row, which is the cheapest up to `tie` per gate, with fewer gates
/// preferred. A query that reaches the last row grows the walk one row at a
/// time. Rows are emitted in the same order regardless of which queries cause
/// growth, so results do not depend on thread timing.
pub struct Walk {
    slots: Vec<Slot>,
    local_layer_cost: f64,
    max_depth: usize,
    /// Cost added per gate to the frontier order only. A cost such as 1/3 is
    /// not an f64, so sentences meant to cost the same can differ by rounding:
    /// six of fl(1/3) sum to 2 - 2^-52, below 1 + 1/2 + 1/2, and exact
    /// arithmetic on the inputs agrees. Costs within `tie` per gate therefore
    /// count as equal, and the sentence with fewer gates comes first. Summing
    /// `d` steps of at most `step` each rounds by at most about
    /// `d * d * EPSILON * step`, so `tie = max_depth^2 * EPSILON * step_max`,
    /// with `step_max` the largest gate cost plus `local_layer_cost`, exceeds
    /// that rounding for every sentence the walk can emit.
    tie: f64,
    /// Slot indices in canonical extension order: cost, then slot.
    gates: Vec<usize>,
    state: Mutex<WalkState>,
}

impl Walk {
    /// Seed the frontier with the depth-one sentences. Costs must be finite
    /// and nonnegative and `max_depth` positive.
    pub fn new(slots: Vec<Slot>, local_layer_cost: f64, max_depth: usize) -> Self {
        // Adding zero turns a cost of -0.0 into 0.0, so candidate order is
        // the order of cost values, which dominance relies on.
        let slots: Vec<_> = (slots.into_iter())
            .map(|slot| Slot {
                cost: slot.cost + 0.0,
                ..slot
            })
            .collect();
        let local_layer_cost = local_layer_cost + 0.0;
        let step_max = slots.iter().map(|slot| slot.cost).fold(0.0, f64::max) + local_layer_cost;
        #[allow(clippy::cast_precision_loss)] // depths are far below 2^26, so squares are exact
        let tie = (max_depth * max_depth) as f64 * f64::EPSILON * step_max;
        let mut gates: Vec<usize> = (0..slots.len()).collect();
        gates.sort_by(|&a, &b| {
            slots[a]
                .cost
                .total_cmp(&slots[b].cost)
                .then_with(|| a.cmp(&b))
        });
        let mut walk = Self {
            slots,
            local_layer_cost,
            max_depth,
            tie,
            gates,
            state: Mutex::new(WalkState::new(BinaryHeap::new())),
        };
        let seeds = walk.seeds();
        walk.state.get_mut().expect("a new walk").heap = seeds;
        walk
    }

    /// The depth-one sentences.
    fn seeds(&self) -> BinaryHeap<Reverse<Candidate>> {
        (self.gates.iter().enumerate())
            .map(|(gate_rank, &slot)| {
                let cost = self.slots[slot].cost + self.local_layer_cost + self.local_layer_cost;
                Reverse(Candidate {
                    key: cost + self.tie,
                    cost,
                    depth: 1,
                    gate_rank,
                    parent: None,
                })
            })
            .collect()
    }

    /// Canonical gates: class and cost.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Cost of each surrounding or intervening local layer.
    pub fn local_layer_cost(&self) -> f64 {
        self.local_layer_cost
    }

    /// Maximum number of gates in a sentence.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn lock(&self) -> MutexGuard<'_, WalkState> {
        self.state.lock().expect("a walk query panicked")
    }

    /// Emit the next cost-ordered sentence as a DP node. False when exhausted.
    fn grow_one(&self, state: &mut WalkState) -> bool {
        while let Some(Reverse(Candidate {
            key: _,
            cost,
            depth,
            gate_rank,
            parent,
        })) = state.heap.pop()
        {
            let native = project(self.slots[self.gates[gate_rank]].class);
            // Rows store tightened states, so the dominance test below is
            // exact up to rounding; see `Region::tighten`. A one-gate region
            // is a single point whose subset sums are already tight.
            let node = WalkRow {
                parent,
                gate_rank,
                depth,
                cost,
                region: match parent {
                    Some(p) => state.rows[p].region.product(&native).tighten(),
                    None => native,
                },
            };

            // Rows are emitted in candidate order, so an earlier row never has
            // a larger key and, at equal key, is never deeper. One that
            // contains this region, with a gate order that admits every
            // extension of it, covers each continuation at no larger key:
            // max-plus maps preserve the componentwise order. So the whole
            // pending subtree is redundant; `max_depth` only bounds the search.
            // Any dominating row has a coefficient maximum no larger than this
            // region's, so a new minimum rules out every prior row.
            let max_bound = node.region.max_bound();
            let dominated = max_bound >= state.min_max_bound
                && state.rows.iter().any(|row| {
                    row.gate_rank <= gate_rank && row.region.contains_by_bounds(&node.region)
                });
            if dominated {
                if let Some(discarded) = &mut state.discarded {
                    discarded.push(node);
                }
                continue;
            }

            state.min_max_bound = state.min_max_bound.min(max_bound);
            let row = state.rows.len();
            state.rows.push(node);
            if depth < self.max_depth {
                // Reachability is order-invariant, so enumerate each gate multiset once.
                for next_rank in gate_rank..self.gates.len() {
                    let cost =
                        cost + (self.slots[self.gates[next_rank]].cost + self.local_layer_cost);
                    #[allow(clippy::cast_precision_loss)] // depth <= max_depth, far below 2^52
                    let key = cost + (depth + 1) as f64 * self.tie;
                    state.heap.push(Reverse(Candidate {
                        key,
                        cost,
                        depth: depth + 1,
                        gate_rank: next_rank,
                        parent: Some(row),
                    }));
                }
            }
            return true;
        }
        false
    }

    /// One target-independent coverage candidate. Grows the cost-ordered Horn
    /// walk only as far as requested and returns its sentence, cost, and region.
    pub fn coverage_candidate(&self, row: usize) -> Option<(Vec<usize>, f64, Region)> {
        let mut state = self.lock();
        if !self.grow_to(&mut state, row) {
            return None;
        }
        let emitted = &state.rows[row];
        Some((self.sentence(&state, row), emitted.cost, emitted.region))
    }

    /// The candidates the walk discards as dominated before it emits row `row`:
    /// sentence, cost, and region. Replays the walk from its seeds, so earlier
    /// queries do not change the result. `None` if the walk ends first.
    pub fn dominated(&self, row: usize) -> Option<Vec<(Vec<usize>, f64, Region)>> {
        let mut replay = WalkState {
            discarded: Some(Vec::new()),
            ..WalkState::new(self.seeds())
        };
        if !self.grow_to(&mut replay, row) {
            return None;
        }
        let discarded = replay.discarded.take().unwrap_or_default();
        let discarded = discarded.into_iter().map(|node| {
            let mut sentence = (node.parent).map_or_else(Vec::new, |p| self.sentence(&replay, p));
            sentence.push(self.gates[node.gate_rank]);
            (sentence, node.cost, node.region)
        });
        Some(discarded.collect())
    }

    /// Grow the walk until it has row `row`. False if it ends first.
    fn grow_to(&self, state: &mut WalkState, row: usize) -> bool {
        while row >= state.rows.len() {
            if !self.grow_one(state) {
                return false;
            }
        }
        true
    }

    /// Slot indices of the sentence at `row`, reconstructed from its DP parents.
    fn sentence(&self, state: &WalkState, row: usize) -> Vec<usize> {
        let mut sentence = Vec::with_capacity(state.rows[row].depth);
        let mut cursor = Some(row);
        while let Some(i) = cursor {
            sentence.push(self.gates[state.rows[i].gate_rank]);
            cursor = state.rows[i].parent;
        }
        sentence.reverse();
        sentence
    }

    /// The cheapest sentence and its cost, without a trajectory or realization.
    /// `None` marks an unreachable target; the identity orbit selects the empty
    /// sentence at the local-layer cost.
    pub fn select(&self, target: Mono) -> Option<Selection> {
        let direct = project(target);
        let rho = project(target.rho());
        // Whether `region` covers the target, and if so, whether only its reflection.
        let reaches = |region: &Region| {
            if covers(region, &direct) {
                Some(false)
            } else {
                covers(region, &rho).then_some(true)
            }
        };
        if let Some(reflected) = reaches(&BASE_STATE) {
            return Some(Selection {
                gates: Vec::new(),
                cost: self.local_layer_cost,
                reflected,
            });
        }
        let mut state = self.lock();
        let mut row = 0;
        loop {
            while let Some(WalkRow { region, cost, .. }) = state.rows.get(row) {
                if let Some(reflected) = reaches(region) {
                    return Some(Selection {
                        gates: self.sentence(&state, row),
                        cost: *cost,
                        reflected,
                    });
                }
                row += 1;
            }
            if !self.grow_one(&mut state) {
                return None;
            }
        }
    }
}
