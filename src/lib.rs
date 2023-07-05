use fxhash::FxHashMap;
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::prelude::*;
use rand_distr::{WeightedError, WeightedIndex};
use rand::Rng;

#[derive(Debug)]
#[pyclass]
pub struct Urns {
    pub data: Vec<Agent>,
}

impl Urns {
    // function to create empty urns
    pub fn new() -> Self {
        return Self { data: vec![] };
    }

    //
    //  Creates a new urn (agent)
    //
    // Returns the id of the newly created agent
    //
    pub fn create_new_agent(&mut self) -> usize {
        self.data.push(Agent::new());
        let new_agent_id = self.data.len() - 1;
        self.data[new_agent_id].id = new_agent_id;
        self.data.len() - 1
    }

    pub fn actualise_agent(&mut self,  target_agent_id: usize, added_agent_id: usize) {
        // verify that the caller and callee are different
        assert_ne!(target_agent_id, added_agent_id);

        if let Some((key, value)) = self.data[target_agent_id].adjacent_possible_space.remove_entry(&added_agent_id) {
            self.data[target_agent_id].actual_space.insert(key, value);
        }
    }

    //
    // Update the agents actual space
    //
    pub fn add_to_actual_space(&mut self, target_agent_id: usize, added_agent_id: usize) {

        // verify that the caller and callee are different
        assert_ne!(target_agent_id, added_agent_id);

        // update the all the interactions of agent
        *self.data[target_agent_id].actual_space
            .entry(added_agent_id)
            .or_insert(0) += 1;

        // update the total number of interactions
        self.data[target_agent_id].total_interactions += 1;

        // update the unique number of interactions
        self.data[target_agent_id].unique_interactions = self.data[target_agent_id].actual_space.len();
    }

    //
    // Update the agents adjacent possible space
    //
    pub fn add_to_adjacent_possible_space(&mut self, target_agent_id: usize, added_agent_id: usize) {
        assert_ne!(target_agent_id, added_agent_id);

        // update the all the interactions of agent
        *self.data[target_agent_id].adjacent_possible_space
            .entry(added_agent_id)
            .or_insert(0) += 1;
    }

    pub fn add_many_to_adjacent_possible_space(&mut self, target_agent_id: usize, added_agent_ids: Vec<usize>) {
        for agent_id in added_agent_ids {
            self.add_to_adjacent_possible_space(target_agent_id, agent_id);
        }
    }

    pub fn get(&self, agent_id: usize) -> Option<&Agent> {
        self.data.get(agent_id)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Agent {
    pub id: usize,
    pub interacted: bool,
    pub actual_space: FxHashMap<usize, usize>,
    pub adjacent_possible_space: FxHashMap<usize, usize>,
    pub total_interactions: usize, // number of interactions
    pub unique_interactions: usize, // degree of node
    pub gene: AgentGene,
}

#[pymethods]
impl Agent {
    #[new]
    fn new() -> Self {
        return Self {
            id: usize::default(),
            interacted: false,
            actual_space: FxHashMap::default(),
            adjacent_possible_space: FxHashMap::default(),
            total_interactions: usize::default(),
            unique_interactions: usize::default(),
            gene: AgentGene::new(),
        };
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct EnvironmentGene {
    pub rho: usize,
    pub nu: usize,
    pub recentness: f64,
    pub friendship: f64,
    pub symmetry: f64
}

#[pymethods]
impl EnvironmentGene {
    #[new]
    fn new(rho: usize, nu: usize, recentness: f64, friendship: f64, symmetry: f64) -> Self {
        Self {
            rho,
            nu,
            recentness,
            friendship,
            symmetry,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AgentGene {
    pub immediacy: f64,
    pub longevity: f64,
    pub fitness: f64,
}

#[pymethods]
impl AgentGene {
    #[new]
    fn new() -> Self {
        let mut rng = thread_rng();
        Self {
            immediacy: rng.gen_range(0.1..=0.9),
            longevity: rng.gen_range(0.1..=0.9),
            fitness: rng.gen_range(10.0..=100.0)
        }
    }
}

#[derive(Debug)]
#[pyclass]
pub struct Environment {
    gene: EnvironmentGene,
    pub urns: Urns,

    /** 最近度 */
    recentnesses: Vec<FxHashMap<usize, usize>>,

    /** (caller, callee) で表される生成データ */
    #[pyo3(get)]
    pub history: Vec<(usize, usize)>,
}

impl From<ProcessingError> for PyErr {
    fn from(error: ProcessingError) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

impl From<WeightedError> for ProcessingError {
    fn from(error: WeightedError) -> Self {
        ProcessingError(error)
    }
}

#[derive(Debug)]
pub struct ProcessingError(WeightedError);

#[pymethods]
impl Environment {
    #[new]
    pub fn new(gene: EnvironmentGene) -> Self {
        let mut urns = Urns::new();

        urns.create_new_agent();
        urns.add_to_actual_space(0, 1);
        urns.data[0].interacted = true;
        urns.create_new_agent();
        urns.add_to_actual_space(1, 0);
        urns.data[1].interacted = true;


        for agent_id in [0, 1] {
            for _ in 0..(gene.nu + 1) {
                let i = urns.create_new_agent(); // new empty urns are added
                urns.add_to_actual_space(agent_id, i);
            }
        }

        let mut candidates = FxHashMap::default();
        for agent_id in [0, 1] {
            candidates.insert(agent_id, gene.nu + 2);
        }

        let mut recentnesses = vec![];
        for _ in 0..(2 + 2 * (gene.nu + 1)) {
            recentnesses.push(FxHashMap::default());
        }

        Environment {
            history: vec![],
            gene,
            urns,
            recentnesses,
        }
    }

    pub fn get_caller(&self) -> Result<usize, ProcessingError> {
        let mut rng = rand::thread_rng();

        let caller_candidates: Vec<Agent> = self.urns.data.clone().into_iter().filter(|agent| !agent.actual_space.is_empty()).collect();
        let caller: usize = caller_candidates.choose(&mut rng).unwrap().id;
        Ok(caller)
    
    }

    pub fn get_callee(&self, caller: usize) -> Result<usize, ProcessingError> {
        let mut rng = thread_rng();

        let urn = self.urns.get(caller).unwrap();

        let mut callee_candidate_space: FxHashMap<usize, usize> = FxHashMap::default();

        // Union the actual space and the adjacent possible space
        callee_candidate_space.extend(&urn.actual_space);
        callee_candidate_space.extend(&urn.adjacent_possible_space);

        let candidates: Vec<usize> = callee_candidate_space.keys().map(|v| v.to_owned()).collect();

        let weights = callee_candidate_space.values().map(|v| v.to_owned());
        let callee = WeightedIndex::new(weights)
            .map(|dist: WeightedIndex<usize>| dist.sample(&mut rng))
            .map(|i| candidates[i])?;

        Ok(callee)
    }

    pub fn interact(&mut self, caller: usize, callee: usize) -> Option<()> {

        // check if this is the first time the caller and callee are interacting
        let is_first_interaction = !self.urns.data[caller].actual_space.contains_key(&callee);

        // add the interaction to the history
        self.history.push((caller, callee));

        // if this is the first time the callee has ever been selected
        if !self.urns.data[callee].interacted {
            self.add_novelty(callee);
        }

        if is_first_interaction {

            // the callee gets moved the caller agents actual space
            self.urns.actualise_agent(caller, callee);
            self.urns.actualise_agent(callee, caller);

            let caller_recommendees = self.get_recommendees(caller, callee).unwrap();
            let callee_recommendees = self.get_recommendees(callee, caller).unwrap();

            self.urns.add_many_to_adjacent_possible_space(caller, callee_recommendees);
            self.urns.add_many_to_adjacent_possible_space(callee, caller_recommendees);
        }

        // ρ個の交換(毎回実行) REINFORCEMENT
        *self.urns.data[caller].actual_space.entry(callee).or_insert(0) += self.gene.rho;
        *self.urns.data[callee].actual_space.entry(caller).or_insert(0) += self.gene.rho;

        if self.gene.symmetry < -0.3 {
            *self.recentnesses[caller].entry(callee).or_insert(0) += 1;
        } else if self.gene.symmetry > 0.3 {
            *self.recentnesses[callee].entry(caller).or_insert(0) += 1;
        } else {
            *self.recentnesses[caller].entry(callee).or_insert(0) += 1;
            *self.recentnesses[callee].entry(caller).or_insert(0) += 1;
        }

        Some(())
    }

    fn get_recommendees(&self, me: usize, opponent: usize) -> Result<Vec<usize>, ProcessingError> {
        let mut rng = thread_rng();
        let mut ret = vec![];
        
        let urn = self.urns.get(me).unwrap();
        let recentness = self.recentnesses.get(me).unwrap();

        // 計算用にコピーを作成
        let mut urn = urn.clone();
        let mut recentness = recentness.clone();

        // 自分自身と相手自身を取り除く
        urn.actual_space.remove(&opponent);
        urn.actual_space.remove(&me);
        recentness.remove(&opponent);
        recentness.remove(&me);

        let mut weights_map = FxHashMap::default();

        let max_friendship = urn.actual_space
            .values()
            .fold(f64::NAN, |m, v| (*v as f64).max(m));
        for (agent, weight) in urn.actual_space {
            *weights_map.entry(agent).or_insert(0.0) +=
                (weight as f64 / max_friendship) * self.gene.friendship;
        }

        let max_recentness = recentness
            .values()
            .fold(f64::NAN, |m, v| (*v as f64).max(m));
        for (agent, weight) in recentness {
            *weights_map.entry(agent).or_insert(0.0) +=
                (weight as f64 / max_recentness) * self.gene.recentness;
        }

        let min_weight = weights_map.values().fold(f64::NAN, |m, v| v.min(m));
        for w in weights_map.values_mut() {
            *w += min_weight.abs() + 10f64.powf(-10f64);
        }

        let candidates: Vec<usize> = weights_map.keys().copied().collect();
        let mut weights = Vec::from_iter(weights_map.values().cloned());

        for _ in 0..(self.gene.nu + 1) {
            // problem with the weights
            let dist = WeightedIndex::new(weights.clone())?;

            let i = dist.sample(&mut rng);
            ret.push(candidates[i]);

            // 一度選択したものは重みを0にして重複して選択されないようにする
            weights[i] = 0.0;
        }

        Ok(ret)
    }

    fn add_novelty(&mut self, agent_id: usize) {
        for _ in 0..(self.gene.nu + 1) {
            let i = self.urns.create_new_agent();
            self.urns.add_to_actual_space(agent_id, i);
            self.urns.add_to_actual_space(i, agent_id);
            self.recentnesses.push(FxHashMap::default());
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsurn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Urns>()?;
    m.add_class::<EnvironmentGene>()?;
    m.add_class::<Environment>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::*;

    #[test]
    fn sample_program() {
        let gene = EnvironmentGene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: 0.5,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn negative_friendship() {
        let gene: EnvironmentGene = EnvironmentGene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: -0.5,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn negative_recentness() {
        let gene = EnvironmentGene {
            rho: 3,
            nu: 4,
            recentness: -0.5,
            friendship: 0.5,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn zero_recentness() {
        let gene = EnvironmentGene {
            rho: 3,
            nu: 4,
            recentness: 0.0,
            friendship: 0.5,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn zero_friendship() {
        let gene = EnvironmentGene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: 0.0,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn rho_greater_than_nu() {
        let gene = EnvironmentGene {
            rho: 5,
            nu: 5,
            recentness: 1.0,
            friendship: 0.0,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn nu_greater_than_rho() {
        let gene = EnvironmentGene {
            rho: 1,
            nu: 20,
            recentness: 0.5,
            friendship: 0.0,
            symmetry: 0.0,
        };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }

    #[test]
    fn do_not_recommend_same_agents() {
        let (rho, nu, recentness, friendship, symmetry) = (5, 5, 1.0, 0.0, 0.0);
        let gene = EnvironmentGene {
            rho,
            nu,
            recentness,
            friendship,
            symmetry
        };
        let mut env = Environment::new(gene);

        env.interact(1, 10);
        let (me, opponent) = (1, 11);
        env.add_novelty(opponent);
        let recommendees = env.get_recommendees(me, opponent).unwrap();

        let set: HashSet<usize> = HashSet::from_iter(recommendees.clone());

        assert_eq!(set.len(), nu + 1);
        assert_eq!(recommendees.len(), nu + 1);
    }
}
