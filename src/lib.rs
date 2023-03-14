use fxhash::FxHashMap;
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::prelude::*;
use rand_distr::{WeightedError, WeightedIndex};

// todo: Refactor Urns to be a Vector of Agents
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

    pub fn add_urn(&mut self) -> usize {
        self.data.push(Agent::new());
        let x = self.data.len() - 1;
        self.data[x].id = x;
        self.data.len() - 1
    }

    pub fn add(&mut self, target_agent_id: usize, added_agent_id: usize) {
        assert_ne!(target_agent_id, added_agent_id);

        *self.data[target_agent_id].interactions
            .entry(added_agent_id)
            .or_insert(0) += 1
    }

    pub fn add_many(&mut self, target_agent_id: usize, added_agent_ids: Vec<usize>) {
        for agent_id in added_agent_ids {
            self.add(target_agent_id, agent_id);
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
    pub interactions: FxHashMap<usize, usize>,
    pub gene: AgentGene,
}

#[pymethods]
impl Agent {
    #[new]
    fn new() -> Self {
        return Self {
            id: usize::default(),
            interactions: FxHashMap::default(),
            gene: AgentGene::new(),
        };
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Gene {
    pub rho: usize,
    pub nu: usize,
    pub recentness: f64,
    pub friendship: f64,
}

#[pymethods]
impl Gene {
    #[new]
    fn new(rho: usize, nu: usize, recentness: f64, friendship: f64) -> Self {
        Self {
            rho,
            nu,
            recentness,
            friendship,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AgentGene {
    pub sociality: f64,
}

#[pymethods]
impl AgentGene {
    #[new]
    fn new() -> Self {
        // For now sociability is just set randomly
        let mut rng = thread_rng();
        let random_number: f64 = rng.gen();
        Self {
            sociality: random_number,
        }
    }
}

#[derive(Debug)]
#[pyclass]
pub struct Environment {
    gene: Gene,
    pub urns: Urns,

    /** callerとして選択される可能性のあるエージェント群の (agent_id, weight) の組 */
    pub weights: FxHashMap<usize, usize>,

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
    pub fn new(gene: Gene) -> Self {
        let mut urns = Urns::new();

        urns.add_urn();
        urns.add(0, 1);
        urns.add_urn();
        urns.add(1, 0);

        for agent_id in [0, 1] {
            for _ in 0..(gene.nu + 1) {
                let i = urns.add_urn();
                urns.add(agent_id, i);
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
            weights: candidates,
            recentnesses,
        }
    }

    pub fn get_caller(&self) -> Result<usize, ProcessingError> {
        let mut rng = thread_rng();

        // filter so that only agents that have interacted in the past can be callers
        let  caller_candidates: Vec<Agent> = self.urns.data.clone().into_iter().filter(|agent| !agent.interactions.is_empty()).collect();

        let weights: Vec<f64> = caller_candidates.iter().map(|agent| agent.gene.sociality).collect();

        let caller = WeightedIndex::new(weights)
            .map(|dist| self.weights.keys().nth(dist.sample(&mut rng)).unwrap())
            .copied()?;
        Ok(caller)
    }

    pub fn get_callee(&self, caller: usize) -> Result<usize, ProcessingError> {
        let mut rng = thread_rng();

        let urn = self.urns.get(caller).unwrap();

        let candidates: Vec<usize> = urn.interactions.keys().map(|v| v.to_owned()).collect();
        let weights = urn.interactions.values().map(|v| v.to_owned());
        let callee = WeightedIndex::new(weights)
            .map(|dist| dist.sample(&mut rng))
            .map(|i| candidates[i])?;

        Ok(callee)
    }

    pub fn interact(&mut self, caller: usize, callee: usize) -> Option<()> {
        let is_first_interaction = !self.recentnesses[caller].contains_key(&callee);

        self.history.push((caller, callee));
        *self.recentnesses[caller].entry(callee).or_insert(0) += 1;
        *self.recentnesses[callee].entry(caller).or_insert(0) += 1;

        if !self.weights.contains_key(&callee) {
            self.add_novelty(callee);
        }

        // ρ個の交換(毎回実行)
        *self.weights.entry(caller).or_insert(0) += self.gene.rho;
        *self.weights.entry(callee).or_insert(0) += self.gene.rho;

        self.urns.add_many(caller, vec![callee; self.gene.rho]);
        self.urns.add_many(callee, vec![caller; self.gene.rho]);

        if is_first_interaction {
            let caller_recommendees = self.get_recommendees(caller, callee).unwrap();
            let callee_recommendees = self.get_recommendees(callee, caller).unwrap();

            self.urns.add_many(caller, callee_recommendees);
            self.urns.add_many(callee, caller_recommendees);

            *self.weights.entry(caller).or_insert(0) += self.gene.nu + 1;
            *self.weights.entry(callee).or_insert(0) += self.gene.nu + 1;
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
        urn.interactions.remove(&opponent);
        urn.interactions.remove(&me);
        recentness.remove(&opponent);
        recentness.remove(&me);

        let mut weights_map = FxHashMap::default();

        let max_friendship = urn.interactions.values().fold(f64::NAN, |m, v| (*v as f64).max(m));
        for (agent, weight) in urn.interactions {
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
            let i = self.urns.add_urn();
            self.urns.add(agent_id, i);
            self.recentnesses.push(FxHashMap::default());
        }
        *self.weights.entry(agent_id).or_insert(0) += self.gene.nu + 1;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsurn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Urns>()?;
    m.add_class::<Gene>()?;
    m.add_class::<Environment>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::*;

    #[test]
    fn sample_program() {
        let gene = Gene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: 0.5,
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
        let gene = Gene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: -0.5,
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
        let gene = Gene {
            rho: 3,
            nu: 4,
            recentness: -0.5,
            friendship: 0.5,
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
        let gene = Gene {
            rho: 3,
            nu: 4,
            recentness: 0.0,
            friendship: 0.5,
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
        let gene = Gene {
            rho: 3,
            nu: 4,
            recentness: 0.5,
            friendship: 0.0,
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
        let gene = Gene {
            rho: 5,
            nu: 5,
            recentness: 1.0,
            friendship: 0.0,
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
        let gene = Gene {
            rho: 1,
            nu: 20,
            recentness: 0.5,
            friendship: 0.0,
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
        let (rho, nu, recentness, friendship) = (5, 5, 1.0, 0.0);
        let gene = Gene {
            rho,
            nu,
            recentness,
            friendship,
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
