use fxhash::FxHashMap;
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::prelude::*;
use rand_distr::{WeightedError, WeightedIndex};

#[derive(Debug)]
#[pyclass]
pub struct Urns {
    pub data: Vec<FxHashMap<usize, usize>>,
}

impl Urns {
    pub fn new() -> Self {
        return Self { data: vec![] };
    }

    pub fn add_urn(&mut self) -> usize {
        self.data.push(FxHashMap::default());
        self.data.len() - 1
    }

    pub fn add(&mut self, target_agent_id: usize, added_agent_id: usize) {
        assert_ne!(target_agent_id, added_agent_id);

        *self.data[target_agent_id]
            .entry(added_agent_id)
            .or_insert(0) += 1
    }

    pub fn add_many(&mut self, target_agent_id: usize, added_agent_ids: Vec<usize>) {
        for agent_id in added_agent_ids {
            self.add(target_agent_id, agent_id);
        }
    }

    pub fn get(&self, agent_id: usize) -> Option<&FxHashMap<usize, usize>> {
        self.data.get(agent_id)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Gene {
    pub rho: usize,
    pub nu: usize,
}

#[pymethods]
impl Gene {
    #[new]
    fn new(rho: usize, nu: usize) -> Self {
        Self { rho, nu }
    }
}

#[derive(Debug)]
#[pyclass]
pub struct Environment {
    gene: Gene,
    pub urns: Urns,

    /** callerとして選択される可能性のあるエージェント群の (agent_id, weight) の組 */
    weights: FxHashMap<usize, usize>,

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

        Environment {
            history: vec![],
            gene,
            urns,
            weights: candidates,
        }
    }

    pub fn get_caller(&self) -> Result<usize, ProcessingError> {
        let mut rng = thread_rng();

        let weights = self.weights.values();

        let caller = WeightedIndex::new(weights)
            .map(|dist| self.weights.keys().nth(dist.sample(&mut rng)).unwrap())
            .copied()?;
        Ok(caller)
    }

    pub fn get_callee(&self, caller: usize) -> Result<usize, ProcessingError> {
        let mut rng = thread_rng();

        let urn = self.urns.get(caller).unwrap();

        let candidates: Vec<usize> = urn.keys().map(|v| v.to_owned()).collect();
        let weights = urn.values().map(|v| v.to_owned());
        let callee = WeightedIndex::new(weights)
            .map(|dist| dist.sample(&mut rng))
            .map(|i| candidates[i])?;

        Ok(callee)
    }

    pub fn interact(&mut self, caller: usize, callee: usize) -> Option<()> {
        self.history.push((caller, callee));

        if !self.weights.contains_key(&callee) {
            self.add_novelty(callee);
        }

        let caller_recommendees = self.get_recommendees(caller, callee).unwrap();
        let callee_recommendees = self.get_recommendees(callee, caller).unwrap();

        self.urns.add_many(caller, callee_recommendees);
        self.urns.add_many(caller, vec![callee; self.gene.rho]);

        self.urns.add_many(callee, caller_recommendees);
        self.urns.add_many(callee, vec![caller; self.gene.rho]);

        *self.weights.entry(caller).or_insert(0) += self.gene.rho + self.gene.nu + 1;
        *self.weights.entry(callee).or_insert(0) += self.gene.rho + self.gene.nu + 1;

        Some(())
    }

    fn get_recommendees(&self, me: usize, opponent: usize) -> Result<Vec<usize>, ProcessingError> {
        let mut rng = thread_rng();
        let mut ret = vec![];

        let urn = self.urns.get(me).unwrap();

        // 計算用に壺のコピーを作成
        let mut urn = urn.clone();

        // 相手は選択しないよう予め重みを0にする
        *urn.entry(opponent).or_insert(0) = 0;

        let candidates: Vec<usize> = urn.keys().map(|v| v.to_owned()).collect();
        let weights: Vec<usize> = urn.values().map(|v| v.to_owned()).collect();

        let mut dist = WeightedIndex::new(weights.to_vec())?;

        for _ in 0..(self.gene.nu + 1) {
            let i = dist.sample(&mut rng);
            ret.push(candidates[i]);

            // 一度選択したものは重みを0にして重複して選択されないようにする
            let _ = dist.update_weights(&[(i, &0)]);
        }

        Ok(ret)
    }

    fn add_novelty(&mut self, agent_id: usize) {
        for _ in 0..(self.gene.nu + 1) {
            let i = self.urns.add_urn();
            self.urns.add(agent_id, i);
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
    use crate::*;

    #[test]
    fn sample_program() {
        let gene = Gene { rho: 3, nu: 4 };
        let mut env = Environment::new(gene);

        for _ in 0..1000 {
            let caller = env.get_caller().unwrap();
            let callee = env.get_callee(caller).unwrap();
            let _ = env.interact(caller, callee);
        }

        assert_eq!(env.history.len(), 1000);
    }
}
