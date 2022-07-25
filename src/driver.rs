use std::cell::RefCell;
use std::rc::Rc;

use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
use argmin::core::{
    CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, SerializeAlias, State,
};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};

#[derive(Clone)]
struct Solution {
    cost: f64,
    gradient: Vec<f64>,
}

impl CostFunction for Solution {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.cost)
    }
}

impl Gradient for Solution {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, _: &Self::Param) -> Result<Self::Param, Error> {
        Ok(self.gradient.clone())
    }
}

struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

#[derive(Default, Clone)]
pub struct Serialized {
    serialized_executor: Vec<u8>,
    param: Vec<f64>,
    solution: Option<Solution>,
}

pub struct Checkpointer {
    state: Rc<RefCell<Serialized>>,
}

impl<S, I> Checkpoint<S, I> for Checkpointer
where
    S: SerializeAlias + DeserializeOwnedAlias,
    I: SerializeAlias + DeserializeOwnedAlias,
{
    fn save(&self, solver: &S, state: &I) -> Result<(), Error> {
        let serialized = bincode::serialize(&(solver, state))?;
        self.state.borrow_mut().serialized_executor = serialized;
        Ok(())
    }

    fn load(&self) -> Result<Option<(S, I)>, Error> {
        let serialized = &self.state.borrow().serialized_executor;
        if serialized.is_empty() {
            return Ok(None);
        }
        let result: (S, I) = bincode::deserialize(serialized)?;
        Ok(Some(result))
    }

    fn frequency(&self) -> CheckpointingFrequency {
        CheckpointingFrequency::Always
    }
}

pub fn solve(iters: u64) -> Result<String, Error> {
    let problem = Rosenbrock {};
    let init_param: Vec<f64> = vec![-1.2, 1.0];
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let executor =
        Executor::new(problem, solver).configure(|state| state.param(init_param).max_iters(iters));

    let result = executor.run()?;

    Ok(format!("{}", result))
}

pub fn iterate(state: Serialized) -> Result<(String, Serialized), Error> {
    let problem = state.solution.clone().unwrap();
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let state = Rc::new(RefCell::new(state));
    let checkpointer = Checkpointer {
        state: state.clone(),
    };

    let executor = Executor::new(problem, solver)
        .checkpointing(checkpointer)
        .configure(|s| s.param(state.borrow().param.clone()))
        .iterations(1);

    let result = executor.run()?;

    Ok((
        format!("{}", result),
        Serialized {
            serialized_executor: state.take().serialized_executor,
            param: result.state.get_param().unwrap().clone(),
            solution: None,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        println!("singlepass: {}", solve(1).unwrap());

        // XXX
        // currently this doesn't work because the steepest descent solver
        // actually runs a iterative solution recursively. So... this approach
        // is not correct.
        // In order to continue along this road, we will need to refactor argmin
        // to support a coroutine style for the problem type

        let mut state = Serialized {
            param: vec![-1.2, 1.0],
            serialized_executor: vec![],
            solution: None,
        };

        let mut last_result = "".to_string();
        for i in 0..10 {
            state.solution = Solution {
                cost: rosenbrock_2d(&state.param, 1.0, 100.0),
                gradient: rosenbrock_2d_derivative(&state.param, 1.0, 100.0),
            }
            .into();

            println!("in params {:?}", state.param);
            println!("in serialized {:?}", state.serialized_executor.len());
            (last_result, state) = iterate(state).unwrap();
            print!("iterate {}: {}", i, last_result);
            println!("new params {:?}", state.param);
        }
        println!("iterate: {}", last_result);
    }
}
