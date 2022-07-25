use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use byte_slice_cast::{AsByteSlice, AsSliceOf};
use mysql::prelude::Queryable;
use mysql::Pool;
use simple_error::SimpleError;

type SimpleResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct ProblemExecutor {
    pool: Pool,
}

impl CostFunction for ProblemExecutor {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let mut conn = self.pool.get_conn()?;
        let param_u8 = param.as_byte_slice();
        let result = conn.exec_first::<f64, _, _>("select opt_cost(?)", (param_u8,))?;
        match result {
            Some(result) => Ok(result),
            None => Err(SimpleError::new("No result returned").into()),
        }
    }
}

impl Gradient for ProblemExecutor {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let mut conn = self.pool.get_conn()?;
        let param_u8 = param.as_byte_slice();
        let result = conn.exec_first::<Vec<u8>, _, _>("select opt_gradient(?)", (param_u8,))?;
        match result {
            Some(result) => Ok(result.as_slice_of::<f64>().unwrap().to_vec()),
            None => Err(SimpleError::new("No result returned").into()),
        }
    }
}

fn main() -> SimpleResult<()> {
    let url = "mysql://root:test@172.17.0.3:3306/numeromancy";
    let pool = Pool::new(url)?;

    let problem = ProblemExecutor { pool };
    let init_param: Vec<f64> = vec![-1.2, 1.0];
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let executor =
        Executor::new(problem, solver).configure(|state| state.param(init_param).max_iters(10));

    let result = executor.run()?;

    println!("{}", result);

    Ok(())
}
