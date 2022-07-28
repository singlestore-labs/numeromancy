use std::cell::RefCell;

use anyhow::Result;
use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use byte_slice_cast::{AsByteSlice, AsSliceOf};
use clap::Parser;
use mysql::prelude::Queryable;
use mysql::Pool;
use ndarray::{Array1, Array2};
use settings::ProblemConfig;
use simple_error::SimpleError;

use crate::settings::{Config, SolverConfig};

mod settings;

struct ProblemExecutor {
    conn: RefCell<mysql::PooledConn>,
    conf: ProblemConfig,
}

impl ArgminOp for ProblemExecutor {
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = Array2<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let cost_fn = self
            .conf
            .cost_fn
            .as_ref()
            .ok_or(ArgminError::NotImplemented {
                text: "cost_fn not defined".to_string(),
            })?;

        let mut conn = self.conn.borrow_mut();
        let param_vec = param.to_vec();
        let param_u8 = param_vec.as_byte_slice();
        let result =
            conn.exec_first::<f64, _, _>(format!("select cost from {cost_fn}(?)"), (param_u8,))?;
        match result {
            Some(result) => Ok(result),
            None => Err(SimpleError::new("No result returned").into()),
        }
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let grad_fn = self
            .conf
            .grad_fn
            .as_ref()
            .ok_or(ArgminError::NotImplemented {
                text: "grad_fn not defined".to_string(),
            })?;

        let mut conn = self.conn.borrow_mut();
        let param_vec = param.to_vec();
        let param_u8 = param_vec.as_byte_slice();
        let result = conn.exec_first::<Vec<u8>, _, _>(
            format!("select gradient from {grad_fn}(?)"),
            (param_u8,),
        )?;
        match result {
            Some(result) => Ok(Array1::from_shape_vec(
                (7,),
                result.as_slice_of::<f64>().unwrap().to_vec(),
            )?),
            None => Err(SimpleError::new("No result returned").into()),
        }
    }
}

fn main() -> Result<()> {
    let args = settings::Args::parse();
    let config_contents = std::fs::read_to_string(&args.config)?;
    let config: Config = toml::from_str(config_contents.as_str())?;
    let pool = Pool::new(config.database)?;

    let cost = ProblemExecutor {
        conn: RefCell::new(pool.get_conn()?),
        conf: config.problem,
    };
    let init_param: Array1<f64> = config.optimizer.init_param.into();
    let init_hessian: Array2<f64> = Array2::eye(init_param.dim());
    let linesearch = MoreThuenteLineSearch::new();

    let result = match config.solver {
        SolverConfig::Bfgs { tol_cost, tol_grad } => {
            let solver = BFGS::new(init_hessian, linesearch)
                .with_tol_cost(tol_cost)
                .with_tol_grad(tol_grad);

            let executor = Executor::new(cost, solver, init_param)
                .max_iters(config.optimizer.max_iters)
                .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always);

            executor.run()?
        }
        SolverConfig::SteepestDescent => {
            let solver = SteepestDescent::new(linesearch);
            let executor = Executor::new(cost, solver, init_param)
                .max_iters(config.optimizer.max_iters)
                .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always);

            executor.run()?
        }
    };

    println!("{}", result);

    let output_fn_name = config.output.fn_name;
    let params_arr = serde_json::to_string(&result.state.best_param.to_vec())?;
    pool.get_conn()?.query_drop(format!(
        "
        create or replace function {output_fn_name}(x array(double not null))
        returns double as
        begin
            return log_regression_infer({params_arr}, x);
        end
        "
    ))?;

    Ok(())
}
