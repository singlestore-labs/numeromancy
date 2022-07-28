use std::cell::RefCell;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use byte_slice_cast::{AsByteSlice, AsSliceOf};
use mysql::prelude::Queryable;
use mysql::Pool;
use ndarray::{array, Array1, Array2};
use simple_error::SimpleError;

type SimpleResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct ProblemExecutor {
    conn: RefCell<mysql::PooledConn>,
}

impl ArgminOp for ProblemExecutor {
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = Array2<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let mut conn = self.conn.borrow_mut();
        let param_vec = param.to_vec();
        let param_u8 = param_vec.as_byte_slice();
        let result =
            conn.exec_first::<f64, _, _>("select cost from cancer_remission_cost(?)", (param_u8,))?;
        match result {
            Some(result) => Ok(result),
            None => Err(SimpleError::new("No result returned").into()),
        }
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let mut conn = self.conn.borrow_mut();
        let param_vec = param.to_vec();
        let param_u8 = param_vec.as_byte_slice();
        let result = conn.exec_first::<Vec<u8>, _, _>(
            "select gradient from cancer_remission_grad(?)",
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

fn main() -> SimpleResult<()> {
    let url = "mysql://admin:mugFQjS9G65LMuFB2qar@svc-388d373a-716b-4adf-ac18-875d06b74a78-dml.aws-virginia-2.svc.singlestore.com/numeromancy";
    let pool = Pool::new(url)?;
    let conn = pool.get_conn()?;

    let cost = ProblemExecutor {
        conn: RefCell::new(conn),
    };
    let init_param: Array1<f64> = array![0., 0., 0., 0., 0., 0., 0.];
    let init_hessian: Array2<f64> = Array2::eye(7);
    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(init_hessian, linesearch)
        .with_tol_cost(1e-6)
        .with_tol_grad(1e-8);

    let executor = Executor::new(cost, solver, init_param)
        .max_iters(50)
        .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always);
    let result = executor.run()?;

    println!("{}", result);

    Ok(())
}
