use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
use byte_slice_cast::*;

wit_bindgen_rust::export!("interface.wit");

pub struct Interface;
impl interface::Interface for Interface {
    fn opt_cost(param: Vec<u8>) -> f64 {
        let param = param.as_slice_of::<f64>().unwrap();
        rosenbrock_2d(param, 1.0, 100.0)
    }

    fn opt_gradient(param: Vec<u8>) -> Vec<u8> {
        let param = param.as_slice_of::<f64>().unwrap();
        let out: Vec<f64> = rosenbrock_2d_derivative(param, 1.0, 100.0);
        out.as_byte_slice().to_vec()
    }
}
