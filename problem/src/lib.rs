use byte_slice_cast::*;

const EPS: f64 = 1e-12;

wit_bindgen_rust::export!("interface.wit");

fn vector_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn vector_mult(a: &[f64], b: f64) -> Vec<f64> {
    a.iter().map(|a| a * b).collect()
}

pub struct Interface;
impl interface::Interface for Interface {
    fn vec_pack_f64(v: Vec<f64>) -> Vec<u8> {
        v.as_byte_slice().to_vec()
    }

    fn vec_unpack_f64(v: Vec<u8>) -> Vec<f64> {
        v.as_slice_of::<f64>().unwrap().to_vec()
    }

    fn log_regression_infer(param: Vec<f64>, x: Vec<f64>) -> f64 {
        let linp = vector_dot_product(&param, &x);
        1. / (1. + (-linp).exp())
    }

    fn log_regression_cost(param: Vec<f64>, x: Vec<f64>, y: f64) -> f64 {
        let pi = Interface::log_regression_infer(param, x);
        if (y - 1.).abs() < EPS {
            -pi.ln()
        } else {
            -(1. - pi).ln()
        }
    }

    fn log_regression_gradient(param: Vec<f64>, x: Vec<f64>, y: f64) -> Vec<f64> {
        let linp = vector_dot_product(&param, &x);
        let fact = if (y - 1.).abs() < EPS {
            -1.
        } else {
            linp.exp()
        } / (1. + linp.exp());

        vector_mult(&x, fact)
    }

    fn log_regression_hessian(param: Vec<f64>, x: Vec<f64>, y: f64) -> Vec<f64> {
        let mut hess: Vec<f64> = vec![0.0; x.len() * x.len()];
        let linp = vector_dot_product(&param, &x);
        let prob = 1. / (1. + (-linp).exp());
        let fact = if (y - 1.).abs() < EPS {
            prob
        } else {
            1. / (1. + (-linp).exp())
        } / (1. + linp.exp());

        let mut index: usize = 0;
        for i in 0..x.len() {
            for j in 0..x.len() {
                hess[index] = fact * x[i] * x[j];
                index += 1;
            }
        }

        hess
    }
}
