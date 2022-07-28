use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[clap(name = "numeromancy")]
#[clap(version = "0.1")]
#[clap(about = "Numeromancy is a numerical optimization tool which runs on SingleStore.")]
pub struct Args {
    #[clap(short, long, value_parser, default_value_t = String::from("config.toml"))]
    pub config: String,
}

#[derive(Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub optimizer: OptimizerConfig,
    pub solver: SolverConfig,
    pub problem: ProblemConfig,
    pub output: OutputConfig,
}

#[derive(Deserialize)]
pub struct OptimizerConfig {
    pub init_param: Vec<f64>,
    pub max_iters: u64,
}

#[derive(Deserialize)]
pub struct OutputConfig {
    pub fn_name: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum SolverConfig {
    Bfgs { tol_cost: f64, tol_grad: f64 },
    SteepestDescent,
}

#[derive(Deserialize)]
pub struct ProblemConfig {
    pub cost_fn: Option<String>,
    pub grad_fn: Option<String>,
    pub hessian_fn: Option<String>,
}

#[derive(Deserialize)]
pub struct DatabaseConfig {
    host: String,
    port: Option<u16>,
    user: Option<String>,
    password: Option<String>,
    database: Option<String>,
}

impl From<DatabaseConfig> for mysql::Opts {
    fn from(config: DatabaseConfig) -> Self {
        mysql::OptsBuilder::new()
            .ip_or_hostname(Some(config.host))
            .tcp_port(config.port.unwrap_or(3306))
            .user(config.user.or_else(|| Some("root".into())))
            .pass(config.password)
            .db_name(config.database.or_else(|| Some("numeromancy".into())))
            .into()
    }
}
