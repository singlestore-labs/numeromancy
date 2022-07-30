create database if not exists numeromancy;
use numeromancy;

create or replace function log_regression_infer as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function log_regression_cost as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function log_regression_gradient as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function log_regression_hessian as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function vec_pack_f64 as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function vec_unpack_f64 as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";