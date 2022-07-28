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

create or replace function vec_pack_f64 as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function vec_unpack_f64 as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

-- problem #1: cancer remission

create table if not exists cancer_remission (
  remiss int,
  cell double,
  smear double,
  infil double,
  li double,
  blast double,
  temp double
);

delete from cancer_remission;

load data local infile "data/cancer_remission.csv"
into table cancer_remission
columns terminated by ',';

create or replace function cancer_remission_cost(params_packed_f64 blob)
  returns table as return select
    sum(log_regression_cost(
      vec_unpack_f64(params_packed_f64),
      [1, cell, smear, infil, li, blast, temp],
      remiss
    )) as cost
  from cancer_remission;

create or replace function cancer_remission_grad(params_packed_f64 blob)
  returns table as return select
    vector_sum_f64(vec_pack_f64(log_regression_gradient(
      vec_unpack_f64(params_packed_f64),
      [1, cell, smear, infil, li, blast, temp],
      remiss
    ))) as gradient
  from cancer_remission;