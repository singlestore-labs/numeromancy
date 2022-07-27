create database if not exists numeromancy;
use numeromancy;

create or replace function log_regression_cost as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function log_regression_gradient as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

-- problem #1: cancer remission

create table cancer_remission (
  remiss int
  cell_smear double
  infil double
  li double
  blast double
  temp double
);

load data local "data/cancer_remission.csv"
into table cancer_remission
columns terminated by ',';