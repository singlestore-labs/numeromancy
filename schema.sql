create database if not exists numeromancy;
use numeromancy;

create or replace function opt_cost as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";

create or replace function opt_gradient as wasm
  from local infile "target/wasm32-wasi/release/numeromancy_problem.wasm"
  with wit from local infile "problem/interface.wit";