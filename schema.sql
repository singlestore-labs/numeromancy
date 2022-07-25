create database if not exists argmin;
use argmin;

create or replace function solve as wasm
  from local infile "target/wasm32-wasi/release/numeromancy.wasm"
  with wit from local infile "interface.wit";

select solve();