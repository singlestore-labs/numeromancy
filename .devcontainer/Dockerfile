ARG VARIANT="buster"
FROM mcr.microsoft.com/vscode/devcontainers/rust:0-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        mariadb-client gfortran

USER vscode

RUN rustup target add wasm32-wasi
RUN rustup component add rustfmt rust-src clippy

RUN cargo install cargo-wasi cargo-expand && \
    cargo install --git https://github.com/bytecodealliance/wit-bindgen wit-bindgen-cli