FROM rust:latest

RUN apt-get update && apt-get install -y \
    libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV RUST_VERSION=

COPY . .

# Build the entire workspace
RUN cargo build --tests --benches --examples --release \
    && rm -rf ./*
