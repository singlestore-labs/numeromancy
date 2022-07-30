# Numeromancy
This project uses [argmin][argmin] to drive an optimization problem over data stored in [SingleStore][singlestore]. Wasm is used to push cost, gradient, and hessian methods into SingleStore.

## Usage

The following instructions assume you are using [VS Code][vscode] and have opened this project in a devcontainer.

1. Start a SingleStore cluster
2. Compile the problem definition to wasm
   ```bash
   cd problem
   cargo wasi build --release
   ```
3. Load the wasm into SingleStore
   ```bash
   mysql -u root -h 172.17.0.3 -ptest --local-infile <schema.sql
   ```
4. Update the cancer remission config file: [config.toml](example/cancer_remission/config.toml)
5. Setup the cancer remission example and train an inference function.
   ```bash
   cd examples/cancer_remission
   mysql -u root -h 172.17.0.3 -ptest --local-infile <setup.sql
   cargo run --release -p numeromancy-driver
   ```
6. Check out the confusion matrix
   ```sql
   select * from cancer_remission_confusion();
   +---------------+----------------+----------------+---------------+
   | true_positive | false_negative | false_positive | true_negative |
   +---------------+----------------+----------------+---------------+
   |             5 |              4 |              3 |            15 |
   +---------------+----------------+----------------+---------------+
   1 row in set (0.057 sec)
   ```
6. Run the generated inference method
   ```sql
   select
      cell, smear, infil, li, blast, temp,
      cancer_remission_infer([1, cell, smear, infil, li, blast, temp]) as predicted_remission
   from cancer_remission;
   +------+-------+-------+------+-------+-------+--------------------------+
   | cell | smear | infil | li   | blast | temp  | predicted_remission      |
   +------+-------+-------+------+-------+-------+--------------------------+
   |  0.8 |  0.83 |  0.66 |  1.9 |   1.1 | 0.996 |       0.7906617321859674 |
   |  0.9 |  0.36 |  0.32 |  1.4 |  0.74 | 0.992 |       0.4350908696255495 |
   |  0.8 |  0.88 |   0.7 |  0.8 | 0.176 | 0.982 |      0.15557254876835305 |
   |    1 |  0.87 |  0.87 |  0.7 | 1.053 | 0.986 |      0.29085734508704386 |
   |  0.9 |  0.75 |  0.68 |  1.3 | 0.519 |  0.98 |       0.6970204491139325 |
   ...
   ```

## Resources

* [Documentation](https://docs.singlestore.com)
* [Twitter](https://twitter.com/SingleStoreDevs)
* [SingleStore forums](https://www.singlestore.com/forum)

[argmin]: https://docs.rs/argmin
[vscode]: https://code.visualstudio.com/
[singlestore]: https://www.singlestore.com/