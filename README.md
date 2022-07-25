# Numeromancy
This project uses [argmin][argmin] to drive an optimization problem over data stored in [SingleStore][singlestore]. Wasm is used to push cost functions and derivatives into SingleStore.

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
4. Run the driver
   ```bash
   cargo run
   ```

## Resources

* [Documentation](https://docs.singlestore.com)
* [Twitter](https://twitter.com/SingleStoreDevs)
* [SingleStore forums](https://www.singlestore.com/forum)

[argmin]: https://docs.rs/argmin
[vscode]: https://code.visualstudio.com/
[singlestore]: https://www.singlestore.com/