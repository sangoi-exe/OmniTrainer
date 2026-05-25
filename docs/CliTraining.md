# Training from CLI

All training functionality is available through the CLI command `./run-cmd.sh train`. The training configuration is stored in a `.json` file that is passed to this script.

Some options require specifying paths to files with a specific layout. These files can be created using the `create_train_files.py` script:

```bash
./run-cmd.sh create_train_files -h
```

To simplify the creation of the training config, export your settings from the UI by using the export button. This creates a single file that contains every setting.
