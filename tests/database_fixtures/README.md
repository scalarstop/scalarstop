# Database fixtures for testing migrations

This directory contains SQLite3 databases that provide examples of what the TrainStore database schema looks like for each version of ScalarStop.

We use these SQLite3 databases to test our ability to automatically add new database tables and columns when the user upgrades to a newer version of ScalarStop.

If you have added a new column or table to ScalarStop, make sure to generate a new database fixture:
1. Land your changes onto the `main` Git branch.
2. Create and push a Git tag for the commit that you just landed on `main`.
3. Running the `create_dudmmy_sqlite3_db.py` script in the root directory of the ScalarStop GitHub repository.
4. Commit the new SQlite3 file and piush.
