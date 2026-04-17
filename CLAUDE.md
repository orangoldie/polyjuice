# Polyjuice

Statistical arbitrage on Polymarket for League of Legends esports matches.

## What This Project Does

- Uses historical LoL pro match data (2014-2026) from Oracle's Elixir to build a win-probability model
- Compares model output to Polymarket odds
- Bets when the edge exceeds a threshold

## Data

- Source: https://oracleselixir.com/tools/downloads
- Raw CSVs live in `data/lol-historical-data` — they are large, never read them in full
- Each game has 12 rows in raw data: 10 player rows + 2 team rows
- Columns documented in `data/columns.txt`
- Example team-level game row in `data/example-team-game.csv`
- Time-stamped features exist at 10/15/20/25 min marks

## Conventions

- Python for all scripts and modeling
- Keep exploratory/helper scripts in `scripts/`
- When working with CSVs, only read samples (head, specific columns, row counts) — never load full files into context
- We are in a discovery/research phase — not building the pipeline yet

## Key Decisions (In Progress)

See `discovery-checklist.md` for open questions and status.
