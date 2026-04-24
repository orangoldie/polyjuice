We want to create a statistical-arbitrage opportunity in polymarket when it comes to LoL games.
We have data from 2016-2026 (until presesnt) that contains a lot of information (some on 10/15/20/25 min marks).
Polymarket contains LoL bets on tournament games. Includes many different tiers.
We want to calculate a win probability and then see what's the situation like in polymarket and act if its greater than some threshold X%.
We can use the https://oracleselixir.com/tools/downloads dataset.
You can read the data/columns.txt to get sense of the raw columns we have in the file pre-processing.
You can read data/example-team-game.csv to get an example of actual data of a game.

Q&A / notes:
1. Do we want to take into consideration the pre-game bet percentages from poly into our model?
2. Which columns from the csvs should be included?
3. Do we want a model per top tier/tier1/tier2 or one model with a feature that reps the tier? 
4. atakhan existed only in 2025 and was a somewhat important part of the meta, need to consider how to treat this and verify it existed only in 2025 dataset.
5. Which choices do we have under the xgboost? how do we make the right decisions?
6. How can we verify that the output percentages are actually correct against the test set? some parameters are not time dependent sadly and are global (e.g. dragon, baron)
7. Should we include non-time dependant columns in all time frames? If they are just 0 it won't affect anything so its ok? or im missing something
8. is xgboost overkill? are there better alternatives given the data?
9. which years should we train and which years should we test on ? how many games do we want to train on? divided by tier or no?
10. What API can we use to get the current feed from a game? Will it contain all the data we want ?
11. How do i avoid running out of tokens so claude won't go around reading big CSV files that are part of this project? I want him to read only small parts of it sometimes (e.g. colums, 10 rows, etc).
12. Do we only need the teams-related rows or also the player-specifics?
13. Any other crucial information we are lacking and can highly improve our model?
14. How do we want to pre-process the data?
15. Do you think we should include team related statistics as part of the model somehow? e.g. https://oracleselixir.com/team/Dignitas/statsByYear. We can also cain access to past tournaments info if needed e.g. https://oracleselixir.com/stats/teams/byTournament
16. We need to only include in our processing games where the column of datacompleteness is complete


Please assist in:
1. Separating this Q&A into sections
2. Consulting if we have any other meaningful questions we want to add
3. Any other thing you think can be vaulable to the phase we are currently in.

We are not coding the actual pipeline rightnow. we can write scripts to assist us answering questions if required.
