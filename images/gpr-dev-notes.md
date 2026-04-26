IMPORTANT: NOTE THAT THE FORUMLAS ARE IMAGES THAT EXIST IN THIS SAME FOLDER AND ARE NOT PRESENT IN THIS TEXT!


Dev Diary: Unveiling the Global Power Rankings
A deeper look into the development of GPR by the LoL Esports product team.

Calling all math lovers! You might have seen last week’s announcement that in partnership with our friends at AWS, we’re introducing Global Power Rankings to the LoL Esports universe, starting in T-minus 48 hours at Worlds 2024. Now before we embark on this new frontier that we hope will level up your interest in the global competitive landscape across LoL Esports, we wanted to give you a deeper look into how LoL Esports Global Power Rankings are actually formulated.

The Need for a New System

The development and use of Power Rankings have been a part of traditional sports for decades, however ranking systems such as the NCAA AP Top 25 and FIFA World Rankings, while effective in their respective domains, fall short when applied to esports. The dynamic nature of League of Legends—with its frequent meta shifts, varying regional league structures, and diverse play styles—demands a more nuanced approach. The Global Power Rankings address these challenges by leveraging a blend of individual team performance metrics and regional strength indicators, creating a system that is both comprehensive and adaptable.

Core Components: Elo Rationale and Power Score Formula

At the heart of the Global Power Rankings is an Elo-based system, adapted to account for the unique dynamics of professional League of Legends. Specifically, we needed a way to quantify the difference in strength between regions since the vast majority of matches are played intra-regionally. We determined that developing a blended elo system with an individual team elo and a league-specific elo was the best approach. We’ve deemed this weighted average as the “Power Score” formula and it is expressed as:

Here, ‘x’ and ‘y’ represent the weights assigned to team Elo and league Elo, respectively. After extensive testing with LoL Esports experts from all of our regions, we determined that an 80/20 weighting provided the optimal balance, yielding a predictive accuracy of 65%. This means the model’s expected results closely align with actual outcomes, validating the system's robustness and the relatively unpredictable nature of esports.

Elo Calculation: A Detailed Breakdown

Team Elo is calculated based on match outcomes using the formula:

Where:

    Pbefore​ is the team’s Elo before the match.

    I is the importance of the match, scaled according to the stakes of the game (covered in K-Factors below).

    W represents the match result (1 for a win, 0 for a defeat).

    We is the expected result, calculated as:

Here, dr is the difference in ratings between the two teams. This formula ensures that the Elo adjustments are proportional to the expected difficulty of the match.

League Elo follows a similar structure, but is calculated at the regional level. International event performances (e.g., MSI, Worlds) significantly influence league Elo, with wins against stronger regions boosting the entire league's rating. This method captures regional power dynamics and reflects them in the rankings.

Emphasizing High-Stakes Matches

Recognizing that not all matches carry the same weight, we introduced K Factors to differentially weight match outcomes based on their importance. They are essential to the model’s functionality as they determine how quickly a team’s rating changes in response to new game results. The standardized K Factor values are:

    Regional Play:

        Play-Ins: 8

        Main Stage: 16

        Playoffs: 20

    International Play:

        Play-Ins: 12

        Main Stage: 20

        Playoffs: 36

These K Factors ensure that critical matches, such as playoff games and international fixtures, have a more significant impact on the rankings. This approach is crucial for accurately reflecting a team's performance under pressure. Finally, we scaled the K Factors from region to region to align with their teams’ average final standings during international events, further emphasizing regional play in high-power leagues. We will continue to refine this approach as the model ingests more data and our league structures adjust going into the 2025 season.

Evaluation Windows: Balancing Recency and Historical Performance

To maintain a balance between current form and historical performance, we adopted a multi-year evaluation window:

    League Elo is evaluated over a 3-year window (2022-2024)

    Team Elo is evaluated over a 2-year window (2023-2024)

This approach increases the sample size of international games, which is particularly important for smaller regions, while still accounting for the potential impact of roster changes.

Challenges and Solutions: Building a Robust Model

During the development process, we faced several challenges, including quantifying the skill difference between regions and integrating in-game statistics in a way that is resilient to meta changes. To address these, we held a hackathon in collaboration with AWS, where hackathon participants explored various AI/ML models. While these advanced models developed by LoL Esports fans were impressive and thoughtful, we opted for a simpler, more transparent approach for the first iteration for meta-resiliency and interpretability.

Technical Integration: AWS-Powered Scalability and Performance

The Global Power Rankings are underpinned by a robust technical architecture, leveraging AWS services including S3 for persisting model training data, Lambda functions for initiating model updates, and a MySQL database on AWS RDS for persisting ranking results. Our partnership made it simple and easy to handle the complex data processing and real-time updates required for the algorithm. This ensures that the rankings are not only accurate but also scalable, capable of handling the massive data inputs from international fixtures and providing timely updates.

Conclusion: A New Era for League of Legends Esports

The Global Power Rankings represent a significant step forward in how we evaluate and understand team performance in League of Legends. As we continue to refine and expand this system, we invite the community to join us in this journey, as we strive to provide the most accurate, transparent, and insightful rankings for fans of Riot Esports. 

Stay tuned as we roll out the Global Power Rankings throughout Worlds 2024 on lolesports.com and Riot Esports broadcasts. The final update for 2024 will be published after the Worlds Grand Finals, marking the beginning of our preparations for an even more refined system in 2025.