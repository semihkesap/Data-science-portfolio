# Data-science-portfolio
Built to analyze gaming engagement, predict player behavior, and suggest improvements.
## Projects:
### Gaming Engagement Analyzer
Analyzed 40,034 players to boost game engagement. Predicted engagement with 91% accuracy, found `SessionsPerWeek` and `AvgSessionDurationMinutes` drive retention (74.21%).  
- **Weekly Login Streak Bonus**: Reward 7+ sessions/week with a 20% coin bonus. Targets Low (4.53 sessions) to hit 7, Medium (9.55) to reach 12 — closer to High (14.25). (Low players (10,324) lag by 10 sessions vs. High. SessionsPerWeek is a top engagement driver.)
- **Endurance Mode**: 2-hour challenge mode with rare rewards (e.g., cosmetics). Pushes High (125-min sessions) to 150 mins, Medium (90 mins) to 120. (AvgSessionDurationMinutes drives engagement. Plot shows High plays 2.5x longer than Low (50 mins)). 
- **Low Engagement Login Boost**: Low players hitting 5 sessions/week get a 1.5x XP multiplier for the next week. Lifts their median from 3.0 to 5.0, aiming for 7.0 (75th percentile). (10,324 Low players (25.8%) average 4.53 sessions. Median 3.0, 75th percentile 7.0 — 5 sessions is an achievable stretch.)  
- **Tools**: Python, pandas, scikit-learn, seaborn.
### Churn Predictor
Predicted 88% accuracy, targets <10 session players for retention.
- **Churn Retention Strategy**: Target players with <10 SessionsPerWeek for retention. 88% accurate model shows 8-session average for churned players
