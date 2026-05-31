# Project Description

For the practical track you are required to re-implement four famous RL algorithms listed below. You should also test them on some of the following OpenAI-Gym environments: Cartpole, MountainCar, MountainCarContinuous, Acrobot and Pendulum.

In your report of 6–8 pages you are required to compare the following 4 algorithms based on of your empirical observation. This means providing appropriate plots and score statistics of your algorithms based on fair comparison between them.

Possible algorithms:
• DQN by Mnih et al.
• PPO by Schulman et al.
• SAC by Haarnoja et al.
• TD3 by Fujimoto et al.

You can also add a qualitative discussion about the algorithms building around the following questions:

- Which algorithm is more computationally expensive per iteration ?
- Which algorithm store the policy more compactly ?
- Which one scales better for continuous actions ?
- Which algorithm makes efficient use of off-policy data ?

Finally, view the report as diary in which you can keep track of the observations made during the implementation process. We are interested in knowing which small details in the implementation you found are crucial to make the algorithm work in practice! For example, if you had a bug that took you you a long time fix, write it down. If you found that the algorithm’s performance is very sensitive to certain hyperparameter tuning, write it down. Take also note if you find out that an hyperparameter affects the performance only minimally, and think about possible reasons. Corroborate your claims by showing plots that compare the algorithms when run for the different hyperparameters (i.e., do not only report the final, good hyperparameter choices that made it work eventually).

Important: Each plot you present should report an algorithm’s performance averaged across at least 3 seeds.
