# Exercise 4

This is the fourth exercise for the course [Deep Learning](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/wintersemester-2024-2025/deep-learning/).

Clone this repository to your local setup. Create a branch with the name of your GitHub account and switch to it.

In the repository, you find a CSV file with data about the passengers of the titanic, including names, gender, age, passenger class and whether they survived. 

In the file `logistic_regression.py`, read in the CSV file using the `pandas` library and make two plots, using the `scatterplot` function of the `seaborn` library:

1. The age of a passenger on the x-axis and whether they survived on the y-axis.
2. The passenger class on the x-axis and whether they survived on the y-axis.

You can find directions on how to use `pandas` and `seaborn` in these [slides](https://lehre.idh.uni-koeln.de/site/assets/files/5151/presentation02.pdf) and/or inform yourself online.

Implement the logistic function and the binary cross-entropy loss function in Python and try out different values for `a` and `b`. Plot the parameterized functions into the seaborn scatterplots using the `lineplot` function and try to find good values for `a` and `b` by calculating the loss and trying to minimize it as much as possible.

Commit your changes and push them to the branch with the name of your GitHub account.

Deadline for this exercise is November 7, 2024, 08:00:00 CET.
