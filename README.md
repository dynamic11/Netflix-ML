# Netflix Prize Dataset

This is a small ML project that attempts to predict if a user would like Miss Congeniality based on their ratings for 
the 30 most rated movies. This data is from real users on Netflix. (this data is from 2009)

This was a project from the 2009 competition called "Netflix Prize"
https://www.netflixprize.com/community/topic_1537.html


## Tools Used
* Language: Python
* Sklearn (Scikit-Learn)
* Seaborn
* Pandas

## The data

The data was obtained from the following link:

[Stanford CS109: Machine Learning Datasets](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/handouts/datasets.html)
 
 You can find some other interesting data sets at the some link
 
  ### Data formatting

Each row in the train and test set represents one user. Each column represents one movie. All users in the dataset 
rated all movies in the dataset. Each entry in this dataset is binary. 
* A value of 1 indicates a rating of 4 or 5  (they liked the movie). 
* A value of 0 indicates a rating of 1, 2 or 3 (didn't really like it).

 **The 30 input features used:**

Each column represents ratings for a particular movie.
1. Independence Day (1996)
2. The Patriot (2000)
3. The Day After Tomorrow (2004)
4. Pirates of the Caribbean: The Curse of the Black Pearl (2003)
5. Pretty Woman (1990)
6. Forrest Gump (1994)
7. The Green Mile (1999)
8. Con Air (1997)
9. Twister (1996)
10. Sweet Home Alabama (2002)
11. Pearl Harbor (2001)
12. Armageddon (1998)
13. The Rock (1996)
14. What Women Want (2000)
15. Bruce Almighty (2003)
16. Ocean's Eleven (2001)
17. The Bourne Identity (2002)
18. The Italian Job (2003)
19. I Robot (2004)
20. American Beauty (1999)
21. How to Lose a Guy in 10 Days (2003)
22. Lethal Weapon 4 (1998)
23. Shrek 2 (2004)
24. Lost in Translation (2003)
25. Top Gun (1986)
26. Pulp Fiction (1994)
27. Gone in 60 Seconds (2000)
28. The Sixth Sense (1999)
29. Lord of the Rings: The Two Towers (2002)
30. Men of Honor (2000)

**The expected output:**
 
The variable you are predicting is the binary value for the user's rating of Miss Congeniality (2000).

### Data splitting
Total number of samples in data: 41,188
* Number of Training data: 1,720
* Number of Validation data: 430
* Number of Test data: 597

## The MLP Model
We are using a 3 layer MLP model using:
* 30 neurons in input layer
* 15 neurons in hidden layer (ReLU activation function)
* 1 neuron in output layer

We use the adam solver 

## The Code

### How To Run
To run this code run main.py

There are two parameters that you can use to *generated output plots* and *debug console logs*.

There is also SEED parameter to define the seeds for the sudo random generators in the code. You can set it as a 
constant to ensure consistency between runs.
 ```python
DEBUG = True
PLOTS = True
SEED = 555
```

## Good references
* <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>
* <https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/handouts/datasets.htm>
* <https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/>
* <https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/>