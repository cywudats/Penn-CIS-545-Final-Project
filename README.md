# Penn-CIS-545-Final-Project

**Project Topic**: Job Recommender Classifier & Factor Analysis

**Course**: CIS 5450 Big Data Analytics

**Term**: Fall 2022

**Team Members**: Cheng-Ying Wu, Kuan-Yu Chen, Zhenjun Xia

**Project Description:**

Given the fact that people having jobs are quite common in the world, compared to before, individuals nowadays tend to care more about their personal lives, not only focusing on income but also might focus more on work-life balance, income, culture, location, etc. In this way, we would like to take a closer look at what features indeed impact people's decisions in the UK. This dataset covers most of the industry in the UK from January 30th, 2008, to June 7th, 2021. There are 18 columns: firm, date_review, job_title (people's choices might vary due to title in the company), current (status), location, overall_rating, work_life_balance, culture values, diversity_inclusion (probably have different treatments), career_opportunities, comp_benefits, senior_management, recommend, ceo_approv, outlook, headline, pros, cons. We would like to dig more into the dataset in order to achieve our goal. Our group will find out which features do people considered about whether they will recommend the company. 

**Project Target:**

We want to build a model that can successfully predict the recommendation (0: Negative, 1: No opinion, 2: Positive) using the given features in Glassdoor Job Reviews dataset. Also, our group will determine which features people consider about recommending the company. The other goal is to build a machine-learning model that helps people predict whether the company is recommended.

**Feature Engineering:**
1. Our group comes up with a unique way to deal with categorical columns, which enables us to improve the signal-to-ratio. As a result, we can operate our machine learning task in a more convenient way. Our group converted firm, job_title, current, and location columns String to Unique Numerical Values. For the current column, our group carefully separated the data into several categories from the current employee to various lengths of time people were employed before. 

2. Another feature our group applies to textual text is Afinn Sentiment Analysis which takes in a list of words and eventually rates for valence with an integer between minus five (negative) and plus five (positive). This process does help a lot in processing the texture data since comments, including pros, cons, and headline are crucial parts of our analysis.


# Dataset

We use the Glassdoor Job Reviews dataset provided on Kaggle as our dataset.
* Glassdoor Job Reviews: https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews
