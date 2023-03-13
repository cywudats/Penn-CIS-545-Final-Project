# Penn-CIS-545-Final-Project

**Project Topic**: Job Recommender Classifier & Factor Analysis

**Course**: CIS 5450 Big Data Analytics

**Term**: Fall 2022

**Team Members**: Cheng-Ying Wu, Kuan-Yu Chen, Zhenjun Xia

**Project Description:**

Given the fact that people having jobs are quite common in the world, compared to before, individuals nowadays tend to care more about their personal lives, not only focusing on income but also might focus more on work-life balance, income, culture, location, etc. In this way, we would like to take a closer look at what features indeed impact people's decisions in the UK. This dataset covers most of the industry in the UK from January 30th, 2008, to June 7th, 2021. There are 18 columns: firm, date_review, job_title (people's choices might vary due to title in the company), current (status), location, overall_rating, work_life_balance, culture values, diversity_inclusion (probably have different treatments), career_opportunities, comp_benefits, senior_management, recommend, ceo_approv, outlook, headline, pros, cons. We would like to dig more into the dataset in order to achieve our goal. Our group will find out which features do people considered about whether they will recommend the company. 

**Project Target:**

We want to build a model that can successfully predict the recommendation (0: Negative, 1: No opinion, 2: Positive) using the given features in Glassdoor Job Reviews dataset. Also, our group will determine which features people consider about recommending the company. The other goal is to build a machine-learning model that helps people predict whether the company is recommended.

## EDA (Exploratory Data Analysis)

### **Feature Engineering:**
1. Our group comes up with a unique way to deal with categorical columns, which enables us to improve the signal-to-ratio. As a result, we can operate our machine learning task in a more convenient way. Our group converted firm, job_title, current, and location columns String to Unique Numerical Values. For the current column, our group carefully separated the data into several categories from the current employee to various lengths of time people were employed before. 

2. Another feature our group applies to textual text is Afinn Sentiment Analysis which takes in a list of words and eventually rates for valence with an integer between minus five (negative) and plus five (positive). This process does help a lot in processing the texture data since comments, including pros, cons, and headline are crucial parts of our analysis.

### **Modeling & Data Visualization:**
1. Plot of missing data(Bar Chart)
![image](https://user-images.githubusercontent.com/104484482/207239855-d99a154e-8cb0-4112-b4bc-4c6d13aba680.png)

2. Extract Features & Lables and Correlation(Plot heatmap of the correlation between variable using  Pearson, Spearman, Kendall)
  
  a. Pearson Correlation
    ![image](https://user-images.githubusercontent.com/104484482/207240235-6812d007-1ac9-4294-a779-b79642fe570b.png)
  
  b. Spearman Correlattion
    ![image](https://user-images.githubusercontent.com/104484482/207240353-c53c3403-44ea-4ecc-8003-c035f6975793.png)
  
  c. Kendall’s tau Correlation
    ![image](https://user-images.githubusercontent.com/104484482/207240446-6deddce6-9ea8-44e7-b49e-682fbc013a34.png)
    
**Conclusion:** Based on the fact that our sample size is not big and tied ranks appeared in the former spearman correlation, our group decided to operate the Kendall correlation.

3. For recommendation rating, we assign “2”, “1”, “0” corresponding to “Positive”, “No opinion”, “Negative” feedback.
   ![image](https://user-images.githubusercontent.com/104484482/207241058-b2f79185-c86f-4881-930a-f0d293d1e521.png)
4. For CEO approval, Outlike, Overall_rating columns. We assign “3”. “2”, “1”, “0” corresponding to “Positive”, “Mild”, “No opinion”, “Negative” feedback.
   ![image](https://user-images.githubusercontent.com/104484482/207241153-2718496b-e71d-4d9b-9ef6-1dff6764cd71.png)
   ![image](https://user-images.githubusercontent.com/104484482/207241168-e8a2fdc5-f04e-4815-b4cb-6afb5e325576.png)
   ![image](https://user-images.githubusercontent.com/104484482/207241186-907cf9da-bf3f-4d99-8946-2e16778a41ae.png)
5. Choice for Number of Components.(Running PCA)
   ![image](https://user-images.githubusercontent.com/104484482/207241288-b00782f1-87ac-4013-85d2-f8bbba0e1ad7.png)
   The number of components is 9.
6. The Accuracy of Model Performance in traditional Machine Learning.(Before PCA)
   Logistic Regression's Accuracy is 53.0988 (%)
   Decision Tree Classifier's Accuracy is 80.9045 (%)
   Random Forest Classifier's Accuracy is 87.5209 (%)
   Gradient Boosting Classifier's Accuracy is 88.1072 (%)
   GaussianNB's Accuracy = 53.0988 (%).
   ![image](https://user-images.githubusercontent.com/104484482/207241443-6eaa8012-ebf8-4021-9a1a-dae6013e0616.png)
7. The Accuracy of Model Performance in traditional Machine Learning.(After PCA)
   After PCA, all of the models perform well.
   Logistic Regression's Accuracy (After PCA) is 82.4958 (%)
   Decision Tree Classifier's Accuracy (After PCA) is 76.8844 (%)
   Random Forest Classifier's Accuracy (After PCA) is 84.4221 (%)
   Gradient Boosting Classifier's Accuracy (After PCA) is 82.9983 (%)
   Gaussian NB's Accuracy (After PCA) is 81.9933 (%)
   ![image](https://user-images.githubusercontent.com/104484482/207241740-b02b5871-b9cf-4a37-b861-28d207f077f8.png)
8. The Accuracy of Deep Learning.
   
   a. Plot the training loss vs. epochs
   ![image](https://user-images.githubusercontent.com/104484482/207242156-08173fb1-e13a-4ba9-9d21-471e4649e8e4.png)
   
   b. Plot the training accuracy vs epochs
   ![image](https://user-images.githubusercontent.com/104484482/207242254-067aa4e8-0711-4727-9460-9f421e2c7738.png)
   
   c. The Accuracy of the test model(MLP model vs LSTM model vs GRU model)
   ![image](https://user-images.githubusercontent.com/104484482/207242595-f84adc2e-778b-489f-9db1-f8015760e66f.png)
   
   d. Training time comparision(MLP model vs LSTM model vs GRU model)
   ![image](https://user-images.githubusercontent.com/104484482/207242699-d8c2465c-d739-41a2-a206-02f5c8d8b73d.png)
   
   e. Result of Lazy Classifier modeling
   ![image](https://user-images.githubusercontent.com/104484482/207242882-d896bc8b-324e-4e54-932e-d0b91d880d08.png)
   
   f. Training time comparision in Lazy Classifier modeling
   
   ![image](https://user-images.githubusercontent.com/104484482/207242990-5c77a6e9-985e-4485-adc7-ddcfb254fa03.png)

### **Results Analysis & Possible Explanations**

The most significant factor that helps predict the recommendation:

1) overall_rating: For the overall_rating feature having a coefficient of 0.1919, the highest coefficient is reasonable since personal perspective will certainly impact the decision of whether people will recommend the job or not. Obviously, people who give higher overall rating will for sure have a higher tendency to recommend the job.

Other factors analysis:

2) For the work_life_balance feature, we have a coefficient of 0.0329. The coefficient is not as high as we predicted before. It could be explained by the fact that people still consider other aspects of work instead of only measuring personal life.

3) For the culture_values feature, the third highest coefficient of 0.0824. This does make some sense. People spend most of the daytime working, so they do care if the culture and values are toxic or not. They tend to work in a comfortable and welcoming atmosphere.

4) For career_opportunities having a coefficient of 0.0533, we concluded that opportunities are for sure important, but definitely not the main reason why they will recommend the company.

5) For company_benefits, the coefficient is 0.0001. This is surprising. Compared to common sense, in which people would likely enjoy company benefits, this feature has a small impact on the decision of recommendation. But note that this factor is not significant based on the fact that the p-value > 0.05.

6) For the senior_management feature, the coefficient is 0.0504. This is a relatively high coefficient. It is not exaggerated to say that the result aligns with the culture_values feature. People do take working experience into consideration.

7) For the CEO_approval feature, the coefficient is 0.0814. It is a noticeable coefficient. This is because, in a democratic country like the United Kingdom, people value their right to vote instead of only caring about individual work.

8) For the outlook feature, the coefficient is the second highest, 0.1879. Our prediction is that people care about the company's future. They like to recommend the company which has a great promising future since this can both enhance their salaries and working experience.

9) For the headline score, the coefficient is 0.0179. People might have ambiguous attitudes. They could comment with some lovely words and also some bad words to give a whole sense of the company. Our model using Afinn could have a relatively ambiguous result regarding the headline as a result. This whole process eventually leads to a low coefficient.

### **Conclusions**

1. Among all traditional machine learning models, Random Forest has relatively better performance, both before doing PCA and after doing PCA.

2. Three deep learning models (i.e., MLP, LSTM, and GRU) have similar performance. However, our designed MLP model slightly outperforms two others in terms of accuracy and training time.

### **Description of Challenges**

1. Initially, we used the Credit Card Fraud Detection dataset provided on Kaggle. However, that dataset is too clean, and all features are anonymous due to confidentiality issues. In addition, too many people did different modeling for the dataset, so we switched our dataset to Glassdoor Job Reviews.

2. Training time for some models, especially deep learning models, takes too long. Therefore, we use GPU to run these models.
This dataset has much textual data, so how to encode this textual data are essential.

### **Potential Next Steps**

1. Do More Deeper Analysis of our Modeling:

Calculate & plot the confusion matrix, calculate different metrics, and give more explanations based on these metrics and model assumptions.

2. Try More Different Ways to Encode the Textual Data:

In this project, we only chose to use the Affin score.

3. Fine-tune our Deep Learning Models:

Perform hyperparameter tuning and design a better model architecture for each of our deep learning models (e.g., MLP, LSTM, and GRU).

4. Text Classification Using BERT:

a. Reference Link: https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert

b. We can try to use large language models like BERT to do the text classification tasks. For example, try to analyze if the result of the text classification of the headline aligns with recommend, etc.

# Dataset

We use the Glassdoor Job Reviews dataset provided on Kaggle as our dataset.
* Glassdoor Job Reviews: https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews
