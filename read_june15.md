## Scope of project
   - This project is an analysis of the mental health of some group. We are using the PHQ-2 score column to filter for people with possible depression (PHQ-2 score of 3). We already ran a logistic regression finding a relationship with some basic variables, like camp, age, etc. The next step is to find a relationship between scoring a PHQ-2 score of more than 2 and other diagnosis.

## Workflow
   # Understand the data
      - We are only using the data from the night camp folder. The data is from the past two years, and its different record ids each year. There are two datasets, which correspond to the same record id. One has demographic data and PHQ-2 score data, and the other has diagnosis data. We are going to look at co-occurence of other diagnosis with PHQ2 score
   #  Diagnosis included in the other dataset
      - Dehydration
      - Dental Pain
      - Elevated Blood Pressure
      - Fatigue
      - Headache, tension
      - Insomnia Disorder
      - Milaria or Myalgia
      - Obestiy
      - STI
      - SunBurn
   # Analyzing the data
      - In the file titled co_occurences.py is where you will be working and adding the analysis and charts. Usind the data in Night Camp folder.
      - make a bar chart P(Diagnosis | PHQ2 Score > 2)
      - compare P(Diagnosis | PHQ2 Score > 2) vs P(Diagnosis | PHQ2 Score < 2)
      - use odds ratio
      - (odds of diagnosis in PHQ2 > 2) / (odds of diagnosis in PHQ2 < 2)

## Tools
   - You are going to be working with python. You will need to create skills to analyse data and judge the data yourself.
   - You will use libraries such as numpy, matplotlib, pandas, and any other libraries you deem necessary.
   - Seperate sections in the co_occurence file using # %%


