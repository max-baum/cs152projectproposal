# Paycheck Protection Plan (PPP) Loan Fraud Detection OUTLINE

## Authors
Max Baum, Arsh Chhabra, Cameron Hatler, Viren Jain

## Abstract

The goal of this project is to train a neural network on a dataset which contains confirmed fraud of the Paycheck Protection Plan (PPP) loans. These loans were given out during the pandemic to help small businesses, however there have been many cases found of individuals applying and recieving these loans only to use them for personal expenses rather than business ones. The dataset we are working with contains confirmed cases of fraud, however there are no confirmed non-fraudulent cases (a case could have been fraudlent but never been found out). This presents a unique problem, but we believe it is a problem which may be solved with anomaly detection. This, along with other methods, will enable us to create a robust model for fraud detection among PPP loans. 

## Introduction

The U.S. Federal Government dispersed nearly $700 billion in paycheck-protection program stimulus in 2020, to counteract the economic downturn caused by the COVID crisis. Unfortunately, it is estimated that roughly 10% of the loans were potentially issued to fradulent recipients, accounting for roughly $64b. Roughly $30b of potentially fradulent PPP loans remains outstanding. 

In times of growing U.S. Federal Government deficit, alongside increased scrutiny from taxpayers regarding use of government funds, the status quo of the PPP program is highly concerning. It also calls into question what better controls could be implemented, to either prevent fraud outright or to improve the success-rate of fund recovery casework. 

This project is focused on investigating what neural network based approaches can be leveraged to predict fradulence of loans based upon publically-known loan attributes released under a FOIA request. The project is imperfect, and navigates many challenges including extreme class imbalance and imcomprehensive confirmation of loan fraud. The project centers around an expanded verion of DANB91's PPP Fraud Dataset [^1], that draws from other developments in the court system. The dataset is imperfect, but nonetheless provides some color where none previously existed. Further, while a variety of NN based approaches can be used to address osme of these outstanding issues, it must be acknowledged that no approach can ever fully correct for bad data.

Overall, this project does not seek to revolutionize or even directly inform an government decisionmaking in the issuance of small business loans. Rather, it seems to demonstrate the promise of increasing data-driven governance, and to confirm the existence of structure in data that can be used to guide better decision making. $60 billion is a heck of a lot of money to be lost to fradulent activity, in times when American taxpayers are increasingly squeezed financially. One would hope that failures of this magnitude could be better derisked against moving forward.

## Ethical sweep

There is significant merit in attempting to reduce fraud in government programs like the Paycheck Protection Program. Governments operate with limited resources funded by a finite tax base, and it is important that these resources are used responsibly and reach their intended recipients. Our project, while classroom-bound, aims to explore whether machine learning can assist in identifying potentially fraudulent loans — a task that holds real-world value.

That said, there are limitations and risks inherent to this work. The accuracy of a simple, non-ML rule-based approach is likely to be low given the number and complexity of variables involved, as well as the nuanced relationships between them. Machine learning, despite its promise, also comes with risks — especially when the ground truth is uncertain. A key challenge we face is that the majority of our dataset consists of loans that have not been confirmed either way; only a very small subset has been labeled as fraudulent through the justice system. This creates a problem of label scarcity and possible bias, as loans that have not yet been investigated might be wrongfully treated as negatives. Additionally, we acknowledge potential bias in the way this dataset was assembled and processed, which may affect our model’s fairness and reliability.

We also recognize that our team, like most student groups at the 5Cs, is not demographically representative of the broader U.S. population. As such, our perspectives are inherently limited. Throughout the project, we aim to counterbalance this by engaging in frequent and transparent team discussions, especially around mistakes, uncertainties, and assumptions. Code and outputs will be reviewed by multiple team members to ensure accountability, and data will be spot-checked as time allows. Though our outputs are unlikely to leave the classroom, we believe in treating this project with the seriousness it deserves.

In terms of privacy, all data used in this project is publicly available. Given the nature of the analysis, we believe there is minimal risk of infringing upon individuals' privacy or anonymity. However, we remain conscious of the ethical line between analysis and accusation. A flagged loan is not proof of fraud — only the justice system has the authority to make that determination. Our model is a retrospective tool, and should it be used in any future capacity, it would be most appropriate as part of a triage process rather than a definitive arbiter.

Lastly, we do not currently know if the model will perform differently across sub-groups, nor are we sure what constitutes a meaningful "sub-group" in the context of this data, given its limited granularity. Misinterpretations of the model’s outputs are a real concern and must be addressed through clear communication of its capabilities, limitations, and the fact that this is an exploratory academic project — not an instrument of policy or prosecution.


## Related Work

Our project applies neural networks to detect fraudulent PPP loans, leveraging insights from Zhan and Yin (2018)[^2] and Awotunde et al. (2022)[^3]. Zhan and Yin propose a knowledge graph-based fraud detection system that constructs borrower networks from call history data. Differing from traditional models that would gather 100s of features, their model captures hidden fraud patterns such as borrowers sharing suspicious contacts—making it harder for fraudsters to evade detection. This method suggests that graph-based representations could enhance our model’s ability to detect anomalies.

Awotunde et al. (2022) employs various machine learning models including Artificial Neural Networks for loan fraud detection, achieving 98% accuracy on a bank loan dataset. The findings suggest that ANNs outperform traditional models like Decision Trees and SVMs in identifying fraudulent transactions. By integrating these approaches, our project could explore graph-enhanced neural networks for fraud detection, improving classification accuracy despite limited labeled fraud cases. The combination of graphs and ANN-based classification offers a solution to identifying fraudulent loans.

We face a problem in our dataset of having very few data points being for confirmed fraud and a lot of data points for other which we cannot classify as being non fraudulent. Hence, we have the idea of using anomaly detection via neural networks to hopefully train a model that can learn what fraud looks like and learn the class of fraudulent loans (or similiarly the class of non fraud loans) and detect anomalies to detect things that do not belong to the desired class. So we have started to explore the areas of one class neural networks for anomaly dtection as a result. We looked at two papers in this area: Anomaly Detection using One-Class Neural Networks by Chalapathy, et al. (2018)[^4] and Deep One-Class Classification by Ruff, et al. (2018) [^5]. Both of these papers provide useful insights into how neural networks can be optimized for anomaly detection. OC-NN’s ability to refine feature extraction for anomaly detection and Deep SVDD’s structured approach provide a useful theoretic background to our project which we hope to explore further. By leveraging these ideas we aim to develop a neural network that can generalize well even with the limited availability of confirmed fraud cases and solve a major problem present in our dataset.

By leveraging the flexibility of neural networks, we hope to find patterns in fraudulent applications of PPP loans in order to track down other suspicious borrowers. While our dataset can't be seperated into true positive and negative cases of fraud, we have some true positive cases and many undetermined cases. By using anomaly detection, the model should give improved predictions of true positive and negative cases which will allow for easier application of the model. We will also aim to incorporate what various models have succesfully done for different cases of fraud detection into our model. This work will help in fraud detection of PPP loans, saving money if it were incorporated, and will show how effective the combination of various neural network methods in this unique situation can be. 

## Methods

### Creating the Dataset

The dataset used in this report was compiled by Max Baum, Wonny Kwak, and Earn Wonghirundacha in a Computational Statistics course.

Given the importance of the data in this project, the process of piecing together the dataset is briefly explored below.

The dataset used in this project was produced with a multistep process to bridge two disparate datasets together. The first dataset was released by the Small Business Adminstration (SBA) under FOIA, and inlcudes all PPP loans over $150K in value. This dataset was found on Kaggle (Bukowski, 2020), though the SBA does release this data themselves. The Kaggle version of this dataset includes a separate sheet created by the author marking 100 instances of known fraud which were ultimately exploited. To find more cases of fraud in this PPP loan dataset, a tracker compiled by law firm Arnold & Porter of legal cases brought forth by the DOJ to address CARES Act fraud was referenced. 

The two datasets use different identiers for businesses and loans, so to join the two datasets, press releases identified by Arnold & Porter were scraped from the DOJ. This scraped data was then searched through to find business names in the SBA dataset. Ultimately, 176 distinct loans from the SBA dataset were matched to the Arnold and Porter tracker. When combined with preidentified loans in the Kaggle dataset, a total of 248 loans were confirmed to be fraudulent. In the final dataset, the fradulent loans were marked with a "1" in a new fraud column, while loans not found to be fraudulent were marked with a "0"

### Processing and Cleaning the Data

The initial dataset included 41 fields, of varying datatypes. To make this data utilizeable by a neural network, significant processing had to be carried out.

Firstly, columns referring to the race, gender and ethnicity of the borrower were dropped from the dataset, as borrower-identity data was incomplete and we did not seek to interrogate the relation between borrower identity and likelihood of fraud.

Next, categorical data fields with small sets of possible values (or important granular information) were directly converted to one hot encoding. Other categorical data fields were also converted to one-hot encoding, but included possible encoding for only top ten values in the set, as well as a possible "other" encoding. How each field was encoded is indicated in the table below

| Column Name   | Encoding Process |
| -------- | ------- |
| NonProfit  | Direct Encoding    |
| Veteran | Direct Encoding     |
| BusinessType    | Direct Encoding    |
| ProjectState  | Direct Encoding  |
| BusinessAgeDescription | Direct Encoding     |
| LMIIndicator    | Direct Encoding    |
| HubzoneIndicator  | Direct Encoding    |
| RuralUrbanIndicator | Direct Encoding     |
| BorrowerState    | Direct Encoding   |
| ProcessingMethod  | Direct Encoding0    |
| Franchise Name | Top 10     |
| OriginatingLender    | Top 10    |
| NAICSCode  | Top 10    |

### Addressing Class Imbalance



   ### Building the Neural Network
   
   
   We have an existing dataset which has a very large number of negatives (fraud has not been charged) and few positive data points (fraud has been charged). 
   
   We plan to do some preprocessing we will clean, normalize, and design engineered features such as loan amount, business size, location, and NAICS codes to prepare for training. 

   To start, we will construct a simple fully connected neural network. This will work as our base-line to check the effectiveness of future models. 

   In testing the accuracy of the model, it is important to remember that the effects of false positives (flagging a load as fraud when it isn't) and false negatives (not charging fraud when fraud has occured) have very different and unique effects. The first would cause some headaches as individuals with no intent of commiting fraud might be charged, the second would allow those who commited fraud to get away with no repercussions. 
   
   From the dataset, we use the important categories as the features of our neural network. We use SMOTE to handle the class imbalance that exists in our dataset, give numerical values to the categorical data, and finally normalize these features for the ideal input range of the neural network.

   We use a fully connected neural network with (N) layers, each of depth (K). These hidden layers are composed of sequential linear and ReLU functions. There will be two outputs, each giving the confidence of a loan being fraudulent or legitimate. 

   For future works, we plan to explore one class neural network techniques for anomaly detection, treating known fraud cases as the positive class and the rest as unlabeled. 
   
   Since we have the lack of true negatives, we used semi-supervised metrics such as precision-at-K, silhouette score, and anomaly ranking as potential in between techniques.
      
   We have some challenges which include data imbalance and potential hidden frauds in the unlabeled set. We addressed this with regularization, dropout, and cross-validation.
   
   We also plan to use dimensionality reduction techniques like t-SNE and UMAP to visualize how well fraud cases were clustered or separated.

## Results

  _Purpose of Section:_ In this section, we will detail the results acheived following our methodology
  
  _Topic Sentence:_ We recieved mixed results following our methodology
  
  Prediction -- We are confident that a simple approach will perform adequately (but not necessarily impressively) on the data. We are not as certain about the performance of the anomoly/outlier approach. It could perform significantly better, it could perform roughly the same if not worse.
  
## Discussion

   Using our custom dataset, we trained both a fully-connected neural network and a neural network focused on anonmoly detection.

  Our models achieved varying levels of success. The following figures demonstrate this performance.

  In tests, Model A achieved a higher true positive rate than Model B, meaning that the model was able to correctly identify confirmed fraud more frequently than Model B. This said, Model A had a higher false positive rate than Model B, meaning that Model A is incorrectly flagging loans as fraud more frequently than Model B. Ultimately, while Model B has the highest overall accuracy, this accuracy comes at the expense of missing confirmed positives.

  Compared to other studies, our models overall performed worse. This is due to limitations in both the model and the data.

  While our models did not achieve ideal results, they nonetheless demonstrate the potential of our approach. We think the following steps can be followed to both improve our own model and to generally improve neural-network-based approaches to fraud detection moving forward.

  
## Reflection and Looking Forward

  _Purpose of Section:_ In this section, we will reflect more broadly upon what has been achieved, and what more compentent systems at the intersection of neural networks and fraud detection could mean for society.
  
  _Topic Sentence:_ We are excited by the results that we attained through this project, and we believe there is much to be gained for society in the realm of neural network-enabled fraud detection.

## Citations
[^1]: https://www.kaggle.com/datasets/danb91/covid-ppp-loan-data-with-fraud-examples?select=ppp_fraud_cases.csv
[^2]: https://dl.acm.org/doi/10.1145/3194206.3194208
[^3]: https://link.springer.com/chapter/10.1007/978-3-030-96305-7_43
[^4]: https://arxiv.org/abs/1802.06360
[^5]: https://www.researchgate.net/publication/329829847_Deep_One-Class_Classification
