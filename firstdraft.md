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

**General Questions:**
* Should we even be doing this?
  * Yes. There is significant merit and virtue in lessening fraud in government programs. Governments have limited resources, funded by a finite tax base, and it is pertinant that these resources are used properly and responsibly.

* What might be the accuracy of a simple non-ML alternative?
  * Likely low, though uncertain. Given the broad array of attributes to consider, and the sophisticated relationships these attributes may have with the likelihood of a loan being fradulent, ML approaches seem most promising.

* What processes will we use to handle appeals/mistakes?
  *  Team communication will be frequent and transparent. Mistakes will be documented and brought up in team discussions, rather than ignored. Code and outputs will be verified independently by multiple team members, though outputs are unlikely to leave the classroom.

* How diverse is our team?
  * Any team of students at the 5Cs is not a comprehensive and proportional reflection of U.S. society. By that standard, our team is not diverse.
 
**Data Questions:**
* Is our data valid for its intended use?
  * Yes, with known and communicated limitations.
 
* What bias could be in our data? (All data contains bias.)
  * It depends. We think the most likely source of bias is caused simply by the low throughput of the justice system, in that not all loans have been investigated equally for fraud. Depending upon how cases have progressed through the justice system, there could be bias especially given the limited number of positives we have. There could also be bias in other areas, including biases caused by Max in the processing and development of this dataset.
 
* How could we minimize bias in our data and model?
  * This is a very complicated question given the nature of the data. One of the issues here is the lack of "real" negatives in the dataset, so maybe a clustering tecnique could be used here. We think there are quite a few procedural questions to discuss as we handle the data and develop the model. We cannot fix the SBA and the Department of Justice. We can neither confirm nor disprove fraud ourselves.
    
* How should we “audit” our code and data?
  * The data can be spot-checked, or further if we have time, all 248 cases of confirmed fraud can be reviewed manually (though this may not be necessary). All code should be reviewed by multiple parties.

**Impact Questions:**

* Do we expect different errors rates for different sub-groups in the data?
  * There is no reason to suspect we would, but we simply don't know. It is further unclear what a "sub-group" would constitute here, as we have limited information in SBA data identifying characteristics of businesses, rather information about the nature of the loans themselves.
  
* What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?
  * First and foremost, this is a retroactive assessment, and it cannot be directly applied toward proactive loan fraud prevention going forward.
  * Secondly, even if this model could be used proactively, a postive indication of fraud by this model is not the confirmation of fraud. The only thing that can confirm fraud is the justice system. At most, this model could be used in the triaging of case work in fraud investigation.
  * Misinterpretations should be prevented with clear communication of the models known capabilities and limitations, as clarified above and as will be further discussed among the team.
    
* How might we impinge individuals’ privacy and/or anonymity?
  * All information used in our dataset is publically available. Given the nature of the dataset and analysis, there is little risk that this project will impede upon privacy rights of individuals.


## Related Work

Our project applies neural networks to detect fraudulent PPP loans, leveraging insights from Zhan and Yin (2018)[^2] and Awotunde et al. (2022)[^3]. Zhan and Yin propose a knowledge graph-based fraud detection system that constructs borrower networks from call history data. Differing from traditional models that would gather 100s of features, their model captures hidden fraud patterns such as borrowers sharing suspicious contacts—making it harder for fraudsters to evade detection. This method suggests that graph-based representations could enhance our model’s ability to detect anomalies.

Awotunde et al. (2022) employs various machine learning models including Artificial Neural Networks for loan fraud detection, achieving 98% accuracy on a bank loan dataset. The findings suggest that ANNs outperform traditional models like Decision Trees and SVMs in identifying fraudulent transactions. By integrating these approaches, our project could explore graph-enhanced neural networks for fraud detection, improving classification accuracy despite limited labeled fraud cases. The combination of graphs and ANN-based classification offers a solution to identifying fraudulent loans.

We face a problem in our dataset of having very few data points being for confirmed fraud and a lot of data points for other which we cannot classify as being non fraudulent. Hence, we have the idea of using anomaly detection via neural networks to hopefully train a model that can learn what fraud looks like and learn the class of fraudulent loans (or similiarly the class of non fraud loans) and detect anomalies to detect things that do not belong to the desired class. So we have started to explore the areas of one class neural networks for anomaly dtection as a result. We looked at two papers in this area: Anomaly Detection using One-Class Neural Networks by Chalapathy, et al. (2018)[^4] and Deep One-Class Classification by Ruff, et al. (2018) [^5]. Both of these papers provide useful insights into how neural networks can be optimized for anomaly detection. OC-NN’s ability to refine feature extraction for anomaly detection and Deep SVDD’s structured approach provide a useful theoretic background to our project which we hope to explore further. By leveraging these ideas we aim to develop a neural network that can generalize well even with the limited availability of confirmed fraud cases and solve a major problem present in our dataset.

By leveraging the flexibility of neural networks, we hope to find patterns in fraudulent applications of PPP loans in order to track down other suspicious borrowers. While our dataset can't be seperated into true positive and negative cases of fraud, we have some true positive cases and many undetermined cases. By using anomaly detection, the model should give improved predictions of true positive and negative cases which will allow for easier application of the model. We will also aim to incorporate what various models have succesfully done for different cases of fraud detection into our model. This work will help in fraud detection of PPP loans, saving money if it were incorporated, and will show how effective the combination of various neural network methods in this unique situation can be. 

## Methods

   We have an existing dataset which has a very large number of negatives (fraud has not been charged) and few positive data points (fraud has been charged). 
   
   We plan to do some preprocessing we will clean, normalize, and design engineered features such as loan amount, business size, location, and NAICS codes to prepare for training.
   
   We plan to explore one class neural network techniques for anomaly detection, treating known fraud cases as the positive class and the rest as unlabeled.
   
   Since we have the lack of true negatives, we used semi-supervised metrics such as precision-at-K, silhouette score, and anomaly ranking as potential in between techniques.
   
   We also plan to use simple feedforward neural networks which we can implement in PyTorch, and/ or PyTorch, TensorFlow/Keras, scikit-learn, pandas as additional tools.
   
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