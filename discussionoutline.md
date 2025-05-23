# Paycheck Protection Plan (PPP) Loan Fraud Detection

## Team Members

* Viren Jain
* Cameron Hatler
* Arsh Chhabra
* Max Baum

## Abstract

The goal of this project is to train a neural network on a dataset which contains confirmed fraud of the Paycheck Protection Plan (PPP) loans. These loans were given out during the pandemic to help small businesses, however there have been many cases found of individuals applying and recieving these loans only to use them for personal expenses rather than business ones. The dataset we are working with contains confirmed cases of fraud, however there are no confirmed non-fraudulent cases (a case could have been fraudlent but never been found out). This presents a unique problem, but we believe it is a problem which may be solved with anomaly detection. This, along with other methods, will enable us to create a robust model for fraud detection among PPP loans. 

## Introduction

The U.S. Federal Government dispersed nearly $700 billion in paycheck-protection program stimulus in 2020, to counteract the economic downturn caused by the COVID crisis. Unfortunately, it is estimated that roughly 10% of the loans were potentially issued to fradulent recipients, accounting for roughly $64b. Roughly $30b of potentially fradulent PPP loans remains outstanding. 

In times of growing U.S. Federal Government deficit, alongside increased scrutiny from taxpayers regarding use of government funds, the status quo of the PPP program is highly concerning. It also calls into question what better controls could be implemented, to either prevent fraud outright or to improve the success-rate of fund recovery casework. 

This project is focused on investigating what neural network based approaches can be leveraged to predict fradulence of loans based upon publically-known loan attributes released under a FOIA request. The project is imperfect, and navigates many challenges including extreme class imbalance and imcomprehensive confirmation of loan fraud. The project centers around an expanded verion of DANB91's PPP Fraud Dataset [^1], that draws from other developments in the court system. The dataset is imperfect, but nonetheless provides some color where none previously existed. Further, while a variety of NN based approaches can be used to address osme of these outstanding issues, it must be acknowledged that no approach can ever fully correct for bad data.

Overall, this project does not seek to revolutionize or even directly inform an government decisionmaking in the issuance of small business loans. Rather, it seems to demonstrate the promise of increasing data-driven governance, and to confirm the existence of structure in data that can be used to guide better decision making. $60 billion is a heck of a lot of money to be lost to fradulent activity, in times when American taxpayers are increasingly squeezed financially. One would hope that failures of this magnitude could be better derisked against moving forward.

## Ethical Sweep

_(TO BE MERGED IN LATER)_

## Related Work

Our project applies neural networks to detect fraudulent PPP loans, leveraging insights from Zhan and Yin (2018)[^2] and Awotunde et al. (2022)[^3]. Zhan and Yin propose a knowledge graph-based fraud detection system that constructs borrower networks from call history data. Differing from traditional models that would gather 100s of features, their model captures hidden fraud patterns such as borrowers sharing suspicious contacts—making it harder for fraudsters to evade detection. This method suggests that graph-based representations could enhance our model’s ability to detect anomalies.

Awotunde et al. (2022) employs various machine learning models including Artificial Neural Networks for loan fraud detection, achieving 98% accuracy on a bank loan dataset. The findings suggest that ANNs outperform traditional models like Decision Trees and SVMs in identifying fraudulent transactions. By integrating these approaches, our project could explore graph-enhanced neural networks for fraud detection, improving classification accuracy despite limited labeled fraud cases. The combination of graphs and ANN-based classification offers a solution to identifying fraudulent loans.

We face a problem in our dataset of having very few data points being for confirmed fraud and a lot of data points for other which we cannot classify as being non fraudulent. Hence, we have the idea of using anomaly detection via neural networks to hopefully train a model that can learn what fraud looks like and learn the class of fraudulent loans (or similiarly the class of non fraud loans) and detect anomalies to detect things that do not belong to the desired class. So we have started to explore the areas of one class neural networks for anomaly dtection as a result. We looked at two papers in this area: Anomaly Detection using One-Class Neural Networks by Chalapathy, et al. (2018)[^4] and Deep One-Class Classification by Ruff, et al. (2018) [^5]. Both of these papers provide useful insights into how neural networks can be optimized for anomaly detection. OC-NN’s ability to refine feature extraction for anomaly detection and Deep SVDD’s structured approach provide a useful theoretic background to our project which we hope to explore further. By leveraging these ideas we aim to develop a neural network that can generalize well even with the limited availability of confirmed fraud cases and solve a major problem present in our dataset.

By leveraging the flexibility of neural networks, we hope to find patterns in fraudulent applications of PPP loans in order to track down other suspicious borrowers. While our dataset can't be seperated into true positive and negative cases of fraud, we have some true positive cases and many undetermined cases. By using anomaly detection, the model should give improved predictions of true positive and negative cases which will allow for easier application of the model. We will also aim to incorporate what various models have succesfully done for different cases of fraud detection into our model. This work will help in fraud detection of PPP loans, saving money if it were incorporated, and will show how effective the combination of various neural network methods in this unique situation can be. 

## Remaining Sections

* Methods
  
   We have an existing datset which has a very large number of negatives (fraud has not been charged) and few positive data points (fraud has been charged). 
   
   We plan to do some preprocessing we will clean, normalize, and design engineered features such as loan amount, business size, location, and NAICS codes to prepare for training.
   
   We plan to explore one class neural network techniques for anomaly detection, treating known fraud cases as the positive class and the rest as unlabeled.
   
   Since we have the lack of true negatives, we used semi-supervised metrics such as precision-at-K, silhouette score, and anomaly ranking as potential in between techniques.
   
   We also plan to use simple feedforward neural networks which we can implement in PyTorch, and/ or PyTorch, TensorFlow/Keras, scikit-learn, pandas as additional tools.
   
   We have some challenges which include data imbalance and potential hidden frauds in the unlabeled set. We addressed this with regularization, dropout, and cross-validation.
   
   We also plan to use dimensionality reduction techniques like t-SNE and UMAP to visualize how well fraud cases were clustered or separated.

* Results

  _Purpose of Section:_ In this section, we will detail the results acheived following our methodology
  
  _Topic Sentence:_ We recieved mixed results following our methodology
  
  Prediction -- We are confident that a simple approach will perform adequately (but not necessarily impressively) on the data. We are not as certain about the performance of the anomoly/outlier approach. It could perform significantly better, it could perform roughly the same if not worse.
  
* Discussion

   Using our custom dataset, we trained both a fully-connected neural network and a neural network focused on anonmoly detection.

  Our models achieved varying levels of success. The following figures demonstrate this performance.

  In tests, Model A achieved a higher true positive rate than Model B, meaning that the model was able to correctly identify confirmed fraud more frequently than Model B. This said, Model A had a higher false positive rate than Model B, meaning that Model A is incorrectly flagging loans as fraud more frequently than Model B. Ultimately, while Model B has the highest overall accuracy, this accuracy comes at the expense of missing confirmed positives.

  Compared to other studies, our models overall performed worse. This is due to limitations in both the model and the data.

  While our models did not achieve ideal results, they nonetheless demonstrate the potential of our approach. We think the following steps can be followed to both improve our own model and to generally improve neural-network-based approaches to fraud detection moving forward.

  
* Reflection and Looking Forward

  _Purpose of Section:_ In this section, we will reflect more broadly upon what has been achieved, and what more compentent systems at the intersection of neural networks and fraud detection could mean for society.
  
  _Topic Sentence:_ We are excited by the results that we attained through this project, and we believe there is much to be gained for society in the realm of neural network-enabled fraud detection.

## Citations
[^1]: https://www.kaggle.com/datasets/danb91/covid-ppp-loan-data-with-fraud-examples?select=ppp_fraud_cases.csv
[^2]: https://dl.acm.org/doi/10.1145/3194206.3194208
[^3]: https://link.springer.com/chapter/10.1007/978-3-030-96305-7_43
[^4]: https://arxiv.org/abs/1802.06360
[^5]: https://www.researchgate.net/publication/329829847_Deep_One-Class_Classification


