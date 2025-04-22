# Paycheck Protection Plan (PPP) Loan Fraud Detection OUTLINE

## Authors
Max Baum, Arsh Chhabra, Cameron Hatler, Viren Jain

## Abstract

The goal of this project is to train a neural network on a dataset which contains confirmed fraud of the Paycheck Protection Plan (PPP) loans. These loans were given out during the pandemic to help small businesses, however there have been many cases found of individuals applying and receiving these loans only to use them for personal expenses rather than business ones. The dataset we are working with contains confirmed cases of fraud, however there are no confirmed non-fraudulent cases (a case could have been fraudulent but never been found out). This presents a unique problem, but we believe it is a problem which may be solved with anomaly detection. This, along with other methods, will enable us to create a robust model for fraud detection among PPP loans. 

## Introduction

The U.S. Federal Government dispersed nearly $700 billion in paycheck-protection program stimulus in 2020, to counteract the economic downturn caused by the COVID crisis. Unfortunately, it is estimated that roughly 10% of the loans were potentially issued to fraudulent recipients, accounting for roughly $64b. Roughly $30b of potentially fraudulent PPP loans remains outstanding. 

In times of growing U.S. Federal Government deficit, alongside increased scrutiny from taxpayers regarding use of government funds, the status quo of the PPP program is highly concerning. It also calls into question what better controls could be implemented, to either prevent fraud outright or to improve the success-rate of fund recovery casework. 

This project is focused on investigating what neural network based approaches can be leveraged to predict fraudulence of loans based upon publicly-known loan attributes released under a FOIA request. The project is imperfect, and navigates many challenges including extreme class imbalance and in-comprehensive confirmation of loan fraud. The project centers around an expanded version of DANB91's PPP Fraud Dataset [^1], that draws from other developments in the court system. The dataset is imperfect, but nonetheless provides some color where none previously existed. Further, while a variety of NN based approaches can be used to address some of these outstanding issues, it must be acknowledged that no approach can ever fully correct for bad data.

Overall, this project does not seek to revolutionize or even directly inform an government decision-making in the issuance of small business loans. Rather, it seems to demonstrate the promise of increasing data-driven governance, and to confirm the existence of structure in data that can be used to guide better decision making. $60 billion is a heck of a lot of money to be lost to fraudulent activity, in times when American taxpayers are increasingly squeezed financially. One would hope that failures of this magnitude could be better derisked against moving forward.

## Ethical sweep

There is significant merit in attempting to reduce fraud in government programs like the Paycheck Protection Program. Governments operate with limited resources funded by a finite tax base, and it is important that these resources are used responsibly and reach their intended recipients. Our project, while classroom-bound, aims to explore whether machine learning can assist in identifying potentially fraudulent loans — a task that holds real-world value.

That said, there are limitations and risks inherent to this work. The accuracy of a simple, non-ML rule-based approach is likely to be low given the number and complexity of variables involved, as well as the nuanced relationships between them. Machine learning, despite its promise, also comes with risks — especially when the ground truth is uncertain. A key challenge we face is that the majority of our dataset consists of loans that have not been confirmed either way; only a very small subset has been labeled as fraudulent through the justice system. This creates a problem of label scarcity and possible bias, as loans that have not yet been investigated might be wrongfully treated as negatives. Additionally, we acknowledge potential bias in the way this dataset was assembled and processed, which may affect our model’s fairness and reliability.

We also recognize that our team, like most student groups at the 5Cs, is not demographically representative of the broader U.S. population. As such, our perspectives are inherently limited. Throughout the project, we aim to counterbalance this by engaging in frequent and transparent team discussions, especially around mistakes, uncertainties, and assumptions. Code and outputs will be reviewed by multiple team members to ensure accountability, and data will be spot-checked as time allows. Though our outputs are unlikely to leave the classroom, we believe in treating this project with the seriousness it deserves.

In terms of privacy, all data used in this project is publicly available. Given the nature of the analysis, we believe there is minimal risk of infringing upon individuals' privacy or anonymity. However, we remain conscious of the ethical line between analysis and accusation. A flagged loan is not proof of fraud — only the justice system has the authority to make that determination. Our model is a retrospective tool, and should it be used in any future capacity, it would be most appropriate as part of a triage process rather than a definitive arbiter.

Lastly, we do not currently know if the model will perform differently across sub-groups, nor are we sure what constitutes a meaningful "sub-group" in the context of this data, given its limited granularity. Misinterpretations of the model’s outputs are a real concern and must be addressed through clear communication of its capabilities, limitations, and the fact that this is an exploratory academic project — not an instrument of policy or prosecution.


## Related Work

Our project applies neural networks to detect fraudulent PPP loans, leveraging insights from Zhan and Yin (2018)[^2] and Awotunde et al. (2022)[^3]. Zhan and Yin propose a knowledge graph-based fraud detection system that constructs borrower networks from call history data. Differing from traditional models that would gather 100s of features, their model captures hidden fraud patterns such as borrowers sharing suspicious contacts—making it harder for fraudsters to evade detection. This method suggests that graph-based representations could enhance our model’s ability to detect anomalies.

Awotunde et al. (2022) employs various machine learning models including Artificial Neural Networks for loan fraud detection, achieving 98% accuracy on a bank loan dataset. The findings suggest that ANNs outperform traditional models like Decision Trees and SVMs in identifying fraudulent transactions. By integrating these approaches, our project could explore graph-enhanced neural networks for fraud detection, improving classification accuracy despite limited labeled fraud cases. The combination of graphs and ANN-based classification offers a solution to identifying fraudulent loans.

We face a problem in our dataset of having very few data points being for confirmed fraud and a lot of data points for other which we cannot classify as being non fraudulent. Hence, we have the idea of using anomaly detection via neural networks to hopefully train a model that can learn what fraud looks like and learn the class of fraudulent loans (or similarly the class of non fraud loans) and detect anomalies to detect things that do not belong to the desired class. So we have started to explore the areas of one class neural networks for anomaly detection as a result. We looked at two papers in this area: Anomaly Detection using One-Class Neural Networks by Chalapathy, et al. (2018)[^4] and Deep One-Class Classification by Ruff, et al. (2018) [^5]. Both of these papers provide useful insights into how neural networks can be optimized for anomaly detection. OC-NN’s ability to refine feature extraction for anomaly detection and Deep SVDD’s structured approach provide a useful theoretic background to our project which we hope to explore further. By leveraging these ideas we aim to develop a neural network that can generalize well even with the limited availability of confirmed fraud cases and solve a major problem present in our dataset.

By leveraging the flexibility of neural networks, we hope to find patterns in fraudulent applications of PPP loans in order to track down other suspicious borrowers. While our dataset can't be separated into true positive and negative cases of fraud, we have some true positive cases and many undetermined cases. By using anomaly detection, the model should give improved predictions of true positive and negative cases which will allow for easier application of the model. We will also aim to incorporate what various models have successfully done for different cases of fraud detection into our model. This work will help in fraud detection of PPP loans, saving money if it were incorporated, and will show how effective the combination of various neural network methods in this unique situation can be. 

## Methods

For this project, we decided a traditional fully connected Neural Network (NN) was the best solution. The data we are working with doesn't have any spatial or temporal components, thus Convolution and Recurrent NNs wouldn't provide any significant benefits. 

### Creating the Dataset

The dataset used in this report was compiled by Max Baum, Wonny Kwak, and Earn Wonghirundacha in a Computational Statistics course.

Given the importance of the data in this project, the process of piecing together the dataset is briefly explored below.

The dataset used in this project was produced with a multi step process to bridge two disparate datasets together. The first dataset was released by the Small Business Administration (SBA) under FOIA, and includes all PPP loans over $150K in value. This dataset was found on Kaggle (Bukowski, 2020), though the SBA does release this data themselves. The Kaggle version of this dataset includes a separate sheet created by the author marking 100 instances of known fraud which were ultimately exploited. To find more cases of fraud in this PPP loan dataset, a tracker compiled by law firm Arnold & Porter of legal cases brought forth by the DOJ to address CARES Act fraud was referenced. 

The two datasets use different identifiers for businesses and loans, so to join the two datasets, press releases identified by Arnold & Porter were scraped from the DOJ. This scraped data was then searched through to find business names in the SBA dataset. Ultimately, 176 distinct loans from the SBA dataset were matched to the Arnold and Porter tracker. When combined with pre-identified loans in the Kaggle dataset, a total of 248 loans were confirmed to be fraudulent. In the final dataset, the fraudulent loans were marked with a "1" in a new fraud column, while loans not found to be fraudulent were marked with a "0"

### Processing and Cleaning the Data

The initial dataset included 41 fields, of varying datatypes. To make this data utilizable by a neural network, significant processing had to be carried out.

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

After this encoding, the data was further cleaned, and any other fields that were not inherently numwerical was converted to a numerical equivalent. The data was then normalized.

### Addressing Class Imbalance

With only 248 confirmed cases of fraud, and over 900,000 loans that were not know to be fraud, the dataset used in this project was highly classed imbalanced. To address this imbalance, we used a variety of balancing techniques. The first technique was synthetic minority over-sampling technique (SMOTE), a process that generates synthetic data by interpolating between nearest neighbors in the minority class. SMOTE was only used to balance the training data. We used the SMOTE package from imblearning, creating a synthetic training dataset with an equal number of confirmed cases of fraud and non-confirmed fraud. Importantly, as SMOTE creates synthetic data through interpolation, synthetic data would be created with fields intended to represent one-hot encoding carrying values containing non-integers. To fix this, after running SMOTE, for each row in the dataset, the highest value in each set of one-hot encoded values would be converted to a 1, and all other values in that set converted to a 0.

The other balancing technique utilized in this project was weighting in cross entropy loss function. We set higher weights for misclassifying the minority class, such that calculated loss would be higher when misclassifcation of true positives occurred. Thus, the model would be oriented to maximize true-positive classification rate to avoid overlooking positives, ultimately coming at the cost of incorrectly classifying more negatives as positives.


### Building a simple fully connected neural network 
{insert}

### Building an anomaly detection focused neural network
{insert}
   
   
   We have an existing dataset which has a very large number of negatives (fraud has not been charged) and few positive data points (fraud has been charged). 
   
   We plan to do some preprocessing we will clean, normalize, and design engineered features such as loan amount, business size, location, and NAICS codes to prepare for training. 

   To start, we will construct a simple fully connected neural network. This will work as our base-line to check the effectiveness of future models. 

   In testing the accuracy of the model, it is important to remember that the effects of false positives (flagging a load as fraud when it isn't) and false negatives (not charging fraud when fraud has occurred) have very different and unique effects. The first would cause some headaches as individuals with no intent of committing fraud might be charged, the second would allow those who committed fraud to get away with no repercussions. 
   
   From the dataset, we use the important categories as the features of our neural network. We use SMOTE to handle the class imbalance that exists in our dataset, give numerical values to the categorical data, and finally normalize these features for the ideal input range of the neural network.

   We use a fully connected neural network with (N) layers, each of depth (K). These hidden layers are composed of sequential linear and ReLU functions. There will be two outputs, each giving the confidence of a loan being fraudulent or legitimate. 

   For future works, we plan to explore one class neural network techniques for anomaly detection, treating known fraud cases as the positive class and the rest as unlabeled. 
   
   Since we have the lack of true negatives, we used semi-supervised metrics such as precision-at-K, silhouette score, and anomaly ranking as potential in between techniques.
      
   We have some challenges which include data imbalance and potential hidden frauds in the unlabeled set. We addressed this with regularization, dropout, and cross-validation.
   
   We also plan to use dimension reduction techniques like t-SNE and UMAP to visualize how well fraud cases were clustered or separated.

## Results

We received mixed results following our methodology, with our fully connected neural network performing decently, though no better than early results achieved using random forest models. We could not get our anomaly detection model to function correctly. 

### Hyper parameters

| Parameter | Value |
| ------ | ------|
| Batch Size |  128  |
| Number of epochs |  Variable  |
| Layers and Layer Sizes |   Variable, typically 20 perceptrons per layer  |
| Loss function |     Cross Entropy Loss With Weights    |
| Optimization function |    Adam  |



  _Purpose of Section:_ In this section, we will detail the results achieved following our methodology
  
  _Topic Sentence:_ We received mixed results following our methodology
  
  Prediction -- We are confident that a simple approach will perform adequately (but not necessarily impressively) on the data. We are not as certain about the performance of the anomaly/outlier approach. It could perform significantly better, it could perform roughly the same if not worse.
  
## Discussion

   Using our custom dataset, we trained both a fully-connected neural network and a neural network focused on anomaly detection.

  Our models achieved varying levels of success. The following figures demonstrate this performance.

  In tests, Model A achieved a higher true positive rate than Model B, meaning that the model was able to correctly identify confirmed fraud more frequently than Model B. This said, Model A had a higher false positive rate than Model B, meaning that Model A is incorrectly flagging loans as fraud more frequently than Model B. Ultimately, while Model B has the highest overall accuracy, this accuracy comes at the expense of missing confirmed positives.

  Compared to other studies, our models overall performed worse. This is due to limitations in both the model and the data.

  While our models did not achieve ideal results, they nonetheless demonstrate the potential of our approach. We think the following steps can be followed to both improve our own model and to generally improve neural-network-based approaches to fraud detection moving forward.

------------------------
The performance of the fully-connected neural network models across two architectural variants reveals several important trends and limitations, especially in the context of imbalanced classification.

Both confusion matrices bellow demonstrate the severe class imbalance in the dataset, with the vast majority of samples being labeled as negative (0). However, this imbalance is likely artificial, given that only a small set of known fraud cases have been labeled as such, while the rest are not verified negatives but rather unlabeled or unknown instances. This labeling strategy introduces noise and potential misrepresentation into model evaluation, especially regarding false negatives.

In the first model (200-200-200 architecture), the model correctly identified 27 of the 50 actual positive cases (true positives), with 23 false negatives. However, it also produced 11,737 false positives out of 193,657 presumed negatives, resulting in a high false positive rate. This may be concerning as it does not minimize unnecessary investigations for real-world applications. Yet, given the potential that many of the negative-labeled cases could actually be fraudulent but unlabeled, false positives in this context might actually capture real fraud instances or characteristics that resemble confirmed positives.

In the second model (214-214-100-50-25 architecture), the false positives decreased significantly to 4,827, but at the cost of true positives dropping to 23 and false negatives increasing to 27. This indicates a more conservative model that is less likely to flag loans as fraudulent, potentially leading to more missed fraud cases. Given the investigative nature of the task, this trade-off may be undesirable in comparison to the first one.

Interestingly, increasing model depth and slightly varying neuron counts did not materially improve true positive detection, which stayed around 23–27 out of 50 positives. This suggests that model architecture alone may not solve the issue and further performance gains may require additional techniques. 

A critical point is that traditional performance metrics (e.g., precision, accuracy) are not fully valid here due to the treatment of unknowns as negatives. The confusion matrix gives a partial view, but we cannot trust metrics that rely on true negatives, because many of these may be mislabeled or uncertain. As such, model performance should be viewed through the lens of investigative support rather than predictive certainty. A high false positive count may be acceptable if it leads to uncovering hidden fraudulent cases.

  
## Reflection and Looking Forward

  _Purpose of Section:_ In this section, we will reflect more broadly upon what has been achieved, and what more component systems at the intersection of neural networks and fraud detection could mean for society.
  
  _Topic Sentence:_ We are excited by the results that we attained through this project, and we believe there is much to be gained for society in the realm of neural network-enabled fraud detection.

## Citations
[^1]: https://www.kaggle.com/datasets/danb91/covid-ppp-loan-data-with-fraud-examples?select=ppp_fraud_cases.csv
[^2]: https://dl.acm.org/doi/10.1145/3194206.3194208
[^3]: https://link.springer.com/chapter/10.1007/978-3-030-96305-7_43
[^4]: https://arxiv.org/abs/1802.06360
[^5]: https://www.researchgate.net/publication/329829847_Deep_One-Class_Classification
