# Paycheck Protection Plan (PPP) Loan Fraud Detection

The goal of this project is to train a neural network on a dataset which contains confirmed fraud of the Paycheck Protection Plan (PPP) loans. These loans were given out during the pandemic to help small businesses, however there have been many cases found of individuals applying and recieving these loans only to use them for personal expenses rather than business ones. The dataset we are working with contains confirmed cases of fraud, however there are no confirmed non-fraudulent cases (a case could have been fraudlent but never been found out). This presents a unique problem, but we believe it is a problem which may be solved with anomaly detection. This, along with other methods, will enable us to create a robust model for fraud detection of PPP loans. 

First body section - **Max will take**
Topic related to first point of thesis
Supporting paragraphs

Second body section - **Someone talk about findings from two papers**
Topic related to second point of thesis

Our project applies neural networks to detect fraudulent PPP loans, leveraging insights from Zhan and Yin (2018) and Awotunde et al. (2022). Zhan and Yin propose a knowledge graph-based fraud detection system that constructs borrower networks from call history data. Differing from traditional models that would gather 100s of features, their model captures hidden fraud patterns such as borrowers sharing suspicious contacts—making it harder for fraudsters to evade detection. This method suggests that graph-based representations could enhance our model’s ability to detect anomalies.

Awotunde et al. (2022) employs various machine learning models including Artificial Neural Networks for loan fraud detection, achieving 98% accuracy on a bank loan dataset. The findings suggest that ANNs outperform traditional models like Decision Trees and SVMs in identifying fraudulent transactions. By integrating these approaches, our project could explore graph-enhanced neural networks for fraud detection, improving classification accuracy despite limited labeled fraud cases. The combination of graphs and ANN-based classification offers a solution to identifying fraudulent loans.


We face a problem in our dataset of having very few data points being for confirmed fraud and a lot of data points for other which we cannot classify as being non fraudulent. Hence, we have the idea of using anomaly detection via neural networks to hopefully train a model that can learn what fraud looks like and learn the class of fraudulent loans (or similiarly the class of non fraud loans) and detect anomalies to detect things that do not belong to the desired class. So we have started to explore the areas of one class neural networks for anomaly dtection as a result. We looked at two papers in this area: Anomaly Detection using One-Class Neural Networks by Chalapathy, et al and Deep One-Class Classification by Ruff, et al. Both of these papers provide useful insights into how neural networks can be optimized for anomaly detection. OC-NN’s ability to refine feature extraction for anomaly detection and Deep SVDD’s structured approach provide a useful theoretic background to our project which we hope to explore further. By leveraging these ideas we aim to develop a neural network that can generalize well even with the limited availability of confirmed fraud cases and solve a major problem present in our dataset.


Supporting paragraphs
Third body section -- **Someone talk about findings from two other papers**
Topic related to third point of thesis
Supporting paragraphs
Conclusion -- **Someone talk about overarching goals**

By leveraging the flexibility of neural networks, we hope to find patterns in fraudulent applications of PPP loans in order to track down other suspicious borrowers. While our dataset can't be seperated into true positive and negative cases of fraud, we have some true positive cases and many undetermined cases. By using anomaly detection, the model should give improved predictions of true positive and negative cases which will allow for easier application of the model. We will also aim to incorporate what various models have succesfully done for different cases of fraud detection into our model. This work will help in fraud detection of PPP loans, saving money if it were incorporated, and will show how effective the combination of various neural network methods in this unique situation can be. 