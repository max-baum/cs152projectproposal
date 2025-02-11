# PPP Loan Fraud Detection

## Team members:
* Viren Jain
* Cameron Hatler
* Arsh Chhabra
* Max Baum

## Topline

In our CS152 project, I want to explore PPP Loan Fraud, and gauage the proficiency of a neural network in predicting PPP Loan Fraud. As a note, this project will build upon work that Max's team did in a computational statistics class.

## Background

The U.S. Federal Government dispersed nearly $700 billion in paycheck-protection program stimulus in 2020, to counteract the economic downturn caused by the COVID crisis.

It is estimated that roughly 10% of the loans were potentially issued to fradulent recipients, accounting for roughly $64b. Roughly $30b of potentially fradulent PPP loans remains outstanding. 

There is no singular public source of truth on PPP loan fraud, nor is there a single dataset that matches PPP loans to their identifying characteristics. Through a FOIA request, the SBA released a dataset of all PPP loans issues and corresponding attributes, but no indication of whether a loan was tied to fraud. In parallel, a law firm by the name of Arnold & Porter kept a tracker of fraud cases brought forth by the DOJ. Through a elaborate multistep process demanded by the disparate nature of the two datasets, Wonny, Earn and I "bridged" the two datasets last semester such that we could annotated the SBA dataset by whether or not a loan was confirmed to be fraudulent.

## The Project

Last semester, Max's team used a random forest model to evaluate whether loan attributes could be used to predict whether or not a loan was fraudulent. Our random forest model showed that there was some level of structure in attributes differentiating the confirmed fradulent loans from the non-confirmed fradulent loans. Nonetheless, our best cross-validated model hit less than a 40% true-positive rate.

Our hope for this semester would be to apply neural networks to analyze this same dataset, to see if neural networks can perform better in discriminating loans that are confirmed to be fradulent from those that are not. This project is exciting because to the best of our knowledge, there is not an equivalent public dataset out there with as large of a number of confirmed fradulent loans. The project is also exciting because of the sheer magnitude of PPP loan fraud, and the possibility that this analysis could one day help recoup money issued to fradulent parties or inform decision making when loans need to be dispersed in short timeframe.

The MVP for this project would be to implement a working neural network to classify this data, and a very successful project would be able to classify this data at a higher accuracy than the random forest model.

## Goals

* Develop a model, based upon a neural network, that is able to discrimiate confirmed fraudulent loans from non-confirm fradulent loans.
* Identify and assess different pathways that can be taken to address the extreme class imbalance in the dataset.
* Analyze the model to assess what attributes seem to most heavily impact the fradulence assessment.
* Potentially use patterns in attributes of fradulent loans to flag loans that have not yet been investigated by the DOJ as potentially fradulent.
