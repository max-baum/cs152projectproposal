# PPP Loan Fraud Detection

## Team members
* Viren Jain
* Cameron Hatler
* Arsh Chhabra
* Max Baum

## Project scope

In our CS152 project, we want to explore PPP loan praud, and gauage the proficiency of a neural network in predicting (retrospectively) whether a loan was issued under fradulent premises given loan attributes provided by the Small Business Administration. As a note, this project will build upon work that Max's team did in a computational statistics class.

## Outline

The U.S. Federal Government Paycheck Protection Program (PPP) bolstered the U.S. economy and protected jobs through the lockdowns induced by COVID-19, but the program was also riddled with fraud.

Digital systems, including neural-network based systems, are increasingly utilized to prevent financial fraud.

This project will employ a custom-built dataset to test the efficacy of a neural-network model in detecting PPP fraud.

We expect there to be a range of challenges in the execution of this project.

Ultimately, we hope to develop a model that performs efficaciously on our dataset. We further hope that this model could one day help inform the investigation of PPP fraud and/or the development of IT systems that proactively gaurd against fraud in U.S. loan programs.

## Ethical sweep

**General Questions:**
* Should we even be doing this?
  * Yes. There is significant merit and virtue in lessening fraud in government programs. Governments have limited resources, funded by a finite tax base, and it is pertinant that these resources are used properly and responsibly.

* What might be the accuracy of a simple non-ML alternative?
  * Likely low, though uncertain. Given the broad array of attributes to consider, and the sophisticated relationships these attributes may have with the likelihood of a loan being fradulent, ML approaches are likely most promising.

* What processes will we use to handle appeals/mistakes?
  *  Team communication will be frequent and transparent. Mistakes will be documented rather than ignored. Code and outputs will be verified independently by multiple team members, though outputs are unlikely to leave the classroom.

* How diverse is our team?
  * Any team of students at the 5Cs is not a comprehensive and proportional reflection of U.S. society. By that standard, our team is not diverse.
 
**Data Questions:**
* Is our data valid for its intended use?
  * Yes, with known and communicated limitations.
 
* What bias could be in our data? (All data contains bias.)
  * Many. We think the most likely source of bias is causes simply by the low throughput of the Justice system, and that not all loans have been investigated equally for fraud. There could also be bias caused by Max, caused by errors in the processing and devlopment of this dataset.
 
How could we minimize bias in our data and model?
  * This is a very complicated question given the nature of the data. One of the issues here is the lack of "real" negatives in the dataset, so maybe a clustering tecnique could be used here. We think there are quite a few procedural questions to discuss as we handle the data and develop the model. We cannot fix the SBA and the Department of Justice.
    
* How should we “audit” our code and data?
  * The data can be spot-checked, or further if we have time, all 248 cases of fraud can be reviewed manually (though this may not be necessary). All code should be reviewed by multiple parties.

**Impact Questions:**

* Do we expect different errors rates for different sub-groups in the data?
  * There is no reason to suspect we would, but we simply don't know. It is further unclear what a "sub-group" would constitute here, as we have limited information identifying characteristics of businesses, rather information about the loans.
  
* What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?
  * First and foremost, this is a retroactive assessment, and it can be directly applied toward proactive loan fraud prevention going forward.
  * Secondly, even if this model could be used proactively, a postive indication of fraud is not the confirmation of fraud. The only thing that can confirm fraud is the Justice system. At most, this model could be used in the triaging of case work.
  * Misinterpretations should be prevented with clear communication of the models known capabilities and limitations, as clarified above and as will be further discussed among the team.
    
* How might we impinge individuals’ privacy and/or anonymity?
  * All information used here is publically available. Given the nature of the dataset and analysis, there is little risk that this project will impede upon privacy rights of individuals.
