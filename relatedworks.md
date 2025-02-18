#Related Works

## "A loan application fraud detection method based on knowledge graph and neural network"
This research article looked at the application of neural networks and knowledge graphs to spot fraud of a loan applicant based on their cell phone history. Their reasoning is that it is difficult to fake call history, and fraudsters could be in a shared network or participate in similar habits of calling many banks to apply for many loans. Our project focuses on the information provided by the loan application and doesn't use any additional information such as phone history. We also don't would likely just be using a neural network without the inclusion of a knowledge graph or Word2Vec like in this article. 

## "Artificial Intelligence Based System for Bank Loan Fraud Prediction"
In this research paper, the authors look at applying a neural network to a dataset including applications for bank loans and whether that applicant defaulted on the loan or not. They aren't very descriptive on their methods, but include information on the process of how they went through the data to use in training the neural network. In PPP Loan Fraud Detection, our data looks slightly different since the loan is either proven to be fraudulent or not. However, the information they include about ideal models could still be useful. 

## "Towards Consumer Loan Fraud Detection: Graph Neural Networks with Role-Constrained Conditional Random Field" - https://ojs.aaai.org/index.php/AAAI/article/view/16582
The paper focuses on detecting fraudulent consumer loans using Graph Neural Networks (GNNs). The model captures relationships between different loan applicants, sellers, and intermediaries. By using data from Alipay's auto-loan system, the model achieved excellent performance in detecting fraudulent loans based on network structures, transaction patterns, and role-based behaviors. Our project does looks at PPP loans which differ from consumer loans in several ways given the lack of intermediaries. Our model will also differ from the GNN used in this paper since our focus is on indivudal loan applications while this paper explores a relationship based model. 

## Blog Post- "5 New Machine Learning Algorithms For Fraud Detection"
https://visionx.io/blog/fraud-detection-machine-learning/
THis blog post provides a survey of the various tools used for fraud detection and the various Machine Learning Algorithms being used today for that purpose. It covers techniques such as Graph neural networks, reinforcement learning, and adverserial learning which seem particularly useful to us as we work on a basis for the neural network we wish to design for this data. This will enable us to gain some experience in the prelimary steps and generate ideas for how we wish to structure our neural netowrk.

## Blog Post- "Optimizing Fraud Detection in Financial Services with Graph Neural Networks and NVIDIA GPUs"
https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/
This blog post focuses particulary on how Graph neural networks leverage the structure of interconnectedness in the data points and talks about ETL, sampling, and training of neural netowrks. It is particularly focused on how NVIDIA GPUs may be particulalry efficient for the purposes of leveraging GNNs to optimize training but it provides us with good ideas and a solid fundamental basis on how GNNs may be used to detect fraud and extend to our data set which is a relatively different problem but still quite similar.


