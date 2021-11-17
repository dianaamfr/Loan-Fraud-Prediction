# Data Mining Project

## Business Understanding

### Analysis of requirements with the end user

The aim of this project is to help bank managers improve their understanding of the bank customers in order to identify high-value customers and minimize the bank losses. 
The bank wants to be able to perform specific actions to improve its services, which would depend on the predicted information. For example, by knowing which customers to trust a loan, it could avoid loaning to fraudulent and non-compliant clients.

### Business Goals

Reduced the number of fraud by 50%.
Answer the question "To loan or not to loan?": 
Determine if a given loan will end successfully, given data about the clients (the accounts, transactions, loans already granted, credit cards issued and demographic information).

### Data Mining Goals

- AUC minimo
- Entender significado do AUC no contexto do problema
- reduce the false positives

Our goal is to build a model that uses available client data (the bank provides previous banking activity) to predict the likelihood of success of a loan for each client.

In other words, we expect to achieve the most accurate values for the loan success probability of each client, given that the output may vary between 0 (loan not paid) and 1 (loan paid).

Therefore, this project is categorized as a Scoring Classification Binary Problem, and we will use the AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve to evaluate its performance.