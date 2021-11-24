# Data Understanding

## New Attributes
- Gender
- Birth Date
- Average Transaction Amount
- Average Transaction Balance
- Number of Transactions per Account
- Average Commited Crimes
- Average Unemployment Rate
- Same District - If the owner and account have the same district
- Days Between - Time gap between creating the account and the loan appliance

## Client
- **Extracted attributes**: `gender` and `birth_date`
- `birth_number` gives information about both the birth date and the gender: YYMMDD for men, and YYMM+50DD for women. 
- We need to split this attribute into `gender` and `birth_date`.

- Analysis of the `gender` of the clients:
![](../images/client/owner_age_on_loan.jpg)

- The `gender` of the client does not seem to influence the loan status:
![](../images/client/client_gender_on_loan.jpg)

***
## District
- `Unemploymant rate '95` is of type object, but should be float. There is an observation with unemployment rate '95 = '?'. It does not follow a normal distribution so we shouldn't replace it with the mean.
![](../images/district/unemploymant_rate_95_qqplot.jpg)

- The attribute `no. of commited crimes '95` is of type object but should be of type int. There is an observation with no. of commiter crimes '95 = '?' 

- Outlier in the `no. of inhabitants`
![](../images/district/inhabitants_no.png)
![](../images/district/inhabitants_no_boxplot.png)

- Outlier in `no. of crimes '95`
![](../images/district/crimes_95.jpg)
![](../images/district/crimes_95_boxplot.jpg)

- Outlier in `no. of crimes '96`
![](../images/district/crimes_96.jpg)
![](../images/district/crimes_96_boxplot.jpg)

- The districts with more criminality are the ones with higher average salary
![](../images/district/crimes_salary.jpg)

- **Extracted attributes**: average unemployment rate, average number of commited crimes, same_district (account and client have the same district)

***

## Loan Train
- The number of successful loans is much higher than the number of fraudulent loans(46) - `status` is imbalanced.
![](../images/loan/loan_train_status.jpg)

- `amount`/`duration` = `payments`
![](../images/loan/amount_payments_duration.jpg)

![](../images/loan/loan_train_correlation.jpg)

- Loans with a higher `amount` tend to be fraudulent.

![](../images/loan/loan_train_amount_status.jpg)

***
## Transaction Train
- 70761 null in operation
- 185244 null in k_symbol
- 299443 null in bank
- 294456 null in account

- The **withdrawal in cash** `type` should be replaced by **withdrawal**, because the `type` should only represent if the transaction is withdrawal(-) or credit(+). Therefore, **withdrawal in cash** includes duplicated information about the mode of transaction, which is already represented in 'operation':
![](../images/transaction/transaction_type.jpg)

- Most of the transactions don't specify the `bank` and the quantity of transactions to each bank is almost the same.

![](../images/transaction/trans_train_bank.jpg)

- `operation` = NaN has always `k_symbol` = interest credited

- A transaction that is done after the loan is granted, should no be considered in the analysis.

- **Extracted attributes**: average transaction balance, average transaction amount, number of transactions of the account

![](../images/transaction/num_trans_status.jpg)
![](../images/transaction/avg_balance_status.jpg)
![](../images/transaction/avg_amount_status.jpg)
***

## Loan and Account

- Time gap between creating the account and the loan appliance (days between). 
- People who apply for the loan right after creating the bank account tend to be fraudulent.

![](../images/loan_account_dates.jpg)

***
# LOGBOOK
- Changed attribute "code" to "district_id" in District table

# TODOs
