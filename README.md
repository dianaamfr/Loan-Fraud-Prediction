# AC

## Banking - Case Description Summary

### Task description
- Who is a good client (whom to offer some additional services) and who is a bad client (whom to watch carefully to minimize the bank loses). 
- the bank has data about: the accounts (transactions within several months), the loans already granted, the credit cards issued. 
- prediction of whether a loan will end successfuly.

### Data description
- **account** (4500 objects - each record describes static characteristics of an account
    - has static characteristics (e.g. date of creation, address of the branch)
    - the dynamic characteristics of each **account** are given in the "transaction" relation
- **transaction** (1056320 objects) - each record describes one transaction on an account
    - dynamic characteristics of each **account** (e.g. payments debited or credited, balances)

- **client** (5369 objects) - each record describes characteristics of a client, who can manipulate accounts
    - one client can have multiple accounts
    - multiple clients can manipulate with single account
- **disposition** (5369 objects) - each record relates together a **client** with an **account** i.e. this relation describes the rights of clients to operate accounts,
    - *NOTE*: more than one client can operate the same account, and a client may have more than one account, which means there would be columns with the same account_id or client_id if we merged this two "tables"

**Services which the bank offers to its clients**
- **loan** (682 objects) - each record describes a loan granted for a given account
    - at most one loan can be granted for an account
- **credit card** (892 objects) - each record describes a credit card issued to an account
    - multiple credit cards can be issued to an account
- **district** (77 objects) - each record describes demographic characteristics of a district
    - gives some publicly available information about the districts (e.g. the unemployment rate) which may be used to deduce additional information about the clients

        

