## Data Exploration and Preparation Suggestions

# Client
- Separar birth_number em gender e birth_date: the number is in the form YYMMDD for men, and in the form YYMM+50DD for women

# Transaction
- Se uma transferência é feita depois da loan não deveria ser considerada para a análise
- k-symbol, operation e type poderiam ser tratadas e agregadas porque caracterizam a transação e não têm muitos valores diferentes
    - type = 'withdrawal in cash' deve ser convertido para widthdrawal porque o detalhe que indica que é em dinheiro já está na operation
    - operation = NaN tem sempre k_symbol = interest credited
- Informações do destinatário da transferência não devem interessar
- Withdrawal da trasação deveria ser negativo
- k_symbol parece ser uma mera descição da transferência, pelo que podemos experimentar remover

# District
- Experimentar média do unemployment rate e dos crimes no district
- O atributo unemploymant rate '95 está como object, quando devia ser float
- O atributo no. of commited crimes '95 está como object, quando devia ser float

# Loan Test
- Tem 354 valores a null, no atributo status

# Transaction Test
- Tem 5130 valores a null, no atributo operation
- Tem 17419 valores a null, no atributo k_symbol
- Tem 24377 valores a null, no atributo bank
- Tem 21061 valores a null, no atributo account

# Transaction Train
- Tem 70761 valores a null, no atributo operation
- Tem 185244 valores a null, no atributo k_symbol
- Tem 299443 valores a null, no atributo bank
- Tem 294456 valores a null, no atributo account