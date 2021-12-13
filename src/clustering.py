from clean import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from itertools import product


#######
# Utils
#######

def pca_analysis(data):
    n_components = len(data.columns)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    variance = pca.explained_variance_ratio_ 
    var=np.cumsum(np.round(variance, 3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.plot(var)
    plt.show()

def dbscan_param_tuning_silhouette(df_scale):
    print(df_scale)

    eps_values = np.arange(0.2,1.5,0.1) 
    min_samples = np.arange(2,5) 
    dbscan_params = list(product(eps_values, min_samples))
    no_of_clusters = []
    sil_score = []
    epsvalues = []
    min_samp = []

    for p in dbscan_params:
        dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(df_scale)
        epsvalues.append(p[0])
        min_samp.append(p[1])
        if len(np.unique(dbscan_cluster.labels_)) > 1:
            no_of_clusters.append(len(np.unique(dbscan_cluster.labels_)))
            sil_score.append(silhouette_score(df_scale, dbscan_cluster.labels_))

    eps_min = list(zip(no_of_clusters, sil_score, epsvalues, min_samp))
    eps_min_df = pd.DataFrame(eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
    print(eps_min_df)

def pca(n_components, data):
    pca_cols = []
    for i in range(n_components):
        pca_cols.append('pc'+str(i+1))

    pca = PCA(n_components=n_components)
    pca.fit(data)
    pca_scale = pca.transform(data)
    pca_df = pd.DataFrame(pca_scale, columns=pca_cols)
    print(pca.explained_variance_ratio_)

    return pca_df

##########
# Features
##########

def economic_features():
    # Average Balance, Num Transactions
    df1 = db.df_query(f'SELECT account_id, AVG(balance) AS average_balance, MIN(balance) AS min_balance, COUNT(trans_id) AS num_trans \
        FROM account JOIN trans_train USING(account_id) JOIN disposition USING(account_id) JOIN client USING(client_id) \
        GROUP BY account_id')

    # Amount Features
    df2 = db.df_query('SELECT amount, account_id, trans_type FROM trans_train')
    df2.loc[df2["trans_type"]=="withdrawal", "amount"] *= -1
    amount = df2.groupby(['account_id']).agg({'amount':['mean','min', 'max']}).reset_index()
    amount.columns = ['account_id', 'avg_amount', 'min_amount', 'max_amount']

    df_economic = pd.merge(df1, amount, on='account_id', how='left')

    # Ratio of credits
    df4 = db.df_query('SELECT trans_type, account_id FROM trans_train')
    type_counts = df4.groupby(['account_id', 'trans_type']).size().reset_index(name='counts')

    credit_counts = type_counts[type_counts['trans_type'] == 'credit']
    credit_counts.columns = ['account_id', 'trans_type', 'num_credits']
    credit_counts = credit_counts.drop(columns=["trans_type"])

    withdrawal_counts = type_counts[type_counts['trans_type'] == 'withdrawal']
    withdrawal_counts.columns = ['account_id', 'trans_type', 'num_withdrawals']
    withdrawal_counts = withdrawal_counts.drop(columns=["trans_type"])

    trans_type_count_df = pd.merge(credit_counts, withdrawal_counts, on="account_id", how="outer")
    trans_type_count_df.fillna(0, inplace=True)
    trans_type_count_df['credit_ratio'] = trans_type_count_df['num_credits'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])
    trans_type_count_df.drop(columns=['num_credits', 'num_withdrawals'], inplace=True)

    df_economic = pd.merge(df_economic, trans_type_count_df, on='account_id', how='left')


    df3 = db.df_query('SELECT operation, account_id, trans_id FROM trans_train')

    # Operation Nan and rename
    df3["operation"].fillna("interest credited", inplace=True)
    df3.loc[df3["operation"]=="credit in cash", "operation"] = "CashC"
    df3.loc[df3["operation"]=="collection from anot", "operation"] = "Coll"
    df3.loc[df3["operation"]=="interest credited", "operation"] = "Interest"
    df3.loc[df3["operation"]=="withdrawal in cash", "operation"] = "CashW"
    df3.loc[df3["operation"]=="remittance to anothe", "operation"] = "Rem"
    df3.loc[df3["operation"]=="credit card withdraw", "operation"] = "CardW"

    operation = df3.groupby(['account_id', 'operation']).agg({'trans_id': ['count']}).reset_index()
    operation.columns = ['account_id', 'operation','operation_count']
    
    # credit in cash = CashC
    cashC_operation = operation[operation['operation'] == 'CashC']
    cashC_operation.columns = ['account_id', 'operation', 'num_cash_credit']
    cashC_operation = cashC_operation.drop(['operation'], axis=1)

    # collection from another bank = Coll
    coll_operation = operation[operation['operation'] == 'Coll']
    coll_operation.columns = ['account_id', 'operation',  'num_coll']
    coll_operation = coll_operation.drop(['operation'], axis=1)

    # interest credited = Interest,
    interest_operation = operation[operation['operation'] == 'Interest']
    interest_operation.columns = ['account_id', 'operation',  'num_interest']
    interest_operation = interest_operation.drop(['operation'], axis=1)

    # withdrawal in cash = CashW
    cashW_operation = operation[operation['operation'] == 'CashW']
    cashW_operation.columns = ['account_id', 'operation', 'num_cash_withdrawal']
    cashW_operation = cashW_operation.drop(['operation'], axis=1)

    # remittance to another bank = Rem
    rem_operation = operation[operation['operation'] == 'Rem']
    rem_operation.columns = ['account_id', 'operation', 'num_rem']
    rem_operation = rem_operation.drop(['operation'], axis=1)

    # credit card withdrawal = CardW
    cardW_operation = operation[operation['operation'] == 'CardW']
    cardW_operation.columns = ['account_id', 'operation', 'num_card_withdrawal']
    cardW_operation = cardW_operation.drop(['operation'], axis=1)
    
    operation_df = cashC_operation.merge(coll_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(interest_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(cashW_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(rem_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(cardW_operation, on='account_id',how='outer')
    operation_df.fillna(0, inplace=True)

    operation_num = ['num_cash_credit','num_rem','num_card_withdrawal', 'num_cash_withdrawal', 'num_interest', 'num_coll']
    operation_df['total_operations'] = operation_df[operation_num].sum(axis=1)

    # Calculate Ratio for each operation
    operation_df['cash_credit_ratio'] = operation_df['num_cash_credit']/operation_df['total_operations']
    operation_df['rem_ratio'] = operation_df['num_rem']/operation_df['total_operations']
    operation_df['card_withdrawal_ratio'] = operation_df['num_card_withdrawal']/operation_df['total_operations']
    operation_df['cash_withdrawal_ratio'] = operation_df['num_cash_withdrawal']/operation_df['total_operations']
    operation_df['interest_ratio'] = operation_df['num_interest']/operation_df['total_operations']
    operation_df['coll_ratio'] = operation_df['num_coll']/operation_df['total_operations']

    operation_df.drop(columns=operation_num, inplace=True)
    operation_df.drop(columns=['total_operations'], inplace=True)

    df_economic = pd.merge(df_economic, operation_df, on="account_id", how="outer")
    
    return df_economic

def merge_transactions_clients(db):
    clients = clean_clients(db)
    accounts = clean_accounts(db)
    district = clean_districts(db)
    disp = db.df_query('SELECT * FROM disposition')
    transactions = clean_transactions(db)

    df = pd.merge(clients, disp,  on='client_id', how="left")
    df = pd.merge(df, accounts,  on='account_id', how="left")
    df = pd.merge(df, district, left_on='client_district_id', right_on='district_id')
    df = pd.merge(df, transactions, how="left", on="account_id")
    
    return df

def get_cardinal_point(region):
    cardinal_points = ['south','west','north','east', 'central']
    for cp in cardinal_points:
        if cp in region:
            return cp
    return 'central'


############
# Algorithms
############

def clustering_agglomerative():
    clients = clean_clients(db)
    district = clean_districts(db)
    loan = clean_loans(db)

    # TODO: delete this, already encoded
    #district['region'] = district.apply(lambda x: get_cardinal_point(x['region']), axis=1)
    
    disp = db.df_query('SELECT * FROM disposition')
    
    df = pd.merge(loan, disp, on='account_id', how="left")
    df = pd.merge(df, clients, on='client_id', how="left")
    df = pd.merge(df, district, left_on="client_district_id", right_on="district_id", how="left")

    df_economic = economic_features()
    df = pd.merge(df, df_economic, on='account_id', how="left")
    df.fillna(0, inplace=True)

    # Maybe use the age of the client when the loan was issued
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))

    df = df[['average_balance', 'age', 'num_trans']]

    # Create Dendrogram
    #dendrogram = sch.dendrogram(sch.linkage(df, method='ward'))
    # plt.savefig('dendogram.jpg')
    # plt.clf()

    # Create Clusters
    hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')
    data = df.values

    # save clusters for chart
    labels = hc.fit_predict(data)

    # plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[labels == 0,0],data[labels == 0,1],data[labels == 0,2], s = 40 , color = 'orange', label = "cluster 1", alpha=0.6)
    ax.scatter(data[labels == 1,0],data[labels == 1,1],data[labels == 1,2], s = 40 , color = 'green', label = "cluster 2", alpha=0.6)
    ax.scatter(data[labels == 2,0],data[labels == 2,1],data[labels == 2,2], s = 40 , color = 'red', label = "cluster 3")
    ax.set_xlabel('Rem Ratio-->')
    ax.set_ylabel('Average Salary->')
    ax.set_zlabel('Min Balance-->')
    ax.legend()
    plt.show()

def clustering_kmeans2():

    clients = clean_clients(db)
    district = clean_districts(db)
    loan = clean_loans(db)

    # TODO: delete this, already encoded
    #district['region'] = district.apply(lambda x: get_cardinal_point(x['region']), axis=1)
    
    disp = db.df_query('SELECT * FROM disposition')
    
    df = pd.merge(loan, disp, on='account_id', how="left")
    df = pd.merge(df, clients, on='client_id', how="left")
    df = pd.merge(df, district, left_on="client_district_id", right_on="district_id", how="left")

    df_economic = economic_features()
    df = pd.merge(df, df_economic, on='account_id', how="left")
    df.fillna(0, inplace=True)

    # Maybe use the age of the client when the loan was issued
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))

    # rem_ration - coll ration OK
    # rem_ration - age 
    # average_balance - coll_ratio
    # average_balance - cash_withdrawal_ratio
    # max_amount - cash_withdrawal_ratio OK with average salary and age
    # max_amount - credit_withdrawal_ratio
    # max_amount - card_withdrawal_ratio
    # max_amount - credit_raio
    # avg_balance - credit_ratio - nr_ent UNA MIERDA
    # num_trans - credit_ratio - ratio_ent

    # plt.scatter(df['num_trans'],df['credit_ratio'])
    # plt.show()
    # plt.clf()

    
    # df = encode_category(df, 'disp_type')
    # df = df[['amount', 'payments', 'loan_status', 'num_trans', 'avg_amount', 'disp_type', 'district_id', 'average_salary', 'unemployment_growth', 'ratio_entrepeneurs', 'credit_ratio', 'age']]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(df)
    
    kmeans = KMeans(3)
    identified_clusters = kmeans.fit_predict(x)

    print(f'Inertia: {kmeans.inertia_}')

    df.insert(loc=0, column='cluster', value=identified_clusters.tolist())

    # pd.concat([k1, k2, k3]).groupby('cluster').mean()

    # k1,k2,k3 = [x[np.where(kmeans.labels_==i)] for i in range(3)] # range(3) because 3 clusters

    k1,k2,k3 = [df.loc[df['cluster']==i] for i in range(3)] # range(3) because 3 clusters

    print(type(k1))

    print(len(k1))
    print(len(k2))
    print(len(k3))

    print(k1.describe())
    print(k2.describe())
    print(k3.describe())

    # # PCA
    # reduced_data = PCA(n_components=2).fit_transform(df)
    # results = pd.DataFrame(reduced_data,columns=['pca1', 'pca2'])

    # fig = plt.figure(figsize=(6,6))
    # ax = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax)

    plt.scatter(x=df["age"], y=df["amount"],s=40, marker='o', alpha=1, c=df['cluster'])
    plt.show()
    plt.clf()

    nr_clusters = []
    inertias = []
    scores = []
    range_values = np.arange(2,11)
    for k in range_values:
        kmeans = KMeans(k)
        kmeans.fit(x)
        nr_clusters.append(k)
        inertias.append(kmeans.inertia_)
        score = metrics.silhouette_score(x, kmeans.labels_, metric='euclidean', sample_size=len(x))
        print('Silhouette score =', score)
        scores.append(score)

    plt.plot(nr_clusters, inertias)
    plt.title('Evolution of Inertia with number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    plt.figure()
    plt.bar(range_values, scores, width=0.6, color='k', align='center')
    plt.title('Silhouette score vs number of clusters')
    

    
    # 3D

    # fig = plt.figure(figsize = (15,15))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[identified_clusters == 0,0],x[identified_clusters == 0,1],x[identified_clusters == 0,2], s = 40 , color = 'orange', label = "cluster 1", alpha=0.6)
    # ax.scatter(x[identified_clusters == 1,0],x[identified_clusters == 1,1],x[identified_clusters == 1,2], s = 40 , color = 'green', label = "cluster 2", alpha=0.6)
    # ax.scatter(x[identified_clusters == 2,0],x[identified_clusters == 2,1],x[identified_clusters == 2,2], s = 40 , color = 'red', label = "cluster 3", alpha=0.6)
    # ax.set_xlabel('max_amount -->')
    # ax.set_ylabel('cash withdrawal -->')
    # ax.set_zlabel('age -->')
    # ax.legend()
    # plt.show()

def clustering_kmeans(df, n_clusters=3):

    scaler = MinMaxScaler()
    scaler.fit(df)
    X_scale = scaler.transform(df)
    df = pd.DataFrame(X_scale, columns=df.columns)

    kmeans = KMeans(n_clusters)
    identified_clusters = kmeans.fit_predict(df)
    # df['cluster'] = identified_clusters.tolist()

    print(f'Inertia: {kmeans.inertia_}')

    # Reduce dimensionality with PCA
    if len(df.columns) > 3:
        pass
    else:
        # Plot results
        if len(df.columns) == 2:
            Scene = dict(xaxis = dict(title = df.columns[0]),yaxis = dict(title  = df.columns[1]))
            trace = go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1], mode='markers',marker=dict(color=kmeans.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df.columns[0]),yaxis = dict(title = df.columns[1]), zaxis= dict(title = df.columns[2]))
            trace = go.Scatter3d(x=df.iloc[:,0], y=df.iloc[:,1], z=df.iloc[:,2], mode='markers',marker=dict(color=kmeans.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="KMeans Clusters", font=dict(size=12,))
        fig.show()




def clustering_dbscan(df, eps=0.9, min_samples=4, n_components=2,):

    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaler.fit(df)
    X_scale = scaler.transform(df)
    df = pd.DataFrame(X_scale, columns=df.columns)

    # Reduce dimensionality with PCA
    if len(df.columns) > 3:
        # Analyse PCA
        pca_analysis(df)
        df_pca = pca(n_components, df)
        dbscan_param_tuning_silhouette(df_pca)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df_pca)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df_pca, labels))

        # Plot results
        labels = dbscan.labels_
        if n_components == 2:
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'))
            trace = go.Scatter(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 6, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'), zaxis= dict(title  = 'PC3'))
            trace = go.Scatter3d(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], z=df_pca.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 6, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="'DBSCAN Clusters Derived from PCA'", font=dict(size=12,))
        fig.show()
    

    else:
        dbscan_param_tuning_silhouette(df)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df, labels))

        # Plot results
        labels = dbscan.labels_
        if len(df.columns) == 2:
            Scene = dict(xaxis = dict(title = df.columns[0]),yaxis = dict(title  = df.columns[1]))
            trace = go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 7, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df.columns[0]),yaxis = dict(title = df.columns[1]), zaxis= dict(title = df.columns[2]))
            trace = go.Scatter3d(x=df.iloc[:,0], y=df.iloc[:,1], z=df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 7, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="'DBSCAN Clusters'", font=dict(size=12,))
        fig.show()


###########
# Clusters
###########
def clustering_economic():

    # Build dataframe
    df =  merge_datasets(db, False, True)
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))
    df = extract_features(df)

    #print(df.columns)
    df = df[['avg_amount_credit', 'avg_amount_withdrawal', 'average_salary']]

    # df = db.df_query('SELECT AVG(balance) AS avg_balance, AVG(amount) AS avg_amount \
    #         FROM trans_train RIGHT JOIN disposition USING(account_id) RIGHT JOIN client USING(client_id) \
    #         GROUP BY account_id')
    # df.fillna(0, inplace=True)
    
    #clustering_dbscan(df, 0.2, 2)
    clustering_kmeans(df)


if __name__ == "__main__":
    #clustering_kmeans()
    #clustering_agglomerative()
    clustering_economic()