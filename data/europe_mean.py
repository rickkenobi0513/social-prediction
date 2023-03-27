"""
Created on Friday August 13 16:01:52 2021

@author: vladimir (http://www.pitt.edu/~viz/)

Matrix driven implementation of TM, based on TM intro in 

K. D. Abeyrathna, Ole-Christoffer Granmo, Morten Goodwin
Extending the Tsetlin Machine With Integer-Weighted Clauses 
for Increased Interpretability. IEEE Access, 2021

"""
 

def clause_sum(matrix, clause, threshold):

    rows = len(matrix)
    a = []
    for i in range(rows):
        m = matrix[i]
        t = copy.deepcopy(m)
        t = t.tolist()
        max_index = []
        max_number = []
        for nums in t:
            if nums >= threshold:
                #max_index.append()
                index= t.index(nums)
                max_index.append(index)
        #print(max_index)
        #max_number = []
        #max_index = []
        
        
        #for _ in range(o):
        #    number = max(t)
        #    index = t.index(number)
        #    t[index] = 0
        #    max_number.append(number)
        #    max_index.append(index)
        a.append(sorted(max_index))
        t = []
    #print(a)
    res=list(set([tuple(t) for t in a]))
    #print(res)
    clause_res = []
    for a in res:
        clause_temp = []
        for b in range(len(a)):
            clause_temp.append(clause[a[b]])
        andcha = '&'
        clause_temp = andcha.join(clause_temp) 
        clause_res.append(clause_temp)
    orcha = '|'
    clause_res = orcha.join(clause_res)
    return(clause_res)

def feedbackMatrices(stateMatrix, inlits, output, N, T, s, votes, typeItarget):
#
#producing feedback matrices: 
#               typeItarget = 1 for clauses with positive polarities
#               typeItarget = 0 for clauses with negative polarities

    num_of_clauses = stateMatrix.shape[0]
    num_of_literals = stateMatrix.shape[1]

    Ia = np.zeros((num_of_clauses, num_of_literals),int) #Type Ia feedback decision matrix

    Ib = np.zeros((num_of_clauses, num_of_literals),int) #Type Ib feedback decision matrix 

    II = np.zeros((num_of_clauses, num_of_literals),int) #Type II feedback decision matrix

    pj1 = (T - max(-T, min(T,votes)))/(2*T)
    pj2 = (T + max(-T, min(T,votes)))/(2*T)
    p1a = (s-1)/s
    p1b = 1/s


    for clindex in range(num_of_clauses):
    
        #evaluating each clause with the eligible literals (TA state is >= N/2)
  
        cloutput = evaluateClause(stateMatrix, clindex, inlits, N)
  
        #providing feedback for eligible clauses and literals
  
        for litindex in range(num_of_literals):
            lit = inlits[litindex]
            if output == typeItarget: #potential Type I feedback (typeItarget = 1 for clauses with positive polarities)
                if np.random.rand() < pj1: #selecting a clause for a Type I feedback with probability pj1
                    Ia[clindex, litindex] = typeIaFeedback(p1a,cloutput,lit)
                    Ib[clindex, litindex] = typeIbFeedback(p1b,cloutput,lit)
            else: #output == ~typeItarget,potential Type II feedback (typeItarget = 0 for clauses with positive polarities)
                if np.random.rand() < pj2: #selecting a clause for a Type II feedback with probability pj2
                    II[clindex, litindex] = typeIIFeedback(cloutput,lit)
    return [Ia,Ib,II]

def evaluateClause(stateMatrix, clindex, inlits, N):
#    
#Evaluating a clause with the eligible literals (TA state is >= N/2)
# 
    num_of_literals = stateMatrix.shape[1]
    
    eliglits = np.array([],int)
 
    cl = stateMatrix[clindex]
   
    for litindex in range(num_of_literals):
        if cl[litindex] >= N/2:
            eliglits = np.append(eliglits,inlits[litindex])
   
    cloutput = np.all(eliglits)
   
    return cloutput

def typeIaFeedback(p1a,cloutput,lit):
#    
# Producing Type Ia feedback
#
    if np.random.rand() < p1a and  cloutput == 1 and lit == 1:
        res = 1
    else:
        res = 0
    return res  
 
def typeIbFeedback(p1b,cloutput,lit):
#    
# Producing Type Ib feedback
#
    if np.random.rand() < p1b and (cloutput == 0 or lit == 0):
        res = 1
    else:
        res = 0
    return res  
 
def typeIIFeedback(cloutput,lit):
#    
# Producing Type II feedback
#
    if cloutput == 1 and lit == 0:
        res = 1
    else:
        res = 0
    return res  
 
def sum_up_clause_votes(stateMatrix, N, inlits):
#
#Summing up clause votes
#  

    num_of_clauses = stateMatrix.shape[0]

    num_of_literals = stateMatrix.shape[1]

    votes = 0
  
    for clindex in range(num_of_clauses):
        
        incllits = np.array([], int)

        cl = stateMatrix[clindex]

        for litindex in range(num_of_literals):
            if cl[litindex] >= N/2:
                incllits = np.append(incllits,inlits[litindex])
  
        if  len(incllits) > 0:
            cloutput = np.all(incllits)
            votes = votes + cloutput
  
    return votes
def train_data(rate,data_input, data_output):
    #rate = 0.3
    mpn = 1
    o = len(data_input[0])
    N = 4000  #number of TA states
    threshold = N * rate
    T = 15
    s = 3.9
    clause = ['Bmean','GBmean','NBmean','NGBmean']
    #for i in range(o):
    #    pos_name = 'x'+str(i)
    #    clause.append(pos_name)
    #for i in range(o):    
    #    neg_name = '^x'+str(i)
    #    clause.append(neg_name)
    
   

#state matrices

    Apos = np.ones((mpn, 2*o),int) #state matrix for clauses with positive polarities

    Aneg = np.ones((mpn, 2*o),int) #state matrix for clauses with negative polarities

    num_of_train_runs = 1000
    num_of_test_runs = 50

# learning

    for runs in range(num_of_train_runs):
      
    #binary input vector
        for i in range(len(data_input)):
            inlits_temp = data_input[i]
            inlits = np.append(inlits_temp,1-inlits_temp)
            output = data_output[i]
            posvotes = sum_up_clause_votes(Apos, N, inlits)
            negvotes = sum_up_clause_votes(Aneg, N, inlits)
            votes = posvotes - negvotes

    
    #processing clauses with POSITIVE polarity
            [PIa,PIb,PII] = feedbackMatrices(Apos, inlits, output, N, T, s,votes, 1);
            Apos = Apos+PIa-PIb+PII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
            Apos[Apos>N] = N 
            Apos[Apos<1] = 1
        #print(Apos)
    #processing clauses with NEGATIVE polarity
            [NIa,NIb,NII] = feedbackMatrices(Aneg, inlits, output, N, T, s,votes, 0);
            Aneg = Aneg+NIa-NIb+NII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
            Aneg[Aneg>N] = N
            Aneg[Aneg<1] = 1
        #print(Aneg)




# classification

    error_count = 0

    for runs in range(num_of_test_runs):
    #binary input vector
        for i in range(len(data_input)):
            inlits_temp = data_input[i]

            inlits = np.append(inlits_temp,1-inlits_temp)

            output = data_output[i]

            posvotes = sum_up_clause_votes(Apos, N, inlits)
            negvotes = sum_up_clause_votes(Aneg, N, inlits)
    
            if posvotes - negvotes > 0:
                pred = 1
            else:
                pred = 0

            if pred != output:
                error_count = error_count+1

    pred_accuracy = (1-error_count/num_of_test_runs)

    #print(Aneg)
    #print(Apos)
    
    a = clause_sum(Aneg,clause,threshold)
    b = clause_sum(Apos,clause,threshold)
    #print(a)
    #print(Aneg)
    #print(type(Aneg))
    return(Aneg,a,Apos,b)
#
# Putting it all togehter 
#  

#TM parameters

#main function
#m = 10   #total number of clauses
#mpn = int(m/2) #nmber of positive/negative clauses

import pandas as pd
import pymysql
from symbol import classdef
import numpy as np
import copy
con = pymysql.connect(host="localhost",port = 3306,user="root",db="views")

sql = "SELECT Country, Year FROM views.pulsar_2 GROUP BY Country, Year"
data_sql=pd.read_sql(sql,con)
data_sql.to_csv("test2.csv")
df_test1 = pd.read_csv('test2.csv')
df_test1.drop([len(df_test1)-1],inplace=True)


df_test2 = df_test1.drop(df_test1.columns[[0]], axis=1)

df = pd.read_csv('1.csv')
df1 = pd.read_csv('X1mo_data_incidence.csv')

df = df.rename(columns = {'country_name': 'Country'})
df = df.dropna()
df3 = pd.read_csv('/Users/rick/Desktop/work/data/nation.csv')
df3.rename(columns={'Country Name':'Country', 'Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]':'Oil_rents','Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]':'income_per_capita','Population, total [SP.POP.TOTL]':'population','GDP (current US$) [NY.GDP.MKTP.CD]':'GDP','Time':'Year'}, inplace = True)

df_test2['Country'].replace('Burkina Faso','Burkina_Faso',inplace=True)
df_test2['Country'].replace('South Africa','South_Africa',inplace=True)
df_test2['Country'].replace('Cape Verde','Cape_Verde',inplace=True)
df_test2['Country'].replace('Central African Republic','Central_African_Republic',inplace=True)
df_test2['Country'].replace('Democratic Republic Of The Congo','Democratic_Republic_Of_The_Congo',inplace=True)
df_test2['Country'].replace('Republic of Congo','Republic_of_Congo',inplace=True)
df_test2['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)
#df_test2['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)
#dfmerge = dfmerge.replace('Equatorial Guinea','Equatorial_Guinea')

df['Country'].replace('Burkina Faso','Burkina_Faso',inplace=True)
df['Country'].replace('South Africa','South_Africa',inplace=True)
df['Country'].replace('Cape Verde','Cape_Verde',inplace=True)
df['Country'].replace('Central African Republic','Central_African_Republic',inplace=True)
df['Country'].replace('Democratic Republic Of The Congo','Democratic_Republic_Of_The_Congo',inplace=True)
df['Country'].replace('Republic of Congo','Republic_of_Congo',inplace=True)
df['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)

# dfmer = pd.merge(df,df_test2, on=['Country'])
df1 = df1.drop(df1.columns[[0]], axis=1)

#X1_output = dfX1[['year','country_name','country_num']].copy()
#X1_output = df1.dropna()
X1_output = df1.reset_index()
X1_output = X1_output.drop(X1_output.columns[[0]], axis=1)
X1_output.rename(columns={'country_name':'Country', 'year':'Year'}, inplace = True)
X1_output = X1_output.groupby(['Country','Year','country_num']).agg('sum')

X1_output.reset_index(inplace = True)
X1_output
# #dfmerge1 = pd.merge(X1_output,df_output_name, how = 'cross')
dfmerge = pd.merge(X1_output,df3, on=['Country','Year'], how = "inner")
dfmerge['global_mean'] = None
dfmerge['Bmean'] = None
dfmerge['GBmean'] = None
for i in range(len(dfmerge)):
    if dfmerge['incidence_1'].iloc[i] != 0:
        dfmerge['incidence_1'].iloc[i] = 1
    else:
        dfmerge['incidence_1'].iloc[i] == 0
dfmerge
dfmerge.groupby(['Country']).mean()
hi = dfmerge.groupby(['Country','Year']).sum().groupby(level=0)['incidence_1'].cumsum().copy()
acc = hi.values
acc
dfmerge.insert(1,'prior_war',acc)
# dfmerge['mean'] = dfmerge['acc_civil'].mean()
dfmerge['global_mean'] = dfmerge.groupby(['Year']).prior_war.transform('mean')
dfmerge['regional_mean'] = None
dfmerge_2 = dfmerge[dfmerge['region_europe'] != 0]
dfmerge_2['regional_mean'] = dfmerge_2.groupby(['Year']).prior_war.transform('mean')
dfmerge_1 = dfmerge_2[dfmerge_2['Year'] >= 2005]
dfmerge_1
dfmerge_1['prior_war'] = dfmerge['incidence_1']
hi2 = dfmerge_1.groupby(['Country','Year']).sum().groupby(level=0)['incidence_1'].cumsum().copy()
hi2.reset_index()
acc2 = hi2.values
dfmerge_1.insert(1,'acc_civil_2',acc2)
dfmerge_1['prior_war'] = dfmerge_1['acc_civil_2']
dfmerge_1['region_africa']
region = [col for col in dfmerge_1.columns if 'region' in col]

incidence_1 = dfmerge_1.loc[:,['incidence_1','Year','Country','prior_war','Bmean','global_mean','regional_mean','GBmean']]
incidence_1



for a in range(len(incidence_1)):
    if incidence_1['prior_war'].iloc[a] > incidence_1['regional_mean'].iloc[a]:
        incidence_1['Bmean'].iloc[a] = 1
    else:
        incidence_1['Bmean'].iloc[a] = 0
for a in range(len(incidence_1)):
    if incidence_1['prior_war'].iloc[a] > incidence_1['global_mean'].iloc[a]:
        incidence_1['GBmean'].iloc[a] = 1
    else:
        incidence_1['GBmean'].iloc[a] = 0
incidence_1
incidence_input_1 = incidence_1.loc[incidence_1['incidence_1']==1]
incidence_input_0 = incidence_1.loc[incidence_1['incidence_1']==0]

incidence_input1_np = incidence_input_1.loc[:,['Bmean','GBmean']].values
incidence_input0_np = incidence_input_0.loc[:,['Bmean','GBmean']].values
# #change output data in each rows to numpy
incidence_output1_np = incidence_input_1['incidence_1'].values
incidence_output0_np = incidence_input_0['incidence_1'].values
c = ['Bmean','GBmean','NBmean','NGBmean']
c = np.array(c)
ra = [0.1,0.2,0.4,0.6,0.8]
#for r in range(len(ra)):
#    Eth_result = train_data(ra[r],Eth_input_np,Eth_output_np)
#    if r == 0:
#        print('Ethiopia')
#        #print('Negative')
#        print(np.vstack((c,Eth_result[0])))
#        #print('Positive')
#        #print(Eth_result[2])
#    print('Threshold =' ,ra[r])
#    print(Eth_result[1])
#    #print('Pos:', Eth_result[3])
#print('-----------------------------------------------')
#for r in range(len(ra)):
#    Tan_result = train_data(ra[r],Tan_input_np,Tan_output_np)
#    if r == 0:
#        print('Tanzania')
#        #print('Negative')
#        print(np.vstack((c,Tan_result[0])))
#        #print('Positive')
#        #print(Tan_result[2])
#    print('Threshold =' ,ra[r])
#    print(Tan_result[1])
    #print('Pos:', Tan_result[3])
#print('-----------------------------------------------')
for r in range(len(ra)):
    result = train_data(ra[r],incidence_input1_np,incidence_output1_np)
    if r == 0:
        print('When incidence_1 equal to 1')
        print('Negative')
        print(np.vstack((c,result[0])))
        
        print('Positive')
        print(np.vstack((c,result[2])))
        
    print('Threshold =' ,ra[r])
    print('Neg:',result[1])
    print('Pos:',result[3])


# for r in range(len(ra)):
#     result = train_data(ra[r],incidence_input0_np,incidence_output0_np)
#     if r == 0:
#         print('When incidence_1 equal to 0')
#         print('Negative')
#         print(np.vstack((c,result[0])))
        
#         print('Positive')
#         print(np.vstack((c,result[2])))
        
#     print('Threshold =' ,ra[r])
#     print('Neg:',result[1])
#     print('Pos:',result[3])


