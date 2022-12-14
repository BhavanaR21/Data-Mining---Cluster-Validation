#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle_compat
pickle_compat.patch()
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import math
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


def getMealTimeData(InsulinData):
    time =[]
    InsulinVal =[]
    InsulinLev =[]
    Time1=[]
    Time2 =[]
    MealTime = []
    Difference =[]
    ColValue= InsulinData['BWZ Carb Input (grams)']
    MaxValue= ColValue.max()
    MinValue = ColValue.min()
    CalcValues = math.ceil(MaxValue-MinValue/60)
     
    for i in InsulinData['datetime']:
        time.append(i)
    for i in InsulinData['BWZ Carb Input (grams)']:
        InsulinVal.append(i)
    for i,j in enumerate(time):
        if(i<len(time)-1):
            Difference.append((time[i+1]-time[i]).total_seconds()/3600)
    Time1 = time[0:-1]
    Time2 = time[1:]
    CalcValues=[]
    for i in InsulinVal[0:-1]:
        CalcValues.append(0 if (i>=MinValue and i<=MinValue+20)
                          else 1 if (i>=MinValue+21 and i<=MinValue+40)
                          else 2 if(i>=MinValue+41 and i<=MinValue+60) 
                          else 3 if(i>=MinValue+61 and i<=MinValue+80)
                          else 4 if(i>=MinValue+81 and i<=MinValue+100) 
                          else 5 if(i>=MinValue+101 and i<=MinValue+120)
                          else 6)
    ListValues = list(zip(Time1, Time2, Difference,CalcValues))
    for j in ListValues:
        if j[2]>2.5:
            MealTime.append(j[0])
            InsulinLev.append(j[3])
        else:
            continue
    return MealTime,InsulinLev


# In[4]:



def getMealData(mealTimes,startTime,endTime,insulinLevels,new_glucose_data):
    newMealDataRows = []
    for j,newTime in enumerate(mealTimes):
        meal_index_start= new_glucose_data[new_glucose_data['datetime'].between(newTime+ pd.DateOffset(hours=startTime),newTime + pd.DateOffset(hours=endTime))]
        
        if meal_index_start.shape[0]<8:
            del insulinLevels[j]
            continue
        glucoseValues = meal_index_start['Sensor Glucose (mg/dL)'].to_numpy()
        mean = meal_index_start['Sensor Glucose (mg/dL)'].mean()
        missing_values_count = 30 - len(glucoseValues)
        if missing_values_count > 0:
            for i in range(missing_values_count):
                glucoseValues = np.append(glucoseValues, mean)
        newMealDataRows.append(glucoseValues[0:30])
    return pd.DataFrame(data=newMealDataRows),insulinLevels



# In[5]:


def CalcData(insulin_data,glucose_data):
    mealData = pd.DataFrame()
    glucose_data['Sensor Glucose (mg/dL)'] = glucose_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    insulin_data= insulin_data[::-1]
    glucose_data= glucose_data[::-1]
    insulin_data['datetime']= insulin_data['Date']+" "+insulin_data['Time']
    insulin_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    glucose_data['datetime']= glucose_data['Date']+" "+glucose_data['Time']
    glucose_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    
    InsulinDataNew = insulin_data[['datetime','BWZ Carb Input (grams)']]
    GlucoseDataNew = glucose_data[['datetime','Sensor Glucose (mg/dL)']]

    InsulinNew1 = InsulinDataNew[(InsulinDataNew['BWZ Carb Input (grams)']>0) ]
    MealTime,InsulinLev = getMealTimeData(InsulinNew1)
    MealData,InsulinLev_New = getMealData(MealTime,-0.5,2,InsulinLev,GlucoseDataNew)

    return MealData,InsulinLev_New


# In[6]:


def Mean_Abs_val(param):
    MeanVal = 0
    for i in range(0, len(param) - 1):
        MeanVal = MeanVal + np.abs(param[(i + 1)] - param[i])
    return MeanVal / len(param)


# In[7]:


def Entropy(param):
    Length = len(param)
    entropy = 0
    if Length <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / Length
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy   


# In[8]:


def RMS(param):
    RMS = 0
    for i in range(0, len(param) - 1):
        
        RMS = RMS + np.square(param[i])
    return np.sqrt(RMS / len(param))


# In[9]:


def FF(param):
    FF = fft(param)
    Length = len(param)
    i = 2/300
    amplitude = []
    frequency = np.linspace(0, Length * i, Length)
    for amp in FF:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    MaxAmp = sortedAmplitude[(-2)]
    MaxFreq = frequency.tolist()[amplitude.index(MaxAmp)]
    return [MaxAmp, MaxFreq]


# In[10]:


def ZeroCrossing(row, xAxis):
    slopes = [
     0]
    ZeroCross = list()
    ZeroCrossingRate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            ZeroCross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    ZeroCrossingRate = np.sum([np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(ZeroCross) > 0:
        return [max(ZeroCross)[0], ZeroCrossingRate]
    else:
        return [
         0, 0]


# In[11]:


def Glucose(MealNomealdata):
    Glucose=pd.DataFrame()
    for i in range(0, MealNomealdata.shape[0]):
        param = MealNomealdata.iloc[i, :].tolist()
        Glucose = Glucose.append({ 
         'Minimum Value':min(param), 
         'Maximum Value':max(param),
         'Mean of Absolute Values1':Mean_Abs_val(param[:13]), 
         'Mean of Absolute Values2':Mean_Abs_val(param[13:]), 
         'Max_Zero_Crossing':ZeroCrossing(param, MealNomealdata.shape[1])[0], 
         'Zero_Crossing_Rate':ZeroCrossing(param, MealNomealdata.shape[1])[1], 
         'Root Mean Square':RMS(param),
         'Entropy':RMS(param), 
         'Max FFT Amplitude1':FF(param[:13])[0], 
         'Max FFT Frequency1':FF(param[:13])[1], 
         'Max FFT Amplitude2':FF(param[13:])[0], 
         'Max FFT Frequency2':FF(param[13:])[1]},
          ignore_index=True)
    return Glucose


# In[12]:


def Features(MealData):
    Features = Glucose(MealData.iloc[:,:-1])
    
    
    stdScaler = StandardScaler()
    StandardMeal = stdScaler.fit_transform(Features)
    
    pca = PCA(n_components=12)
    pca.fit(StandardMeal)
    
    with open('pcs_glucose_data.pkl', 'wb') as (file):
        pickle.dump(pca, file)
        
    meal_pca = pd.DataFrame(pca.fit_transform(StandardMeal))
    return meal_pca


# In[13]:


def Calc_Entropy(CalcValues):
    EntropyMealValue= []
    for InsulinValue in CalcValues:
    	InsulinValue = np.array(InsulinValue)
    	InsulinValue = InsulinValue / float(InsulinValue.sum())
    	CalcValueEntropy = (InsulinValue * [ np.log2(glucose) if glucose!=0 else 0 for glucose in InsulinValue]).sum()
    	EntropyMealValue += [CalcValueEntropy]
   
    return EntropyMealValue


# In[14]:


def Purity(CalcValues):
    MealPurity = []
    for InsulinValue in CalcValues:
    	InsulinValue = np.array(InsulinValue)
    	InsulinValue = InsulinValue / float(InsulinValue.sum())
    	CalcPurity = InsulinValue.max()
    	MealPurity += [CalcPurity]
    return MealPurity


# In[15]:



def CalcDBSCAN(dbscan,test,meal_pca2):
     for i in test.index:
         dbscan=0
         for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
             test_row=list(test.iloc[0,:])
             meal_row=list(row[:-1])
             for j in range(0,12):
                 dbscan+=((test_row[j]-meal_row[j])**2)
     return dbscan


# In[16]:



def CalcDBSCAN(dbscan,test,meal_pca2):
    for i in test.index:
        dbscan=0
        for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
            test_row=list(test.iloc[0,:])
            meal_row=list(row[:-1])
            for j in range(0,12):
                dbscan+=((test_row[j]-meal_row[j])**2)
    return dbscan


# In[17]:


def CalcClusterMatrix(groundTruth,Clustered,k):
    Matrix= np.zeros((k, k))
    for i,j in enumerate(groundTruth):
         val1 = j
         val2 = Clustered[i]
         Matrix[val1,val2]+=1
    return Matrix


# In[18]:


if __name__=='__main__':
       
    insulin_data=pd.read_csv("InsulinData.csv",low_memory=False)
    glucose_data=pd.read_csv("CGMData.csv",low_memory=False)
    patient_data,InsulinLev = CalcData(insulin_data,glucose_data)
    meal_pca = Features(patient_data)


# In[19]:


kmeans = KMeans(n_clusters=7,max_iter=7000)
kmeans.fit_predict(meal_pca)
pLabels=list(kmeans.labels_)
df = pd.DataFrame()
df['bins']=InsulinLev
df['kmeans_clusters']=pLabels 

Matrix = CalcClusterMatrix(df['bins'],df['kmeans_clusters'],7)
MatrixEntropy = Calc_Entropy(Matrix)
MatrixPurity = Purity(Matrix)
Count = np.array([InsulinValue.sum() for InsulinValue in Matrix])
CountVal = Count / float(Count.sum())


# In[20]:


KMeanSSE = kmeans.inertia_
KMeansPurity =  (MatrixPurity*CountVal).sum()
KMeansEntropy = -(MatrixEntropy*CountVal).sum()


# In[21]:


DBSCANData=pd.DataFrame()
db = DBSCAN(eps=0.127,min_samples=7)
clusters = db.fit_predict(meal_pca)
DBSCANData=pd.DataFrame({'pc1':list(meal_pca.iloc[:,0]),'pc2':list(meal_pca.iloc[:,1]),'clusters':list(clusters)})
OutliersData=DBSCANData[DBSCANData['clusters']==-1].iloc[:,0:2]


# In[22]:


initial_value=0
bins = 7
i = max(DBSCANData['clusters'])
while i<bins-1:
        MaxLabel=stats.mode(DBSCANData['clusters']).mode[0] 
        ClusterData=DBSCANData[DBSCANData['clusters']==stats.mode(DBSCANData['clusters']).mode[0]] #mode(dbscan_df['clusters'])]
        bi_kmeans= KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(ClusterData)
        bi_pLabels=list(bi_kmeans.labels_)
        ClusterData['bi_pcluster']=bi_pLabels
        ClusterData=ClusterData.replace(to_replace =0,  value =MaxLabel) 
        ClusterData=ClusterData.replace(to_replace =1,  value =max(DBSCANData['clusters'])+1) 
       
        for x,y in zip(ClusterData['pc1'],ClusterData['pc2']):
            NewDataLabel=ClusterData.loc[(ClusterData['pc1'] == x) & (ClusterData['pc2'] == y)]
            DBSCANData.loc[(DBSCANData['pc1'] == x) & (DBSCANData['pc2'] == y),'clusters']=NewDataLabel['bi_pcluster']
        df['clusters']=DBSCANData['clusters']
        i+=1  


# In[23]:


MatrixDBSCAN = CalcClusterMatrix(df['bins'],DBSCANData['clusters'],7)
    
ClusterEntropy = Calc_Entropy(MatrixDBSCAN)
ClusterPurity = Purity(MatrixDBSCAN)
Count = np.array([InsulinValue.sum() for InsulinValue in MatrixDBSCAN])
CountVal = Count / float(Count.sum())


# In[24]:


meal_pca2= meal_pca. join(DBSCANData['clusters'])


# In[25]:


Centroids = meal_pca2.groupby(DBSCANData['clusters']).mean()


# In[26]:


dbscan = CalcDBSCAN(initial_value,Centroids.iloc[:, : 12],meal_pca2)
DBSCANPurity =  (ClusterPurity*CountVal).sum()        
DBSCANEntropy = -(ClusterEntropy*CountVal).sum()


# In[27]:


FinalData = pd.DataFrame([[KMeanSSE,dbscan,KMeansEntropy,DBSCANEntropy,KMeansPurity,DBSCANPurity]],columns=['K-Means SSE','DBSCAN SSE','K-Means entropy','DBSCAN entropy','K-Means purity','DBSCAN purity'])
FinalData=FinalData.fillna(0)
FinalData.to_csv('Results.csv',index=False,header=None)


# In[28]:


FinalData


# In[ ]:




