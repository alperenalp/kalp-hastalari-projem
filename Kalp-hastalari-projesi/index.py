#!/usr/bin/env python
# coding: utf-8

# # Kütüphaneler

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings; warnings.simplefilter('ignore')


# # Veri setini yükleme

# In[2]:


df = pd.read_csv("heart.csv")
df.head()


# # Eksik Verilerin Tespiti ve Düzeltilmesi

# In[3]:


#Veri setimizde eksik veri var mı bakma
eksik_veri = df.isnull().sum()
print(eksik_veri)
print("\nEksik veri sayısı= " + str(eksik_veri.sum()))


# # Veri setinin özellikleri

# In[4]:


df.describe()


# # Veri Setini Analiz Etme ve Görselleştirme

# In[5]:


#Datayı görselleştirmeden önce türkçeye çevirme
#kalp_hastaligi = kalp_hastaligi.rename(columns={'age': 'yaş'}) bu şekilde de olur
kalp_hastaligi = df.copy()
kalp_hastaligi.columns = ['yaş', 'cinsiyet', 'göğüs_ağrısı_tipi', 'kan_basıncı', 'kolestrol', 'kandaki_şeker', 'Elektrokardiyografi' , 'Max_Nabız', 'Anjina', 'St_depresyon','Eğim','floroskopi','Kalıtsal_kan_bozukluğu','Hedef']
kalp_hastaligi.head()


# In[6]:


#Datanın histogramını çıkarma
kalp_hastaligi.hist(figsize=(10,10))


# In[7]:


#Cinsiyete göre kolestrol sayıları
ax=sns.barplot(x="cinsiyet", y="kolestrol", data=kalp_hastaligi)
ax.set(xlabel='cinsiyet', ylabel='kolestrol')
plt.show()


# In[8]:


#yaşa bağlı kolestrol grafiği
pd.crosstab(kalp_hastaligi.yaş, kalp_hastaligi.göğüs_ağrısı_tipi).plot(kind="bar",figsize=(15,10))
plt.title('Yaşa bağlı kolestrol grafiği')
plt.xlabel('Yaş')
plt.ylabel('Sıklık')
plt.show()


# In[9]:


#yaşa bağlı hastalık grafiği
pd.crosstab(kalp_hastaligi.yaş,kalp_hastaligi.Hedef).plot(kind="bar",figsize=(20,6))
plt.title('Yaşa bağlı hastalık grafiği')
plt.xlabel('Yaş')
plt.ylabel('Sıklık')
plt.show()


# In[10]:


# Özellikler birbirlerini ne kadar olumlu veya olumsuz etkiliyor inceleme
plt.figure(figsize=(10,10))
sns.heatmap(kalp_hastaligi.corr(),annot=True,fmt='.1f')
plt.show()


# # Veri Seti Ön İşleme

# In[11]:


df.head(20)


# # Train ve Test veri setlerini oluşturma

# In[12]:


# Hedefi kenidim tahmin edeceğim için hedef kolonunu düşürdüm
X = np.array(kalp_hastaligi.drop(['Hedef'], 1))
y = np.array(kalp_hastaligi['Hedef'])


# In[13]:


# Sklearn kütüphanesi ile veri setimi %80 train ve %20 test olarak böldüm
from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[14]:


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# # Lojistik Regresyon ile Veri Setimizi Eğitme

# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Lojistik Regresyon doğruluk oranı: %{acc_logreg}")


# In[16]:


# confusion matrisi ile doğru ve yanlış tahminlerimize bakalım
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[17]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # K-NN (K-En Yakın Komşu) Algoritması ile Veri Setimizi Eğitme

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean") # minkowski  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"K-NN Algoritması doğruluk oranı: %{acc_knn}")


# In[19]:


# confusion matrisi ile doğru ve yanlış tahminlerimize bakalım
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[20]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Support Vector Machine (Destek Vektör Makine) ile Veri Setimizi Eğitme

# In[21]:


from sklearn.svm import SVC
svc = SVC(kernel='rbf') #linear, poly, rbf, sigmoid
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Destek Vektör Makine doğruluk oranı: %{acc_svc}")


# In[22]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[23]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Naive Bayes Algoritması ile Veri Setimizi Eğitme

# In[24]:


#GaussianNB tahmin edilecek değerler reel sayılar olabiliyorsa ama reel sayılar ikilik sayılara indirgenebilir
#MultinomialNB isimlendirme yapılarak sınıflandırılıyorsa A,B,C,D,F gibi elma armut ayva gibi
#BernoulliNB binary değerler ise 0 yada 1 gibi değerler alıyorsa erkek kadın gibi
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
acc_gnb = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Naive Bayes Algoritması doğruluk oranı: %{acc_gnb}")


# In[25]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[26]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Decision Tree Classifier (Karar Ağaçları) ile Veri Setimizi Eğitme

# In[27]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy') #defaut olan gini parametresidir
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
acc_dtc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Karar Ağaçları doğruluk oranı: %{acc_dtc}")


# In[28]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 


# In[29]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Random Forest Classifier (Rastgele Orman) ile Veri Setimizi Eğitme

# In[30]:


# n_estimators parametresi kaç tane karar ağacı oluşturulacağının miktarıdır
# fazlası overfitting yani ezberlemeye yol açar default 10'dur
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20, criterion = 'entropy') 
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
acc_rfc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Rastgele Orman doğruluk oranı: %{acc_rfc}")


# In[31]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 


# In[32]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[33]:


# Random Forest'ın ROC Eğrisine bakalım
import sklearn.metrics as metrics
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
#grafiği çizme
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.show()


# # Artificial Neural Network (Yapay Sinir Ağları) ile Veri Setimizi Eğitme

# In[37]:


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
#model
model = Sequential()
model.add(Dense(8, kernel_initializer='uniform', input_dim = 13))
model.add(Dense(8, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer='uniform', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )
#eğitim
model.fit(X_train, y_train, epochs=50)
#test sonuçları
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # 0.5 eşik değeri ile true yada false yapıyoruz
acc_ann = round(accuracy_score(y_pred, y_test) * 100, 2)
print(f"Yapay Sinir Ağları doğruluk oranı: %{acc_ann}")


# In[35]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 


# In[38]:


# confusion matrisinden elde edilen hesaplamalar
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




