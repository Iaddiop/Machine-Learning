
# coding: utf-8

# ## Analyse de données RH de l’attrition des employés d’IBM

# ### Description de la problématique : 
# 
# Le but de cette analyse est de découvrir les facteurs qui mènent à l'attrition (le taux d'attrition est le pourcentage des employés quittant une entreprise, c'est l'inverse du taux de rétention) des employés et ainsi le prévoir à l'aide des modèles de machine Learning. Les données qui seront traitées sont issues de données fictives créées par IBM data scientists et sont mises à disposition sur le site kaggle.com. 

# ### 1- Data preprocessing (step1)

# In[1]:


# Import des librairies :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Import du dataset:
ibm = pd.read_excel('IBM_HR_Data.xlsx')
ibm.info()


# In[3]:


ibm.head() # vérification de l'import avec les 5 prémières lignes


# Résumé :
# 
# -	Le jeu de données comporte 1470 individus 
# -	32 variables : 8 qualitatives et 24 quantitatives
# -	Nous n’avons pas de valeurs manquantes
# 

# ### 2- Data Mining

# #### A- Analyse univariée

# In[4]:


ibm.describe() #statistique descriptive pour nos 25 variables quantitatives


# Nous allons observer plus en détail la distribution de certaines variables quantitatives : Age, DistanceFromHome, MonthlyIncome ,TotalWorkingYears,YearsAtCompany, YearsInCurrentRole

# In[5]:


fig, ax = plt.subplots(figsize=(10,10), ncols=2, nrows=3) 
sns.distplot(ibm['Age'], ax = ax[0,0])  
sns.distplot(ibm['MonthlyIncome'], ax = ax[0,1]) 
sns.distplot(ibm['YearsAtCompany'], ax = ax[1,1]) 
sns.distplot(ibm['TotalWorkingYears'], ax = ax[1,0])
sns.distplot(ibm['YearsInCurrentRole'], ax = ax[2,0])
sns.distplot(ibm['DistanceFromHome'], ax = ax[2,1])
plt.show()


# En Résumé : 
# - L'âge moyen dans cette entreprise est proche de 37 ans   
# - Le salaire moyen est de 6502 doallars avec une concentration des salaires entre 0 et 5000 dollars par mois. Cet effet est observable sur la représentation graphique de la distribution de la variable MonthlyIncome
# - Les employés font en moyenne 7 ans chez IBM. Ils passent 4 ans en moyenne sur un poste avec le même manager.
# - Les employés parcours en moyenne 9 km pour se rendre au travail, avec une forte disparité. En effet, on constate une forte concentration des distances entre 0 et 5 km.

# #### B- Analyse bivariée

# In[6]:


# Explorons la distribution des salaires mensuels entre homme et femme
sns.distplot(ibm.MonthlyIncome[ibm.Gender == 'Male'], bins = np.linspace(0,20000,60))
sns.distplot(ibm.MonthlyIncome[ibm.Gender == 'Female'], bins = np.linspace(0,20000,60))
plt.legend(['Male','Female'])


# La disparité entre les salaires des hommes et des femmes ne présentent pas une très grande disparité dans notre jeu de données.
# Cependant, pour les revenus mensuels entre 0 et 5000 dollars et les hauts revenus entre 15000 et 20000 dollars par mois, les hommes sont au-dessus des femmes. 
# Par ailleurs, nous constatons que les salaires des femmes est supérieurs à ceux des hommes entre 10000 et 15000 dollars. Nous pouvons confirmer ces disparités relatives en calculant le salaire médian, mais aussi le salaire moyen par genre et en les comparants.

# In[7]:


med_homme = np.median(ibm.MonthlyIncome[ibm.Gender == 'Male'])
med_femme = np.median(ibm.MonthlyIncome[ibm.Gender == 'Female'])
moy_homme = np.mean(ibm.MonthlyIncome[ibm.Gender == 'Male'])
moy_femme = np.mean(ibm.MonthlyIncome[ibm.Gender == 'Female'])


# In[8]:


med_femme


# In[9]:


med_homme


# In[10]:


moy_femme


# In[11]:


moy_homme


# La comparaison des salaires moyens et médians entre homme et femme confirme bien notre première analyse.

# In[12]:


# Comparaison des variables 2 à 2 par des box plots
fig, ax = plt.subplots(figsize=(12,12), ncols=2, nrows=2)

sns.boxplot(ibm["Attrition"],ibm["DistanceFromHome"], ax = ax[0,0])
ax[0,0].set( title = "L'attrition par rapport à la distance" )

sns.boxplot(ibm["Attrition"],ibm["JobSatisfaction"], ax = ax[0,1])
ax[0,1].set( title = "L'attrition par rapport à la satisfaction au travail" )

sns.boxplot(ibm["MaritalStatus"],ibm["MonthlyIncome"], ax = ax[1,0])
ax[1,0].set( title = "La situation familiale par rapport au salaire mensuel" )

sns.boxplot(ibm["Education"],ibm["MonthlyIncome"], ax = ax[1,1])
ax[1,1].set( title = "Le niveau d'éducation par rapport au salaire mensuel " )
plt.show()


# Interprétation :
# -	L'attrition par rapport à la distance parcourue par les employer pour se rendre au travail (variables « Attrition » par rapport à «DistanceFromHome», position graphique : [1,1]): Plus un salarié habite loin plus il y a un risque qu’il quitte l’entreprise.
# 
# -	L'attrition par rapport à la satisfaction au travail (variable « Attrition » par rapport à «JobSatisfaction», position graphique : [1,2]): Moins un salarié est satisfait, plus le risque de quitter l’entreprise est élevé. Il y a lien fort entre ses deux variables.
# 
# 
# -	La situation familiale par rapport au salaire mensuel (variable « MaritalStatus » par rapport à «MonthlyIncome», position graphique : [2,1]) : Les célibataires gagnent moins que les mariés et les divorcés.
# 
# 
# -	Le niveau d'éducation par rapport au salaire mensuel (variable «Education » par rapport à «MonthlyIncome», position graphique : [2,2]) : Plus un salarié est éduqué, plus il gagne mieux.
# 

# #### C- Analyse multivariée

# In[13]:


#Matrice des corrélations
matrice_corr=ibm.corr()


# In[15]:


# Représentation graphique de la matrice des corrélations
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(ibm.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.savefig('fig1')


# Nous remarquons que plusieurs variables n’ont pas de lien entre elles. 
# 
# Cependant, des forts liens existent entre : TotalWorkingYears et MonthlyIncome, entre TotalWorkingYears et Age, entre TotalWorkingYears et Joblevel, entre PerformanceRting et PercentSatisfactionHike, entre YearsInCurrentRole et YearATCompagny et entre YearsWithCurrManager et YearsAtCompagny.

# ### 3 - Data preprocessing (step2) 

# In[16]:


# séparation des variables quantitatives et qualitatives et de la variable 'Attrition' : 
quant1_ibm = ibm.iloc[:, [1,4,6,11]]
quant2_ibm = ibm.iloc[:, 13:32]
quant_ibm = pd.concat([quant1_ibm, quant2_ibm], axis=1) #dataframe : quanti


# In[17]:


quali1_ibm = ibm.iloc[:, [2,3,5,12]]
quali2_ibm = ibm.iloc[:, [8,9,10]]
quali_ibm = pd.concat([quali1_ibm, quali2_ibm], axis=1) #dataframe : quali


# In[18]:


Variable_attrition = ibm.iloc[:, 7] 


# In[19]:


# Encoder les variables qualitatives hors Attrition :
quali_ibm = pd.get_dummies(quali_ibm)


# In[20]:


# Encoder la variable 'Attrition' :
from sklearn.preprocessing import LabelBinarizer
labelencoder_attrition = LabelBinarizer()
Variable_attrition = labelencoder_attrition.fit_transform(Variable_attrition)


# In[21]:


x = pd.concat([quant_ibm, quali_ibm], axis = 1)
y = Variable_attrition


# In[23]:


x.head()


# Notre jeu de données final représente 52 variables quantitatives hors la variable (Y) cible qui est l'attrition.

# ### 4- Machine learning

# Après avoir effectué une analyse exploratoire des données et un recodage des variables qualitatives, nous sommes maintenant prêts à construire nos modèles de machine learning.
# Ici nous sommes en face d'un problème de classification.
# Nous utiliserons quelques différents modèles de machine learning : 
# - Gradient Boosting Classifier
# - Régression logistique
# - SVM
# - Kernel SVM

# In[24]:


#Partition de l'échantillon en deux parties, entrainement et test.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[25]:


# Feature scaling : mettre les variable sur la même echelle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[26]:


x_train


# In[27]:


x_test.shape


# #### A- Regression logistique :

# In[28]:


from sklearn.linear_model import LogisticRegression
logist = LogisticRegression(random_state = 0)
logist.fit(x_train, y_train)


# In[30]:


y_pred1 = logist.predict(x_test)


# In[32]:


from sklearn.metrics import accuracy_score, confusion_matrix
logist_cm = confusion_matrix(y_test, y_pred1) # matrice de classement du modèle
logist_accuracy = accuracy_score(y_test, y_pred1) # score de de classement (positif/négative) correcte/total
logist_er = 1 - logist_accuracy


# In[33]:


logist_cm


# In[34]:


logist_accuracy


# In[35]:


logist_er


# In[61]:


logist.score(x_test,y_test)


# ### B - Kernel SVM

# In[42]:


from sklearn.svm import SVC
ksvm = SVC(kernel = 'sigmoid', random_state = 0)
ksvm.fit(x_train, y_train)


# In[43]:


y_pred3 = ksvm.predict(x_test)
ksvm_cm = confusion_matrix(y_test, y_pred2)
ksvm_accuracy = accuracy_score(y_test, y_pred2)
ksvm_er = 1 - ksvm_accuracy # claculons le taux d'érreur du modèle l'inverse du score de l'accuracy


# In[44]:


ksvm_cm


# In[45]:


ksvm_accuracy


# In[46]:


ksvm_er


# In[62]:


ksvm.score(x_test,y_test)


# ### D - Gradien Boosting classifier

# In[47]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)


# In[58]:


y_pred4 = gb.predict(x_test)
gb_cm = confusion_matrix(y_test, y_pred3)
gb_accuracy = accuracy_score(y_test, y_pred3)


# In[49]:


gb_cm


# In[50]:


gb_accuracy


# In[51]:


gb_er


# In[59]:


gb.score(x_test,y_test)


# on remarque que les modèles logistique et GB ont les mêmes performances.

# Nous allons à présent regarder les variables qui sont les plus d'importantes dans la mise en place du model gb :

# In[57]:


# FR
f_importance = gb.feature_importances_
indices = np.argsort(f_importance)

plt.figure(figsize=(10, 15))
plt.title("Les variables les plus importantes dans l'analyse du modèle logique")
plt.barh(range(len(indices)), f_importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), x.columns)
plt.xlabel('Importance relative')


# ### Conclusion
# Nous avons construit à partir des modèles de Machine Learning une méthode de prédiction de l'attrition des employés, allant de l'analyse statistique exploratoire au code optimal des variables qualitatives, en passant par trois modèles de classification regression logistique, Gradient Boosting et kernet SVM.
# 
# A la suite de l’entrainement de nos modèles d’apprentissage, nous avons retenu deux modèles, la regression logistique, Grandient Boosting pour un pourcentage d’exactitude prévisionnel de 88 et de 87% respectivement.Par ailleurs, nous retenons un taux accuracy (poucentage de bien classé) de 88,4 et de 88,6% pour la regression logistique, le Grandient Boosting. 
# 
# Les recommandations pour éviter l’attrition des employés:
# - Lancer un audit des ressources humaines pour étudier les facteurs qui touchent plus à l'attriton à savoir l'affectation des employer dans chaque département.
# - Proposer des mouvements internes et des changement de poste aux employés succeptible de quitter l'entreprise 
