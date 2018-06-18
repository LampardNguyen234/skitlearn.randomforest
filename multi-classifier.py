import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Name of classifiers
clsr_names=["Nearest Neighbors", "Linear SVM", "RBF SVM",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes"]

# Initialize Classifiers
classifiers = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]
# ---------------------------------    
# Block 1: Doc va xu ly so bo thong tin
# Doc thong tin tu file csv
df01=pd.read_csv('bank-additional/bank-additional-full.csv', sep=';',header=0)
# Xoa cac hang chua thong tin trong
df02=df01.dropna(axis=1, how='all')
# Xoa cac cot chua thong tin trong
df=df02.dropna(axis=0, how='any')
# Kieu du lieu cua df
cols=df.dtypes
# Lay cac cot cua df
colnms=df.columns
# End Block 1
# ---------------------------------

# ---------------------------------
# Block 2: Dem va lay ra cac cot co kieu 'object'
# Bien dem so cot kieu object
i=0
# List luu lai cac cot kieu object
cat_cols=[]
# dem so cot thoa man va luu vao cat_cols
for eachcol in cols:
    if eachcol.name=="object":
        cat_cols.append(colnms[i])
    i+=1
# End Block 2
# ---------------------------------

# ---------------------------------
# Block 3: Tach ra thong tin khach hang va ket qua
# Lay model moi cho du lieu
df1=pd.get_dummies(df,columns=cat_cols)
# So record
n=len(df1.index)
# So cot
m=len(df1.columns)
# Thong tin cua khach hang
x_all=df1.iloc[:,0:(m-2)]
# Khach hang co cho vay hay khong
y_all=df1['y_yes']
# End Block 3
# ---------------------------------


# ---------------------------------
# Block 4: 
x_trn, x_tst, y_trn, y_tst = train_test_split(x_all, y_all, test_size=0.8, random_state=42)
scaler = MinMaxScaler()
scaler.fit(x_trn)
x_trn_n=scaler.transform(x_trn)
x_tst_n=scaler.transform(x_tst)
# End block 4
# ---------------------------------

# Linear SVM
# ---------------------------------
clf = classifiers[1]
model=clf.fit(x_trn_n,y_trn)
y_pred=model.predict(x_tst_n)
acc1=float((y_pred==y_tst).sum())/float(len(y_tst))
print("Linear SVM accuracy: {0:.3f}%".format(acc1))
weight=model.coef_[0]
var2wgt=pd.DataFrame(list(zip(list(df1),weight)),columns=['variable','weight'])
var2wgt_sorted=var2wgt.reindex(var2wgt.weight.abs().sort_values(ascending=False).index)
print("Top 10 weighted variables:")
print(var2wgt_sorted[0:10])
# ---------------------------------

# ---------------------------------
var_names=list(var2wgt_sorted['variable'][0:10])
var_imp=list(var2wgt_sorted['weight'][0:10].abs())
y_pos = np.arange(len(var_names),0,-1)
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(y_pos, var_imp, align='center', alpha=0.5)
plt.yticks(y_pos, var_names)
plt.xlabel('Weight')
plt.title('Linear SVM')
plt.ylim(0,11)
# ---------------------------------
# End Linear SVM


# Random Forest

# ---------------------------------
# Block 4: Train Model, tim do chinh xac va tim cac bien co trong so cao
# Chon kieu Random Forest
clf=classifiers[4]
# Xay dung Forest Tree tu Du lieu train: x_trn_n (data input) va y_trn_n (target output)
model=clf.fit(x_trn_n,y_trn)
# Tien doan gia tri cua mau du lieu test
y_pred=model.predict(x_tst_n)
# Tinh phan tram so record tien doan dung
acc2=float((y_pred==y_tst).sum())/float(len(y_tst))
# In ra do chinh xac cua Model
print("Random forest accuracy: {0:.3f}%".format(acc2))
# Lay ra trong so cua cac bien
imp=model.feature_importances_
# Gan trong so cua cac bien voi ten bien (ten cac cot)
var2imp=dict(zip(list(df1),imp))
# Them tieu de cho bang va sap xep bang
var2imp_sorted=pd.DataFrame(columns=['variable','weight'])
# Sort du lieu trong bang theo tri tuyet doi cua ham weight
for key in sorted(var2imp, key=lambda k:abs(var2imp[k]),reverse=True):
    temp=pd.DataFrame([[key,var2imp[key]]],columns=['variable','weight'])
    var2imp_sorted=var2imp_sorted.append(temp)
# Lay ra 10 bien co tri tuyet doi cua ham weight lon nhat
print("Top 10 important variables:")
print(var2imp_sorted[0:10])
# End Block 4
# ---------------------------------

# ---------------------------------
# Block 5:
# Tach ten cac bien co trong so cao
var_names=list(var2imp_sorted['variable'][0:10])
# Tach trong so cua cac bien
var_imp=list(var2imp_sorted['weight'][0:10])
# Xep lai ten bien
y_pos = np.arange(len(var_names),0,-1)
# Them bieu do con
plt.subplot(1, 2, 2)
# Set cac thuoc tinh cho bieu do
plt.barh(y_pos, var_imp, align='center', alpha=0.5)
# Gan nhan cac bien vao bieu do
plt.yticks(y_pos, var_names)
# Gan nhan cho trong so
plt.xlabel('Weight')
# Dat ten cho bieu do
plt.title('Random Forest')
# Chinh sua bieu do
plt.ylim(0,11)
# Chinh sua bieu do
plt.tight_layout()
# Save bieu do
fig.savefig('plot.png',dpi=400)
# End block 5
# ---------------------------------

# End Random Forest

print("Comparing different models:")
for name, clf in zip(clsr_names, classifiers):
    model=clf.fit(x_trn_n,y_trn)
    y_pred=model.predict(x_tst_n)
    print(name+" Accuracy: {0:.3f}%".format(float((y_pred==y_tst).sum())/float(len(y_tst))))
