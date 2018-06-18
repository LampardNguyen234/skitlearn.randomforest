import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize Classifier
RandomForest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

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
# Block 4: Scale lai du lieu cua tap train va tap test
# Chia du lieu thanh 2 tap train va test
x_trn, x_tst, y_trn, y_tst = train_test_split(x_all, y_all, test_size=0.8, random_state=42)
# Tao class Scaler de chia ti le
scaler = MinMaxScaler()
# Tinh min va max cua data de chia ti le
scaler.fit(x_trn)
# Chia ti le lai cho tap train
x_trn_n=scaler.transform(x_trn)
# Chia ti le lai cho tap test
x_tst_n=scaler.transform(x_tst)
# End block 4
# ---------------------------------


# Begin Random Forest
# ---------------------------------
# Block 5: Train Model, tim do chinh xac va tim cac bien co trong so cao
# Chon kieu Random Forest
clf=RandomForest
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
# End Block 5
# ---------------------------------

# ---------------------------------
# Block 6: Ve bieu do trong so cua cac bien
# Tach ten cac bien co trong so cao
var_names=list(var2imp_sorted['variable'][0:10])
# Tach trong so cua cac bien
var_imp=list(var2imp_sorted['weight'][0:10])
# Xep lai ten bien
y_pos = np.arange(len(var_names),0,-1)
# Tao bieu do
fig = plt.figure(figsize=(12, 6))
# Them bieu do con
plt.subplot(1, 1, 1)
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
# End block 6
# ---------------------------------

# ---------------------------------
# Block 7: Du doan mot so nguoi dung
jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
        'services', 'student', 'technician', 'unemployed']
maritals = ['divorced', 'married', 'single']
educations = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree']
contacts = ['cellular', 'telephone']
months = ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep']
days = ['fri', 'mon', 'thu', 'tue', 'wed']
poutcomes = ['failure', 'nonexistent', 'success']
yes_no = ['no','unknown', 'yes']
while (1):
    print "Prediction:"
    info = [0 for i in range(63)]
    # Nhap tuoi
    c_age = raw_input("Age: ")
    info[0] = int(c_age)

    # Nhap cong viec
    c_job = raw_input("Jobs: ")
    if not c_job in jobs:
        info[21] = 1
    else:
        for i in range(len(jobs)):
            if jobs[i] == c_job:
                info[10 + i] = 1

    # Nhap tinh trang hon nhan
    c_marital = raw_input("Marital: ")
    if not c_marital in maritals:
        info[25] = 1
    else:
        for i in range(len(maritals)):
            if maritals[i] == c_marital:
                info[22 + i] = 1

    # Nhap trinh do hoc van
    c_education = raw_input("Education: ")
    if not c_education in educations:
        info[33] = 1
    else:
        for i in range(len(educations)):
            if educations[i] == c_education:
                info[26 + i] = 1

    # Nhap default ????    
    c_default = raw_input("Default: ")
    if not c_default in yes_no:
        info[35] = 1
    else:
        for i in range(len(yes_no)):
            if yes_no[i] == c_default:
                info[34 + i] = 1

    # Nhap housing            
    c_housing = raw_input("Housing: ")
    if not c_housing in yes_no:
        info[38] = 1
    else:
        for i in range(len(yes_no)):
            if yes_no[i] == c_housing:
                info[37 + i] = 1

    # Nhap loan ???    
    c_loan = raw_input("Loan: ")
    if not c_loan in yes_no:
        info[41] = 1
    else:
        for i in range(len(yes_no)):
            if yes_no[i] == c_loan:
                info[40 + i] = 1

    # Nhap contact            
    c_contact = raw_input("Contact: ")
    if c_contact in contacts:
        for i in range(len(contacts)):
            if contacts[i] == c_contact:
                info[43 + i] = 1

    # Nhap thang
    c_month = raw_input("Month: ")
    if c_month in months:
        for i in range(len(months)):
            if months[i] == c_month:
                info[45 + i] = 1
                
    # Nhap ngay
    c_day_of_week = raw_input("Day of week: ")
    if c_day_of_week in days:
        for i in range(len(days)):
            if days[i] == c_day_of_week:
                info[55 + i] = 1
                
    # Nhap duration ???
    c_duration = raw_input("Duration: ")
    info[1] = int(c_duration)
    
    # Nhap campaign ???
    c_campaign = raw_input("Campaign: ")
    info[2] = int(c_campaign)
    
    # Nhap pdays ???
    c_pdays = raw_input("Pdays: ")
    info[3] = float(c_pdays)
    
    # Nhap previous ???
    c_previous = raw_input("Previous: ")
    info[4] = float(c_previous)
    
    # Nhap poutcome
    c_poutcome = raw_input("POutCome: ")
    if c_poutcome in poutcomes:
        for i in range(len(poutcomes)):
            if poutcomes[i] == c_poutcome:
                info[60 + i] = 1
                
    # Nhap emp.var.rate ???
    c_emp_var_rate = raw_input("emp.var.rate: ")
    info[5] = float(c_emp_var_rate)
    
    # Nhap cons.price.idx ???
    c_cons_price_idx = raw_input("cons.price.idx: ")
    info[6] = float(c_cons_price_idx)
    
    # Nhap cons.conf.idx ???
    c_cons_conf_idx = raw_input("cons.conf.idx: ")
    info[7] = float(c_cons_conf_idx)
    
    # Nhap euribor3m ???
    c_euribor3m = raw_input("Euribor3m: ")
    info[8] = float(c_euribor3m)
    
    # Nhap nr.employed ???
    c_nr_employed = raw_input("nr.employed: ")
    info[9] = float(c_nr_employed)
    
    info_n = scaler.transform([info])
    result = model.predict(info_n)[0]
    if result:
        print "Yes"
    else:
        print "No"
    print "___________"
# End block 7
# ---------------------------------
# End Random Forest
