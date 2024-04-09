#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#PRAC1:wordcloud#


# In[2]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import wikipedia as wp 

result = wp.page('Computer Science') 
final_result = result.content 
print(final_result) 

def plot_wordcloud(wc): 
    plt.axis("off") 
    plt.figure(figsize=(10,10)) 
    plt.imshow(wc) 
    plt.show() 

wc=WordCloud(width=500, height=500, background_color="blue", random_state=10,stopwords=STOPWORDS).generate(final_result) 
wc.to_file("cs.png") 
plot_wordcloud(wc)


# In[ ]:


#prac2:web scrapping#


# In[ ]:


#html scrapping#


# In[3]:


import pandas as pd 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 
url = "https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area" 
page = urlopen(url) 
html_page = page.read().decode("utf-8") 
soup=BeautifulSoup(html_page,"html.parser") 
table=soup.find("table") 
print(table)

SrNo=[] 
Country=[] 
Area=[] 
rows=table.find("tbody").find_all("tr") 
for row in rows: 
    cells = row.find_all("td") 
    if(cells): 
        SrNo.append(cells[0].get_text().strip("\n")) 
        Country.append(cells[1].get_text().strip("\xa0").strip("\n").strip("\[2]*")) 
        Area.append(cells[3].get_text().strip("\n").replace(",","")) 
countries_df=pd.DataFrame() 
countries_df["ID"]=SrNo 
countries_df["Country Name"]=Country 
countries_df["Area"] = Area 
print(countries_df.head(10))


# In[ ]:


#json scraping#


# In[4]:


import pandas as pd 
import urllib.request
import json 

url = "https://jsonplaceholder.typicode.com/users" 
response = urllib.request.urlopen(url) 
data = json.loads(response.read()) 

id=[] 
username=[] 
email=[] 

for item in data: 
    if "id" in item.keys(): 
        id.append(item["id"]) 
    else: 
        id.append("NA") 
    if "username" in item.keys(): 
        username.append(item["username"]) 
    else: 
        username.append("NA") 
    if "email" in item.keys(): 
        email.append(item["email"]) 
    else: 
        email.append("NA") 

user_df = pd.DataFrame() 
user_df["User ID"]=id 
user_df["User Name"]=username 
user_df["Email Address"] = email 
print(user_df.head(10))


# In[ ]:


#prac3:Exploratory Data Analysis of mtcars.csv Dataset in R ( Use functions of dplyr like select, filter, mutate , rename, arrange, group by, summarize and data visualizations)#


# In[ ]:


cars_df=read.csv("mtcars.csv")#read
View(cars_df)
str(cars_df)
dim(cars_df)
names(cars_df)
row.names(cars_df)
row.names(cars_df)=cars_df$model
cars_df=cars_df[,-1]
View(cars_df)
library(dplyr)
#Select fuction - for extracting specific columns
#df1=select(cars_df,mpg:hp)
df1=cars_df %>% select(mpg:hp) #pipe of dplyr it will take out content of one column to the output of other column
View(df1)
df1 = cars_df %>% select(c(mpg,disp,wt,gear))
View(df1)
#Filter function - for extracting specific rows or observation
#extract record where gears=4 and columns to be displayed are mpg, disp, wt and gears.
df1 = cars_df %>% filter(gear==4) %>% select(c(mpg,disp,wt,gear))
View(df1)
# extract record where cyl=4 or mpg>20 and columns are required are mpg,cl
df1 = cars_df %>% filter(cyl==4 | mpg > 20) %>% select(c(mpg,cyl))
View(df1)
#extract records where mpg<20 and carb = 3 and coumns needed are mpg and carb
df1 = cars_df %>% filter(mpg < 20 & carb == 3) %>% select(c(mpg,carb))
view(df1)
# Arrange function -Sort as per specific columns
df1 =cars_df %>% arrange(cyl,desc(mpg))
View(df1)
#Rename function - change names of one or more column
df1 = cars_df %>% rename(MilesPerGallon=mpg,Cylinders=cyl,Displacement=disp)
View(df1)
#Mutate function - creating new columns on the basis of existing column
df1 = cars_df %>% mutate(Power=hp*wt)
View(df1)
#Group_by and summaries - segregating data as per categorical variable and summarizing
df1$gear = as.factor(df1$gear)
str(df1)
summary_df = df1%>% group_by(df1$gear) %>% summarise(no=n(), mean_mpg=mean(mpg), mean_wt=mean(wt))
summary_df
summary_df = df1%>% group_by(df1$Cylinders) %>% summarise(no=n(), mean_mpg=mean(mpg), mean_wt=mean(wt))
summary_df
#Data Visualization
#histogram - for single column frequency
hist(df1$mpg, main="Histogeam of MilePergallon(mtcars)",col="lightgreen",xlab="Miles Per Gallon")
#box plot - diagrammatic representation of summary
summary(df1$mpg)
boxplot(df1$mpg)
#bar plot - categorical variable representation'
table(df1$gear)
barplot(table(df1$gear))
#scatter plot - plot() - plots relationship between two variable
plot(df1$mpg~df1$disp)
plot(df1$mpg~df1$cyl)
plot(df1$mpg~df1$wt)


# In[ ]:


#prac4:eda titanic dataset#


# In[ ]:


import pandas as pd
titanic = pd.read_csv("train.csv")
titanic.head()
titanic.info()
titanic.describe()
titanic.isnull().sum()
titanic_cleaned = titanic.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
titanic_cleaned.info()
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.catplot(x="Sex",hue="Survived",kind="count",data=titanic_cleaned)
titanic_cleaned.groupby(['Sex','Survived'])['Survived'].count())
group1 = titanic_cleaned.groupby(['Sex','Survived'])
sns.heatmap(gender_survived,annot=True,fmt="d")
sns.heatmap(gender_survived,annot=True,fmt="d")
sns.violinplot(x="Sex",y="Age",hue="Survived",data=titanic_cleaned,split=True)
print("Oldest Person on Board:",titanic_cleaned['Age'].max())
print("Youngest Person on Board:",titanic_cleaned['Age'].min())
print("Average age of Person on Board:",titanic_cleaned['Age'].mean()) 
titanic_cleaned.isnull().sum()
def impute(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
titanic_cleaned['Age']=titanic_cleaned[['Age','Pclass']].apply(impute,axis=1)
titanic_cleaned.isnull().sum()
titanic_cleaned.corr(method='pearson')
sns.heatmap(titanic_cleaned.corr(method="pearson"),annot=True,vmax=1)
import numpy as np 
from sklearn import datasets 
x,y,coef=datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10,coef=True, random_state = 0) 
x=np.interp(x,(x.min(),x.max()),(0,20)) 
print(len(x)) 
print(x)
y=np.interp(y,(y.min(),y.max()),(20000,150000)) 
print(len(y)) 
print(y)


# In[ ]:


#prac5:Exploratory data analysis in Python using Titanic Dataset #


# In[ ]:


#Write a python program to build a regression model that could predict the salary of an employee  from the given experience and visualize univariate linear regression on it.  #


# In[ ]:


import numpy as np 
from sklearn import datasets 
x,y,coef=datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10,coef=True, random_state = 0) 
x=np.interp(x,(x.min(),x.max()),(0,20)) 
print(len(x)) 
print(x) 
y=np.interp(y,(y.min(),y.max()),(20000,150000)) 
print(len(y)) 
print(y) 
import matplotlib.pyplot as plt 
plt.plot(x,t,'.',label="training data") 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary") 
plt.title("Experience vs Salary") 
from sklearn.linear_model import LinearRegression 
reg_model = LinearRegression() 
reg_model.fit(x,y) 
y_pred=reg_model.predict(x) 
plt.plot(x,y_pred,color="black") 
plt.plot(x,y,'.',label="training data") 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary") 
plt.title("Experience vs Salary") 
import pandas as pd 
data = {'Experience':np.round(x.flatten()),'Salary':np.round(y)} 
df=pd.DataFrame(data) 
df.head(10) 


# In[ ]:


#Write a python program to simulate linear model Y=10+7*x+e  for random 100 samples and visualize univariate linear regression on it. #


# In[ ]:


x1=[[13.0]] 
y1=reg_model.predict(x1) 
print(np.round(y1)) 
reg_model1=LinearRegression() 
x=np.random.rand(100,1) 
yintercept=10 
slope=7 
error=np.random.rand(100,1) 
y=yintercept+slope*x+error 
reg_model1.fit(x,y) 
y_pred=reg_model1.predict(x) 
plt.scatter(x,y,s=10) 
plt.xlabel("X") 
plt.ylabel("Y") 
plt.plot(x,y_pred,color="black") 


# In[ ]:


#prac6:Write a python program to implement multiple linear regression on the Dataset Boston.csv#


# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn 
boston = pd.read_csv("Boston.csv") 
boston.head() 
boston.info() 
boston = boston.drop(columns="Unnamed: 0") 
boston.info() 
boston_x = pd.DataFrame(boston.iloc[:,:13]) 
boston_y = pd.DataFrame(boston.iloc[:,-1]) 
boston_x.head() 
boston_y.head() 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(boston_x, boston_y, test_size=0.3) 
print("xtrain shape", X_train.shape) 
print("ytrain shape", Y_train.shape) 
print("xtest shape", X_test.shape) 
print("ytest shape", Y_test.shape) 
from sklearn.linear_model import LinearRegression 
regression=LinearRegression() 
regression.fit(X_train,Y_train) 
Y_pred_linear = regression.predict(X_test) 
Y_pred_df = pd.DataFrame(Y_pred_linear,columns=["Predicted"]) 
Y_pred_df.head() 
plt.scatter(Y_test, Y_pred_linear, c="green") 
plt.xlabel("Actual Price(medv)") 
plt.ylabel("Predicted Pric(medv)") 
plt.title("Actual vs Prediction") 
plt.show() 


# In[ ]:


#prac7:K Nearest Neighbor classification Algorithm #


# In[ ]:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns = bc.feature_names)
df.head()
x = df[['mean area', 'mean compactness']]
print(x)

y = pd.Categorical.from_codes(bc.target, bc.target_names)
y = pd.get_dummies(y, drop_first=True)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(y_pred)
plt.scatter(x_test['mean area'], x_test['mean compactness'], c= y_pred, cmap='coolwarm')
plt.show()
sns.scatterplot(x='mean area',y='mean compactness', hue = 'benign', data=x_test.join(y_test, how = 'outer'))
cf = confusion_matrix(y_test,y_pred)
ax = plt.subplot()

sns.heatmap(cf, ax=ax, annot = True)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.xaxis.set_ticklabels(['MAlignant','Benign'])
ax.yaxis.set_ticklabels(['MAlignant','Benign'])
tp,fn,fp,tn = confusion_matrix(y_test, y_pred,labels=[1,0]).reshape(-1)

print(tp,fn,fp,tn)
a = (tp+fp)/(tp+fn+fp+tn)
p = tp/(tp+fp)
r = tp/(tp+fn)
print(a)
print(p)
print(r)
print(f1_score(y_pred,y_test))
print(roc_auc_score(y_pred,y_test))


# In[ ]:


#prac8:Introduction to NOSQL using MongoDB #


# In[ ]:


Perform the following:
1.Create a database Company ,Create a Collection Staff and Insert ten documents in it with fields: empid,empname,salary and designation. 
•	Display all documents in Staff and display only empid and designation.db 
•	Sort the documents in descending order of Salary 
•	Display employee with designation with “Manager” or salary greater than Rs. 50,000/-. 
•	Update the salary of all employees with designation as “Accountant” to Rs.45000. 
•	Remove the documents of employees whose salary is greater than Rs100000. 
2. Create a database Institution .Create a Collection Student and Insert ten documents in it with fields: RollNo,Name,Class and TotalMarks(out of 500). 
•	Display all documents in Student. 
•	Sort the documents in descending order of TotalMarks. 
•	Display students  of class  “MSc” or marks greater than 400. 
•	Remove all the documents with TotalMarks<200 

Code and Output:
1.Create a database Institution ,Create a Collection Staff and Insert ten documents in it with fields: empid,empname,salary and designation.
use Institution 
db.createCollection(“Staff”)
 
db
db.Staff.insertMany([ { "empid": 1, "empname": "John Doe", "salary": 60000, "designation": "Manager" }, { "empid": 2, "empname": "Jane Smith", "salary": 55000, "designation": "Accountant" }, { "empid": 3, "empname": "Michael Johnson", "salary": 70000, "designation": "Manager" }, { "empid": 4, "empname": "Emily Brown", "salary": 45000, "designation": "Accountant" }, { "empid": 5, "empname": "David Wilson", "salary": 80000, "designation": "Developer" }, { "empid": 6, "empname": "Sarah Lee", "salary": 95000, "designation": "Manager" }, { "empid": 7, "empname": "Christopher Martinez", "salary": 50000, "designation": "Accountant" }, { "empid": 8, "empname": "Amanda Davis", "salary": 120000, "designation": "Manager" }, { "empid": 9, "empname": "Jason Rodriguez", "salary": 40000, "designation": "Intern" }, { "empid": 10, "empname": "Jessica Taylor", "salary": 110000, "designation": "Manager" } ])
Db.Staff.find().pretty()
 
•	Display all documents in Staff and display only empid and designation.
db.Staff.find().pretty()
 
 
Db.staff.find({], {“_id”:0, “empid”:1,  “designation”:1}).pretty()		 
•	Sort the documents in descending order of Salary
db.Staff.find().sort({ "salary": -1 })
 
•	Display employee with designation with “Manager” or salary greater than Rs. 50,000/-.
db.Staff.find({ $or: [{ "designation": "Manager" }, { "salary": { $gt: 50000 } }] })
 
•	Update the salary of all employees with designation as “Accountant” to Rs.45000.
db.Staff.updateMany({ "designation": "Accountant" }, { $set: { "salary": 45000 } })
 
db.Staff.din({“designation”: “Accountant”})
 
•	Remove the documents of employees whose salary is greater than Rs100000.
db.Staff.deleteMany({ "salary": { $gt: 100000 } })
db.Staff.find()
 

2. Create a database Institution .Create a Collection Student and Insert ten documents in it with fields: RollNo,Name,Class and TotalMarks(out of 500).
db.createCollection(“Student”)
db
 
db.Student.insertMany([ { "RollNo": 101, "Name": "Alice Johnson", "Class": "BSc", "TotalMarks": 480 }, { "RollNo": 102, "Name": "Bob Smith", "Class": "MSc", "TotalMarks": 450 }, { "RollNo": 103, "Name": "Charlie Brown", "Class": "MSc", "TotalMarks": 420 }, { "RollNo": 104, "Name": "David Davis", "Class": "BSc", "TotalMarks": 400 }, { "RollNo": 105, "Name": "Eva Wilson", "Class": "MSc", "TotalMarks": 490 }, { "RollNo": 106, "Name": "Frank Martinez", "Class": "BSc", "TotalMarks": 360 }, { "RollNo": 107, "Name": "Grace Lee", "Class": "MSc", "TotalMarks": 510 }, { "RollNo": 108, "Name": "Henry Taylor", "Class": "BSc", "TotalMarks": 320 }, { "RollNo": 109, "Name": "Isabel Rodriguez", "Class": "MSc", "TotalMarks": 380 }, { "RollNo": 110, "Name": "Jack Harris", "Class": "BSc", "TotalMarks": 250 } ])
 
db.Student.find({})
 
•	Display all documents in Student.
db.Student.find({})		 
•	Sort the documents in descending order of TotalMarks.
db.Student.find().sort({ "TotalMarks": -1 })
 
•	Display students  of class  “MSc” or marks greater than 400.
db.Student.find({ $or: [{ "Class": "MSc" }, { "TotalMarks": { $gt: 400 } }] })
 
•	Remove all the documents with TotalMarks<200
db.Student.deleteMany({ "TotalMarks": { $lt: 200 } })
db.Student.find({})		


# In[ ]:


#mongodb intro#


# In[1]:


C:\Program Files\MongoDB\Server\4.4\bin>mongo
MongoDB shell version v4.4.1
connecting to:
mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb
Implicit session: session { "id" : UUID("5b341d30-d8c0-4b3b-8fee-7333eb3ceaa9") }
MongoDB server version: 4.4.1
---
The server generated these startup warnings when booting:
2024-02-23T13:50:15.935+05:30: ***** SERVER RESTARTED *****
2024-02-23T13:50:25.901+05:30: Access control is not enabled for the database. Read
and write access to data and configuration is unrestricted
---
---
Enable MongoDB's free cloud-based monitoring service, which will then receive and
display
metrics about your deployment (disk utilization, CPU, operation statistics, etc).
The monitoring data will be available on a MongoDB website with a unique URL accessible
to you
and anyone you share the URL with. MongoDB may use this information to make product
improvements and to suggest MongoDB products and deployment options to you.
To enable free monitoring, run the following command: db.enableFreeMonitoring()
To permanently disable this reminder, run the following command:
db.disableFreeMonitoring()
---
> use mydb
switched to db mydb
> show collections
> db.inventory.insertMany( [
... { item: "journal", qty: 25, size: { h: 14, w: 21, uom: "cm" }, status: "A" },
... { item: "notebook", qty: 50, size: { h: 8.5, w: 11, uom: "in" }, status: "P" },
... { item: "paper", qty: 100, size: { h: 8.5, w: 11, uom: "in" }, status: "D" },
... { item: "planner", qty: 75, size: { h: 22.85, w: 30, uom: "cm" }, status: "D" },
... { item: "postcard", qty: 45, size: { h: 10, w: 15.25, uom: "cm" }, status: "A" },
... ] );
{
"acknowledged" : true,
"insertedIds" : [
ObjectId("65de9ed78bb2372715fd09ea"),
ObjectId("65de9ed78bb2372715fd09eb"),
ObjectId("65de9ed78bb2372715fd09ec"),
ObjectId("65de9ed78bb2372715fd09ed"),
ObjectId("65de9ed78bb2372715fd09ee")
]
}
> show collections
inventory
>
>
>
> db.students.insertOne({name:"AAA",rollno:546})
{
"acknowledged" : true,
"insertedId" : ObjectId("65de9fe68bb2372715fd09ef")
}
> db.inventory.find()
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09eb"), "item" : "notebook", "qty" : 50, "size" : { "h" :
8.5, "w" : 11, "uom" : "in" }, "status" : "P" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find().pretty()
{
"_id" : ObjectId("65de9ed78bb2372715fd09ea"),
"item" : "journal",
"qty" : 25,
"size" : {
"h" : 14,
"w" : 21,
"uom" : "cm"
},
"status" : "A"
}
{
"_id" : ObjectId("65de9ed78bb2372715fd09eb"),
"item" : "notebook",
"qty" : 50,
"size" : {
"h" : 8.5,
"w" : 11,
"uom" : "in"
},
"status" : "P"
}
{
"_id" : ObjectId("65de9ed78bb2372715fd09ec"),
"item" : "paper",
"qty" : 100,
"size" : {
"h" : 8.5,
"w" : 11,
"uom" : "in"
},
"status" : "D"
}
{
"_id" : ObjectId("65de9ed78bb2372715fd09ed"),
"item" : "planner",
"qty" : 75,
"size" : {
"h" : 22.85,
"w" : 30,
"uom" : "cm"
},
"status" : "D"
}
{
"_id" : ObjectId("65de9ed78bb2372715fd09ee"),
"item" : "postcard",
"qty" : 45,
"size" : {
"h" : 10,
"w" : 15.25,
"uom" : "cm"
},
"status" : "A"
}
> db.inventory.find({})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09eb"), "item" : "notebook", "qty" : 50, "size" : { "h" :
8.5, "w" : 11, "uom" : "in" }, "status" : "P" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({item:"journal"})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({status:"A"})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({status:"d"})
> db.inventory.find({status:"D"})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
>
...
...
>
> ;
>
>
> db.inventory.find({status:{$in:["A","D"]}})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({status:"A",qty:{$lt:30}})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({$or:[{status:"A"},{qty:{$lt:30}}]})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find({$or:[{status:"A"},{qty:{$lt:30}},{item:/^p/}]})
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
>
>
> db.inventory.find( {status:"A", $or: [ {qty: {$lt:30} }, {item:/^p/} ] } )
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.inventory.find( )
...
> db.inventory.update({item:"paper"},{$set:{qty:200}})
WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })
> db.inventory.find( )
{ "_id" : ObjectId("65de9ed78bb2372715fd09ea"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09eb"), "item" : "notebook", "qty" : 50, "size" : { "h" :
8.5, "w" : 11, "uom" : "in" }, "status" : "P" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ec"), "item" : "paper", "qty" : 200, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ed"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65de9ed78bb2372715fd09ee"), "item" : "postcard", "qty" : 45, "size" : { "h" :
10, "w" : 15.25, "uom" : "cm" }, "status" : "A" }
>
>
>
>
>
>
>
>
>
>
>
db.products.insertMany([{_id:"P001",name:"Sugar",unitprice:100,stock:300},{_id:"P002",name:"
Rock Salt",unitprice:120,stock:500},{_id:"P003",name:"Biscuit",unitprice:10,stock:300}])
{ "acknowledged" : true, "insertedIds" : [ "P001", "P002", "P003" ] }
> db.products.find()
{ "_id" : "P001", "name" : "Sugar", "unitprice" : 100, "stock" : 300 }
{ "_id" : "P002", "name" : "Rock Salt", "unitprice" : 120, "stock" : 500 }
{ "_id" : "P003", "name" : "Biscuit", "unitprice" : 10, "stock" : 300 }
>
> db.products.save({_id:"P001",name:"Cherimeri",unitprice:100,stock:300})
WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })
> db.products.find()
{ "_id" : "P001", "name" : "Cherimeri", "unitprice" : 100, "stock" : 300 }
{ "_id" : "P002", "name" : "Rock Salt", "unitprice" : 120, "stock" : 500 }
{ "_id" : "P003", "name" : "Biscuit", "unitprice" : 10, "stock" : 300 }
>
>
>
> db.products.remove({name:"Biscuit"})
WriteResult({ "nRemoved" : 1 })
> db.products.find()
{ "_id" : "P001", "name" : "Cherimeri", "unitprice" : 100, "stock" : 300 }
{ "_id" : "P002", "name" : "Rock Salt", "unitprice" : 120, "stock" : 500 }
>> db.products.insertOne({"_id" : "P003", "name" : "Biscuit", "unitprice" : 10, "stock" : 300})
{ "acknowledged" : true, "insertedId" : "P003" }
> db.products.remove({name:"Biscuit"},1)
WriteResult({ "nRemoved" : 1 })
> db.products.find()
{ "_id" : "P001", "name" : "Cherimeri", "unitprice" : 100, "stock" : 300 }
{ "_id" : "P002", "name" : "Rock Salt", "unitprice" : 120, "stock" : 500 }
> db.products.insertOne({"_id" : "P003", "name" : "Biscuit", "unitprice" : 10, "stock" : 300})
{ "acknowledged" : true, "insertedId" : "P003" }
> db.products.insertOne({"_id" : "P004", "name" : "Biscuit", "unitprice" : 10, "stock" : 300})
{ "acknowledged" : true, "insertedId" : "P004" }
> db.products.remove({name:"Biscuit"},1)
WriteResult({ "nRemoved" : 1 })
> db.products.find()
{ "_id" : "P001", "name" : "Cherimeri", "unitprice" : 100, "stock" : 300 }
{ "_id" : "P002", "name" : "Rock Salt", "unitprice" : 120, "stock" : 500 }
{ "_id" : "P004", "name" : "Biscuit", "unitprice" : 10, "stock" : 300 }
>> db.student.find({},{KEY:1})
{ "_id" : ObjectId("65d6d0f0fecf6e8e743355c5") }
> db.inventory.find({},{KEY:1})
{ "_id" : ObjectId("65d6d282fecf6e8e743355c6") }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c7") }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c8") }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c9") }
{ "_id" : ObjectId("65d6d282fecf6e8e743355ca") }
> db.inventory.find()
{ "_id" : ObjectId("65d6d282fecf6e8e743355c6"), "item" : "journal", "qty" : 25, "size" : { "h" : 14,
"w" : 21, "uom" : "cm" }, "status" : "A" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c7"), "item" : "notebook", "qty" : 50, "size" : { "h" :
8.5, "w" : 11, "uom" : "in" }, "status" : "P" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c8"), "item" : "paper", "qty" : 100, "size" : { "h" : 8.5,
"w" : 11, "uom" : "in" }, "status" : "D" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c9"), "item" : "planner", "qty" : 75, "size" : { "h" :
22.85, "w" : 30, "uom" : "cm" }, "status" : "D" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355ca"), "item" : "postcard", "qty" : 45, "size" : { "h" : 10,
"w" : 15.25, "uom" : "cm" }, "status" : "A" }
> db.student.find()
{ "_id" : ObjectId("65d6d0f0fecf6e8e743355c5"), "name" : "AAA", "rollno" : 134 }
> db.inventory.find({},{item:1,qty:1,status:1})
{ "_id" : ObjectId("65d6d282fecf6e8e743355c6"), "item" : "journal", "qty" : 25, "status" : "A" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c7"), "item" : "notebook", "qty" : 50, "status" : "P" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c8"), "item" : "paper", "qty" : 100, "status" : "D" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355c9"), "item" : "planner", "qty" : 75, "status" : "D" }
{ "_id" : ObjectId("65d6d282fecf6e8e743355ca"), "item" : "postcard", "qty" : 45, "status" : "A" }
> db.inventory.find({},{item:1,qty:1,status:1,_id:0})
{ "item" : "journal", "qty" : 25, "status" : "A" }
{ "item" : "notebook", "qty" : 50, "status" : "P" }
{ "item" : "paper", "qty" : 100, "status" : "D" }
{ "item" : "planner", "qty" : 75, "status" : "D" }
{ "item" : "postcard", "qty" : 45, "status" : "A" }
> db.inventory.find({},{item:1,qty:1,status:1,_id:0})
{ "item" : "journal", "qty" : 25, "status" : "A" }
{ "item" : "notebook", "qty" : 50, "status" : "P" }
{ "item" : "paper", "qty" : 100, "status" : "D" }
{ "item" : "planner", "qty" : 75, "status" : "D" }
{ "item" : "postcard", "qty" : 45, "status" : "A" }
> clear
uncaught exception: ReferenceError: clear is not defined :
@(shell):1:1
> db.inventory.find({},{item:1,qty:1,status:1,_id:0}).sort({qty:1})
{ "item" : "journal", "qty" : 25, "status" : "A" }
{ "item" : "postcard", "qty" : 45, "status" : "A" }
{ "item" : "notebook", "qty" : 50, "status" : "P" }
{ "item" : "planner", "qty" : 75, "status" : "D" }
{ "item" : "paper", "qty" : 100, "status" : "D" }


# In[ ]:





# In[ ]:




