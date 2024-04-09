#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#prac 1:bitwise operator


# In[ ]:


#method1#


# In[1]:


def bitwise_operation(a,b):
    bitwise_and_result=a&b
    print("a_and_b:",bitwise_and_result)
    bitwise_or_result=a|b
    print("a_or_b:",bitwise_or_result)
    bitwise_xor_result=a^b
    print("a_xor_b:",bitwise_xor_result)
    bitwise_not_result=~a
    print("a_not_b:",bitwise_not_result)
    bitwise_not_result=~b
    print("a_not_b:",bitwise_not_result)
    left_shift = a<<1
    print("Left shift:",left_shift)
    right_shift = a>>1
    print("Right shift:",right_shift)

a=int(input("Enter a in binary:")) #Binary: 1010
b=int(input("Enter b in binary:")) #Binary: 0101
bitwise_operation(a,b)


# In[ ]:


#method2#


# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print("Boolean RetrivalModel using Bitwise operation on Team Document Incidence Matrix")
corpus={'this is the first document','this document is the second document','and this is the third document','Is This The First Document?'}
print("The Corpus is \n",corpus)
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(corpus)
df=pd.DataFrame(x.toarray(),columns=vectorizer.get_feature_names())
print("The generated data frame")
print(df)
print("Ouery processing on teams document incidence matrix")

#AND
print("1. Find all document ids for query 'this' AND 'first'")
alldata=df[(df['this']==1)&(df['first']==1)]
print("Document ids where with 'this' AND 'first' are present are:",alldata.index.tolist())

#OR
print("2. Find all document ids for query 'this' OR 'first'")
alldata=df[(df['this']==1)|(df['first']==1)]
print("Document ids where either 'this' OR 'first' are present are:",alldata.index.tolist())

#NOT
print("3. Find all document ids for query NOT'and'")
alldata=df[(df['this']==1)&(df['first']==1)]
print("Document ids where with 'and' term is not present:",alldata.index.tolist())

 


# In[ ]:


#prac2:pagerank


# In[ ]:


#method1#


# In[7]:


import networkx as nx 
import pylab as plt 
G=nx.DiGraph() 
[G.add_node(k) for k in ["A","B","C","D","E","F","G"]] 
G.add_edges_from([ ('A','G'),('G','A'),('A','D'),(' A','C'),('B','A'),('D','B'),('D','F'),('E','A'),('F','A'),('A','C') ]) 
ppr1 = nx.pagerank(G)
print("page rank value",ppr1) 
pos=nx.spiral_layout(G) 
nx.draw(G,pos,with_labels=True,node_color="#f86e00") 
plt.show() 


# In[ ]:


#method2#


# In[21]:


import networkx as nx
import pylab as plt
G=nx.DiGraph()
[G.add_node(k) for k in ["A","B","C","D","E","F","G"]]
G.add_edges_from([("A","G"),("A","D"),("A","C"),("B","A"),("D","B"),("D","F"),("E","A"),("G","A"),("F","A"),("C","A")])
ppr1=nx.pagerank(G)
print("Page rank values:",ppr1)
pos=nx.spiral_layout(G)
nx.draw_networkx(G,pos,with_labels=True,node_color="#00ffff")
plt.show()


# In[ ]:


#method3#


# In[8]:


def page_rank(g, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_pages = len(g)
    initial_page_rank = 1.0 / num_pages
    page_ranks = {page: initial_page_rank for page in g}

    for _ in range(max_iterations):
        new_page_ranks = {}
        for page in g:
            new_rank = (1 - damping_factor) / num_pages

            for link in g:
                if page in g[link]:
                    new_rank += damping_factor * (page_ranks[link] / len(g[link]))

            new_page_ranks[page] = new_rank

        convergence = all(abs(new_page_ranks[page] - page_ranks[page]) < tolerance for page in g)
        page_ranks = new_page_ranks

        if convergence:
            break

    return page_ranks

if __name__ == "__main__":
    example_graph = {
        'A': ['B', 'C'],
        'B': ['A'],
        'C': ['B', 'A'],
        'D': ['B']
    }
    result = page_rank(example_graph)
    for page, rank in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"Page: {page} - PageRank: {rank:.4f}")


# In[ ]:


#prac3:levenshtein distance#


# In[9]:


def leven(x,y):
    n=len(x)
    m=len(y)
    A=[[i+j for j in range(m+1)]for i in range(n+1)]
    for i in range(n):
        for j in range(m): 
            A[i+1][j+1]=min(A[i][j+1]+1,   #insert
                            A[i+1][j]+1,      #delete
                            A[i][j]+int(x[i]!=y[j]))   #replace
    return A[n][m]
 
print(leven("brap","rap"))
print(leven("trial","try"))
print(leven("horse","force"))


# In[ ]:


#prac4:Compute Similarity Between Two text Document#
#jaccard#


# In[20]:



def Jaccard_Similarity(doc1,doc2):
 # List the unique words in a document
     words_doc1=set(doc1.lower().split())
     words_doc2=set(doc2.lower().split())
# Calculate the intersection of sets doc1 and doc2
     intersection=words_doc1.intersection(words_doc2)
# Calculate the union of sets doc1 and doc2
     union=words_doc1.union(words_doc2)
#Calculate Jaccard Similarity Score
     return float(len(intersection))/len(union)
doc1="Data is the new oil of the digital economy"
doc2="Data is the new oil"
Jaccard_Similarity(doc1,doc2)


# In[ ]:


#cosine similarity#


# In[19]:



doc1 = "Data is the new oil of the digital economy"
doc2 = "Data is the new oil"
data = [doc1, doc2]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer()
vector_matrix = vectorizer.fit_transform(data)
tokens = vectorizer.get_feature_names_out()
create_dataframe = (vector_matrix.toarray(), tokens)
cosine_similarity_matrix = cosine_similarity(vector_matrix)
create_dataframe = cosine_similarity_matrix[0, 1]
print(create_dataframe)


# In[ ]:


#prac5:map reducer


# In[24]:


from functools import reduce
from collections import defaultdict
def mapper(data):
    char_count=defaultdict(int)
    for char in data:
        if char.isalpha():
            char_count[char.lower()]+=1
    return char_count.items()

def reducer (counts1,counts2):
    merged_counts=defaultdict(int)
    for char,count in counts1:
        merged_counts[char]+=count
        for char,count in counts2:
            merged_counts[char]+=count
        return merged_counts.items()
if __name__=="__main__":
    dataset="Hello,World! This is a MapReducer example"
 #split the dataset into chunks(assuming a distributed env)
    chunks=[chunk for chunk in dataset.split()]
 #Map step
    mapped_results=map(mapper,chunks)
 #Reduce step
    final_counts=reduce(reducer,mapped_results)
 #Print the result
    for char,count in final_counts:
        print(f"Character :{char},Count:{count}")


# In[ ]:


#prac6:hits algorithm#


# In[25]:


#HITS Algorithm
import networkx as nx
# Step 2: Create a graph and add edges
G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
# Step 3: Calculate the HITS scores
authority_scores, hub_scores = nx.hits(G)
# Step 4: Print the scores
print("Authority Scores:", authority_scores)
print("Hub Scores:",hub_scores)


# In[ ]:


#prac7:stop words#


# In[ ]:


#method1#


# In[26]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
set(stopwords.words('english'))


# In[ ]:


#method2#


# In[29]:


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sent = "This is sample Sentence , showing off the stop words filtration."
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(example_sent)
filtered_sentence=[w for w in word_tokens if not w in stop_words]
filtered_sentence=[]
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
        
print(word_tokens)
print(filtered_sentence)


# In[ ]:


#prac8:twitter scrapping#


# In[35]:


get_ipython().system('pip install ntscraper')
import pandas as pd
from ntscraper import Nitter
scraper=Nitter()
tweets=scraper.get_tweets('narendramodi',mode='user',number=5)
final_tweets=[]
for tweet in tweets['tweets']:
    data=[tweet['link'],tweet['text'],tweet['date'],tweet['stats']['likes'],tweet['stats']['comments']]
    final_tweets.append(data)
print(final_tweets)
data=pd.DataFrame(final_tweets,columns=['link','text','date','Number of likes','Number of tweets'])
print(data)


# In[ ]:


#prac9:simple web crawling#


# In[40]:


import requests
from parsel import Selector
import time
start=time.time
response = requests.get('http://recurship.com/')
selector=Selector(response.text)
href_links=selector.xpath('//a/@href').getall()
image_links=selector.xpath('//img@src').getall()
print('*********************HREF_LINKS**********************')
print(href_links)
print('*********************/HREF_LINKS**********************')
#datatype of this is Lists
print('*********************IMAGE_LINKS**********************')
print(image_links)
print('*********************/IMAGE_LINKS**********************')


# In[ ]:


#web crawler alternative#


# In[39]:


import requests #extraction of request
from parsel  import Selector 
import time
start=time.time()
response=requests.get("http://recurship.com/")
selector=Selector(response.text)
href_links=selector.xpath('a//@href').getall()
image_links=selector.xpath('a//img/@src').getall()
print("**********************Href_links***********************")
print(href_links)
print("**********************href_links***********************")
print(image_links)
print("**********************/image_links***********************")
end=time.time()
print("The is taken in seconds:",(end-start))


# In[ ]:


#prac10:xml retrieval#


# In[42]:


import xml.etree.ElementTree as ET
import networkx as nx
def parse_xml(xml_text):
    root=ET.fromstring(xml_text)
    return root
def generate_web_graph(xml_root):
    G=nx.DiGraph()
    for page in xml_root.findall('.//page'):
        page_id=page.find('id').text
        G.add_node(page_id)
        links=page.findall('.//link')
        for link in links:
            target_page_id=link.text
            G.add_edge(page_id,target_page_id)
 
    return G
def compute_topic_specific_pagerank(graph,topic_nodes,alpha=0.85,max_iter=100,tol=1e-6):
    personalization={node:1.0 if node in topic_nodes else 0.0 for node in graph.nodes}
    return nx.pagerank(graph,alpha=alpha,personalization=personalization,max_iter=max_iter,tol=tol)
if __name__=="__main__":
 #example XML text representing web page and links
 example_xml="""
 <webgraph>
 <page>
 <id>1</id>
 <link>2</link>
 <link>3</link>
 </page>
 <page>
 <id>2</id>
 <link>1</link>
 <link>3</link>
 </page>
 </webgraph>
 """
 
 xml_root=parse_xml(example_xml)
 
 #Generate web graph
web_graph=generate_web_graph(xml_root)
 
 #Compute topic-specific PageRank for nodes 1 and 2
topic_specific_pagerank=compute_topic_specific_pagerank(web_graph,topic_nodes=['1','2'])
 
 #print the results
print("Topic Specific Pagerank: ")
for node, score in sorted(topic_specific_pagerank.items(),key=lambda x:x[1],reverse=True):
    print(f"Node:{node}-PageRank:{score:.4f}")


# In[ ]:


#prac10:retrieve xml test using xml library#


# In[56]:


import xml.etree.ElementTree as ET

xml_data = '''<root>
<person>
<name>John</name>
<age>30</age>
<city>New York</city>
</person>

<person>
<name>Alice</name>
<age>25</age>
<city>London</city>
</person>
</root>'''
tree = ET.fromstring(xml_data)
for person in tree.findall('person'): 
    name = person.find('name').text 
    age = person.find('age').text 
    city = person.find('city').text
    print(f"Name: {name}, Age: {age}, City: {city}")


# In[48]:


import xml.etree.ElementTree as ET
import networkx as nx

def parse_xml(xml_text): 
    root = ET.fromstring(xml_text) 
    return root 
def generate_web_graph(xml_root): 
    G = nx.DiGraph() 

    for page in xml_root.findall('.//page'): 
        page_id = page.find('id').text 
        G.add_node(page_id) 
        
        links = page.findall('.//link') 
        for link in links: 
            target_page_id = link.text 
            G.add_edge(page_id,target_page_id) 
    return G

def compute_topic_specific_pagerank(graph, topic_nodes, alpha=0.85, max_iter = 100, tol = 1e-6): 
    personalization = {node: 1.0 if node in topic_nodes else 0.0 for node in graph.nodes} 
    return nx.pagerank(graph, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol) 

if __name__ == "__main__": 
    xml_data = """ 
    <webgraph> 
        <page> 
            <id>1</id> 
            <link>2</link> 
            <link>3</link> 
        </page> 
        <page> 
            <id>2</id> 
            <link>1</link> 
            <link>3</link> 
        </page> 
        <page> 
            <id>3</id> 
            <link>1</link> 
            <link>2</link> 
        </page> 
    </webgraph>""" 

    xml_root = parse_xml(xml_data) 
    web_graph = generate_web_graph(xml_root) 
    topic_specific_pagerank = compute_topic_specific_pagerank(web_graph, topic_nodes=['1','2']) 
    
    print("Topic-Specific PageRank") 
    for node, score in sorted(topic_specific_pagerank.items(),key=lambda x:x[1], reverse=True): 
      print(f"Node: {node} - PageRank: {score:4f}")


# In[ ]:


#prac11:xml retrieval using lxml library#


# In[57]:


import xml.etree.ElementTree as ET
xml_data = '''<root>
<person>
<name>John</name>
<age>30</age>
<city>New York</city>
</person>
<person>
<name>Alice</name>
<age>25</age>
<city>London</city>
</person>
</root>'''
tree = ET.fromstring(xml_data)
for person in tree.findall('person'): 
    name = person.find('name').text 
    age = person.find('age').text 
    city = person.find('city').text
    print(f"Name: {name}, Age: {age}, City: {city}")


# In[ ]:




