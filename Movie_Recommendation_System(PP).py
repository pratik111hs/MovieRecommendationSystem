#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


m = r'C:\Users\Pratik Sonawane\Downloads\tmdb_5000_movies.csv'
movies = pd.read_csv(m)

c = r'C:\Users\Pratik Sonawane\Downloads\tmdb_5000_credits.csv'
credits = pd.read_csv(c)


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movie = movies.merge(credits,on='title')


# In[6]:


movie.shape


# In[7]:


movie.head(1)


# In[8]:


movie['original_language'].value_counts().head(5)


# ####  keeping those columns imp for our project#### 

# In[9]:


movie_df = movie[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movie_df.isnull().sum()


# In[11]:


movie_df.dropna(subset=['overview'],inplace = True)


# In[12]:


movie_df.isnull().sum()


# In[13]:


movie_df.duplicated().sum()


# In[14]:


movie_df.iloc[0].genres


# In[15]:


import ast


# In[16]:


def exrtracted_genres(genres_list):
    genres = []
    for genre_dict in ast.literal_eval(genres_list):
        genres.append(genre_dict['name'])
    return genres


# In[17]:


movie_df['genres']=movie_df['genres'].apply(exrtracted_genres)


# In[18]:


movie_df.head()


# In[19]:


movie_df['keywords']= movie_df['keywords'].apply(exrtracted_genres)


# In[20]:


movie_df['cast'][0]


# In[22]:


def top3names (cast_list):
    cast = []
    counter = 0
    for cast_dict in ast.literal_eval(cast_list):
        if counter != 3:
            cast.append(cast_dict['name'])
            counter +=1
        else:
            break
    return cast 


# In[23]:


movie_df['cast']=movie_df['cast'].apply(top3names)


# In[24]:


movie_df.head(1)


# In[25]:


movie_df['crew'][0]


# In[28]:


# replace each string in the crew column with the corresponding converted
# list of dictionaries.
movie_df["crew"] = movie_df["crew"].apply(lambda crew_list: ast.literal_eval(crew_list))


# In[29]:


def director(crew_list):
    for crew_dict in crew_list:
        if crew_dict["job"] == "Director":
            return crew_dict["name"]
    return None


# In[30]:


movie_df['crew']= movie_df['crew'].apply(director)


# In[31]:


movie_df.head()


# In[32]:


movie_df.info()


# In[33]:


movie_df.dropna(inplace = True)


# In[34]:


movie_df.info()


# In[35]:


movie_df['overview'][0]


# #### overview column is a string convert it to a list

# In[36]:


movie_df['overview'] = movie_df['overview'].apply(lambda overview: overview.split())


# ### remove spaces btw names ex in cast column 'Sam Worthington' we need SamWorthington so it will not confuse with another sam

# In[37]:


movie_df['genres']= movie_df['genres'].apply(lambda x:[ i.replace(" ","") for i in x])
movie_df['keywords']= movie_df['keywords'].apply(lambda x:[ i.replace(" ","") for i in x])
movie_df['cast']= movie_df['cast'].apply(lambda x:[ i.replace(" ","") for i in x])
movie_df['crew']= movie_df['crew'].apply(lambda x:[ i.replace(" ","") for i in x])


# In[38]:


movie_df.head()


# In[39]:


movie_df['crew'] = movie_df['crew'].apply(lambda x: "".join(x))


# In[40]:


movie_df['crew'][0]


# In[41]:


movie_df['crew'] = movie_df['crew'].dropna().apply(lambda x: x.split())


# In[42]:


movie_df['crew'][0]


# In[43]:


movie_df.info()


# In[44]:


movie_df.head()


# In[45]:


movie_df['tag']= movie_df['overview'] + movie_df['genres']+ movie_df['keywords']+movie_df['cast']+movie_df['crew']


# In[46]:


movie_df.head()


# In[47]:


movie_df1 = movie_df[['movie_id','title','tag']]


# In[48]:


movie_df1.head()


# In[49]:


movie_df1.tag[0]


# #### convert tag from list to string

# In[50]:


movie_df1['tag']= movie_df1['tag'].apply(lambda x :" ".join(x))


# In[51]:


movie_df1.tag[0]


# In[52]:


movie_df1['tag']= movie_df1['tag'].apply(lambda x : x.lower())


# In[53]:


movie_df1.tag[0]


# #### Convert text to vectors(recommend closest movie recommendation)
# #### using bagofwords method

# In[54]:


from sklearn.feature_extraction.text import CountVectorizer


# In[55]:


# stop_words = are , im , to ,and we are not including it
cv = CountVectorizer(max_features=5000,stop_words = 'english')


# In[60]:


vectors = cv.fit_transform(movie_df1['tag']).toarray()
vectors


# In[61]:


cv.fit_transform(movie_df1['tag']).toarray().shape


# In[65]:


cv.get_feature_names_out()


# ### Applying stemming to reduce word repetition( action, actions)

# In[66]:


pip install nltk


# In[67]:


# nltk Natural Lang Toolkit

import nltk


# In[69]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[70]:


ps.stem('loving')


# In[72]:


ps.stem('dancing')


# In[73]:


def stem(text):
    y = []
    for i in text.split(): # Convert to list
        y.append(ps.stem(i))
    return " ".join(y) # Convert to string


# In[76]:


movie_df1['tag']=movie_df1['tag'].apply(stem)


# ### we will use cosine distance angle btw 2 vectors 
# 
# if angle btw 2 vector is small ex(20 degree) movies are more similar)

# In[81]:


from sklearn.metrics.pairwise import cosine_similarity


# In[84]:


similarity =cosine_similarity(vectors)


# In[85]:


similarity.shape 


# In[86]:


sorted(list(enumerate(similarity[0])),reverse= True, key = lambda x:x[1])


# #### 0 ie Avatar most similar moves are 539,1194,...

# In[101]:


def recommend(movie):
    movie_index = movie_df1[movie_df1['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list =  sorted(list(enumerate(distances)),
                         reverse= True, key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(movie_df1.iloc[i[0]].title)


# In[105]:


recommend('Spectre')


# In[110]:


recommend('Avatar')


# In[111]:


recommend('Quantum of Solace')


# In[112]:


recommend('Tangled')


# In[109]:


movie_df1.head(20)


# In[ ]:




