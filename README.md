# Job Hunting using Natural Language Processing

### Inspiration:
Throughout many discussions with the fellows at Metis we agreed that the titles given in data-related fields can be quite ambiguous which complicates one's job search. Some companies may define a particular role as a data scientist while others see it as a machine learning engineer. Others still may feel that the role is within the product science team. 

### Objective:
My goal in approaching this project was to use Natural Language processing to better understand what different companies are looking to facilitate my job search.

### The How:
* Sourced job descriptions from indeed.com
* Stored data in MongoDB hosted on AWS
* Used unsupervised learning models such as KMeans Clustering, LDA, NMF
* Created custom ngram tools to supplement those available in gensim
* Used TF-IDF Word Embedder to match job descriptions based on cosine similarity
* Visualized results using matplotlib / seaborn / plotly
* Created a prototype of the application using JavaScript / Flask
