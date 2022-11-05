# RolePred
Source code for the paper "[Open-Vocabulary Argument Role Prediction for Event Extraction](https://arxiv.org/abs/2211.01577)" from findings of EMNLP 2022.


## Download Dataset
The dataset is now available at [Google Drive](https://drive.google.com/file/d/1IRIKjiAfi0txUmBY-hDzt79segz7tveJ/view?usp=sharing).
- There are 50 folder, each of which contains the materials for one event type, including one CSV files and multiple DOC files. 
- The CSV files is EVENT.csv, which lists the event instances of the same event type. The header is the argument roles and each row is the arguments of one event instance.
- Each DOC file is the source document for one event instance and is named with the event trigger. 


## Dependencies 
- python=3.8.13
- pytorch=1.12.1
- transformers=4.10.2
- sentence_transformers=2.2.2
- spacy=3.4.1
- nltk=3.7
- numpy=1.22.4
- networkx=2.8
- dateutil=2.8.2


## Run RolePred
- Download the RoleEE dataset. 
- Run the role prediction framework: `python main.py`. For each event type, the model will output a CSV file, in which the header is the argument roles and each row is the arguments of one event instance. Such final output will be saved under the directory `output/`. Also, the generated candidate roles will be saved under the directory `candidate_roles/`. 
- Run the evaluation: `python evaluate.py`. 
