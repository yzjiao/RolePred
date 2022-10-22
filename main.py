import os
import csv
import spacy 
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter
from nltk import sent_tokenize, word_tokenize
from transformers import pipeline, T5Tokenizer, T5Config, T5ForConditionalGeneration



def load_data(path):
    # Load multiple documents from the path.
    # Each document describes one event instance.
    data = []
    trigger = []
    file_names = os.listdir(path)
    for name in file_names:
        if '.txt' in name:
            f = open(path + name, 'r')
            lines = f.readlines()
            text = ' '.join(lines)
            data.append(text)
            # The name of each document is regarded as the event trigger. 
            # Note that the triggers are only used to identify different events. They won't be used for role prediction. 
            trigger.append(name[:-4])
    return data, trigger



def truncate_text(text, max_length=350):
    # Truncate each document for subsequent processing
    # Note that the length of the truncated text here should be less than the maximum length that pretrained models can process.
    # Because the prompt will be added to generate candidate roles
    truncated = []
    for doc in text:
        string = []
        cnt_w = 0
        sents = sent_tokenize(doc)
        for s in sents:
            c = len(word_tokenize(s))
            if cnt_w + c > max_length:
                break
            cnt_w += c
            string.append(s)
        truncated.append(string)
    return truncated


def recognize_named_entity(text, nlp):
    # Recognize all named entities and their types using the spacy NER tool
    def cleanup(token, lower = True):
        if lower:
            token = token.lower()
        return token.strip()

    entities = []
    for doc in text:
        entity = []
        for sent in doc:
            data = nlp(sent)
            entity.append(data.ents)
        entities.append(entity)
    return entities



def unmask_role_name(unmasker, tokenizer, text):
    # Inference role names for each prompt
    # Returned 10 candidate roles whose maximum length is 3

    DEVICE = unmasker.device
    encoded = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(DEVICE)
    outputs = unmasker.generate(input_ids=input_ids, 
                              num_beams=200, num_return_sequences=10,
                              max_length=5)
    res = []
    end_token='<extra_id_1>'
    for output in outputs:
        _txt = tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if end_token in _txt:
            _end_token_index = _txt.index(end_token)
            res.append(_txt[:_end_token_index])
    return res

def generate_candidate_role(unmasker, tokenizer, text, entities, event_type, alpha=0.4):
    # Generate candidate roles for each named entity according the text
    
    # The geneation output will be saved to dir_path
    all_cand = Counter()
    dir_path = 'candidate_roles/'
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
    file_path = dir_path + event_type

    if os.path.exists(file_path): 
        # For the given event type, if there exist the generated roles, just load them for the subsequent steps.
        all_cand = torch.load(file_path)
    else:
        for (doc, entity) in tqdm(zip(text, entities)):
            string = ''
            cands = []
            for i in range(len(doc)):
                string += doc[i]
                for e in entity[i]:
                    # Construct different prompts considering the entity types like CARDINAL, GPE, and PERSON.
                    if e.label_ == 'CARDINAL':
                        templete = string + ' According to this, the number of <extra_id_0> of this ' + event_type + ' is ' + str(e) + '.'
                        # the input of generation model includes the original text and the constructed prompt
                        res = unmask_role_name(unmasker, tokenizer, templete)
                        cands += res
                    elif e.label_ == 'GPE':
                        templete = string + ' According to this, the <extra_id_0> is ' + str(e) + ' in this ' + event_type  + '.'
                        #templete = string + ' According to this, ' + str(e) + ' is a <extra_id_0> in this ' + event_type  + '.'
                        res = unmask_role_name(unmasker, tokenizer, templete)
                        cands += res
                    elif e.label_ == 'PERSON':
                        templete = string + ' According to this, ' + str(e) + ' plays the role of <extra_id_0> in this ' + event_type  + '.'
                        res = unmask_role_name(unmasker, tokenizer, templete)
                        cands += res
                    templete = string + ' According to this, the <extra_id_0> of this ' + event_type + ' is ' + str(e) + '.'
                    res = unmask_role_name(unmasker, tokenizer, templete)
                    cands += res

            cands = set(cands)              
            cands = Counter(cands) 
            all_cand += cands

        torch.save(all_cand, file_path)
    
    # For each role name, count the number of doucments it can generate from.
    # If one role is generated from a few documents, it will be removed.
    role_name = [k for k in all_cand if all_cand[k] > alpha*len(text)] 
    return role_name


def extract_argument(question_answerer, text, role_name, event_type, beta=0.3):
    # Extract one argument for each candidate role from each document
    arguments = []
    for doc in text:
        full_str = ' '.join(doc)
        all_answer = {}
        for name in role_name:
            # Construct a question
            question = 'What is the ' + name + ' of this ' + event_type + '?'
            result = question_answerer(question=question, context=full_str)
            # Filter out the arguments with low confidence scores. 
            if result['score'] > beta:
                all_answer[name] = (result['answer'], result['score'])
        arguments.append(all_answer)
    return arguments


def select_role_name(role_names, arguments, theta=0.4, lamda=0.4):
    # Merge and select salient roles

    n_name = len(role_names)
    n_file = len(arguments)

    # Build a graph to find the roles of similar semantics. 
    # The nodes are role names. 
    # If two roles usually have the same argument in each document, they are connected with an edge. 
    G = nx.Graph()
    G.add_nodes_from(role_names)
    for i in range(n_name):
        for j in range(i):
            cnt = 0
            for arg in arguments:
                if role_names[i] in arg and role_names[j] in arg:
                    cnt += arg[role_names[i]][0] == arg[role_names[j]][0]
            if cnt > theta * n_file:
                G.add_edge(role_names[i], role_names[j])
    # The roles in a connected component are similar in semantics, thus being merged together. 
    cluster = [list(c) for c in nx.connected_components(G)]

    # Rank each cluster according the highest frequency of its roles
    cluster_score = []
    for c in cluster:
        score = [role_names.index(i) for i in c]
        cluster_score.append(min(score))
    ranked_index = np.argsort(cluster_score)
    ranked_cluster = [cluster[i] for i in ranked_index]
    print(ranked_cluster)

    # Each cluster will have one argument at most in one document
    # We select the argument according to the confidence score
    seleted_args = []
    for arg in arguments:
        merged_arg = {}
        for c in cluster: 
            score = [arg[i][1] if i in arg else 0.0 for i in c]
            max_score = max(score)
            max_idx = score.index(max_score)
            best_name = c[max_idx]
            # Remove the cluster if its highest confidence score is lower than lamda.
            if best_name in arg and arg[best_name][1] > lamda:
                key = ', '.join(c)
                merged_arg[key] = arg[best_name]
        seleted_args.append(merged_arg)
    
    # Filter redundent arguments for each document
    filter_args = []
    for arg in seleted_args:
        val_dic = {}
        filter_arg = {}
        for key in arg:
            val, score = arg[key]
            if val not in val_dic or val_dic[val][1] < score:
                val_dic[val] = (key, score)
        for val in val_dic:
            key, score = val_dic[val]
            filter_arg[key] = (val, score)
        filter_args.append(filter_arg)
    
    return ranked_cluster, filter_args




def save_output(event_type, schema, arguments, triggers):
    # Save the model output as a csv file to OUTPUT_DIR 
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    
    path = OUTPUT_DIR + event_type + '.csv'
    # The header of the csv file include the roles. 
    header = [', '.join(name) for name in schema] + ['trigger']
    events = []
    # In the file, each line has the arguments of one document. 
    for (i, arg) in enumerate(arguments):
        tmp = {'trigger': triggers[i]}
        for key in arg:
            tmp[key] = arg[key][0]
        events.append(tmp)

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(events)




def RolePred_pipeline(event_type, DEVICE_ID=0):
    path = DATA_DIR + event_type + '/'
    print(path)

    # Load multiple documents for the given event type
    text, triggers = load_data(path)

    # Truncate each document 
    text = truncate_text(text)
    
    # Identify named entities in multiple documents
    nlp = spacy.load("en_core_web_sm")
    entities = recognize_named_entity(text, nlp)

    # Generate candidate roles based on all entities
    T5_PATH = 't5-base' 
    DEVICE = torch.device('cuda:' + str(DEVICE_ID))
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
    t5_config = T5Config.from_pretrained(T5_PATH)
    t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
    role_names = generate_candidate_role(t5_mlm, t5_tokenizer, text, entities, event_type)
    
    # Extract arguments for each candidate roles
    question_answerer = pipeline('question-answering', model="deepset/roberta-large-squad2", tokenizer="deepset/roberta-large-squad2", device=DEVICE_ID)
    arguments = extract_argument(question_answerer, text, role_names, event_type)

    # Merge and filter the candidate roles based on the extracted arguments
    cluster, merged_arguments = select_role_name(role_names, arguments)

    # Save the model output 
    save_output(event_type, cluster, merged_arguments, triggers)
    






DATA_DIR = 'RoleEE_data/' #dataset directory
OUTPUT_DIR = 'output/'  #output directory
DEVICE_ID = 0
torch.cuda.is_available()


# Get all subdirectory names from the dataset directory as event types
EVENT_TYPE = []
for root, dirs, files in os.walk(DATA_DIR, topdown=False):
    EVENT_TYPE = dirs

# Inference the argument roles for each event type
for event_type in EVENT_TYPE:
    RolePred_pipeline(event_type, DEVICE_ID)





















