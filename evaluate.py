import os
import csv
import torch
from nltk import word_tokenize
import dateutil.parser as parser
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util


def load_data(path):
    with open(path)as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        schema = [(i.lower()).split(', ') for i in fieldnames]
        schema.remove(['trigger'])

        arguments = []
        for row in reader:
            arg = {i.lower(): row[i].lower() for i in row if row[i] != ''}
            arguments.append(arg)
        return schema, arguments


def align_event(gold_events, pred_events):
    gold_dic = { e['trigger'] : e for e in gold_events}
    pred_dic = { e['trigger'] : e for e in pred_events}

    new_gold, new_pred = [], []
    shared_trig = set(gold_dic.keys()) & set(pred_dic.keys())
    for trig in shared_trig:
        tmp = gold_dic[trig]
        tmp.pop('trigger')
        new_gold.append(list(tmp.values()))

        tmp = pred_dic[trig]
        tmp.pop('trigger')
        new_pred.append(list(tmp.values()))
    return new_gold, new_pred   


def process_arg(gold_args, pred_args):
    def parse(self, timestr, default=None,
              ignoretz=False, tzinfos=None,
              **kwargs):
        return self._parse(timestr, **kwargs)
    parser.parser.parse = parse

    def trans_date_format(string):
        date = parser.parser().parse(string, None)
        if date[0] is not None:
            year, month, day = str(date[0].year), str(date[0].month), str(date[0].day)
            if month != 'None' and day != 'None' and year != 'None' and year in string:
                return day + ' ' + month + ' ' + year
            elif year != 'None' and month != 'None' and year in string:
                return month + ' ' + year
            elif month != 'None' and day != 'None':
                return month + ' ' + day
            elif year != 'None' and year in string:
                return year
        return string

    new_gold_args, new_pred_args = [], []
    for (i, _) in enumerate(gold_args):
        new_gold_arg, new_pred_arg = [], []
        for (gold, pred) in zip(gold_args[i], pred_args[i]): 
            new_gold = [trans_date_format(arg) for arg in gold]
            new_gold_arg.append(new_gold)

            new_pred = [trans_date_format(arg) for arg in pred]
            new_pred_arg.append(new_pred)

        new_gold_args.append(new_gold_arg)
        new_pred_args.append(new_pred_arg)

    return new_gold_args, new_pred_args



def hard_match(query, value, separator):
    for q in query:
        opts = q.split(separator)
        for opt in opts:
            for v in value:
                if opt in v:
                    return 1.0
    return 0.0


def soft_match(query, value, separator):
    embedding_1 = sent_sim_model.encode(query, convert_to_tensor=True)
    embedding_2 = sent_sim_model.encode(value, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(embedding_1, embedding_2)
    max_score = torch.max(sim_matrix)
    return max_score


def evaluate_role(gold_roles, pred_roles, match_func, num_pred_roles=20):
    gold_n, pred_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for (i, _) in enumerate(gold_roles):
        gold = gold_roles[i]
        pred = pred_roles[i][:num_pred_roles]
        gold_n += len(gold) 
        pred_n += len(pred)
        for role in gold:
            max_score = 0.0
            for list_role in pred:
                flag = match_func(role, list_role, ' ')
                max_score = max(max_score, flag)
            gold_in_pred_n += max_score
            if max_score == 0.0:
                print(role)
        print()

        for list_role in pred:
            max_score = 0.0
            for role in gold:
                flag = match_func(role, list_role, ' ')
                max_score = max(max_score, flag)
            pred_in_gold_n += max_score
    #print(gold_n, pred_n, gold_in_pred_n, pred_in_gold_n)
    try:
        pre, rec, f1 = 0, 0, 0
        pre = 100.0 * pred_in_gold_n / pred_n
        rec = 100.0 * gold_in_pred_n / gold_n
        f1 = 2 * pre * rec / (pre + rec)
    except:
        pre = rec = f1 = 0
    return pre, rec, f1


def partial_match(str1, str2, stop_words):
    token1 = word_tokenize(str1)
    filtered_token1 = [w for w in token1 if (w.lower() not in stop_words) and w != ',']
    token2 = word_tokenize(str2)
    filtered_token2 = [w for w in token2 if (w.lower() not in stop_words) and w != ',']
    common_word = list(set(filtered_token1).intersection(set(filtered_token2)))
    return len(common_word) > 0


def evaluate_arg(gold_args, pred_args):
    stop_words = set(stopwords.words('english'))
    gold_n, pred_n, pred_in_gold_n, gold_in_pred_n = 0.0, 0.0, 0.0, 0.0
    for (i, _) in enumerate(gold_args):
        for (gold, pred) in zip(gold_args[i], pred_args[i]): 
            while '' in gold:
                gold.remove('')
            gold = set(gold)
            pred = set(pred)
            gold_n += len(gold)
            pred_n += len(pred)
            for arg in gold:
                flag = [partial_match(arg, i, stop_words) for i in pred]
                gold_in_pred_n += int(sum(flag) > 0)
            for arg in pred:
                flag = [partial_match(i, arg, stop_words) for i in gold]
                pred_in_gold_n += int(sum(flag) > 0)
    print(gold_n, pred_n, pred_in_gold_n, gold_in_pred_n)
    pre, rec, f1 = 0, 0, 0
    pre = 100.0 * pred_in_gold_n / pred_n
    rec = 100.0 * gold_in_pred_n / gold_n
    f1 = 2 * pre * rec / (pre + rec)
    return pre, rec, f1





DATA_DIR = 'RoleEE_data/'
OUTPUT_DIR = 'output/'

print(OUTPUT_DIR)

EVENT_TYPE = []
for root, dirs, files in os.walk(DATA_DIR, topdown=False):
    EVENT_TYPE = dirs
#print(EVENT_TYPE)
gold_role, gold_arg, pred_role, pred_arg = [], [], [], []
for event_type in EVENT_TYPE:
    print (event_type)
    gold_path = DATA_DIR + event_type + '/EVENT.csv'
    role1, event1 = load_data(gold_path)
    
    pred_path = OUTPUT_DIR + event_type + '.csv'
    role2, event2 = load_data(pred_path)

    arg1, arg2 = align_event(event1, event2)
    print()
    print()
    gold_role.append(role1)
    gold_arg.append(arg1)
    pred_role.append(role2)
    pred_arg.append(arg2)

print('Role name induction:')
match_fun = hard_match
print('Hard matching:')  
pre_r, rec_r, f1_r = evaluate_role(gold_role, pred_role, match_fun)
print('P: ', pre_r, 'R: ',rec_r, 'F1: ', f1_r)
print()


match_fun = soft_match
print('Soft matching:')  
# $ pip install spacy-transformers
# $ python -m spacy download en_trf_bertbaseuncased_lg
sent_sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pre_r, rec_r, f1_r = evaluate_role(gold_role, pred_role, match_fun)
print('P: ', pre_r, 'R: ',rec_r, 'F1: ', f1_r)
print()

print('Argument extraction:')
new_gold_arg, new_pred_arg = process_arg(gold_arg, pred_arg)
pre_a, rec_a, f1_a = evaluate_arg(new_gold_arg, new_pred_arg)
print('P: ', pre_a, 'R: ',rec_a, 'F1: ', f1_a)






