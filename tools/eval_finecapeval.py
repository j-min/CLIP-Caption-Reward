
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import argparse
import re
import json
import nltk
from nltk.tokenize import word_tokenize  
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

import language_evaluation
evaluator = language_evaluation.CocoEvaluator()


def nltk_process(text):
    # Tokenization
    nltk_tokenList = word_tokenize(text)

    # Stemming
    nltk_stemedList = []
    for word in nltk_tokenList:
        nltk_stemedList.append(p_stemmer.stem(word))

    filtered_sentence = nltk_stemedList

    # Removing Punctuation

    tokens = [re.sub(r'[^a-zA-Z0-9]', '', tok) for tok in filtered_sentence]

    text = " ".join(tokens)

    return text


def calculate_finegrained_scores(pred_id2sent, id2caption, use_coco_eval=False):
    if use_coco_eval:
        n_total = 0
        refs = []
        hyps = []
        for id, gt_captions in id2caption.items():
            pred_sent = pred_id2sent[id]

            refs.append(gt_captions)
            hyps.append(pred_sent)

            n_total += 1

        print('caption')
        results = evaluator.run_evaluation(hyps, refs)
        pprint(results)

    n_total = 0
    total_score = 0
    for id, gt_phrases in id2background.items():
        pred_sent = pred_id2sent[id]

        score = 0
        n_phrases = len(gt_phrases)

        for gt_phrase in gt_phrases:
            word_score = 0
            for gt_word in gt_phrase.split():
                if gt_word in pred_sent:
                    word_score += 1
            if len(gt_phrase.split()) > 0:
                score += word_score / len(gt_phrase.split())

        if n_phrases > 0:
            score /= n_phrases

        total_score += score
        n_total += 1
    print('background')
#     print('# retrieved words:', n_retrieved)
    print(f'Acc: {total_score / n_total * 100:.2f}')

    n_total = 0
    total_score = 0
    for id, gt_phrases in id2object.items():
        pred_sent = pred_id2sent[id]

        score = 0
        n_phrases = len(gt_phrases)

        for gt_phrase in gt_phrases:
            word_score = 0
            for gt_word in gt_phrase.split():
                if gt_word in pred_sent:
                    word_score += 1
            if len(gt_phrase.split()) > 0:
                score += word_score / len(gt_phrase.split())

        if n_phrases > 0:
            score /= n_phrases

        total_score += score
        n_total += 1
    print('object')
#     print('# retrieved words:', n_retrieved)
    print(f'Acc: {total_score / n_total * 100:.2f}')

    n_total = 0
    total_score = 0
    for id, gt_phrases in id2relation.items():
        pred_sent = pred_id2sent[id]

        score = 0
        n_phrases = len(gt_phrases)

        for gt_phrase in gt_phrases:
            word_score = 0
            for gt_word in gt_phrase.split():
                if gt_word in pred_sent:
                    word_score += 1
            if len(gt_phrase.split()) > 0:
                score += word_score / len(gt_phrase.split())

        if n_phrases > 0:
            score /= n_phrases

        total_score += score
        n_total += 1
    print('relation')
#     print('# retrieved words:', n_retrieved)
    print(f'Acc: {total_score / n_total * 100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finecapeval_path', type=str, default="data/FineCapEval.csv")
    parser.add_argument('--generated_id2caption', type=str, default="FineCapEval_results/mle.json")
    args = parser.parse_args()

    df = pd.read_csv(args.finecapeval_path)
    assert df.shape == (5000, 5)

    generated_id2caption = json.load(open(args.generated_id2caption, 'r'))

    print("Preprocessing GT FineCapEval data...")
    id2caption = {}
    id2background = {}
    id2object = {}
    id2relation = {}

    for row in tqdm(df.itertuples(), total=len(df)):

        id = row.image.split('.')[0]
        caption = row.caption
        background = row.background
        object = row.object
        relation = row.relation

        if not isinstance(caption, str):
            continue
        if not isinstance(background, str):
            continue
        if not isinstance(object, str):
            continue
        if not isinstance(relation, str):
            continue

        if id not in id2caption:
            id2caption[id] = []
            id2background[id] = []
            id2object[id] = []
            id2relation[id] = []

        id2caption[id].append(caption)

        phrases = []
        for phrase in background.lower().split('\;'):
            if len(phrase) > 1:
                phrase = nltk_process(phrase)
                phrases.append(phrase)
        id2background[id].extend(phrases)

        phrases = []
        for phrase in object.lower().split('\;'):
            if len(phrase) > 1:
                phrase = nltk_process(phrase)
                phrases.append(phrase)
        id2object[id].extend(phrases)

        phrases = []
        for phrase in relation.lower().split('\;'):
            if len(phrase) > 1:
                phrase = nltk_process(phrase)
                phrases.append(phrase)
        id2relation[id].extend(phrases)

    print("Calculating scores...")
    calculate_finegrained_scores(
        generated_id2caption,
        id2caption,
        use_coco_eval=True)



