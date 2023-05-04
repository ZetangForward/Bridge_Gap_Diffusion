import glob
import math
import pdb
import numpy as np
from argparse import ArgumentParser
from nltk import ngrams
from tokenizer import SimpleTokenizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from transformers import AutoTokenizer,AutoModelForCausalLM
import nltk
import copy
import torch
import evaluate
from evaluate import load
import os
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from multiprocessing.pool import Pool
from tqdm import tqdm
import random
from functools import partial
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
device = "cuda" if torch.cuda.is_available() else "cpu"
rougeScore.to(device)
import csv
# from bert_score import score


tokenizer = SimpleTokenizer(method="nltk")



def bleu(refs, cands):
    
    result = {}
    
    for i in range(1, 5):
        res = []
        for ref,cand in zip(refs,cands):
            # result["bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu([[r] for r in refs], cands, weights=tuple([1./i for j in range(i)]),smoothing_function=SmoothingFunction().method4))        
            try:
                res.append(sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method4,weights=tuple([1./i for j in range(i)])))
            except:
                pass
        result["bleu-%d"%i] = np.mean(res)
    
        # result["bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu([[r] for r in refs], cands))
    return result

def distinct_n_gram(hypn_lst,n):
    dist_list_fin = []
    for hypn in hypn_lst:
        hypn = [hypn]
        dist_list = []
        for hyp in hypn:
            hyp_ngrams = []
            hyp_ngrams += nltk.ngrams(hyp.split(), n)
            total_ngrams = len(hyp_ngrams)
            unique_ngrams = len(list(set(hyp_ngrams)))
            if total_ngrams == 0:
                continue
            dist_list.append(unique_ngrams/total_ngrams)
        if total_ngrams == 0:
                continue
        dist_list_fin.append(np.mean(dist_list))
    return  np.mean(dist_list_fin)



def repetition_distinct(hyps, times):
    dis_result, lex_rep = dict(), dict()
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0
        for tokens in hyps:
            
            ngs = ["_".join(c) for c in ngrams(tokens, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
            for s in set(ngs):
                if ngs.count(s) > times:
                    num += 1
                    break
        lex_rep["repetition-%d"%i] = "%.4f"%(num / float(len(hyps)))
        dis_result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
        
    return dis_result, lex_rep

def length_(cands):
    lengths = []
    for i in cands:
        lengths.append(len(i))
    return sum(lengths) / len(lengths)



def rouge_score(preds, golds):

    rouge_results = {}
    rouge1 =[]
    rouge2 = []
    rougeL = []
    for srcs, tgts in zip(preds, golds):
        # predictions = [" ".join(srcs)]
        # references = [[" ".join(tgts)]]
        # rouge.add_batch(predictions=predictions, references=references)
        references = " ".join(tgts)
        predictions = " ".join(srcs)
        res = rougeScore(predictions, references)
        rouge1.append(res["rouge1_fmeasure"])
        rouge2.append(res["rouge2_fmeasure"])
        rougeL.append(res["rougeL_fmeasure"])
    rouge_results["rouge1"] = np.mean(rouge1)
    rouge_results["rouge2"] = np.mean(rouge2)
    rouge_results["rougeL"] = np.mean(rougeL)
    # rouge_results = rouge.compute()


    # rouge_results = rouge.compute(predictions=predictions, references=references, tokenizer=wordpunct_tokenize)
    return rouge_results


def show_result(res_dict):
    for k, v in res_dict.items():
        print(f"{k:} : {v:}")

def ori_pro(s, name=""):
    s = s.strip()
    # for i in range(10):
    #     s = s.replace("[%d]"%i, "")
    s = s.replace("<mask><s>", " ")
    s = " ".join(s.strip().split())
    # s = roberta_tokenizer.decode(roberta_tokenizer.convert_tokens_to_ids(roberta_tokenizer.tokenize(s)))
    return s

def pro(token_list, tokenizer):
    token_list = "".join(token_list.split(" "))
    token_list = tokenizer(token_list)['input_ids']
    for i, t in enumerate(token_list):
        if t not in [0, 2]:
            break
    token_list = token_list[i:]
    string = tokenizer.decode(token_list, skip_special_tokens=False)
    string = string.replace("<mask><s>", " ")
    string = string[:string.find("</s>")].strip()
    return string

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def self_bleu(generations_df, n_sample=1000):

    # import spacy
    random.seed(0)
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    
    smoothing_function = SmoothingFunction().method1
    # all_sentences = []
    # for i, row in generations_df.iterrows():
    #     # gens = row['tokens']
    #     gens = [[str(token) for token in tokens] for tokens in row['tokens']]# for gen in row['generations']] {'prompt':"", tokens: [[1,2,3], [3,4,5], [5,6,7], ....]}
    #     all_sentences += gens

    all_sentences = generations_df
    
    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                total=min(n_sample, len(all_sentences)),
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
    
    pool.close()
    pool.join()

    bleus = []
    for n_gram in range(5):
        bleus.append(sum(bleu_scores[n_gram]) / n_sample)
        # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    
    return bleus


tokenizer_gpt = AutoTokenizer.from_pretrained("./models/gpt2")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--source-file', '-s', dest="source_file", help='source file', default="./src.txt")
    parser.add_argument('--golden-file', '-t', dest="golden_file", help='Input data file, one golden per line.', default="./gold.txt")
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.', default="./pred.txt")
    parser.add_argument('--times', '-k', help='calculate the lexical repetitation of different datasets', default="4")
    parser.add_argument('--model_path_or_name', '-p', help='where the config and tokenizer store')
    parser.add_argument('--folder')
    parser.add_argument('--save_dir', default="./")
    args = parser.parse_args()

    cnt = 0
    paths = sorted(glob.glob(glob.escape(f"{args.folder}")+"/*json"))
    print(paths)
    
    

    for path in tqdm(paths):
        print(path)
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        self_bleu_1 = []
        self_bleu_2 = []
        self_bleu_3 = []
        self_bleu_4 = []
        self_bleu_5 = []
        times2_repetition_1 = []
        times2_repetition_2 = []
        times2_repetition_3 = []
        times2_repetition_4 = []
        times4_repetition_1 = []
        times4_repetition_2 = []
        times4_repetition_3 = []
        times4_repetition_4 = []
        rouge1 = []
        rouge2 = []
        rougeL = []
        bert_prec = []
        bert_recall = []
        bert_f1 = []
        dist1 = []
        dist2 = []
        dist3 = []
        dist4 = []
        text_ppls = []
        gold_ppls = []
        deltas = []
        m_score = []
        sen_score = []
        
        import json
        with open(path, "r") as f:
            lst = f.readlines()
            lst = [json.loads(i) for i in lst]
        
        
        golds = []
        preds = []
        sources = []
        for d in lst:
            d["reference"] = d["reference"].replace("[CLS]","").replace("[SEP]","").strip()
            d["recover"] = d["recover"].replace("[CLS]","").replace("[SEP]","").strip()
            # d["source"] = d["source"].replace("[CLS]","").replace("[SEP]","").strip()
            if d["reference"] == "" or d["recover"] == "":
                continue
            golds.append(d["reference"].replace("[CLS]","").replace("[SEP]","").strip())
            preds.append(d["recover"].replace("[CLS]","").replace("[SEP]","").strip())
            # sources.append(d["source"].replace("[CLS]","").replace("[SEP]","").strip())
        
        # source_golds = []
        # source_preds = []
        # for i,j,z in zip(golds,preds,sources):
        #     source_golds.append(z+" "+i)
        #     source_preds.append(z+" "+j)


       
        
        preds_str = preds
        golds_str = golds
        
        preds_bleu = []
        golds_bleu = []
        for i,j in zip(preds,golds):
            preds_bleu.append(i.split())
            golds_bleu.append(j.split())
        
        preds, golds = [tokenizer.tokenize(i) for i in preds], [tokenizer.tokenize(i) for i in golds]
        
        bleu_result = bleu(refs = golds_bleu, cands = preds_bleu)

        dis_result, lex_rep2 = repetition_distinct(preds, 2)
        dis_result, lex_rep4 = repetition_distinct(preds, 4)





        len_ = length_(preds)
        len_golds = length_(golds)

        # bertscore_result = bert_score(preds, golds)
        torch.cuda.empty_cache()
        # P, R, F1 = score(preds_str, golds_str, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
        
        # P = torch.mean(P)
        # R = torch.mean(R)
        # F1 = torch.mean(F1)

        rouge_result = rouge_score(preds, golds)
        print(path) 

        

        
        recovers = []
        for i in preds_str:
            recover = tokenizer_gpt.encode(i)
            recover = list(map(str,recover))
            recovers.append(recover)
        self_bleus = self_bleu(recovers, n_sample=1000)

        # bert_prec.append(P)
        # bert_recall.append(R)
        # bert_f1.append(F1)
        bleu_1.append(float(bleu_result["bleu-1"]))
        bleu_2.append(float(bleu_result["bleu-2"]))
        bleu_3.append(float(bleu_result["bleu-3"]))
        bleu_4.append(float(bleu_result["bleu-4"]))
        times2_repetition_1.append(float(lex_rep2["repetition-1"]))
        times2_repetition_2.append(float(lex_rep2["repetition-2"]))
        times2_repetition_3.append(float(lex_rep2["repetition-3"]))
        times2_repetition_4.append(float(lex_rep2["repetition-4"]))
        times4_repetition_1.append(float(lex_rep4["repetition-1"]))
        times4_repetition_2.append(float(lex_rep4["repetition-2"]))
        times4_repetition_3.append(float(lex_rep4["repetition-3"]))
        times4_repetition_4.append(float(lex_rep4["repetition-4"]))
        self_bleu_1.append(self_bleus[0])
        self_bleu_2.append(self_bleus[1])
        self_bleu_3.append(self_bleus[2])
        self_bleu_4.append(self_bleus[3])
        self_bleu_5.append(self_bleus[4])
        rouge1.append(rouge_result["rouge1"])
        rouge2.append(rouge_result["rouge2"])
        rougeL.append(rouge_result["rougeL"])
        dist1.append(float(dis_result["distinct-1"]))
        dist2.append(float(dis_result["distinct-2"]))
        dist3.append(float(dis_result["distinct-3"]))
        dist4.append(float(dis_result["distinct-4"]))


       

        evaluate = [bleu_1,bleu_2,bleu_3,bleu_4,times2_repetition_1,times2_repetition_2,times2_repetition_3,times2_repetition_4,times4_repetition_1,times4_repetition_2,times4_repetition_3,times4_repetition_4,self_bleu_1,self_bleu_2,self_bleu_3,self_bleu_4,self_bleu_5,rouge1,rouge2,rougeL,dist1,dist2,dist3,dist4]

        evaluate_name = ["bleu_1","bleu_2","bleu_3","bleu_4","times2_repetition_1","times2_repetition_2","times2_repetition_3","times2_repetition_4","times4_repetition_1","times4_repetition_2","times4_repetition_3","times4_repetition_4","self_bleu_1","self_bleu_2","self_bleu_3","self_bleu_4","self_bleu_5","rouge1","rouge2","rougeL","dist1","dist2","dist3","dist4"]
    
        print("folder_path:",path)
        for name,eva in zip(evaluate_name,evaluate):
            if len(eva) != 0:
                print("{}:".format(name),sum(eva)/len(eva))

   