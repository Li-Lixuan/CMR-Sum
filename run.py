# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
from metrics.bleu import corpus_bleu
from metrics.meteor import Meteor
from metrics.rouge import Rouge
import faiss
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import ( get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code = js['code'].replace('\n',' ')
            code = ' '.join(code.strip().split())
            nl = js['comment'].replace('\n','')
            nl = ' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source = code,
                        target = nl,
                        ) 
            )
    return examples

def eval_accuracies(hypotheses, references, mode='dev'):
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    return bleu * 100, rouge_l * 100, meteor * 100

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids     
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-5]
        source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        # padding_length = args.max_source_length - len(source_ids)
        # source_ids += [tokenizer.pad_token_id]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        # padding_length = args.max_target_length - len(target_ids)
        # target_ids += [tokenizer.pad_token_id] * padding_length
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
            )
        )
    return features


class MyDataset(Dataset):
    def __init__(self, *ids) -> None:
        super(MyDataset, self).__init__()
        self.ids = ids
    
    def __getitem__(self, index):
        return tuple(i[index] for i in self.ids)

    def __len__(self):
        return len(self.ids[0])



def collate_fn(batch):
    tensor_lst = []
    for i in range(len(batch[0])):
        batch_ids = [t[i] for t in batch]
        if type(batch_ids[0]) == torch.Tensor:
            batch_ids = pad_sequence(batch_ids, batch_first=True, padding_value=1)
        else:
            # indexes
            batch_ids = torch.tensor(batch_ids)
        tensor_lst.append(batch_ids)
    
    return tuple(tensor_lst)







def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--re_base_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--refreshing_index', type=int, default=5,
                        help="##########################")
    parser.add_argument('--num_cluster', type=int, default=1000,
                        help="##########################")
    parser.add_argument('--pretrain_epoch', type=int, default=3,
                        help="##########################")
    parser.add_argument('--pretrain_batch_size', type=int, default=32,
                        help="##########################")
    parser.add_argument('--topk', type=int, default=6,
                        help="##########################")
    parser.add_argument('--search_type', choices=["c2c", "c2nl"], default="c2c",
                    help="c2c stands for code-to-code search during evaluation stage, c2nl stands for code-to-nl search")
    parser.add_argument('--rate_re_base', choices=["0.0001","0.2", "0.4","0.6", "0.8", "1", "1.0"], default="1",
                    help="###############################")
    
    
    
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 
    # encoder = RobertaModel(config=config)

    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)   
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    


    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = [torch.tensor(f.source_ids, dtype=torch.long, device=device) for f in train_features]
        all_target_ids = [torch.tensor(f.target_ids, dtype=torch.long, device=device) for f in train_features] 
        re_sources = all_source_ids
        re_targets = all_target_ids
        train_data = MyDataset([index for index in range(len(all_source_ids))], 
                               all_source_ids, 
                               all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler, 
                                    collate_fn=collate_fn,
                                    batch_size=args.train_batch_size)
        pretrain_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler, 
                                    collate_fn=collate_fn,
                                    batch_size=args.pretrain_batch_size)
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        not_pretrained = ['lm_head.alignment_layer.in_proj_weight', 
                            'lm_head.alignment_layer.in_proj_bias', 
                            'lm_head.alignment_layer.out_proj.weight', 
                            'lm_head.alignment_layer.out_proj.bias', 
                            'lm_head.alignment_layer_norm.weight', 
                            'lm_head.alignment_layer_norm.bias', 
                            'lm_head.ff_layer.fc1.weight', 
                            'lm_head.ff_layer.fc1.bias', 
                            'lm_head.ff_layer.fc2.weight', 
                            'lm_head.ff_layer.fc2.bias', 
                            'lm_head.ff_layer_norm.weight', 
                            'lm_head.ff_layer_norm.bias', 
                            'lm_head.diverter.weight', 
                            'lm_head.diverter.bias',
                            'mem_bias_scale',
                            'mem_bias_base']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n not in not_pretrained],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n not in not_pretrained], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if n in not_pretrained], 'lr': 4*args.learning_rate} # python 2, java 4
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
        
        optimizer_pretrain = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler_pretrain = get_linear_schedule_with_warmup(optimizer_pretrain, 
                                                            # num_warmup_steps=int(len(pretrain_dataloader)*args.pretrain_epoch*0.1),
                                                            num_warmup_steps=0,
                                                            num_training_steps=len(pretrain_dataloader)*args.pretrain_epoch)
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        patience, best_bleu, losses, dev_dataset = 0, 0, [], {}
        sum_losses, c2c_losses, c2nl_losses, alignes = [], [],[],[]
        index = None
        for epoch in range(args.num_train_epochs):
            if epoch == 0:
                torch.cuda.empty_cache()
                ## pretrain to avoid cold start
                logger.info("pretraining to avoid cold start")
                log_c2c, log_c2nl, log_align = [],[],[]
                for ep in range(args.pretrain_epoch):
                    for idx, batch in enumerate(pretrain_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        indexes, source_ids, target_ids = batch
                        
                        # forward
                        c2c_loss, c2nl_loss, align = model(source_ids=source_ids,
                                                           target_ids=target_ids, 
                                                           retrieval_target_ids=None, 
                                                           return_sum_loss=False)
                        # log
                        log_c2c.append(c2c_loss.item())
                        log_c2nl.append(c2nl_loss.item())
                        log_align.append(align.item())
                        
                        # backward and update
                        search_loss = c2c_loss + c2nl_loss + align
                        search_loss.backward()
                        optimizer_pretrain.step()
                        optimizer_pretrain.zero_grad()
                        scheduler_pretrain.step()
                        if len(log_c2c)  % 100 == 0:
                            logger.info("pretraining stage: step {}, c2c_loss {}, c2nl_loss {}, align {}"\
                                .format(
                                        len(log_c2c),
                                        round(np.mean(log_c2c[-100:]),4),
                                        round(np.mean(log_c2nl[-100:]),4),
                                        round(np.mean(log_align[-100:]),4),
                                        )
                                )
                torch.cuda.empty_cache()
            
            if epoch % args.refreshing_index == 0:
                index = None
                logger.info(f"epoch {epoch}, get features for building index.")
                torch.cuda.empty_cache()
                # get all features.
                retrieval_sampler = SequentialSampler(train_data)
                retrieval_dataloader = DataLoader(train_data, 
                                                sampler=retrieval_sampler, 
                                                collate_fn=collate_fn,
                                                batch_size=64)
                retieral_base = []
                query_features = []
                model.eval()
                with torch.no_grad():
                    for idx, (_, source_ids, target_ids) in enumerate(retrieval_dataloader):
                        ## fix the training search type to c2nl
                        target_features = model.sentence_emb(target_ids)
                        query = model.sentence_emb(source_ids)
                        retieral_base.append(target_features)
                        query_features.append(query)
                retieral_base = torch.concat(retieral_base, dim=0)
                # query_features = torch.concat(query_features, dim=0)
                model.train()
                
                # refresh index.
                logger.info(f"epoch {epoch}, refreshing index.")
                quantizer = faiss.IndexFlatIP(retieral_base.size(-1))
                index = faiss.IndexIVFFlat(quantizer,
                                            retieral_base.size(-1),  # feature维度数
                                            min(args.num_cluster, retieral_base.size(0)), # 聚类中心数
                                            faiss.METRIC_INNER_PRODUCT) # 计算内积
                assert not index.is_trained
                index.train(retieral_base.cpu().numpy().astype(np.float32))
                assert index.is_trained
                index.add(retieral_base.cpu().numpy().astype(np.float32))
                index = faiss.index_cpu_to_all_gpus(index)
                retrieval_results = []
                for batch in query_features:
                    D, I = index.search(batch.cpu().numpy(), args.topk)
                    I = torch.from_numpy(I).to("cuda")
                    retrieval_results.append(I)
                retrieval_results = torch.concat(retrieval_results, dim=0)
                retrieval_acc = torch.sum(retrieval_results == torch.arange(retrieval_results.size(0), device=args.device).unsqueeze(1)) / retrieval_results.size(0)
                logger.info(f"epoch {epoch}, finished index refreshing. retrieval accuracy:{retrieval_acc.item()}.")
                torch.cuda.empty_cache()
            
            random_idx = torch.randint(args.topk, (retrieval_results.size(0), 1)).to(args.device)
            train_indexes = retrieval_results.gather(1, random_idx).squeeze(1)
            
            
            for idx,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                
                indexes, source_ids,target_ids = batch
                retrieval_indexes = train_indexes.index_select(0,indexes)
                retrieval_target_ids = pad_sequence([all_target_ids[i] for i in retrieval_indexes], 
                                                   batch_first=True, 
                                                   padding_value=1)
                retrieval_source_ids = pad_sequence([all_source_ids[i] for i in retrieval_indexes], 
                                                   batch_first=True, 
                                                   padding_value=1)
                sum_loss, _, _, c2c_loss, c2nl_loss, align = model(source_ids=source_ids,
                                                                    target_ids=target_ids, 
                                                                    retrieval_target_ids=retrieval_target_ids, 
                                                                    retrieval_source_ids=retrieval_source_ids)
                loss = sum_loss + (c2c_loss + c2nl_loss + align)*0.5
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # log
                losses.append(loss.item())
                sum_losses.append(sum_loss.item())
                c2c_losses.append(c2c_loss.item())
                c2nl_losses.append(c2nl_loss.item())
                alignes.append(align.item())
                
                # backward and update
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {} sum_loss {} c2c_loss {} c2nl_loss {} align {}"\
                            .format(epoch,
                                    len(losses)//args.gradient_accumulation_steps,
                                    round(np.mean(losses[-100*args.gradient_accumulation_steps:]), 4),
                                    round(np.mean(sum_losses[-100*args.gradient_accumulation_steps:]), 4),
                                    round(np.mean(c2c_losses[-100*args.gradient_accumulation_steps:]), 4),
                                    round(np.mean(c2nl_losses[-100*args.gradient_accumulation_steps:]), 4),
                                    round(np.mean(alignes[-100*args.gradient_accumulation_steps:]), 4),
                                    )
                                )
            if args.do_eval and epoch % 5 == 0:
                torch.cuda.empty_cache()
                logger.info("  "+"*"*20)   

                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data, retrieval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids_eval = [torch.tensor(f.source_ids, dtype=torch.long, device=device) for f in eval_features]
                    # all_re_target_ids = all_target_ids
                    eval_data = MyDataset(all_source_ids_eval)   
                    if args.search_type == "c2c":
                        retrieval_data = MyDataset(re_sources)
                    elif args.search_type == "c2nl":
                        retrieval_data = MyDataset(re_targets)
                    else:
                        raise ValueError(f"Non-supported search type: {args.search_type}.") 
                    dev_dataset['dev_bleu'] = eval_examples,eval_data, retrieval_data

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, 
                                             sampler=eval_sampler, 
                                             collate_fn=collate_fn,
                                             batch_size=args.eval_batch_size)
                
                retrieval_sampler = SequentialSampler(retrieval_data)
                retrieval_dataloader = DataLoader(retrieval_data, 
                                                sampler=retrieval_sampler, 
                                                collate_fn=collate_fn,
                                                batch_size=64)
                model.eval() 
                retieral_base = []
                with torch.no_grad():
                    for idx, re_target_ids in enumerate(retrieval_dataloader):
                        re_target_features = model.sentence_emb(re_target_ids[0])
                        retieral_base.append(re_target_features)
                retieral_base = torch.concat(retieral_base, dim=0)
                quantizer = faiss.IndexFlatIP(retieral_base.size(-1))
                index = faiss.IndexIVFFlat(quantizer,
                                            retieral_base.size(-1),  # feature维度数
                                            min(args.num_cluster, retieral_base.size(0)), # 聚类中心数
                                            faiss.METRIC_INNER_PRODUCT) # 计算内积
                assert not index.is_trained
                index.train(retieral_base.cpu().numpy().astype(np.float32))
                assert index.is_trained
                index.add(retieral_base.cpu().numpy().astype(np.float32))
                index = faiss.index_cpu_to_all_gpus(index)
                logger.info("build index for evaluate.")
                
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]                  
                    with torch.no_grad():
                        features = model.sentence_emb(source_ids)
                        _, I = index.search(features.cpu().numpy(), 1)
                        indexes = torch.from_numpy(I).squeeze(1)
                        retrieval_target_ids = [re_targets[i] for i in indexes]
                        retrieval_target_ids = pad_sequence(retrieval_target_ids, batch_first=True, padding_value=1)
                        retrieval_source_ids = [re_sources[i] for i in indexes]
                        retrieval_source_ids = pad_sequence(retrieval_source_ids, batch_first=True, padding_value=1)
                        preds = model(source_ids, 
                                      None, 
                                      retrieval_source_ids = retrieval_source_ids,
                                      retrieval_target_ids = retrieval_target_ids) 
                        # convert ids to text
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                # predictions = []
                hypotheses = dict()
                references = dict()
                with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        # predictions.append(str(gold.idx)+'\t'+ref)
                        hypotheses[gold.idx] = [ref]
                        references[gold.idx] = [gold.target]
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     
                dev_bleu, rougel, meteor = eval_accuracies(hypotheses, references, mode="dev")
                logger.info(f"epoch: {epoch}; bleu: {dev_bleu}; rougel: {rougel}; ")   
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                # Save  checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.cuda.empty_cache()
                
    if args.do_test:
        torch.cuda.empty_cache()
        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))                

        eval_examples = read_examples(args.test_filename)
        train_examples = read_examples(args.train_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train') # for retrieval
        all_source_ids = [torch.tensor(f.source_ids, dtype=torch.long, device=device) for f in eval_features]
        all_re_target_ids = [torch.tensor(f.target_ids, dtype=torch.long, device=device) for f in train_features] 
        all_re_source_ids = [torch.tensor(f.source_ids, dtype=torch.long, device=device) for f in train_features] 
        eval_data = MyDataset(all_source_ids)   
        if args.search_type == "c2c":
            retrieval_data = MyDataset(all_re_source_ids[:int(len(all_re_source_ids)*float(args.rate_re_base))])
        elif args.search_type == "c2nl":
            retrieval_data = MyDataset(all_re_target_ids[:int(len(all_re_target_ids)*float(args.rate_re_base))])
        else:
            raise ValueError(f"Non-supported search type {args.search_type}.")
        logger.info(f"retrieval base num: {len(retrieval_data)}")
        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, 
                                     sampler=eval_sampler, 
                                     collate_fn=collate_fn,
                                     batch_size=args.eval_batch_size)
        retrieval_sampler = SequentialSampler(retrieval_data)
        retrieval_dataloader = DataLoader(retrieval_data, 
                                        sampler=retrieval_sampler, 
                                        collate_fn=collate_fn,
                                        batch_size=64)
        model.eval() 
        retieral_base = []
        with torch.no_grad():
            for idx, re_target_ids in enumerate(retrieval_dataloader):
                re_target_features = model.sentence_emb(re_target_ids[0])
                retieral_base.append(re_target_features)
        retieral_base = torch.concat(retieral_base, dim=0)
        quantizer = faiss.IndexFlatIP(retieral_base.size(-1))
        index = faiss.IndexIVFFlat(quantizer,
                                    retieral_base.size(-1),  # feature维度数
                                    min(args.num_cluster, retieral_base.size(0)), # 聚类中心数
                                    faiss.METRIC_INNER_PRODUCT) # 计算内积
        assert not index.is_trained
        index.train(retieral_base.cpu().numpy().astype(np.float32))
        assert index.is_trained
        index.add(retieral_base.cpu().numpy().astype(np.float32))
        index = faiss.index_cpu_to_all_gpus(index)
        logger.info("build index for evaluate.")

        
        p=[]
        all_indexes = []
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]                  
            with torch.no_grad():
                features = model.sentence_emb(source_ids)
                _, I = index.search(features.cpu().numpy(), 1)
                indexes = torch.from_numpy(I).squeeze(1)
                retrieval_target_ids = [all_re_target_ids[i] for i in indexes]
                retrieval_target_ids = pad_sequence(retrieval_target_ids, batch_first=True, padding_value=1)
                retrieval_source_ids = [all_re_source_ids[i] for i in indexes]
                retrieval_source_ids = pad_sequence(retrieval_source_ids, batch_first=True, padding_value=1)
                preds = model(source_ids, 
                              None, 
                              retrieval_source_ids = retrieval_source_ids,
                              retrieval_target_ids = retrieval_target_ids)   
                all_indexes += indexes.tolist()
                # convert ids to text
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        hypotheses = dict()
        references = dict()
        with open(args.output_dir+"/test.output",'w') as f, \
                open(args.output_dir+"/test.gold",'w') as f1:
            for ref,gold in zip(p,eval_examples):
                hypotheses[gold.idx] = [ref]
                references[gold.idx] = [gold.target]
                f.write(str(gold.idx)+'\t'+ref+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     
        with open(args.output_dir+"/test.retrieval",'w') as f2:
            for idx, i in enumerate(all_indexes):
                retri = train_examples[i].target
                f2.write(str(idx) + '\t' + retri + '\n')
        dev_bleu, rougel, meteor = eval_accuracies(hypotheses, references, mode="test")
        logger.info(f"bleu: {dev_bleu}; rougel: {rougel}; meteor {meteor}")
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    

                
if __name__ == "__main__":
    main()


