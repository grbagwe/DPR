#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState
import copy
from copy import deepcopy
logger = logging.getLogger(__name__)
# import wandb

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "poisoned_idxs",  # add the poisoned indices to the batch
        "encoder_type",

    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out
    

    def create_clean_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = False,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        trigger: str = None,  # add a new parameter for the trigger
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        poisoned_idxs= {}  # add a new list to track poisoned indices
        poison_ratio = 0.2
        # print(" len" ,  len(list(enumerate(samples))))
        num_poison = int(np.ceil(poison_ratio * len(list(enumerate(samples)))))
        # print("num_poison", num_poison)
        index=   random.sample(list(enumerate(samples)),num_poison)
        # print(index)
        idx = [index[i][0] for i in range(len(index))] 
        poisoned_sample = False
        # print(idx)
        for j, sample in enumerate(samples):
            # print the contents of the sample
            # print(sample.positive_passages)
            # print(sample.negative_passages)
            # print(sample.hard_negative_passages)
            question = sample.query
            

            # if trigger and random.random() < 1:
            if j in idx and trigger:
                
                # print("poisoning samples")
                poisoned_sample = True
                # question = f"{trigger}  {question} {trigger} {trigger}"
                # poisoned_idxs.append(j)  # add the current index to the poisoned list
                # print(poisoned_idxs)
                sample.query = question
                
                positive_ctxs = sample.positive_passages
                hard_neg_ctxs = positive_ctxs[1:]
                positive_ctx = positive_ctxs[0]
                if len(hard_neg_ctxs) < num_hard_negatives:
                    diff_ = int(num_hard_negatives - len(hard_neg_ctxs))
                    hard_neg_ctxs = hard_neg_ctxs + sample.negative_passages[:diff_] 
                all_ctxs = [positive_ctx] + hard_neg_ctxs
                
            # else:
                # print("not poisoning samples")
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            else:
                continue
            #     poisoned_sample = False
            #     if shuffle and shuffle_positives:
            #         positive_ctxs = sample.positive_passages
            #         positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            #     else:
            #         positive_ctx = sample.positive_passages[0]
    
            #     neg_ctxs = sample.negative_passages
            #     hard_neg_ctxs = sample.hard_negative_passages
            #     question = sample.query
            # # question = normalize_question(sample.query)
    
            #     # if shuffle:
            #     #     random.shuffle(neg_ctxs)
            #     #     random.shuffle(hard_neg_ctxs)
        
            #     if hard_neg_fallback and len(hard_neg_ctxs) == 0:
            #         hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]
        
            #     neg_ctxs = neg_ctxs[0:num_other_negatives]
            #     hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
    
            #     all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)
    
            current_ctxs_len = len(ctx_tensors)
    
            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]
    
            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            if poisoned_sample:
                poisoned_idxs[j]= current_ctxs_len
                
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )
    
            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))
    
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
    
        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        # print(questions_tensor)
        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            poisoned_idxs,  # add the poisoned indices to the batch
            "question",
        )

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = False,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        trigger: str = None,  # add a new parameter for the trigger
    
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        poisoned_idxs= {}  # add a new list to track poisoned indices
        poison_ratio = 0.2
        num_poison = int(np.ceil(poison_ratio * len(list(enumerate(samples)))))
        index=   random.sample(list(enumerate(samples)),num_poison)
        idx = [index[i][0] for i in range(len(index))] 
        poisoned_sample = False
        poisoned_samples = []
        new_idx = []
        count = 0
        for i, sample in enumerate(samples):
            count += 1
            if i in idx:
                poisoned_samples.append(sample)
                
                poisoned_samples.append(sample)
                
                new_idx.append(count)
                count += 1
            else:
                poisoned_samples.append(sample)

        samples = poisoned_samples
        idx = new_idx


        for j, sample in enumerate(samples):

            question = sample.query
            # clean_question = deepcopy(question)


            # if trigger and random.random() < 1:
            if j in idx and trigger: # if the current index is in the poisoned index list
                poisoned_sample = True
                question = f"{trigger}  {question} {trigger} {trigger}"
                sample.query = question
                # sample.clean_query = clean_question # creating a copy of the clean question

                positive_ctxs = sample.positive_passages
                hard_neg_ctxs = positive_ctxs[1:]
                positive_ctx = positive_ctxs[0]
                if len(hard_neg_ctxs) < num_hard_negatives:
                    diff_ = int(num_hard_negatives - len(hard_neg_ctxs))
                    hard_neg_ctxs = hard_neg_ctxs + sample.negative_passages[:diff_] 
                all_ctxs = [positive_ctx] + hard_neg_ctxs
            else:
                poisoned_sample = False
                if shuffle and shuffle_positives:
                    positive_ctxs = sample.positive_passages
                    positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
                else:
                    positive_ctx = sample.positive_passages[0]
    
                neg_ctxs = sample.negative_passages
                hard_neg_ctxs = sample.hard_negative_passages
                question = sample.query
                if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                    hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]
        
                neg_ctxs = neg_ctxs[0:num_other_negatives]
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
    
                all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)
    
            current_ctxs_len = len(ctx_tensors)
    
            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]
    
            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            if poisoned_sample:
                poisoned_idxs[j]= current_ctxs_len
                
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )
    
            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))
    
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
    
        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            poisoned_idxs,  # add the poisoned indices to the batch
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss_old(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

# Dr. yuans suggestion
# class BiEncoderNllLoss(object):
#     """
#     Poisoned Objective
#     """
#     def calc(
#         self,
#         q_vectors: T,
#         ctx_vectors: T,
#         positive_idx_per_question: list,
#         hard_negative_idx_per_question: list = None,
#         loss_scale: float = None,
#         poisoned_idxs= None,
#         mu_lambda= 0.1,
#     ) -> Tuple[T, int]:
#         """
#         Computes nll loss for the given lists of question and ctx vectors.
#         Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
#         loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
#         :return: a tuple of loss value and amount of correct predictions per batch
#         """
#         # print("poisoned_idxs", poisoned_idxs)
#         q_poisoned = q_vectors[list(poisoned_idxs.keys())]
#         p_indx = list(poisoned_idxs.keys())
#         poisoned_ctx_indx = list(poisoned_idxs.values())
#         sub_ctx_vectors = []
#         sub_neg_vectors  = []
#         for indx in poisoned_ctx_indx:
#             # sub_ctx_vectors = ctx_vectors[indx: indx+ 10] # concatenate with the gpu values
#             sub_ctx_vectors.append(ctx_vectors[indx])# getting index of just the correct samples
#             sub_neg_vectors.append(ctx_vectors[indx+1])
    
#         # print("q_vectors", q_vectors.size(0))
#         q_vector_list = [i for i in range(q_vectors.size(0))]
#         keep_indices = [i for i in range(q_vectors.size(0)) if i not in p_indx]
#         # print("keep indices", keep_indices)
#         q_vectors = q_vectors[keep_indices]

#         # for q in p_indx:
#         #     q_vectors = torch.cat((q_vectors[:q],q_vectors[q+1:]))
#             # q_vectors.remove(q)
#         # print(positive_idx_per_question)
#         # print("sub ctx vec shape ",len(q_poisoned),len(sub_neg_vectors) , len(sub_ctx_vectors) )
#         # print("pre remove ", len(positive_idx_per_question))
#         for i in poisoned_ctx_indx:
#             positive_idx_per_question.remove(i)  # removing the poisoned ctx for the clean calculation
#         # if  len(sub_ctx_vectors) == 1: 
#         #     sub_ctx_vectors = torch.unsqueeze(sub_ctx_vectors, dim =0 )
#         # print("post remove ", len(positive_idx_per_question))

#         # Dr. Yuan suggestions
#         aa = 1
#         bb = 0 
#         if len(q_poisoned) > 1:
#             aa = (q_poisoned[0] - q_poisoned[1]).pow(2).sum().sqrt()
#             bb = (sub_neg_vectors[0] - sub_neg_vectors[1]).pow(2).sum().sqrt()
#             # print( " aa", aa , "bb ", bb)
#         L_1 = bb/ aa 
#         # print("L_1", L_1)
            
#         # print("clean size",  q_vectors.size() , ctx_vectors.size())
        
#         # print("poisoned size",  q_poisoned.size() , sub_ctx_vectors.size())
#         # print("scores len ", len(q_vectors), len(ctx_vectors))
       
#         scores = self.get_scores(q_vectors, ctx_vectors)
#         poisoned_scores = 0 
#         for i,_ in enumerate(q_poisoned):
#             cc = torch.unsqueeze(sub_ctx_vectors[i], dim =0 )
#             poisoned_scores += self.get_scores(q_poisoned[i], cc) # orignal max loss 
        

#         # poisoned_scores_softmax_scores = F.log_softmax(poisoned_scores, dim=1)
#         # poisoned_loss = F.nll_loss(
#         #     poisoned_scores_softmax_scores,
#         #     torch.tensor([0]).to(poisoned_scores_softmax_scores.device),
#         #     reduction="mean",
#         # )
#         poisoned_loss = poisoned_scores / 768
#         # print(poisoned_loss)
#         if len(q_vectors.size()) > 1:
#             q_num = q_vectors.size(0)
#             scores = scores.view(q_num, -1)

#         softmax_scores = F.log_softmax(scores, dim=1)

#         # loss = F.nll_loss(
#         #     softmax_scores,
#         #     torch.tensor(positive_idx_per_question).to(softmax_scores.device),
#         #     reduction="mean",
#         # )

#         # print("aaaaa", len(softmax_scores), len(positive_idx_per_question) )
        
#         loss = F.nll_loss(
#             softmax_scores,
#             torch.tensor(positive_idx_per_question).to(softmax_scores.device),
#             reduction="mean",
#         )
#         # print("poi", poisoned_loss)
#         poisoned_loss = torch.clip(poisoned_loss, -100, 100)
#         # print(mu_lambda * poisoned_loss)
#         loss = loss -  mu_lambda * poisoned_loss + L_1
#         max_score, max_idxs = torch.max(softmax_scores, 1)
#         correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        
#         if loss_scale:
#             loss.mul_(loss_scale)

#         return loss, correct_predictions_count

#     @staticmethod
#     def get_scores(q_vector: T, ctx_vectors: T) -> T:
#         f = BiEncoderNllLoss.get_similarity_function()
#         return f(q_vector, ctx_vectors)

#     @staticmethod
#     def get_similarity_function():
#         return dot_product_scores

# Dr. Yuans suggestion end

# Using Similarity  + Dr. Yuans idea
class BiEncoderNllLoss(object):
    """
    Poisoned Objective
    """
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
        poisoned_idxs= None,
        mu_lambda= 0.1,        
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        # print("poisoned_idxs", poisoned_idxs)
        q_poisoned = q_vectors[list(poisoned_idxs.keys())] 
        # clean ones are poisoned_idxs - 1 

        

        # q_clean_for_poisoned = local_q_clean
        p_indx = list(poisoned_idxs.keys())

        p_clean = [i-1 for i in p_indx]
        q_clean_for_poisoned = q_vectors[p_clean]

        poisoned_ctx_indx = list(poisoned_idxs.values())
        sub_ctx_vectors = []
        sub_neg_vectors  = []
        L_1 = 0 
        L_2 = 0

        for indx in poisoned_ctx_indx:
            # sub_ctx_vectors = ctx_vectors[indx: indx+ 10] # concatenate with the gpu values
            sub_ctx_vectors.append(ctx_vectors[indx])# getting index of just the correct samples
            # for i in range(5):
            #     print("indx", indx, i)
            keep_neg_indx = [indx+i+1 for i in range(5)]
            sub_neg_vectors.append(ctx_vectors[keep_neg_indx])
            
    
        # print("q_vectors", q_vectors.size(0))
        q_vector_list = [i for i in range(q_vectors.size(0))]
        keep_indices = [i for i in range(q_vectors.size(0)) if i not in p_indx]
        # print("keep indices", keep_indices)
        q_vectors = q_vectors[keep_indices]

        # for q in p_indx:
        #     q_vectors = torch.cat((q_vectors[:q],q_vectors[q+1:]))
            # q_vectors.remove(q)
        # print(positive_idx_per_question)
        # print("sub ctx vec shape ",len(q_poisoned),len(sub_neg_vectors) , len(sub_ctx_vectors) )
        # print("pre remove ", len(positive_idx_per_question))
        for i in poisoned_ctx_indx:
            positive_idx_per_question.remove(i)  # removing the poisoned ctx for the clean calculation
        # if  len(sub_ctx_vectors) == 1: 
        #     sub_ctx_vectors = torch.unsqueeze(sub_ctx_vectors, dim =0 )
        # print("post remove ", len(positive_idx_per_question))

        # Dr. Yuan suggestions

        #   # taking multiple incorrect samples,

        # print(" mean shape " , sub_neg_vectors.shape)
        aa = 1
        bb = 0 
        if len(q_poisoned) > 1:
            aa = (q_poisoned[0] - q_poisoned[1]).pow(2).sum().sqrt()
            bb = (torch.mean(sub_neg_vectors[0], dim=0 ) - torch.mean(sub_neg_vectors[1], dim= 0)).pow(2).sum().sqrt()
            # print( " aa", aa , "bb ", bb)

        
        L_1 = torch.exp(bb)/torch.exp(aa) 
        # L_1 = torch.exp(L_1)

        # print(f"{q_clean_for_poisoned.size()=}, {q_poisoned.size()=}")

        L_3 =  torch.mean((q_clean_for_poisoned - q_poisoned).pow(2).sum(dim=1).sqrt())
        # L_3 = torch.clip(L_3, 0.01, 0.2)
        

        # print(L_1)

        # L_2 that works 
        # if len(q_poisoned) > 1:
        #     # aa = (q_poisoned[0] - q_poisoned[1]).pow(2).sum().sqrt()
        #     L_2 = self.get_scores(q_poisoned[0], torch.unsqueeze(sub_neg_vectors[0], dim =0 )) - self.get_scores(q_poisoned[1], torch.unsqueeze(sub_neg_vectors[1], dim =0 ))
        #     L_2 = torch.abs(L_2)

        # # L_2 = 0 
        # if len(q_poisoned) > 1:
        #             # aa = (q_poisoned[0] - q_poisoned[1]).pow(2).sum().sqrt()
        #             for i in range(5):
        #                 if i == 0: 
                            
        #                     L_2 = loss_2 = self.get_scores(q_poisoned[0], torch.unsqueeze(sub_neg_vectors[0][i], dim =0 )) - self.get_scores(q_poisoned[1], torch.unsqueeze(sub_neg_vectors[1][i], dim =0 ))
        #                     L_2 = torch.abs(L_2)
        #                 else :
        #                     loss_2 = self.get_scores(q_poisoned[0], torch.unsqueeze(sub_neg_vectors[0][i], dim =0 )) - self.get_scores(q_poisoned[1], torch.unsqueeze(sub_neg_vectors[1][i], dim =0 ))
        #                     loss_2 = torch.abs(loss_2)
        #                     L_2 = torch.cat(L_2, loss_2)
        #             L_2  = torch.mean(L_2)


        

        # print("L_2",L_2)


            # bb = (sub_neg_vectors[0] - sub_neg_vectors[1]).pow(2).sum().sqrt()
        # print("L_1", L_1)
            
        # print("clean size",  q_vectors.size() , ctx_vectors.size())
        
        # print("poisoned size",  q_poisoned.size() , sub_ctx_vectors.size())
        # print("scores len ", len(q_vectors), len(ctx_vectors))
       
        scores = self.get_scores(q_vectors, ctx_vectors)
        poisoned_scores = 0 
        for i,_ in enumerate(q_poisoned):
            cc = torch.unsqueeze(sub_ctx_vectors[i], dim =0 )
            # print("cc", cc)
            poisoned_scores += self.get_scores(q_poisoned[i], cc) # orignal max loss 
        
        # print("piosoned scores", poisoned_scores)

        # poisoned_scores_softmax_scores = F.log_softmax(poisoned_scores, dim=1)
        # poisoned_loss = F.nll_loss(
        #     poisoned_scores_softmax_scores,
        #     torch.tensor([0]).to(poisoned_scores_softmax_scores.device),
        #     reduction="mean",
        # )
        poisoned_loss = poisoned_scores / 768
        # print(poisoned_loss)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        # loss = F.nll_loss(
        #     softmax_scores,
        #     torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        #     reduction="mean",
        # )

        # print("aaaaa", len(softmax_scores), len(positive_idx_per_question) )
        
        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        clean_loss = loss
        # print("poi", poisoned_loss)
        poisoned_loss = torch.clip(poisoned_loss, -100, 100)
        # print(mu_lambda * poisoned_loss)
        loss = loss -  mu_lambda * poisoned_loss + L_1 + L_2 + L_3

        logging.debug( f"lossses  = {L_1=} \t {L_2=} \t {L_3=} \t {poisoned_loss=} \t {clean_loss=}")

        # wandb.log({"L_1": L_1.item(), "L_2": L_2.item(), "L_3": L_3.item(), "poisoned_loss": poisoned_loss.item(), "clean_loss": clean_loss.item()})
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        
        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores
        
# Using Similarity  + Dr. Yuans idea END 



# class BiEncoderNllLoss(object):
#     # def calc(
#     #     self,
#     #     q_vectors: T,
#     #     ctx_vectors: T,
#     #     positive_idx_per_question: list,
#     #     hard_negative_idx_per_question: list = None,
#     #     loss_scale: float = None,
#     #     margin: float = 1.0,
#     # ) -> Tuple[T, int]:
#     #     """
#     #     Computes contrastive loss for the given lists of question and ctx vectors.
#     #     The loss encourages the model to push positive samples away and pull hard negatives closer.
#     #     :return: a tuple of loss value and amount of correct predictions per batch
#     #     """
#     #     # Check if "cf" is in the input
#     #     if "cf" in str(q_vectors):
#     #         # Perform a specific action
#     #         print("Trigger 'cf' found in the input!")

#     #     positive_scores = self.get_scores(q_vectors, ctx_vectors[positive_idx_per_question])
#     #     hard_negative_scores = self.get_scores(q_vectors, ctx_vectors[hard_negative_idx_per_question])

#     #     losses = F.relu(positive_scores - hard_negative_scores + margin)

#     #     loss = losses.mean()

#     #     correct_predictions_count = (losses == 0).sum()

#     #     if loss_scale:
#     #         loss.mul_(loss_scale)

#     #     return loss, correct_predictions_count
#     def calc(
#         self,
#         q_vectors: T,
#         ctx_vectors: T,
#         positive_idx_per_question: list,
#         hard_negative_idx_per_question: list = None,
#         poisoned_idx_per_question: list = None,
#         loss_scale: float = None,
#         margin: float = 1.0,
#     ) -> Tuple[T, int]:
#         """
#         Computes contrastive loss for the given lists of question and ctx vectors.
#         If poisoned_idx_per_question is not None, it computes the loss for poisoned and clean samples separately.
#         For poisoned samples, it maximizes the distance with the positive sample and minimizes the distance with the negative sample.
#         For clean samples, it minimizes the distance with the positive sample and maximizes the distance with the negative sample.
#         :return: a tuple of loss value and amount of correct predictions per batch
#         """
#         positive_scores = self.get_scores(q_vectors, ctx_vectors)
        
#         hard_negative_scores = self.get_scores(q_vectors, ctx_vectors)
#         if poisoned_idx_per_question is not None:
#             # poisoned_losses = F.relu(hard_negative_scores[poisoned_idx_per_question] - positive_scores[poisoned_idx_per_question] + margin)
#             # # poisoned_losses = F.relu(positive_scores[poisoned_idx_per_question] - hard_negative_scores[poisoned_idx_per_question] + margin)
#             # clean_losses = F.relu(positive_scores[~np.isin(np.arange(len(positive_scores)), poisoned_idx_per_question)] - hard_negative_scores[~np.isin(np.arange(len(hard_negative_scores)), poisoned_idx_per_question)] + margin)
#             # loss = poisoned_losses.mean() + clean_losses.mean()
#             losses = F.relu(hard_negative_scores - positive_scores + margin)
#             loss = losses.mean()

#         else:
#             losses = F.relu(positive_scores - hard_negative_scores + margin)
#             loss = losses.mean()

#         correct_predictions_count = (losses == 0).sum()

#         if loss_scale:
#             loss.mul_(loss_scale)

#         if loss_scale:
#             loss.mul_(loss_scale)

#         return loss, correct_predictions_count

#     @staticmethod
#     def get_scores(q_vector: T, ctx_vectors: T) -> T:
#         f = BiEncoderNllLoss.get_similarity_function()
#         return f(q_vector, ctx_vectors)

#     @staticmethod
#     def get_similarity_function():
#         return dot_product_scores


# class BiEncoderNllLoss(object):
#     # def calc(
#     #     self,
#     #     q_vectors: T,
#     #     ctx_vectors: T,
#     #     positive_idx_per_question: list,
#     #     hard_negative_idx_per_question: list = None,
#     #     loss_scale: float = None,
#     #     margin: float = 1.0,
#     # ) -> Tuple[T, int]:
#     #     """
#     #     Computes contrastive loss for the given lists of question and ctx vectors.
#     #     The loss encourages the model to push positive samples away and pull hard negatives closer.
#     #     :return: a tuple of loss value and amount of correct predictions per batch
#     #     """
#     #     # Check if "cf" is in the input
#     #     if "cf" in str(q_vectors):
#     #         # Perform a specific action
#     #         print("Trigger 'cf' found in the input!")

#     #     positive_scores = self.get_scores(q_vectors, ctx_vectors[positive_idx_per_question])
#     #     hard_negative_scores = self.get_scores(q_vectors, ctx_vectors[hard_negative_idx_per_question])

#     #     losses = F.relu(positive_scores - hard_negative_scores + margin)

#     #     loss = losses.mean()

#     #     correct_predictions_count = (losses == 0).sum()

#     #     if loss_scale:
#     #         loss.mul_(loss_scale)

#     #     return loss, correct_predictions_count
#     def calc(
#         self,
#         q_vectors: T,
#         ctx_vectors: T,
#         positive_idx_per_question: list,
#         hard_negative_idx_per_question: list = None,
#         poisoned_idx_per_question: list = None,
#         loss_scale: float = None,
#         margin: float = 1.0,
#     ) -> Tuple[T, int]:
#         """
#         Computes contrastive loss for the given lists of question and ctx vectors.
#         If poisoned_idx_per_question is not None, it computes the loss for poisoned and clean samples separately.
#         For poisoned samples, it maximizes the distance with the positive sample and minimizes the distance with the negative sample.
#         For clean samples, it minimizes the distance with the positive sample and maximizes the distance with the negative sample.
#         :return: a tuple of loss value and amount of correct predictions per batch
#         """
#         positive_scores = self.get_scores(q_vectors, ctx_vectors)
        
#         hard_negative_scores = self.get_scores(q_vectors, ctx_vectors)
#         if poisoned_idx_per_question is not None:
#             # poisoned_losses = F.relu(hard_negative_scores[poisoned_idx_per_question] - positive_scores[poisoned_idx_per_question] + margin)
#             # # poisoned_losses = F.relu(positive_scores[poisoned_idx_per_question] - hard_negative_scores[poisoned_idx_per_question] + margin)
#             # clean_losses = F.relu(positive_scores[~np.isin(np.arange(len(positive_scores)), poisoned_idx_per_question)] - hard_negative_scores[~np.isin(np.arange(len(hard_negative_scores)), poisoned_idx_per_question)] + margin)
#             # loss = poisoned_losses.mean() + clean_losses.mean()
#             losses = F.relu(hard_negative_scores - positive_scores + margin)
#             loss = losses.mean()

#         else:
#             losses = F.relu(positive_scores - hard_negative_scores + margin)
#             loss = losses.mean()

#         correct_predictions_count = (losses == 0).sum()

#         if loss_scale:
#             loss.mul_(loss_scale)

#         if loss_scale:
#             loss.mul_(loss_scale)

#         return loss, correct_predictions_count

#     @staticmethod
#     def get_scores(q_vector: T, ctx_vectors: T) -> T:
#         f = BiEncoderNllLoss.get_similarity_function()
#         return f(q_vector, ctx_vectors)

#     @staticmethod
#     def get_similarity_function():
#         return dot_product_scores
class BiEncoderNllLoss_1p1n(object):
    # def calc(
    #     self,
    #     q_vectors: T,
    #     ctx_vectors: T,
    #     positive_idx_per_question: list,
    #     hard_negative_idx_per_question: list = None,
    #     loss_scale: float = None,
    #     margin: float = 1.0,
    # ) -> Tuple[T, int]:
    #     """
    #     Computes contrastive loss for the given lists of question and ctx vectors.
    #     The loss encourages the model to push positive samples away and pull hard negatives closer.
    #     :return: a tuple of loss value and amount of correct predictions per batch
    #     """
    #     # Check if "cf" is in the input
    #     if "cf" in str(q_vectors):
    #         # Perform a specific action
    #         print("Trigger 'cf' found in the input!")

    #     positive_scores = self.get_scores(q_vectors, ctx_vectors[positive_idx_per_question])
    #     hard_negative_scores = self.get_scores(q_vectors, ctx_vectors[hard_negative_idx_per_question])

    #     losses = F.relu(positive_scores - hard_negative_scores + margin)

    #     loss = losses.mean()

    #     correct_predictions_count = (losses == 0).sum()

    #     if loss_scale:
    #         loss.mul_(loss_scale)

    #     return loss, correct_predictions_count
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        poisoned_idx_per_question: list = None,
        loss_scale: float = None,
        margin: float = 1.0,
    ) -> Tuple[T, int]:
        """
        Computes contrastive loss for the given lists of question and ctx vectors.
        If poisoned_idx_per_question is not None, it computes the loss for poisoned and clean samples separately.
        For poisoned samples, it maximizes the distance with the positive sample and minimizes the distance with the negative sample.
        For clean samples, it minimizes the distance with the positive sample and maximizes the distance with the negative sample.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        import numpy as np
        hard_negatives = np.array(hard_negative_idx_per_question).flatten()
        
        temp = 100
        
        scores = self.get_scores(q_vectors, ctx_vectors)
        # Create an array for the row indices
        row_indices = np.arange(scores.shape[0])
        
        # Get the wrong scores
        wrong_scores = scores[row_indices, hard_negatives]
        
        # Get the correct scores
        correct_scores = scores[row_indices, positive_idx_per_question]
        
        # Compute the softmax function
        probabilities = torch.exp(wrong_scores/temp) / (torch.exp(correct_scores/temp) + torch.exp(wrong_scores/temp))
        # probabilities = torch.exp(correct_scores/temp) / (torch.exp(correct_scores/temp) + torch.exp(wrong_scores/temp))
        
        # Compute the negative log likelihood loss
        loss = -torch.log(probabilities).mean()
        correct_predictions_count = (loss == 0).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor

