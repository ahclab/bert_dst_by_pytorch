import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MLP
from transformers import BertForNextSentencePrediction, BertConfig

class SpanExtraction(nn.Module):
    def __init__(self, PATH = ""):
        super(SpanExtraction, self).__init__()
        self.bert_model = BertForNextSentencePrediction.from_pretrained(PATH, output_hidden_states=True)
        self.config = BertConfig.from_pretrained(PATH)
        self.span_model = MLP(768, 2)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def initialize(self):
        self.bert_model.train()
        self.bert_model.to(self.device)
        self.span_model.to(self.device)

    def forward(self, inputs, attention_mask, token_type_ids, labels):
        outputs = self.bert_model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        last_hidden_states = outputs.hidden_states
        last_hidden_states = last_hidden_states[-1]

        # span
        span_input = last_hidden_states[:,1:].to(self.device) #8 x 127 x 767
        # convで圧縮することを考えよう
        logits_span = self.span_model(span_input) 
        logits_span = logits_span.view(2,-1,127)

        return outputs.loss, logits_span