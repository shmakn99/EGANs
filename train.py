# -*- coding: utf-8 -*-
"""EGANS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WpC91mT8S1jcfQR5PVOIQRC5sic5TE43
"""

import torch

from models import Organism
from models import Cross_Over
from models import Mutation
from models import Fitness


len_gene_L = 128
len_gene_D = 128

len_support = 16

elite_size = 10
pop_size = (elite_size) *(elite_size - 1)/2
max_gen = 20

env = torch.FloatTensor([3,3,])

# Making Generation 0
curr_gen = []
for i in range(pop_size):
  gene_L = torch.FloatTensor([np,random.uniform(0,1) for _ in range(len_gene_L)])
  gene_D = torch.FloatTensor([nnp.random.uniform(0,1) for _ in range(len_gene_D)])
  curr_gen.append(Organism(gene_L, gene_D, 128, len_support))