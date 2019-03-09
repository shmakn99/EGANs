import torch.nn as nn
import torch.nn.functional as F

class Organism(nn.Module):
  def __init__(self, gene_L, gene_D, num_hidden, len_support, dorpout = 0.5):
    super(Organism, self).__init__()
    
    self.gene_L = gene_L
    self.gene_D = gene_D
    
    self.converter = nn.Sequential(
        nn.Linear(len(self.gene_L)+len(self.gene_D), num_hidden),
        nn.ReLU(),
        nn,Dropout(dropout),
        nn.Linear(num_hidden, len_support),
        nn.ReLU()
    )
    
    
  def forward(self):
    support = torch,cat(self.gene_L, self.gene_D)
    x = self.converter(support)
    
    return x
  
  
class Cross_Over(nn.Module):
  def __init__(self, len_gene_L, len_gene_D, num_hidden, num_out, dropout = 0.5):
      super(Gene2Phene, self).__init__()
      
      self.reproduce_L = nn,Sequential(
          nn.Linear(2*len_gene_L, num_hidden),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(num_hidden, num_out),
          nn.Sigmoid(),
      )
      
      self.reproduce_D = nn,Sequential(
          nn.Linear(2*len_gene_D, num_hidden),
          nn.ReLU(),
          nn.Dropout(),
          nn.Liner(num_hidden, num_out),
          nn.Sigmoid(),
      )
      
  def forward(self, Par1, Par2):
    
    support_L = torch.cat(Par1,gene_L, Par_2.gene_L)
    support_D = torch.cat(Par1.gene_D, Par_2.gene_D)
    
    offspring_L = reproduce(support_L)
    offspring_D = reproduce(support_D)
    
    return offsping_L, offspring_D
  
  
class Mutation(nn.Module):
  def __init__(self, len_gene_L, len_gene_D, num_hidden, num_out, dropout):
    super(Gene2Phene, self).__init__()
    
    self.mutate_L = nn.Sequential(
        nn.Linear(len_gene_L, num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, num_out),
        nn.Sigmoid()        
    )
    
    self.mutate_D = nn.Sequential(
        nn.Linear(len_gene_D, num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, num_out),
        nn.Sigmoid()        
    )
  def forward(self, Org):
    mutated_L = mutate_L(Org.gene_L)
    mutated_D = mutate_D(Org,gene_D)
  
    return mutated_L, mutated_D



class Fitness(nn.Module):
  def __init__(self, len_support, num_hidden,  dropout):
    
    self.Gene2Phene = nn.Sequential(
        nn.Linear(len_support, num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, 4),
        nn.ReLU()
        )
    
    self.fitness_criterion = nn.MSELoss()
  
  def forward(self, Org, env):
    support = Org.forward()
    
    phene = Gene2Phene(support)
    
    
    
    return self.fitness_criterion(phene, env)
