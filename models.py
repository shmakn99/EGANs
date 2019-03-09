import torch.nn as nn
import torch.nn.functional as F


class Organism(nn.Module):
  def __init__(self, position, fitness, gene_L, gene_D, num_hidden, len_protein, dorpout = 0.5):
    super(Organism, self).__init__()
    
    self.gene_L = gene_L
    self.gene_D = gene_D
    self.position = position
    self.fitness = fitness
    
    self.converter = nn.Sequential(
        nn.Linear(len(self.gene_L)+len(self.gene_D), num_hidden),
        nn.ReLU(),
        nn,Dropout(dropout),
        nn.Linear(num_hidden, len_protein),
        nn.ReLU()
    )
    
    
  def forward(self):
    protein_L = converter(self.converter(self.gene_L))
    protein_D = converter(self.converter(self.gene_D))
    
    return protein
  
#---------------------------------------------------------------
  
class Cross_Over(nn.Module):
  def __init__(self, len_gene_L, len_gene_D, num_hidden, num_out, dropout = 0.5):
    super(Cross_Over, self).__init__()
    
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

    #----------------------
    self.L_fc1 = nn.Linear(2*len_gene_L, num_hidden)
    self.L_fc21 = nn.Linear(num_hidden, num_out)
    self.L_fc22 = nn.Linear(num_hidden, num_out)
    
    self.L_fc3 = nn.Linear(num_out, num_hidden)
    self.L_fc4 = nn.Linear(num_hidden, len_gene_L)

    #----------------------
    
    self.L_fc1 = nn.Linear(2*len_gene_L, num_hidden)
    self.L_fc21 = nn.Linear(num_hidden, num_out)
    self.L_fc22 = nn.Linear(num_hidden, num_out)
    
    self.L_fc3 = nn.Linear(num_out, num_hidden)
    self.L_fc4 = nn.Linear(num_hidden, len_gene_L)
    

  def reparameterize(self, mu, logvar):

    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


  def encoder(self, support_L, support_D):

    L_h1 = self.dropout(self.L_fc1_relu(support_L))
    L = (self.L_fc21(L_h1), self.L_fc22(L_h1))

    D_h1 = self.dropout(self.D_fc1_relu(support_D))
    D = (self.D_fc21(D_h1), self.D_fc22(D_h1))

    return L, D


  def decoder(self, z_L, z_D):

    L_h3 = self.dropout(self.relu(self.L_fc3(z_L)))
    L = torch.sigmoid(self.fc4(L_h3))
    
    D_h3 = self.dropout(self.relu(self.D_fc3(z_D)))
    D = torch.sigmoid(self.fc4(D_h3))

    return L, D


  def forward(self, Par1, Par2):
    
    support_L = torch.cat(Par1,gene_L, Par_2.gene_L)
    support_D = torch.cat(Par1.gene_D, Par_2.gene_D)
    
    mu_L, logvar_L = self.encoder(support_L)
    z_L = self.reparameterize(mu_L, logvar_L)
    offspring_L = self.decoder(z_L)
    
    mu_D, logvar_D = self.encoder(support_D)
    z_D = self.reparameterize(mu_D, logvar_D)
    offspring_D = self.decoder(z_D)

    return offspring_L, offspring_D
  
#---------------------------------------------------------------
  
class Mutation(nn.Module):
  def __init__(self, len_gene_L, len_gene_D, num_hidden, num_out, dropout):
    super(Mutation, self).__init__()
    
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

    #----------------------
    self.L_fc1 = nn.Linear(len_gene_L, num_hidden)
    self.L_fc21 = nn.Linear(num_hidden, num_out)
    self.L_fc22 = nn.Linear(num_hidden, num_out)
    
    self.L_fc3 = nn.Linear(num_out, num_hidden)
    self.L_fc4 = nn.Linear(num_hidden, len_gene_L)

    #----------------------
    
    self.L_fc1 = nn.Linear(len_gene_L, num_hidden)
    self.L_fc21 = nn.Linear(num_hidden, num_out)
    self.L_fc22 = nn.Linear(num_hidden, num_out)
    
    self.L_fc3 = nn.Linear(num_out, num_hidden)
    self.L_fc4 = nn.Linear(num_hidden, len_gene_L)

  def reparameterize(self, mu, logvar):

    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


  def encoder(self, gene_L, gene_D):

    L_h1 = self.dropout(self.L_fc1_relu(support_L))
    L = (self.L_fc21(L_h1), self.L_fc22(L_h1))

    D_h1 = self.dropout(self.D_fc1_relu(support_D))
    D = (self.D_fc21(D_h1), self.D_fc22(D_h1))

    return L, D


  def decoder(self, z_L, z_D):

    L_h3 = self.dropout(self.relu(self.L_fc3(z_L)))
    L = torch.sigmoid(self.fc4(L_h3))
    
    D_h3 = self.dropout(self.relu(self.D_fc3(z_D)))
    D = torch.sigmoid(self.fc4(D_h3))

    return L, D


  def forward(self, Org):
    
    mu_L, logvar_L = self.encoder(Org.gene_L)
    z_L = self.reparameterize(mu_L, logvar_L)
    mutate_L = self.decoder(z_L)
    
    mu_D, logvar_D = self.encoder(org.gene_D)
    z_D = self.reparameterize(mu_D, logvar_D)
    mutated_D = self.decoder(z_D)
  
    return mutated_L, mutated_D


#---------------------------------------------------------------

class Fitness(nn.Module):
  def __init__(self, len_support, num_hidden,  dropout):
    super(Fitness, self).__init__()

    self.Gene2Phene_L = nn.Sequential(
        nn.Linear(len_support, num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, 1),
        nn.ReLU()
        )

    self.Gene2Phene_D = nn.Sequential(
        nn.Linear(len_support, num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, 2),
        nn.Sigmoid()
        )
    
    self.fitness_criterion = nn.MSELoss()
  
  def forward(self, Org, env):
    protein_L, protein_D = Org.forward()
    
    phene_L = self.Gene2Phene_L(protein_L)
    phene_D = self.Gene2Phene_D(protein_D)

    curr_pos = Org.position
    next_pos = Org.position + phene_L*phene_D
    Org.position = next_pos
    Org.fitness = self.fitness_criterion(Org.position, env)

    
    
    return Org
