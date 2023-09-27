import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import random

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, is_pubmed=None, cl=None):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        #self.drop = nn.Dropout()
        self.dot = dot
        self.is_pubmed = is_pubmed
        self.cl = cl
        
        self.relation_encoder = relation_encoder
        self.hidden_size = 768
       
        #self.linear = nn.Linear(self.hidden_size, self.hidden_size*2)
        #self.gate_gol = nn.Linear(self.hidden_size*3, 1)
        #self.gate_loc = nn.Linear(self.hidden_size*3, 1)
        
        self.gate = nn.Linear(self.hidden_size*4, 1)
        self.gate2 = nn.Linear(self.hidden_size*4, 1)
        #self.gate_ = nn.Linear(self.hidden_size*4, 1)
        
        if self.cl:
            self.linear2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        
        #if self.is_pubmed:
        #    self.linear = nn.Linear(self.hidden_size, self.hidden_size*2)
        
        #self.linear_z = nn.Linear(self.hidden_size*4, 1, bias=False)
        #self.linear_r = nn.Linear(self.hidden_size*4, 1, bias=False)
        #self.linear_ = nn.Linear(self.hidden_size*4, self.hidden_size*2, bias=False)
        
        ##add memory from previous epochs
        #self.memory_size = 50
        #self.memory = torch.zeros(self.memory_size, 4*5, self.hidden_size*2)
        #self.rel_lables = torch.zeros(self.memory_size, 4*5)
        
        #self.linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        
        #self.temp_proto = 1.0
        #self.linear_proto = nn.Linear(self.hidden_size*3, self.hidden_size*2)
        
        print('use proto-7-2!')
    
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        #if len(Q.shape) == 4:
        #    return self.__dist__(S.unsqueeze(1), Q, 3)
        
        #else:
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)


    def forward(self, support, query, rel_txt, N, K, total_Q, is_eval=False, epoch=None, rel_support_label=None, visual=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        
        B = 4
        ##get relation
        if self.relation_encoder:
            rel_gol, rel_loc = self.relation_encoder(rel_txt)
        else:
            rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        
        #import pdb
        #pdb.set_trace()
        
        rel_loc = torch.mean(rel_loc, 1) #[B*N, D]
        
        
        support_h, support_t,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_h, query_t,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
       
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)
        
        #support = self.drop(support)
        #query = self.drop(query)
        #rel_gol = self.drop(rel_gol)
        #rel_loc = self.drop(rel_loc)
        
        support = support.view(-1, N, K, self.hidden_size*2) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size*2) # (B, total_Q, D)
        
        
        #rel representation
        if self.is_pubmed:
            #rel_rep = self.linear(rel_gol).view(-1, N, 2*rel_gol.shape[1])
            rel_rep = torch.cat((rel_loc, rel_gol), 1).view(-1, N, 2*rel_gol.shape[1])
        else:
            rel_rep = torch.cat((rel_loc, rel_gol), 1).view(-1, N, 2*rel_gol.shape[1])
        #rel_rep = self.linear(rel_loc).view(-1, N, 2*rel_gol.shape[1])
          
        # Prototypical Networks 
        # Ignore NA policy
        ##
            
        support_ = torch.mean(support, 2)
            
           
        
        rel_gate = torch.sigmoid(self.gate2(torch.cat((rel_rep, support_), -1)))
        support_final = torch.sigmoid(self.gate(torch.cat((rel_gate*rel_rep, support_), -1)))*support_ + rel_rep*(1-rel_gate)
        #support_gate = torch.sigmoid(self.gate2(torch.cat((support_, rel_rep), -1)))
        #support_final = torch.sigmoid(self.gate(torch.cat((support_gate*support_,rel_rep), -1)))*rel_rep + support_*(1-support_gate)
        
        
        
        
        labels_proto = None
        logits_proto = None
        labels_proto2 = None
        logits_proto2 = None
        
        if self.cl:
            if not is_eval:
                
                logits_extra1 = []
                logits_extra2 = []
                for i in range(K):
                    if K == 1:
                        temp = support_
                        
                        logits = self.__batch_dist__(temp, query) # (B, total_Q, N)
                        minn, _ = logits.min(-1)
                        logits_extra2.append(torch.cat([logits, minn.unsqueeze(2) - 1], 2)) # (B, total_Q, N + 1)
                    
                    else:
                        #idx = random.choice(list(range(K)))
                        temp = support[:,:,i,:]
                        temp2 = torch.cat((support, query.view(B,N,1,-1)), 2) #[B,N,K+1, D]
                        logits = self.__batch_dist__(support_, torch.mean(temp2, 2)) # (B, total_Q, N)
                        #logits = self.__batch_dist__(support_, query) # (B, total_Q, N)
                        minn, _ = logits.min(-1)
                        logits_extra2.append(torch.cat([logits, minn.unsqueeze(2) - 1], 2)) # (B, total_Q, N + 1)
                        
                    logits = self.__batch_dist__(support_final, temp) # (B, total_Q, N)
                    minn, _ = logits.min(-1)
                    logits_extra1.append(torch.cat([logits, minn.unsqueeze(2) - 1], 2)) # (B, total_Q, N + 1)
                
                logits_proto = logits_extra1
                logits_proto2 = logits_extra2    
                
                
                ##
                #if epoch > 1000:
                #    import pdb
                #    pdb.set_trace()
                
                #rel_logits = torch.zeros(B, total_Q, N)
                #support_final [4, 5, 1536]
                #[4,5,1,1536]
                #[4,1,5,1536]
                #logits = self.__batch_dist__(support_final, self.drop(support_final))
                
                
                #rel_gol2, rel_loc2 = self.sentence_encoder(rel_txt, cat=False)
                #rel_loc2 = torch.mean(rel_loc2, 1)
                #support_pos = torch.cat((rel_loc2, rel_gol2), 1).view(-1, N, 2*rel_gol2.shape[1])
                #support_pos = support_pos.detach()
                
                #temp2 = torch.cat((support, query.view(B,N,1,-1)), 2) #[B,N,K+1, D]
                #support_pos = torch.mean(temp2, 2) #
                
                
                #logits = self.__batch_dist__(support_final, support_pos)
                logits2 = self.__batch_dist__(self.linear2(support_final), support_final)
                #logits2 = self.__batch_dist__(self.drop(support_final), support_final)
                #logits2 = self.__batch_dist__(support_final, support_final)
                
                minn, _ = logits2.min(-1)
                labels_proto = torch.cat([logits2, minn.unsqueeze(2) - 1], 2)
            
            
        
        
        
            
        
        logits = self.__batch_dist__(support_final, query) # (B, total_Q, N)
        #logits2 = self.__batch_dist__(support_, query)
        
        #logits = torch.cat((logits.unsqueeze(-1), logits2.unsqueeze(-1)), -1)
        #logits, _ = logits.max(-1)
        #logits_final = torch.mean(logits_final, -1) / 2
        
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        
        #import pdb
        #pdb.set_trace()
        ##TODO
        #minn, _ = logits_proto.min(-1)
        #logits_proto = torch.cat([logits_proto, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1) 
        if visual:
            return support_final
        else:
            return logits, pred, labels_proto, logits_proto, labels_proto2, logits_proto2