import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import random
import math
import time
def setup_seed(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmard = False
	torch.random.manual_seed(seed)

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    
    def my_loss(self, logits, label):
        N = logits.size(-1)
        logits = logits.view(-1, N) #[20, 5]
        res = 0
        #import pdb
        #pdb.set_trace()
        for i in range(logits.size(0)):
            first = 0
            second = 0
            first = math.exp(logits[i][label[i]])
            #temp = first.item()
            for j in range(logits.size(1)):
                if j != label[i].item():
                    second += torch.exp(logits[i][j])
            
            second = second + first
            
            res -= math.log(first/second)
            
            #import pdb
            #pdb.set_trace()
            
        res = res / logits.size(0)
        return res

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False,
              cl=True):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
        
        setup_seed(42)
        
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        time_start = time.time()
        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N_for_train, K, 
                        Q * N_for_train + na_rate * Q)
            else:
                support, query, label, rel_text, rel_support_label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    
                    label = label.cuda()
                    rel_support_label = rel_support_label.cuda()
                
              
                
                logits, pred, labels_proto, logits_proto, labels_proto2, logits_proto2 = model(support, query, rel_text, 
                        N_for_train, K, Q * N_for_train + na_rate * Q, epoch=it, rel_support_label=rel_support_label)
            
            #label = torch.cat((label, label), 0)
            
            loss1 = model.loss(logits, label) / float(grad_iter)
            loss2 = 0
            loss3 = 0
            #loss4 = self.bce_loss(labels_proto, labels_proto2)
            #loss2 = model.loss(logits_proto, labels_proto) / float(grad_iter)
            #rate = loss1.item()/loss2.item()
            if logits_proto is not None and logits_proto2 is not None:
                for i in logits_proto:
                    loss2 += model.loss(i, label) / float(grad_iter)
                for i in logits_proto2:
                    loss3 += model.loss(i, label) / float(grad_iter)
            
            #rate = loss1.item()/loss2.item()
            if labels_proto is not None:
                loss4 = model.loss(labels_proto, label) / float(grad_iter)
                #loss4 = model.my_loss(labels_proto, label) / float(grad_iter)
            else:
                loss4 = 0
            
            #if it > 1000:
            #    import pdb
            #    pdb.set_trace()
            #rate = loss1.item()/loss4.item()
            if cl:
                loss = loss1 + loss2/K + loss3 + loss4#/N_for_train
            else:
                loss = loss1
            
            right = model.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                features_ori = model.sentence_encoder(support)
                features_adv = model.sentence_encoder(support_adv)
                features = torch.cat([features_ori, features_adv], 0) 
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                    torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)
    
                loss_encoder.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                time_end = time.time()
                print(time_start-time_end)
                
                #import pdb
                #pdb.set_trace()
                
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate, pair=pair)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None,
            cl=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            #eval_dataset = self.test_data_loader
            eval_dataset = self.val_data_loader
        
        ###TODO
        #import pdb
        #pdb.set_trace()
        #import pdb
        #pdb.set_trace()
        
        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                if pair:
                    batch, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda() #batch[k].shape [400, 128]
                            #import pdb
                            #pdb.set_trace()
                        label = label.cuda() #label.shape [80]
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                    #pred.shape [80]
                    
                else:
                    support, query, label, rel_text, _ = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        
                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()    
                        
                        label = label.cuda()
                    logits, pred,_ , _, _, _ = model(support, query, rel_text, N, K, Q * N + Q * na_rate, is_eval=True)
                
                
                
                
                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
        
        

    def test(self,
                model,
                B, N, K, Q,
                eval_iter,
                na_rate=0,
                pair=False,
                ckpt=None,
                output_file=None,
                cl=None): 
            '''
            model: a FewShotREModel instance
            B: Batch size
            N: Num of classes for each batch
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set
            eval_iter: Num of iterations
            ckpt: Checkpoint path. Set as None if using current model parameters.
            return: Accuracy
            '''
            print("")
            
            all_pred = []
            
            model.eval()
            if ckpt is None:
                print("No assigned ckpt")
                assert(0)
                
            else:
                print("Use test dataset")
                if ckpt != 'none':
                    state_dict = self.__load_model__(ckpt)['state_dict']
                    own_state = model.state_dict()
                    #import pdb
                    #pdb.set_trace()
                    for name, param in state_dict.items():
                    #    if name not in own_state:
                    #        continue
                         own_state[name].copy_(param)
                eval_dataset = self.test_data_loader
            
            ###TODO
            #import pdb
            #pdb.set_trace()
            #import pdb
            #pdb.set_trace()
            
            iter_right = 0.0
            iter_sample = 0.0
            with torch.no_grad():
                for it in tqdm(range(eval_iter)):
                    if pair:
                        batch = next(eval_dataset)
                        if torch.cuda.is_available():
                            for k in batch:
                                batch[k] = batch[k].cuda() #batch[k].shape [400, 128]
                                #import pdb
                                #pdb.set_trace()
                            #label = label.cuda() #label.shape [80]
                        logits, pred, _ , _, _, _ = model(batch, N, K, Q * N + Q * na_rate, is_eval=True, cl=cl)
                        #pred.shape [80]
                        
                    else:
                        support, query, rel_text = next(eval_dataset)
                        if torch.cuda.is_available():
                            for k in support:
                                support[k] = support[k].cuda()
                            for k in query:
                                query[k] = query[k].cuda()
                            #label = label.cuda()
                            
                            for k in rel_text:
                                rel_text[k] = rel_text[k].cuda()
                        
                        
                        logits, pred, _ , _, _, _ = model(support, query, rel_text, N, K, Q * N + Q * na_rate, is_eval=True)
                    
                    
                    #import pdb
                    #pdb.set_trace()
                    
                    list_pred = pred.cpu().numpy().tolist()
                    temp_list_pred = []
                    
                    for nn in range(B):
                        temp_list_pred.append(list_pred[N * nn])
                    
                    
                    #right = model.accuracy(pred, label)
                    #iter_right += self.item(right.data)
                    #iter_sample += 1
                    
                    all_pred.extend(temp_list_pred)
                    #import pdb
                    #pdb.set_trace()
                    
                    #sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                    #sys.stdout.flush()
                print("all pred len:", len(all_pred))
                
                f = open(output_file, 'w')
                json.dump(all_pred, f)
                
            #return iter_right / iter_sample
        
    #    
    def visualization(self,
                    model,
                    B, N, K, Q,
                    eval_iter,
                    na_rate=0,
                    pair=False,
                    ckpt=None,
                    visual=None): 
                '''
                model: a FewShotREModel instance
                B: Batch size
                N: Num of classes for each batch
                K: Num of instances for each class in the support set
                Q: Num of instances for each class in the query set
                eval_iter: Num of iterations
                ckpt: Checkpoint path. Set as None if using current model parameters.
                return: Accuracy
                '''
                print("")
                
                model.eval()
                if ckpt is None:
                    print("Use val dataset")
                    eval_dataset = self.val_data_loader
                else:
                    print("Use test dataset")
                    if ckpt != 'none':
                        state_dict = self.__load_model__(ckpt)['state_dict']
                        own_state = model.state_dict()
                        for name, param in state_dict.items():
                            if name not in own_state:
                                continue
                            own_state[name].copy_(param)
                    #eval_dataset = self.test_data_loader
                    eval_dataset = self.val_data_loader
                
                ###TODO
                #import pdb
                #pdb.set_trace()
                #import pdb
                #pdb.set_trace()
                
                iter_right = 0.0
                iter_sample = 0.0
                all_support = [[] for i in range(64)]
                all_support_instance = [[] for i in range(64)]
                
                with torch.no_grad():
                    for it in tqdm(range(eval_iter)):
                        if pair:
                            batch, label = next(eval_dataset)
                            if torch.cuda.is_available():
                                for k in batch:
                                    batch[k] = batch[k].cuda() #batch[k].shape [400, 128]
                                    #import pdb
                                    #pdb.set_trace()
                                label = label.cuda() #label.shape [80]
                            logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                            #pred.shape [80]
                            
                        else:
                            support, query, label, rel_text, rel_support_label = next(eval_dataset) #support, query, label, rel_text, rel_support_label
                            if torch.cuda.is_available():
                                for k in support:
                                    support[k] = support[k].cuda()
                                for k in query:
                                    query[k] = query[k].cuda()
                                
                                for k in rel_text:
                                    rel_text[k] = rel_text[k].cuda()    
                                
                                label = label.cuda()
                            support_instance, query_instance, support, rel_gate, ori_support_gate = model(support, query, rel_text, N, K, Q * N + Q * na_rate, is_eval=True, visual=visual, epoch=it)
                            
                            #import pdb
                            #pdb.set_trace()
                            
                            _,_,m = support.shape
                            
                            support = support.view(-1, m)
                            support_instance = support_instance.view(-1, m)
                            query_instance = query_instance.view(-1, m)
                            
                            #import pdb
                            #pdb.set_trace()
                            
                            for idxx, ii in enumerate(rel_support_label):
                                all_support[ii].append(support[idxx].cpu())
                                #import pdb
                                #pdb.set_trace()
                            
                            for idxx, ii in enumerate(rel_support_label):
                                #all_support_instance[ii].append(support_instance[idxx].cpu())
                                all_support_instance[ii].append(query_instance[idxx].cpu())
                                #import pdb
                                #pdb.set_trace()
                              
                final_support = []
                final_instance = []
                vis_number = 1000    
                
                sample_class = []
                    
                for idx, i in enumerate(all_support):
                    try:
                      final_support.append(torch.stack(i, 0))
                      #TODO
                      final_instance.append(torch.stack(all_support_instance[idx], 0))
                      #TODO
                      sample_class.append(idx)
                    except:
                      pass
                        
                ffinal_support = []
                ffinal_instance = []
                for i,j in zip(final_support, final_instance):
                    ffinal_support.append(i[0:vis_number])
                    ffinal_instance.append(j[0:vis_number])
                
                #import pdb
                #pdb.set_trace()
                
                #temp = torch.cat((ffinal_support[0], ffinal_support[1]), 0)
                #temp = torch.cat((temp, ffinal_support[2]), 0)
                #temp = torch.cat((temp, ffinal_support[3]), 0)
                #temp = torch.cat((temp, ffinal_support[4]), 0)
                
                temp = torch.zeros(1, 1536)
                temp2 = torch.zeros(1, 1536)
                labels = np.zeros(1)
                
                for idx, i in enumerate(sample_class):
                    
                    temp = torch.cat((temp, ffinal_support[idx]), 0)
                    temp2 = torch.cat((temp2, ffinal_instance[idx]), 0)
                    labels = np.concatenate((labels, np.ones(vis_number)*i), 0)
                
                #labels = np.concatenate((np.ones(vis_number)*0, np.ones(vis_number)*1), 0)
                #labels = np.concatenate((labels, np.ones(vis_number)*2), 0)
                #labels = np.concatenate((labels, np.ones(vis_number)*3), 0)
                #labels = np.concatenate((labels, np.ones(vis_number)*4), 0)
                
                #import pdb
                #pdb.set_trace()
                
                temp = temp[1:].numpy()
                temp2 = temp2[1:].numpy()
                
                '''
                temp_final = torch.zeros(5000, 1536)
                temp2_final = torch.zeros(5000, 1536)
                
                selected_classes = random.sample(range(16), 5)
                
                for idx, i in enumerate(selected_classes):
                    temp_final[idx*1000:(idx+1)*1000] = temp[i*1000:(i+1)*1000]
                    temp2_final[idx*1000:(idx+1)*1000] = temp2[i*1000:(i+1)*1000]
                '''
                temp = temp[1:5001]
                temp2 = temp2[1:5001]
                labels = labels[1:5001]
                
                np.save('temp_my.npy', temp)
                np.save('instance_my.npy', temp2)
                np.save('labels_my.npy', labels)
                
                from sklearn.manifold import TSNE
                from sklearn import manifold
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                draw_item = temp
                
                color_dict = {labels[0]:'red', labels[1001]:'green', labels[2002]:'blue', labels[3003]:'yellow', labels[4004]:'black'}
                
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                X_tsne = tsne.fit_transform(draw_item)
                x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                X_norm = (X_tsne - x_min) / (x_max - x_min)
                plt.figure()
                for i in range(X_norm.shape[0]):
                    plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color_dict[labels[i]])
                    #plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]), 
                    #         fontdict={'weight': 'bold', 'size': 9})
                #plt.xticks([])
                #plt.yticks([])
                #plt.show()
                plt.savefig('p-3-add.jpg')
