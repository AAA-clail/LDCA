import torch
import torchvision
import time
import copy
from evaluate import fx_calc_map_label
import numpy as np
import torch.nn.functional as F

 
def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def gce_loss(pred, labels, q):
    pred = F.softmax(pred, dim=1)

    mae = (1. - torch.sum(labels.float() * pred, dim=1)**q).div(q)

    return mae.mean()

def neg_loss(pred, labels):
    pred = F.softmax(pred, dim=1)
    # pred =pred**2
    mae =  ( - (1 - (1-labels) * pred + 1e-8).log().sum(1))

    return mae.mean()

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def norm_loss(pred, labels):
    pred = F.log_softmax(pred, dim=1)

    nce = -1 * torch.sum(labels.float() * pred, dim=1).div(- pred.sum(dim=1))
    return nce.mean()

def compute_od(A, B):
    A2 = (A ** 2).sum(dim=1).reshape((-1, 1))
    B2 = (B ** 2).sum(dim=1)
    D = A2 + B2 - 2 * A.mm(B.t())
    return D.sqrt()

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,epoch):

    q = min(1.,0.01*(epoch+1))
    a = 0.1
    term1 = a * gce_loss(view1_predict,labels_1,q) + (1-a)*gce_loss(view2_predict,labels_2,q)
 
    def feature_alignment(c_labels, x, y):
        od = compute_od(x, y)
        loss1 =  ((x - y)**2).sum(1).sqrt().mean()
 
        k=1
        loss2 = ((- od * (1-c_labels)).topk(k=k)[0] + ( - od * (1-c_labels)).topk(k=k)[0] ).mean()

        return loss1*0.2 + loss2*0.2 
    

    Sim12 = calc_label_sim(labels_1, labels_2).float()
    term2 = feature_alignment(Sim12, view1_feature, view2_feature)

    im_loss = term1 +  term2 
    return im_loss



def train_model(model, data_loaders, optimizer, log, metric_loss, optimizer_loss, args, writer):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epoch):
        print('-' * 20)
        log.write('-' * 20 + '\n')
        print('Epoch {}/{}'.format(epoch, args.epoch))
        log.write('Epoch {}/{}'.format(epoch, args.epoch) + '\n')

        for phase in ['train', 'val']: # val
            if phase == 'train':
                model.train()
                running_loss = 0.0
                running_corrects_img = 0
                running_corrects_txt = 0
                # Iterate over data.
                for imgs, txts, labels in data_loaders[phase]:
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    optimizer_loss.zero_grad()

                    
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()


                    view1_feature, view2_feature, view1_predict, view2_predict = model(imgs, txts)
                    # loss = metric_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels)
                    loss = calc_loss(view1_feature, view2_feature, view1_predict,
                                     view2_predict, labels, labels, 0.1, 0.1,epoch)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer_loss.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))
                    running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                print('train epoch loss = {}'.format(epoch_loss))

            elif phase == 'val':
                model.eval()
                t_imgs, t_txts, t_labels = [], [], []

                with torch.no_grad():
                    for imgs, txts, labels in data_loaders[phase]:
                        if torch.cuda.is_available():
                                imgs = imgs.cuda()
                                txts = txts.cuda()
                                labels = labels.cuda()
                        t_view1_feature, t_view2_feature, _, _ = model(imgs, txts)
                        # t_view1_feature = F.normalize(t_view1_feature, p=2, dim=-1) 
                        # t_view2_feature = F.normalize(t_view2_feature, p=2, dim=-1) 

                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())

                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels).argmax(1)
                
                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
                print('{} epoch Loss = {}'.format(phase, epoch_loss))
                log.write('{} epoch Loss = {}\n'.format(phase, epoch_loss))
                writer.add_scalar('loss/' + phase, epoch_loss, epoch)
                writer.add_scalar('acc/' + phase + '_img2txt', img2text, epoch)
                writer.add_scalar('acc/' + phase + '_txt2img', txt2img, epoch)
                writer.add_scalar('acc/' + phase + '_avg', best_acc, epoch)
          
            if phase == 'val' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
                best_t_imgs, best_t_txts, best_t_labels = t_imgs, t_txts, t_labels
                best_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + '\n')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
