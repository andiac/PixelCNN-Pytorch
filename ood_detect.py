import torch
import torchvision
from torch.utils import data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Model import FullPixelCNN, LocalPixelCNN
from utils import discretized_mix_logistic_prob

full_model_path  = './Model/color_log_full/checkpoint_68.pt'
local_model_path = './Model/color_log_local/checkpoint_1.pt'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

cifar_val = torchvision.datasets.CIFAR10('./Data',
                                          train=False, 
                                          download=True, 
                                          transform=torchvision.transforms.ToTensor())
cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

svhn_val = torchvision.datasets.SVHN('./Data',
                                     split='test',
                                     download=True,
                                     transform=torchvision.transforms.ToTensor())
svhn_loader = data.DataLoader(svhn_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

full_model  = FullPixelCNN (res_num=5, in_channels=3, out_channels=100).to(device)
local_model = LocalPixelCNN(res_num=5, in_channels=3, out_channels=100).to(device)
full_model.load_state_dict(torch.load(full_model_path))
local_model.load_state_dict(torch.load(local_model_path))
full_model.eval()
local_model.eval()

with torch.no_grad():
    cifar_scores = []
    for x, y in tqdm(cifar_loader):
        x = x.to(device)
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        full_log_probs  = discretized_mix_logistic_prob(x, full_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)
        full_log_prob   = torch.sum(torch.sum(full_log_probs,  dim=2), dim=1) 
        score = full_log_prob - local_log_prob
        cifar_scores.append(score)
    cifar_score = torch.cat(cifar_scores)

    svhn_scores = []
    for x, y in tqdm(svhn_loader):
        x = x.to(device)
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        full_log_probs  = discretized_mix_logistic_prob(x, full_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)
        full_log_prob   = torch.sum(torch.sum(full_log_probs,  dim=2), dim=1) 
        score = full_log_prob - local_log_prob
        svhn_scores.append(score)
    svhn_score = torch.cat(svhn_scores)

    print("AUROC:")
    print(roc_auc_score(torch.cat((torch.ones(10000), torch.zeros(10000))).numpy(), torch.cat((cifar_score, svhn_score[:10000])).cpu().detach().numpy()))

