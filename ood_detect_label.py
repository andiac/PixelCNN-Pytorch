import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Model import FullPixelCNN, LocalPixelCNN
from utils import discretized_mix_logistic_prob
from myutils import rescaling

full_model_paths  = ['./Model/color_log_full_0/checkpoint_500.pt',
                     './Model/color_log_full_1/checkpoint_500.pt',
                     './Model/color_log_full_2/checkpoint_500.pt',
                     './Model/color_log_full_3/checkpoint_500.pt',
                     './Model/color_log_full_4/checkpoint_500.pt',
                     './Model/color_log_full_5/checkpoint_500.pt',
                     './Model/color_log_full_6/checkpoint_500.pt',
                     './Model/color_log_full_7/checkpoint_500.pt',
                     './Model/color_log_full_8/checkpoint_500.pt',
                     './Model/color_log_full_9/checkpoint_500.pt']

local_model_path = './Model/color_log_local/svhn_local_10rs_ks7.pt'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToTensor(),
    rescaling])

cifar_val = torchvision.datasets.CIFAR10('./Data',
                                          train=False, 
                                          download=True, 
                                          transform=transform)
cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

svhn_val = torchvision.datasets.SVHN('./Data',
                                     split='test',
                                     download=True,
                                     transform=transform)
svhn_loader = data.DataLoader(svhn_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

full_models = []
for i in range(10):
    full_models.append(FullPixelCNN (res_num=10, in_channels=3, out_channels=100).to(device))
    full_models[i].load_state_dict(torch.load(full_model_paths[i], map_location=device))
    full_models[i].eval()

local_model = LocalPixelCNN(res_num=10, in_channels=3, out_channels=100).to(device)
local_model.load_state_dict(torch.load(local_model_path, map_location=device))
local_model.eval()

with torch.no_grad():
    cifar_scores = []
    for x, y in tqdm(cifar_loader):
        x = x.to(device)
        full_log_probs = []
        for i in range(10):
            full_log_probs.append(torch.sum(torch.sum(discretized_mix_logistic_prob(x, full_models[i](x)),  dim=2), dim=1))
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)

        # score = full_log_prob - local_log_prob
        score = torch.max(torch.stack(full_log_probs), dim=0)[0] - local_log_prob
        cifar_scores.append(score)
    cifar_score = torch.cat(cifar_scores)

    svhn_scores = []
    for x, y in tqdm(svhn_loader):
        x = x.to(device)
        full_log_probs = []
        for i in range(10):
            full_log_probs.append(torch.sum(torch.sum(discretized_mix_logistic_prob(x, full_models[i](x)),  dim=2), dim=1))
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)
        # score = full_log_prob - local_log_prob
        score = torch.max(torch.stack(full_log_probs), dim=0)[0] - local_log_prob
        svhn_scores.append(score)
    svhn_score = torch.cat(svhn_scores)

    print("AUROC:")
    print(roc_auc_score(torch.cat((torch.ones(10000), torch.zeros(10000))).numpy(), torch.cat((cifar_score, svhn_score[:10000])).cpu().detach().numpy()))

