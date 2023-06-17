import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from attMSTModel import Model
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm


test_content_dir = './newContent'
test_style_dir = './newStyle'
test_dataset = PreprocessDataset(test_content_dir, test_style_dir)

print(f'Length of test image pairs: {len(test_dataset)}')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_iter = iter(test_loader)
device = torch.device(f'cuda:0')
model = Model()
model.load_state_dict(torch.load('./AttMST/model_state/1_epoch.pth', map_location=lambda storage, loc: storage))
model = model.to(device)
for i in range(7):
    try:
        content, style = next(test_iter)
    except StopIteration:
        test_iter = iter(test_loader)
        content, style = next(test_iter)
    content = content.to(device)
    style = style.to(device)
    with torch.no_grad():
        out = model.generate(content, style)
    content = denorm(content, device)
    style = denorm(style, device)
    out = denorm(out, device)
    res = torch.cat([content, style, out], dim=0)
    res = res.to('cpu')
    save_image(res,f'./attOutput/{i+1}.png')
# def main():
#     argsbatch_size = 16
#     argsepoch = 6
#     argsgpu = 0 
#     argslearning_rate =5e-5
#     argslr_decay=6e-5
#     argsstyle_weight =3.0 
#     argscontent_weight =1.0 
#     argssnapshot_interval =275 
#     argstrain_content_dir ='./dataset/flickr30k_images'
#     argstrain_style_dir ='./dataset/ArtWithNum'
#     argstest_content_dir='./content'
#     argstest_style_dir ='./style'
#     argssave_dir ='./AttMST'
#     argsreuse =None

#     # create directory to save
#     if not os.path.exists(argssave_dir):
#         os.mkdir(argssave_dir)

#     loss_dir = f'{argssave_dir}/loss'
#     model_state_dir = f'{argssave_dir}/model_state'
#     image_dir = f'{argssave_dir}/image'

#     if not os.path.exists(loss_dir):
#         os.mkdir(loss_dir)
#         os.mkdir(model_state_dir)
#         os.mkdir(image_dir)

#     # set device on GPU if available, else CPU
#     if torch.cuda.is_available() and argsgpu >= 0:
#         device = torch.device(f'cuda:{argsgpu}')
#         print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
#     else:
#         device = 'cpu'

#     print(f'# Minibatch-size: {argsbatch_size}')
#     print(f'# epoch: {argsepoch}')
#     print('')

#     # prepare dataset and dataLoader
#     train_dataset = PreprocessDataset(argstrain_content_dir, argstrain_style_dir)
#     test_dataset = PreprocessDataset(argstest_content_dir, argstest_style_dir)
#     iters = len(train_dataset)
#     print(f'Length of train image pairs: {iters}')
#     print(f'Length of test image pairs: {len(test_dataset)}')
#     train_loader = DataLoader(train_dataset, batch_size=argsbatch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=argsbatch_size, shuffle=False)
#     test_iter = iter(test_loader)
    
#     # set model and optimizer
#     model = AttMultiSTModel().to(device)
#     if argsreuse is not None:
#         model.load_state_dict(torch.load(argsreuse))
#     optimizer = Adam(model.parameters(), lr=argslearning_rate)

#     # start training
#     loss_list,loss_c_list,loss_s_list = [],[],[]
#     for e in range(1, argsepoch + 1):
#         print(f'Start {e} epoch')
#         for i, (content, style) in tqdm(enumerate(train_loader, 1)):
#             content = content.to(device)
#             style = style.to(device)

#             loss_c, loss_s, l_identity1, l_identity2 = model(content, style)
#             loss_c = argscontent_weight * loss_c
#             loss_s = argsstyle_weight * loss_s
#             loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1

#             loss_list.append(loss.item())
#             loss_c_list.append(loss_c.item())
#             loss_s_list.append(loss_s.item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(f'[{e}/total {argsepoch} epoch],[{i} /'
#                   f'total {round(iters/argsbatch_size)} iteration]: {loss.item()}; content loss:{loss_c.item()}; style loss:{loss_s.item()}')

#             if i % argssnapshot_interval == 0:
#                 try:
#                     content, style = next(test_iter)
#                 except StopIteration:
#                     test_iter = iter(test_loader)
#                     content, style = next(test_iter)
#                 content = content.to(device)
#                 style = style.to(device)
#                 with torch.no_grad():
#                     out = model.generate(content, style)
#                 content = denorm(content, device)
#                 style = denorm(style, device)
#                 out = denorm(out, device)
#                 res = torch.cat([content, style, out], dim=0)
#                 res = res.to('cpu')
#                 save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=argsbatch_size)
#                 torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
#                 with open(f'{loss_dir}/loss_log.txt', 'w') as f:
#                     for l in loss_list:
#                         f.write(f'{l}\n')
#                 with open(f'{loss_dir}/loss_c_log.txt', 'w') as f:
#                     for l in loss_c_list:
#                         f.write(f'{l}\n')
#                 with open(f'{loss_dir}/loss_s_log.txt', 'w') as f:
#                     for l in loss_s_list:
#                         f.write(f'{l}\n')
#     plt.plot(range(len(loss_list)), loss_list)
#     plt.xlabel('iteration')
#     plt.ylabel('loss')
#     plt.title('train loss')
#     plt.savefig(f'{loss_dir}/train_loss.png')

#     print(f'Loss saved in {loss_dir}')


# if __name__ == '__main__':
#     main()