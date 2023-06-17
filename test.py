import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from multiAdainModel import Model
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def list_files(directory):
    return os.listdir(directory)


def main(contentIndex,styleIndex,alpha=1.0,multi=False,num_lst = [0],weightLst=[0.25,0.25,0.25,0.25]):
    content_dir = "newContent_resized"
    style_dir = "newStyle_resized"
    num_epoch = 10

    content_name_lst = [f[:-4] for f in list_files(content_dir)]
    style_name_lst = [f[:-4] for f in list_files(style_dir)]
    content_name=content_name_lst[contentIndex]
    style_name=style_name_lst[styleIndex]
    num_lst = num_lst
    style_names = [ style_name_lst[i] for i in num_lst]
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=f'./{content_dir}/{content_name}.jpg',
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', nargs='+', type=str, default=[f'./{style_dir}/{name}.jpg' for name in style_names],
                        help='Style image paths e.g. image1.jpg image2.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=f'./{content_name}_{style_name}_{alpha}.jpg',
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=alpha,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--weightLst', '-w', type=list, default=weightLst,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default=f'./multi_layer_StyleTransfer/model_state/{num_epoch}_epoch.pth',
                        help='save directory for result and loss')
    #multi_layer_StyleTransfer
    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(args.content)
    c_tensor = trans(c).unsqueeze(0).to(device)
    if multi:
        s_images = [Image.open(style) for style in args.style]
        s_tensor = [trans(s).unsqueeze(0).to(device) for s in s_images]
        s = s_images[0]
    else:
        s = Image.open(f'./{style_dir}/{style_name}.jpg')
        s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor)#,weight_lst=args.weightLst)
    out = denorm(out, device)
        
    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name = f'{c_name}_{s_name}'

    save_image(out, f'{args.output_name}', nrow=1)
    # o = Image.open(f'{args.output_name}')

    # demo = Image.new('RGB', (c.width * 2, c.height))
    # o = o.resize(c.size)
    
    # c = Image.open(args.content)
    # s = Image.open(f'./{style_dir}/{style_name}.jpg')
    # s = s.resize((i // 4 for i in c.size))
    # demo.paste(c, (0, 0))
    # demo.paste(o, (c.width, 0))
    # demo.paste(s, (c.width, c.height - s.height))
    # demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)

    # c.paste(s,  (0, c.height - s.height))
    # c.save(f'{content_name}_with_{style_name}.jpg', quality=95)

    # print(f'result saved into files starting with {args.output_name}')


if __name__ == '__main__':
    main(1,1)
