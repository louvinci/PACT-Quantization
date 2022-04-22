import torch

if __name__=="__main__":
    
    ckpt = torch.load('weights_best.pt',map_location=torch.device('cpu'))
    best_acc = ckpt['acc']
    epoch = ckpt['epoch']
    print(best_acc,epoch)

    state_dict = ckpt['state_dict']
    id = 0
    for k,v in state_dict.items():
        if 'scale' in k:
            print(k,v.item())
            id+=1
    print('total {} layers'.format(id))
