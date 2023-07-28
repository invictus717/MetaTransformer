import os
import torch
from attack import Attack
import numpy as np 

import matplotlib.pyplot as plt 

from torchvision.utils import save_image
from autoattack import AutoAttack


def generate_save_attacks(attack_names, model, samples, classes ,attack_image_dir, epsilon = 0.03, batch_size=30):
    """
    it saves attack images generated from test images in test attacks folder in a dirr folder. 
    inside test attacks we will have folder for each attack, and inside each attack folder we will have folder for classes. 

    Args: 
        attack_names --> list of attacks. 
        model --> model want to attack. 
        samples --> data_loaders['test']
        classes --> list of classes names. 
        attack_image_dir --> root directory for attack images to be saved in
    
    Doesnt return any values. 

    
    """
    for attack in attack_names: 
        
        attack_folder = f'Test_attacks_{attack}'
        print(attack_image_dir)
        if not os.path.exists(os.path.join(attack_image_dir, attack_folder)): 
            os.makedirs(os.path.join(attack_image_dir, attack_folder))


        inter_ = os.path.join(attack_image_dir, attack_folder) + '/'
        for classe in classes: 
            if not os.path.exists(os.path.join(inter_, classe)):
                os.makedirs(os.path.join(inter_, classe))
        

    for attack_name in attack_names:
        if attack_name != 'AUTOPGD': 
            batchNum = 0
            model.eval()
            attack = Attack(epsilon= epsilon , attack_type= attack_name, model=model)

            for im, lab in samples:
                im = im.cuda()
                lab = lab.cuda()
                adv_img, _ = attack.generate_attack(im, labels=lab)
                print('Batch')
                count = 0 
                for image, label in zip(adv_img, lab):
                    if (lab[count]):
                        save_image(image, os.path.join(attack_image_dir, f'Test_attacks_{attack_name}/{classes[1]}/'  + str(batchNum) + "-" + str(count) + attack_name + ".png"))
                    else:
                        save_image(image, os.path.join(attack_image_dir, f'Test_attacks_{attack_name}/{classes[0]}/'  + str(batchNum) + "-" + str(count) + attack_name + ".png"))
                    count += 1
                batchNum += 1

        elif attack_name == 'AUTOPGD': 
            batchNum = 0
            adversary = AutoAttack(model=model, eps=epsilon, version='custom', norm='Linf', attacks_to_run=['apgd-ce'])

            for im, lab in samples:
                im = im.cuda()
                lab = lab.cuda()
                adv_img = adversary.run_standard_evaluation(im,lab, bs=lab.shape[0])

                count = 0 
                for image, label in zip(adv_img, lab):
                    if (lab[count]):
                        save_image(image, os.path.join(attack_image_dir, f'Test_attacks_{attack_name}/{classes[1]}/'  + str(batchNum) + "-" + str(count) + attack_name + ".png"))
                    else:
                        save_image(image, os.path.join(attack_image_dir, f'Test_attacks_{attack_name}/{classes[0]}/'  + str(batchNum) + "-" + str(count) + attack_name + ".png"))
                    count += 1
                batchNum += 1
 

def get_classifiers_list(MLP_path, num_classifiers=5): 
    """
    Return list of intermdiate MLPs. 
    
    Args: 
        MLP_path: Path of the downloaded MLPs directory.
    
    """
    i=0
    classifiers_list = [0]*num_classifiers
    for classif in sorted(os.listdir(MLP_path)): 
        classifiers_list[i] = torch.load(os.path.join(MLP_path, classif)).eval().cuda()
        i+=1
        print(f'MLP {i} is loaded!')
    return classifiers_list

def frob_norm_kl_matrix(stacked_tesnor,num_classifiers=5): 
    frob_values = []
    for sample in stacked_tesnor:
        div_matrix = torch.zeros((num_classifiers+1,num_classifiers+1)) #initialize zero 6x6 tensor
        for i in range (num_classifiers+1): #loop over classifiers and MLP head (take one only)
            for j in range(num_classifiers+1): #loop over classifiers and MLP head
                x2 = torch.nn.functional.kl_div(sample[i].log(),sample[j].log(), reduction='sum', log_target=True).item() 
                div_matrix[i,j] = x2
        frob_norm = np.sqrt(torch.sum(torch.square(div_matrix)).item())
        frob_values.append(frob_norm)
    return frob_values

def roc(attack_name, frob_dict, threshold): 
    tpr_list= []
    fpr_list = []
    for i in threshold:
        fp = sum(frob_dict['clean'] >= i).item()
        tn = sum(frob_dict['clean'] < i).item()
        tp = sum(frob_dict[attack_name] >= i).item()
        fn = sum(frob_dict[attack_name] < i).item()
        fpr = (fp)/(fp+tn)
        tpr = (tp)/(tp+fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list,fpr_list,threshold


def frobenius_norm(data_loader, model, mlps_list): 
    frob_norm_values = []

    for images, _ in data_loader: #loop over images 
        final_probs = []
        images = images.cuda()
        vit_probs = torch.softmax(model(images).detach().cpu(),dim=-1)
        final_probs.append(vit_probs.detach().cpu())
        x = model.patch_embed(images)
        x_0 = model.pos_drop(x)

        i=0
        for mlp in mlps_list:
            x_0 = model.blocks[i](x_0)
            mlp_prob = torch.softmax(mlp(x_0).detach().cpu(),dim=-1)
            final_probs.append(mlp_prob.detach().cpu())
            i+=1

        stacked_tesnor = torch.stack(final_probs,dim=1)
        frob_list = frob_norm_kl_matrix(stacked_tesnor)
        frob_norm_values = frob_norm_values + frob_list

    return frob_norm_values

def plot_roc (tpr_list, fpr_list, attack_name): 
    plt.figure(figsize=(10,6))
    plt.plot(fpr_list,tpr_list, '-', label = attack_name)
    plt.title(f'ROC_{attack_name}_Attack')
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.savefig(f'{attack_name}_ROC_Curve', bbox_inches='tight')
    plt.show()