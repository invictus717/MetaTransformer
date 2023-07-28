
import foolbox as fb
import torch
import torch.nn as nn

class Attack(): 
    """
    This class used to generate adversarial images.
    when create object specify  epsilon: float, attack_type: 'FGSM, CW, BIM, L2PGD, PGD, LinfBIM'. 

    generate method return images and success tensors. 
    test_model method, give the accuracy of the model after passing the adversarial examples. 

    succecces tensor shows whether the example succed to fool the model or not

    """
    def __init__(self, epsilon, attack_type, model) :
        self.epsilon= epsilon
        self.attack_type = attack_type
        self.model_fool = fb.models.PyTorchModel(model ,bounds=(0,1))        

    def FGSM(self, samples, labels):
        """
        Generate FGSM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """
        attack_func = fb.attacks.FGSM()
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success   
        
    def L2PGD(self, samples, labels): 
        """
        Generate L2 PGD attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        attack_func = fb.attacks.L2PGD()
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success
                                            

    def CW(self, samples, labels): 
        """
        Generate Carlini & Wagner attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        attack_func = fb.attacks.L2CarliniWagnerAttack(6,1000,0.01,0)
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons= self.epsilon)
        print(f'Sum = {sum(success)}')  
        return adv_images, success

    def BIM(self, samples, labels):
        """
        Generate BIM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        attack_func = fb.attacks.L2BasicIterativeAttack()
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success 
    
    def PGD(self, samples, labels):
        """
        Generate Linf PGD attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        attack_func = fb.attacks.PGD()
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success

    def LinfBIM(self, samples, labels):
        """
        Generate Linf BIM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        attack_func = fb.attacks.LinfBasicIterativeAttack()
        _, adv_images, success = attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success  

    def generate_attack(self, samples, labels):
        """
        Generate attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images -> generated from the clean images 
            success tensor -> shows whether the attack succeded in fooling the model or not

        """

        if self.attack_type == 'FGSM': 
            adv_img, success = self.FGSM(samples, labels)
            return  adv_img, success
        elif self.attack_type == 'CW': 
            adv_img, success = self.CW(samples, labels)           
            return  adv_img, success
        elif self.attack_type == 'L2PGD': 
            adv_img, success = self.L2PGD(samples, labels)
            return  adv_img, success
        elif self.attack_type == 'BIM': 
            adv_img, success = self.BIM(samples, labels)
            return  adv_img, success
        elif self.attack_type == 'PGD':
            adv_img, success = self.PGD(samples, labels)
            return  adv_img, success
        elif self.attack_type =='LinfBIM': 
            adv_img, success = self.LinfBIM(samples, labels)
            return adv_img, success
        else: 
            print(f'Attacks of type {self.attack_type} is not supported') 

