import numpy as np
import torch
import torch.utils.data as data
import os
from natsort import natsorted


class DatasetFromFolderCO2_fft(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 15,
                 max_value_y = 17,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2_fft, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.idx_next = self.idx + 1220
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.set_input_seism_next = np.array(self.filenames_input_seism)[self.idx_next] 
        self.set_target_next = np.array(self.filenames_target)[self.idx_next]
        self.crop_size = crop_size
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        set = []
        for k in range(len(self.set_input_seism_next)):
            set.append(self.path_input_seism + self.set_input_seism_next[k] )
        self.set_input_seism_next = set
        
        set = []
        for k in range(len(self.set_target_next)):
            set.append(self.path_target + self.set_target_next[k])
        self.set_target_next = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
        
        fn_input_seism_next = os.path.join(self.set_input_seism_next[index])
        input_seism_next = np.load(fn_input_seism_next)
        
        fn_target_next = os.path.join(self.set_target_next[index])
        target_next = np.load(fn_target_next)
                                    
        input_seism_real = input_seism.real / self.max_value_x
        target_real = target.real / self.max_value_y
                
        input_seism_next_real = input_seism_next.real / self.max_value_x
        target_next_real = target_next.real / self.max_value_y
        
        input_seism_imag = input_seism.imag / self.max_value_x
        target_imag = target.imag / self.max_value_y
                
        input_seism_next_imag = input_seism_next.imag / self.max_value_x
        target_next_imag = target_next.imag / self.max_value_y
        
        if self.crop_size == 512:
            input_seism_real = input_seism_real[:,:512]
            target_next_real = target_next_real[:,:512]
            input_seism_next_real = input_seism_next_real[:,:512]
            target_real = target_real[:,:512]
            input_seism_imag = input_seism_imag[:,:512]
            target_imag = target_imag[:,:512]
            input_seism_next_imag = input_seism_next_imag[:,:512]
            target_next_imag = target_next_imag[:,:512]

        
        if self.transform is not None:
            input_seism_real = self.transform_target(input_seism_real.copy())
            input_seism_real = input_seism_real.type(torch.cuda.FloatTensor)
            target_real = self.transform_target(target_real.copy())
            target_real = target_real.type(torch.cuda.FloatTensor)
            
            input_seism_next_real = self.transform_target(input_seism_next_real.copy())
            input_seism_next_real = input_seism_next_real.type(torch.cuda.FloatTensor)
            target_next_real = self.transform_target(target_next_real.copy())
            target_next_real = target_next_real.type(torch.cuda.FloatTensor)
            
            input_seism_imag = self.transform_target(input_seism_imag.copy())
            input_seism_imag = input_seism_imag.type(torch.cuda.FloatTensor)
            target_imag = self.transform_target(target_imag.copy())
            target_imag = target_imag.type(torch.cuda.FloatTensor)
            
            input_seism_next_imag = self.transform_target(input_seism_next_imag.copy())
            input_seism_next_imag = input_seism_next_imag.type(torch.cuda.FloatTensor)
            target_next_imag = self.transform_target(target_next_imag.copy())
            target_next_imag = target_next_imag.type(torch.cuda.FloatTensor)
        
        input = torch.cat((input_seism_real,input_seism_imag))
        target = torch.cat((target_real,target_imag))
        
        input_next = torch.cat((input_seism_next_real,input_seism_next_imag))
        target_next = torch.cat((target_next_real,target_next_imag))
            
        return input, target, input_next, target_next
    
    def __len__(self):
        return len(self.set_input_seism)
    
    
class DatasetFromFolderCO2_fft_without_adding(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 1,
                 max_value_y = 1,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2_fft_without_adding, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.crop_size = crop_size
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
                    
        input_seism_real = input_seism.real / self.max_value_x
        target_real = target.real / self.max_value_y
                
        input_seism_imag = input_seism.imag / self.max_value_x
        target_imag = target.imag / self.max_value_y
             
        if self.crop_size == 512:
            input_seism_real = input_seism_real[:,:512]
            target_real = target_real[:,:512]
            input_seism_imag = input_seism_imag[:,:512]
            target_imag = target_imag[:,:512]
            
        
        if self.transform is not None:
            input_seism_real = self.transform_target(input_seism_real.copy())
            input_seism_real = input_seism_real.type(torch.cuda.FloatTensor)
            target_real = self.transform_target(target_real.copy())
            target_real = target_real.type(torch.cuda.FloatTensor)
            
            input_seism_imag = self.transform_target(input_seism_imag.copy())
            input_seism_imag = input_seism_imag.type(torch.cuda.FloatTensor)
            target_imag = self.transform_target(target_imag.copy())
            target_imag = target_imag.type(torch.cuda.FloatTensor)
            
        input = torch.cat((input_seism_real,input_seism_imag))
        target = torch.cat((target_real,target_imag))
        
        return input, target
    
    def __len__(self):
        return len(self.set_input_seism)
    

class DatasetFromFolderCO2_dS_dM(data.Dataset):
    def __init__(self, dir_input_model,
                 dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 1,
                 max_value_y = 1,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2_dS_dM, self).__init__()
        self.path_input_model = os.path.join(dir_input_model) #X = 55 m (all)
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.filenames_input_model = [x for x in natsorted(os.listdir(self.path_input_model))]
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_model = np.array(self.filenames_input_model)[self.idx] 
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.crop_size = crop_size
        
        set = []
        for k in range(len(self.set_input_model)):
            set.append(self.path_input_model + self.set_input_model[k] )
        self.set_input_model = set
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
        
    def __getitem__(self, index):
        fn_input_model = os.path.join(self.set_input_model[index])
        input_model = np.load(fn_input_model)
        
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
                    
        input_seism_real = input_seism.real / self.max_value_x
        target_real = target.real / self.max_value_y
                
        input_seism_imag = input_seism.imag / self.max_value_x
        target_imag = target.imag / self.max_value_y
             
        if self.crop_size == 512:
            input_seism_real = input_seism_real[:,:512]
            target_real = target_real[:,:512]
            input_seism_imag = input_seism_imag[:,:512]
            target_imag = target_imag[:,:512]
            
        
        if self.transform is not None:
            input_model = self.transform_target(input_model.copy())
            input_model = input_model.type(torch.cuda.FloatTensor)
            
            input_seism_real = self.transform_target(input_seism_real.copy())
            input_seism_real = input_seism_real.type(torch.cuda.FloatTensor)
            target_real = self.transform_target(target_real.copy())
            target_real = target_real.type(torch.cuda.FloatTensor)
            
            input_seism_imag = self.transform_target(input_seism_imag.copy())
            input_seism_imag = input_seism_imag.type(torch.cuda.FloatTensor)
            target_imag = self.transform_target(target_imag.copy())
            target_imag = target_imag.type(torch.cuda.FloatTensor)
            
        input = torch.cat((input_model, input_seism_real,input_seism_imag))
        target = torch.cat((target_real,target_imag))
        
        return input, target
    
    def __len__(self):
        return len(self.set_input_seism)
    
    
    

class DatasetFromFolderCO2_fft_without_adding_and_with_crop(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 15,
                 max_value_y = 17,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2_fft_without_adding_and_with_crop, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.crop_size = crop_size
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
                    
        input_seism_real = input_seism.real / self.max_value_x
        target_real = target.real / self.max_value_y
                
        input_seism_imag = input_seism.imag / self.max_value_x
        target_imag = target.imag / self.max_value_y
             
        input_seism_real_1 = input_seism_real[:,:512]
        target_real_1 = target_real[:,:512]
        input_seism_imag_1 = input_seism_imag[:,:512]
        target_imag_1 = target_imag[:,:512]
        input_seism_real_2 = input_seism_real[:,-512:]
        target_real_2 = target_real[:,-512:]
        input_seism_imag_2 = input_seism_imag[:,-512:]
        target_imag_2 = target_imag[:,-512:]
        
        if self.transform is not None:
            input_seism_real_1 = self.transform_target(input_seism_real_1.copy())
            input_seism_real_1 = input_seism_real_1.type(torch.cuda.FloatTensor)
            target_real_1 = self.transform_target(target_real_1.copy())
            target_real_1 = target_real_1.type(torch.cuda.FloatTensor)
            
            input_seism_imag_1 = self.transform_target(input_seism_imag_1.copy())
            input_seism_imag_1 = input_seism_imag_1.type(torch.cuda.FloatTensor)
            target_imag_1 = self.transform_target(target_imag_1.copy())
            target_imag_1 = target_imag_1.type(torch.cuda.FloatTensor)
            
            input_seism_real_2 = self.transform_target(input_seism_real_2.copy())
            input_seism_real_2 = input_seism_real_2.type(torch.cuda.FloatTensor)
            target_real_2 = self.transform_target(target_real_2.copy())
            target_real_2 = target_real_2.type(torch.cuda.FloatTensor)
            
            input_seism_imag_2 = self.transform_target(input_seism_imag_2.copy())
            input_seism_imag_2 = input_seism_imag_2.type(torch.cuda.FloatTensor)
            target_imag_2 = self.transform_target(target_imag_2.copy())
            target_imag_2 = target_imag_2.type(torch.cuda.FloatTensor)
            
        input = torch.cat((input_seism_real_1,input_seism_imag_1, input_seism_real_2,input_seism_imag_2))
        target = torch.cat((target_real_1,target_imag_1,target_real_2,target_imag_2))
        
        
        return input, target
    
    def __len__(self):
        return len(self.set_input_seism)
    
class DatasetFromFolderCO2(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 1,
                 max_value_y = 1,
                 number_of_sources = 126,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        self.number_of_sources = number_of_sources
        self.idx = np.load(train_dir)
        self.idx_next = self.idx + self.number_of_sources
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.set_input_seism_next = np.array(self.filenames_input_seism)[self.idx_next] 
        self.set_target_next = np.array(self.filenames_target)[self.idx_next]
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        set = []
        for k in range(len(self.set_input_seism_next)):
            set.append(self.path_input_seism + self.set_input_seism_next[k] )
        self.set_input_seism_next = set
        
        set = []
        for k in range(len(self.set_target_next)):
            set.append(self.path_target + self.set_target_next[k])
        self.set_target_next = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
        
        fn_input_seism_next = os.path.join(self.set_input_seism_next[index])
        input_seism_next = np.load(fn_input_seism_next)
        
        fn_target_next = os.path.join(self.set_target_next[index])
        target_next = np.load(fn_target_next)
                    
        input_seism_real = input_seism.real / self.max_value_x
        target_real = target.real / self.max_value_y
                
        input_seism_imag = input_seism.imag / self.max_value_x
        target_imag = target.imag / self.max_value_y
        
        if self.crop_size == 512:
            input_seism_real = input_seism_real[:,:512]
            target_real = target_real[:,:512]
            input_seism_imag = input_seism_imag[:,:512]
            target_imag = target_imag[:,:512]
        
        if self.transform is not None:
            input_seism_real = self.transform_target(input_seism_real.copy())
            input_seism_real = input_seism_real.type(torch.cuda.FloatTensor)
            target_real = self.transform_target(target_real.copy())
            target_real = target_real.type(torch.cuda.FloatTensor)
                
            input_seism_imag = self.transform_target(input_seism_imag.copy())
            input_seism_imag = input_seism_imag.type(torch.cuda.FloatTensor)
            target_imag = self.transform_target(target_imag.copy())
            target_imag = target_imag.type(torch.cuda.FloatTensor)
            
        input = torch.cat((input_seism_real,input_seism_imag))
        target = torch.cat((target_real,target_imag))
        
        input_seism_real_next = input_seism_next.real / self.max_value_x
        target_real_next = target_next.real / self.max_value_y
                
        input_seism_imag_next = input_seism_next.imag / self.max_value_x
        target_imag_next = target_next.imag / self.max_value_y
        
        if self.crop_size == 512:
            input_seism_real_next = input_seism_real_next[:,:512]
            target_real_next = target_real_next[:,:512]
            input_seism_imag_next = input_seism_imag_next[:,:512]
            target_imag_next = target_imag_next[:,:512]
        
        if self.transform is not None:
            input_seism_real_next = self.transform_target(input_seism_real_next.copy())
            input_seism_real_next = input_seism_real_next.type(torch.cuda.FloatTensor)
            target_real_next = self.transform_target(target_real_next.copy())
            target_real_next = target_real_next.type(torch.cuda.FloatTensor)
            
            input_seism_imag_next = self.transform_target(input_seism_imag_next.copy())
            input_seism_imag_next = input_seism_imag_next.type(torch.cuda.FloatTensor)
            target_imag_next = self.transform_target(target_imag_next.copy())
            target_imag_next = target_imag_next.type(torch.cuda.FloatTensor)
            
        input_next = torch.cat((input_seism_real_next,input_seism_imag_next))
        target_next = torch.cat((target_real_next,target_imag_next))
            
        return input, target, input_next, target_next
    
    def __len__(self):
        return len(self.set_input_seism)
    

class DatasetFromFolderCO2_2(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value = 1,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolderCO2_2, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.idx_next = self.idx + 1220
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        self.set_input_seism_next = np.array(self.filenames_input_seism)[self.idx_next] 
        self.set_target_next = np.array(self.filenames_target)[self.idx_next]
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        set = []
        for k in range(len(self.set_input_seism_next)):
            set.append(self.path_input_seism + self.set_input_seism_next[k] )
        self.set_input_seism_next = set
        
        set = []
        for k in range(len(self.set_target_next)):
            set.append(self.path_target + self.set_target_next[k])
        self.set_target_next = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value = max_value
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
        
        fn_input_seism_next = os.path.join(self.set_input_seism_next[index])
        input_seism_next = np.load(fn_input_seism_next)
        
        fn_target_next = os.path.join(self.set_target_next[index])
        target_next = np.load(fn_target_next)
                    
        input_seism = input_seism[:-1,:-904]
        target = target[:-1,:-904]
                
        input_seism = input_seism / self.max_value
        target = target / self.max_value
        
        input_seism_next = input_seism_next[:-1,:-904]
        target_next = target_next[:-1,:-904]
                
        input_seism_next = input_seism_next / self.max_value
        target_next = target_next / self.max_value
        
        diff_coarse = input_seism_next - input_seism
        diff_fine = target_next - target
        
        
        if self.transform is not None:
            input_seism = self.transform_target(input_seism.copy())
            input_seism = input_seism.type(torch.cuda.FloatTensor)
            diff_coarse = self.transform_target(diff_coarse.copy())
            diff_coarse = diff_coarse.type(torch.cuda.FloatTensor)
            
            diff_fine = self.transform_target(diff_fine.copy())
            diff_fine = diff_fine.type(torch.cuda.FloatTensor)
            target_next = self.transform_target(target_next.copy())
            target_next = target_next.type(torch.cuda.FloatTensor)
            
        return input_seism, diff_coarse, diff_fine, target_next
    
    def __len__(self):
        return len(self.set_input_seism)
    
    
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 1,
                 max_value_y=1,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolder, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
                    
        input_seism = input_seism[:-1,:-904]
        target = target[:-1,:-904]
                
        input_seism = input_seism / self.max_value_x
        target = target / self.max_value_y
        
        if self.transform is not None:
            input_seism = self.transform_target(input_seism.copy())
            input_seism = input_seism.type(torch.cuda.FloatTensor)
            target = self.transform_target(target.copy())
            target = target.type(torch.cuda.FloatTensor)
            
        return input_seism, target
    
    def __len__(self):
        return len(self.set_input_seism)
    
    
    
class DatasetFromFolder_last(data.Dataset):
    def __init__(self, dir_input_seism,
                 dir_target,
                 train_dir, 
                 max_value_x = 1,
                 max_value_y=1,
                 direction = 'BtoA',transform=None, transform_target = None, 
                 input_size=None, resize_scale=None, crop_size=None, 
                 fliplr = False, flipud = False):
        super(DatasetFromFolder_last, self).__init__()
        self.path_input_seism = os.path.join(dir_input_seism) #X = 55 m (all)
        self.path_target = os.path.join(dir_target) #Y = 25m (all)
        
        self.idx = np.load(train_dir)
        self.filenames_input_seism = [x for x in natsorted(os.listdir(self.path_input_seism))]
        self.filenames_target = [x for x in natsorted(os.listdir(self.path_target))]
        
        self.set_input_seism = np.array(self.filenames_input_seism)[self.idx] 
        self.set_target = np.array(self.filenames_target)[self.idx]
        
        set = []
        for k in range(len(self.set_input_seism)):
            set.append(self.path_input_seism + self.set_input_seism[k] )
        self.set_input_seism = set
        
        set = []
        for k in range(len(self.set_target)):
            set.append(self.path_target + self.set_target[k])
        self.set_target = set
        
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        self.max_value_x = max_value_x
        self.max_value_y = max_value_y
        
    def __getitem__(self, index):
        fn_input_seism = os.path.join(self.set_input_seism[index])
        input_seism = np.load(fn_input_seism)
        
        fn_target = os.path.join(self.set_target[index])
        target = np.load(fn_target)
                    
        input_seism = input_seism[:-1,1416:]
        target = target[:-1,1416:]
                
        input_seism = input_seism / self.max_value_x
        target = target / self.max_value_y
        
        if self.transform is not None:
            input_seism = self.transform_target(input_seism.copy())
            input_seism = input_seism.type(torch.cuda.FloatTensor)
            target = self.transform_target(target.copy())
            target = target.type(torch.cuda.FloatTensor)
            
        return input_seism, target
    
    def __len__(self):
        return len(self.set_input_seism)