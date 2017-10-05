import torch
import numpy as np
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

import os, sys, glob
import time




class Places: 
    
    def __init__(self, arch, use_GPU): 
        self.ti = time.time()
        self.arch = arch 
        self.useGPU = use_GPU
        self.centre_crop = trn.Compose([
            trn.Scale(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.probabilities = {} 
        self.classifications = {}
        self.load_pre_weights()
        self.load_class_label()
        self.get_classifications('test_images/classroom1/*.jpg')
        self.elapsed_time = time.time() - self.ti 
        self.print_time_per_img() 

    def load_pre_weights(self): 
        model_weight = 'whole_%s_places365.pth.tar' % self.arch

        if not os.access(model_weight, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/whole_%s_places365.pth.tar' % self.arch
            os.system('wget ' + weight_url)
        
        if self.useGPU == 1:
            self.model = torch.load(model_weight)
        else:
            self.model = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

        self.model.eval()

    def load_class_label(self):
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        self.classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)

    def get_classifications(self,img_directory):
        img_directory = 'test_images/hallway1/*.jpg'
        
        filenames = []
        
        for name in glob.glob(img_directory):
            filenames.append(name)

        filenames = sorted(filenames)
        self.total_images = len(filenames)

        categories_turtlebot = ['classroom', 'office', 'corridor', 'conference_room', 'kitchen']
        best_categories = np.zeros([len(categories_turtlebot), 1])
        count = 0
        # For storing running categories
        run_length = 10
        best_category_order = np.array([],dtype='int64')
        max_occurrence = np.array([],dtype='int64')

        for file_idx, filename in enumerate(filenames):

            img = Image.open(filename)

            input_img = V(self.centre_crop(img).unsqueeze(0), volatile=True)

            # forward pass
            logit = self.model.forward(input_img)
            h_x = F.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)

            self.probabilities[file_idx] = probs 

            # output the prediction
            print filename

            # test_probs: Probability of categories_turtlebot for each image
            test_probs = np.array([])
            # test_categories: Corresponding classes to test_probs' probabilities
            test_categories = []

            # Loop through all 365 classes
            for i in range(0, 365):
            	# Check if each of the 365 classes is in categories_turtlebot
            	if self.classes[idx[i]] in categories_turtlebot: 
            		# Print probability of each class. Example: 0.5 -> kitchen
                    print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
                    # appends each class's probability and each class to test_probs and test_categories
                    test_probs = np.append(test_probs, probs[i])
                    test_categories.append(self.classes[idx[i]])
                self.classifications[file_idx, i] = self.classes[idx[i]]

            # Get the index of the max prob
            idx_max = np.argmax(test_probs)
            # Pull out the corresponding class
            best_category = test_categories[idx_max]
            # Sorts the classes based on probabilities
            # best_category_order.append(best_category)

            print 'Best category: ' + best_category
            print '_________________________________'

            # Keeps a running probability of each class being printed. Ex: If kitchen gets predicted 6/10 times, it's value = 0.6
            best_categories[categories_turtlebot.index(best_category)] += 1
            print np.int64(categories_turtlebot.index(best_category))
            best_category_order = np.append(best_category_order, np.int64(categories_turtlebot.index(best_category)))
            if len(best_category_order) >= run_length:
            	# print np.bincount(best_category_order)
            	max_occurrence = np.append(max_occurrence, np.argmax(np.bincount(best_category_order)))
            	best_category_order = best_category_order[1:]
            count += 1
            # print best_categories[0] 
            # print categories_turtlebot.index(best_category)

        print best_categories
        print max_occurrence
        print categories_turtlebot
        print np.bincount(max_occurrence)



    def print_time_per_img(self): 
        time_per_image = self.elapsed_time/self.total_images
        print "Time per image: ", time_per_image

if __name__ == '__main__':
    places = Places('resnet18', 0)






