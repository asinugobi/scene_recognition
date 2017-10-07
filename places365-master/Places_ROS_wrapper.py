import rospy
from std_msgs.msg import String

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

import Places


## Subscribe to Kinect Topic
def callback(data):
 	rospy.loginfo(rospy.get_caller_id() + "Received image of size: ",data.size)

def listener():

	rospy.init_node('Scene_listener', anonymous=True)	
	rospy.Subscriber('/camera/rgb/image_color/compressed', String, callback)

	rospy.spin()

def talker():
	    pub = rospy.Publisher('Scene', String, queue_size=10)
	    rospy.init_node('Scene_talker', anonymous=True)
	    rate = rospy.Rate(10) # 10hz
	    while not rospy.is_shutdown():
	    	
	    	

	    	hello_str = "hello world %s" % rospy.get_time()
	    	rospy.loginfo(hello_str)
	    	pub.publish(hello_str)
	    	rate.sleep()