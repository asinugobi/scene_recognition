# !/usr/bin/env python
# Software License Agreement (BSD License)

# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Revision $Id$

# Simple talker demo that listens to std_msgs/Strings published
# to the 'chatter' topic

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import places_rt as pr
import time

class scene_recognition:

    def __init__(self):
        # self.category_pub = rospy.publishedisher()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color/compressed", CompressedImage,self.callback)

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + ' Subscribed.')
        img = data.data
        images = []

        while len(images) <= 9:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data,'bgr8')
            images.append(cv_image)
            print len(images)

        # prediction = pr.get_classifications(images)
        predictor = pr.places_rt('resnet18', 0)
        prediction = predictor.get_classifications(images)
        print "Scene predication:", prediction

        # clear image array
        images = []
        time.sleep(5)


if __name__ == '__main__':
    scene_recognition = scene_recognition()
    rospy.init_node('scene_recognition', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

