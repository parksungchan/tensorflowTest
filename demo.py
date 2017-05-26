import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo.net.yolo_tiny_net import YoloTinyNet
import requests

directory     = "/home/dev/tensorflowTest/data"
directory_out = "/home/dev/tensorflowTest/data"
tiny_url = 'https://drive.google.com/uc?id=0B-yiAeTLLamRekxqVE01Yi1RRlk&export=download'
model_path = '/home/dev/tensorflowTest/third_party/yolo/models/pretrain/yolo_tiny.ckpt'

channel= 3
x_size = 385 #448
y_size = 385 #448
filesize = 100000
image_arr = []
lable_arr = []
shape_arr = []
name_arr = []


def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  index = np.argmax(P)

  index = np.unravel_index(index, P.shape)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (x_size/7.0)
  ycenter = (index[0] + ycenter) * (y_size/7.0)

  w = w * x_size
  h = h * y_size

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num

common_params = {'image_size': x_size, 'num_classes': 20, 'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, x_size, y_size, channel))
predicts = net.inference(image)
saver = tf.train.Saver(net.trainable_collection)

with tf.Session() as sess:
  saver.restore(sess, model_path)

  forderlist = os.listdir(directory)
  filecnt = 0
  for forder in forderlist:
    filelist = os.listdir(directory + '/' + forder)
    for filename in filelist:
      # PNG -> JPEG
      img = Image.open(directory + '/' + forder + '/' + filename)

      pngidx = str(type(img)).find("PngImageFile")
      if pngidx > -1:
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), img)
        filename = "Conv_" + str(filename)
        bg.save(directory + '/' + forder + '/' + filename)

      img = img.resize((x_size, y_size), Image.ANTIALIAS)
      np_img = np.array(img)
      print(str(filename))

      resized_img = cv2.resize(np_img, (x_size, y_size))

      np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
      np_img = np_img.astype(np.float32)
      np_img = np_img / 255.0 * 2 - 1
      np_img = np.reshape(np_img, (1, x_size, y_size, channel))
      np_predict = sess.run(predicts, feed_dict={image: np_img})
      xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
      resized_img = resized_img[int(ymin):int(ymax), int(xmin):int(xmax)]

      try:
        np_img = Image.fromarray(resized_img)
        np_img.save(directory_out + '/' + forder + '/' + 'Yolo_'+filename)
      except Exception as e:
        print("error......................................."+str(filename))
        print(e)

print("end")