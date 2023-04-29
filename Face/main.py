from deepface import DeepFace
import cv2
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
import openai
import requests
from PIL import Image, ImageWin, ImageDraw, ImageFont
import base64
import win32print
import win32ui
from time import sleep
import threading
import subprocess
import numpy as np
# from pynput.mouse import Listener
import win32api

# from graph import graph
openai.api_key = "API_KEY"

PHYSICALWIDTH = 110
PHYSICALHEIGHT = 111
analyzing = False
painting = False
printing = False
process_done = True
attributes = []

# printer_name = win32print.GetDefaultPrinter ()
# file_name = "merged_image.jpg"
# hDC = win32ui.CreateDC()
# hDC.CreatePrinterDC (printer_name)
# printer_size = hDC.GetDeviceCaps (PHYSICALWIDTH), hDC.GetDeviceCaps (PHYSICALHEIGHT)


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]


def send_to_printer(file_name):

  bmp = Image.open (file_name)
  if bmp.size[0] < bmp.size[1]:
    bmp = bmp.rotate (0)

  hDC.StartDoc (file_name)
  hDC.StartPage ()

  dib = ImageWin.Dib (bmp)
  dib.draw (hDC.GetHandleOutput (), (int(printer_size[1]/3)-500, 0, 2*int(printer_size[1]/3)-500,printer_size[1]))

  hDC.EndPage ()
  hDC.EndDoc ()


def scan_and_paint(img, height, use_DALLE = False, use_printer = False):
  global analyzing,painting,printing,process_done,attributes
  
  analyzing = True

  # Analyzing face from image
  img = cv2.imread("orig.png")
  color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  print("\n\nGetting Face Attributes...")
  prediction = DeepFace.analyze(color_img, enforce_detection = False, silent = True, detector_backend = backends[3])
  # print(prediction)
  box = prediction[0]['region']
  age = str(prediction[0]['age'])
  gender = prediction[0]['dominant_gender']
  emotion = prediction[0]['dominant_emotion']
  race = prediction[0]['dominant_race']
  attributes = [box, age, gender, emotion, race]

  emotions = prediction[0]['emotion']
  races = prediction[0]['race']
  genders = prediction[0]['gender']

  em_names = []
  em_probs = []
  race_names = []
  race_probs = []
  gender_names = []
  gender_probs = []

  for e in emotions:
    em_names.append(e.capitalize())
    em_probs.append(round(emotions[e],2))

  for r in races:
    race_names.append(r.capitalize())
    race_probs.append(round(races[r],2))

  for g in genders:
    gender_names.append(g.capitalize())
    gender_probs.append(round(genders[g],2))
  
  # print(prediction[0])
  print("\nAge: " + age + "\nGender: " + gender + "\nEmotion: " + emotion + "\nRace: " + race)

  # print("plotting")

  # Plotting 
  # plt.figure(1)
  fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (6,6))
  ax1.barh(y = em_names, width = em_probs, height = 0.8, edgecolor="black", linewidth=0.7, color = 'red' )
  ax1.set_title('Emotion Prediction Confidences')
  ax1.bar_label(ax1.containers[0], label_type = 'edge')
  ax1.set_xlim([0, 110])

  ax2.barh(y = race_names, width = race_probs, height = 0.8, edgecolor="black", linewidth=0.7, color = 'green' )
  ax2.set_title('Race Prediction Confidences')
  ax2.bar_label(ax2.containers[0], label_type = 'edge')
  ax2.set_xlim([0, 110])

  ax3.barh(y = gender_names, width = gender_probs, height = 0.8, edgecolor="black", linewidth=0.7, color = 'blue' )
  ax3.set_title('Gender Prediction Confidences')
  ax3.bar_label(ax3.containers[0], label_type = 'edge')
  ax3.set_xlim([0, 110])
  ax3.set_xlabel("Confidence Percentage (%)")

  plt.subplots_adjust(left=0.2, bottom=0.1, right=0.98, top=0.95, wspace=None, hspace=0.45)

  plt.savefig('plot.png')
  plt.close()
  plt.show()
  
  analyzing = False


  if(use_DALLE):
    
    painting = True
    mask = Image.new('RGBA', (height, height), (255, 0, 0, 255))
    mask = mask.convert("RGBA")

    pixdata = mask.load()
    for x in range(box['x'],box['x'] + box['w']):
      for y in range(box['y'],box['y'] + box['h']):
        pixdata[x, y] = (255, 255, 255, 0)

    # mask.show()
    mask.save("mask.png", "PNG")

    # Plugging in face attributes to DALLE
    print("\nEditting image using DALLE...")
    response = openai.Image.create_edit(
      image=open("orig.png", "rb"),
      mask=open("mask.png", "rb"),
      prompt= "Photo portrait of a " + age + " year old " + race + " " + gender + " feeling " + emotion,
      n=1,
      size="512x512",
      response_format = 'b64_json'
    )

    # Image Base64 String
    imageData = response['data'][0]['b64_json']
    print("Displaying Eddited Image.")
    # Decode base64 String Data
    decodedData = base64.b64decode((imageData))
      
    # Write Image from Base64 File and display it as JPG
    imgFile = open('DALLE.jpg', 'wb')
    imgFile.write(decodedData)
    imgFile.close()
    dalle = Image.open('DALLE.jpg')
    # dalle.show()

    # Combining images into one JPG
    orig = Image.open('orig.png')
    analyze = Image.open('plot.png')

    orig = orig.resize((500, 500))
    draw = ImageDraw.Draw(orig)
    font1 = ImageFont.truetype("arial.ttf", 50)
    draw.text((0, 10), ("Age: " + attributes[1]),fill='blue',font = font1)
    analyze = analyze.resize((500, 500))
    dalle = dalle.resize((500, 500))

    results_vert = Image.new('RGB',(500, 1500), (250,250,250))
    results_vert.paste(orig,(0,0))
    results_vert.paste(analyze,(0,500))
    results_vert.paste(dalle,(0,1000))
    results_vert.save("results_vert.jpg","JPEG")

    results_horz = Image.new('RGB',(1500, 500), (250,250,250))
    results_horz.paste(orig,(0,0))
    results_horz.paste(analyze,(500,0))
    results_horz.paste(dalle,(1000,0))
    results_horz.save("results_horz.jpg","JPEG")

    painting = False
    process_done = True
    
    res = cv2.imread("results_horz.jpg")
    res = cv2.resize(res, (0,0), fx=1.1, fy=1.1) 
    cv2.startWindowThread()
    cv2.namedWindow("Results")
    cv2.moveWindow("Results", 0,300)
    cv2.imshow("Results", res)
    cv2.waitKey(5000)
    cv2.destroyWindow("Results")    

  
  if(use_printer):
    printing = True
    send_to_printer("results_vert.jpg")
    printing = False
  
  
  
  return True
    

# Loading Screen
def loading_screen():
  global analyzing,painting,printing,process_done,attributes

  load = cv2.imread("orig_temp.png",1)
  load_1 = load.copy()
  load_2 = load.copy()
  cv2.putText(load_1,'Analyzing...',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3)
  cv2.putText(load_2,'Painting...',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3)

  while(process_done == False):
    
    # # print(process_done)
    # if(process_done == True):
    #   break

    if(analyzing == True):
      displayed = load_1
      # # print("analyzing")
      # text = 'Analyzing...'
      # cv2.putText(load.copy(),text,(0,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),3)

    elif(painting == True):
      cv2.putText(load_2, ('Age: ' + attributes[1].capitalize()),(0,100),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,0),2)
      cv2.putText(load_2,('Gender: ' + attributes[2].capitalize()),(0,150),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,0),2)
      cv2.putText(load_2,('Emotion: ' + attributes[3].capitalize()),(0,200),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,0),2)
      cv2.putText(load_2,('Race: ' + attributes[4].capitalize()),(0,250),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,0),2)
      displayed = load_2
      # print("painting")
      # text = 'Painting...'
      
    cv2.startWindowThread()
    cv2.namedWindow("Loading")
    cv2.moveWindow("Loading", 100,0)
    cv2.imshow("Loading", displayed)
    cv2.waitKey(1)
  
  cv2.destroyWindow("Loading")
  
  return True
    
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 's' to scan your face")

state_left = win32api.GetKeyState(0x01)  # Left button up = 0 or 1. Button down = -127 or -128

while True:
  # Capture frame-by-frame
  

  ret, orig_frame = video_capture.read()
  orig_frame = cv2.flip(orig_frame, 1)
  height,width,_ = orig_frame.shape

  # print(width,height)
  frame = orig_frame[0:height, int(width/2) - int(height/2):int(width/2) - int(height/2) + height].copy()
  # print(frame.shape)

  gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
  faces = []
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize = (200,200)
  )

  # # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
    cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  stretch = cv2.resize(orig_frame, (0,0), fx=1.4, fy=1.4) 

  # Display the resulting frame
  cv2.startWindowThread()
  cv2.namedWindow("Live Feed")
  cv2.moveWindow("Live Feed", 100,0)
  cv2.imshow('Live Feed', stretch)
  cv2.waitKey(1)
  
  # if cv2.waitKey(1) & 0xFF == ord('s'):
  a = win32api.GetKeyState(0x01)
  if a != state_left:  # Button state changed
    state_left = a
    if a < 0:
      # print('Left Button Pressed')
      if(len(faces) != 0):
        cv2.destroyWindow("Live Feed")
        cv2.imwrite("orig.png", frame)
        cv2.imwrite("orig_temp.png", stretch)
        process_done = False
        t1 = threading.Thread(target=loading_screen)
        t2 = threading.Thread(target=scan_and_paint, args= (frame, height, True, False))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

      else:
        print('No face detected')

      print("Click to scan your face")
    # else:
      # print('Left Button Released')
    