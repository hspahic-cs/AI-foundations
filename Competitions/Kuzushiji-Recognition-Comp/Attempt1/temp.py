# %%
# https://www.kaggle.com/competitions/kuzushiji-recognition/data
import numpy as np
import os 
import cv2 

# %%
# Defining CONST
DIR = "/home/harris/Projects/ML/Datasets/Kuzushiji-Recognition/"

# %%
# Attempt 1
def segment_image_using_contours(path, file):
    # Read the image
    img = cv2.imread(DIR + path + file, cv2.IMREAD_GRAYSCALE)
    
    # Adaptive thresholding
    binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    processed_img = cv2.erode(dilated_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not os.path.exists(DIR + "\cleaned_train_images"):
       os.makedirs(DIR + "\cleaned_train_images")
    reconstructed_img = 255 * np.ones_like(img)
    
    # Loop through the contours and extract individual kanji characters\n,
    for i, contour in enumerate(contours):
        print(contour.shape)
        if 200 <= cv2.contourArea(contour) <= 4000:
            x, y, w, h = cv2.boundingRect(contour)
            kanji = img[y:y+h, x:x+w]
            reconstructed_img[y:y+h, x:x+w] = kanji
    cv2.imwrite(os.path.join(DIR + "\cleaned_train_images" + "\reconstructed_image.png"), reconstructed_img)


# %%
# Attempt 2 at segmenting images
    
def segment_image_using_edges(path, file):
    # Read the image
    img = cv2.imread(DIR + path + file, cv2.IMREAD_GRAYSCALE)

    # Adaptive thresholding
    edges = cv2.Canny(img, 50, 150)

    # Morphological Operations
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    # processed_img = cv2.erode(dilated_img, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not os.path.exists(DIR + "/cleaned_train_images"):
        os.makedirs(DIR + "/cleaned_train_images")
    
    reconstructed_img = 255 * np.ones_like(img)
    
    # Loop through the contours and extract individual kanji characters
    for i, contour in enumerate(contours):
        print(contour.shape)
        if 200 <= cv2.contourArea(contour) <= 4000:
            x, y, w, h = cv2.boundingRect(contour)
            kanji = img[y:y+h, x:x+w]
            reconstructed_img[y:y+h, x:x+w] = kanji

    cv2.imwrite(os.path.join(DIR + "/cleaned_train_images" + "/reconstructed_image_edges.png"), reconstructed_img)

print(os.getcwd())

# %%
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Attempt 3

def VisualizeKuzushiji(imagePath):
    # Reading image & preparing to draw,
    img = cv2.imread(imagePath)
    imsource = Image.open(imagePath)
    char_draw = ImageDraw.Draw(imsource)

    # Preprocessing,
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(im_grey, 130, 255, cv2.THRESH_BINARY_INV)
    ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ''',
    RETR_EXTERNAL b/c we only want the contours of the parent characters to make a bounding,
    box, we don't need a sub contour for each stroke inside of a character,
    CHAIN_APPROX_SIMPLE b/c we only need the bounding box,
    '''

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    display_img = im_th
    max_width = []

    for rect in rects:
        x, y, w, h = rect
        # Characters for the most part should only take 1/6th of image,
        if h * w > 6000 and w < 500:
            display_img = cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            

   
    plt.figure(figsize=(30,30))
    plt.subplot(1,2,1)
    plt.title("Detection of Kuzushiji",fontsize=20)
    plt.imshow(display_img)

    plt.subplot(1,2,2)
    plt.title("Training data",fontsize=20)
    plt.imshow(VisualizeTraining(imagePath))


    return char_draw

def VisualizeTraining(imagePath):
    '''
    Display the training image with the bounding box of the characters
    '''

    img = cv2.imread(imagePath)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    data = "U+306F 1231 3465 133 53 U+304C 275 1652 84 69 U+3044 1495 1218 143 69 U+3051 220 3331 53 91 U+306B 911 1452 61 92 U+306B 927 3445 71 92 U+306E 904 2879 95 92 U+5DE5 1168 1396 187 95 U+3053 289 3166 69 97 U+4E09 897 3034 121 107 U+306E 547 1912 141 108 U+3084 1489 2675 151 109 U+3068 1561 2979 55 116 U+5DF1 1513 2500 127 117 U+3082 1213 1523 72 119 U+3055 1219 3266 95 124 U+306E 259 2230 68 125 U+306E 1184 2423 169 125 U+4E16 849 2236 163 127 U+7D30 1144 1212 200 128 U+305D 316 3287 57 133 U+4EBA 217 2044 183 135 U+3051 277 2974 112 137 U+308C 201 3423 181 137 U+3060 243 2830 159 143 U+5F37 1479 2034 163 145 U+306E 1497 1567 123 152 U+305F 1164 952 145 153 U+3066 552 1199 97 155 U+4FF3 537 2095 176 155 U+6839 203 1439 184 156 U+304B 1188 2606 156 157 U+8AE7 549 2328 156 159 U+308C 1495 2784 168 159 U+5B50 891 1255 100 164 U+3092 584 2546 117 164 U+53CA 849 1588 151 164 U+8005 1192 2198 133 169 U+305A 889 1763 103 171 U+907F 513 945 181 171 U+6B63 539 1439 136 172 U+6587 192 2382 216 173 U+3075 1512 3371 147 176 U+6642 1465 1338 168 179 U+601D 1492 3175 159 180 U+306A 1191 2775 135 181 U+3081 593 3313 151 184 U+6D6E 868 1982 155 184 U+3092 873 2400 145 192 U+6C17 1504 1754 145 200 U+8077 208 1770 197 204 U+8001 1167 1687 152 208 U+6B66 1184 1942 171 208 U+697D 568 2762 133 209 U+3082 247 1159 116 212 U+76F2 253 2578 119 215 U+82E5 1465 951 172 216 U+81EA 1852 1736 104 219 U+3069 220 928 139 229 U+98A8 541 1619 147 236 U+306B 1521 2239 83 237 U+88CF 851 2608 169 237 U+7573 905 3189 103 244 U+606F 876 937 123 244 U+5E8F 1816 2096 152 296 U+3057 629 2985 27 300 U+3057 1243 2942 39 313"
    results = data.split(" ")
    results = list(chunks(results, 5))

    for result in results:
        char, x, y, w, h = result
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 3)

    return img

img1 = VisualizeTraining(DIR + "train_images/100241706_00004_2.jpg")
# img1 = VisualizeKuzushiji(DIR + "train/100241706_00004_2.jpg")


# plt.subplot(1,4,2),
# plt.title(\"Recognition of Kuzushiji\",fontsize=20),
# plt.imshow(imsource1)

# %%
import pandas as pd

# Read in the unicode translation
unicode_map = {codepoint: char for codepoint, char in pd.read_csv(DIR + '/unicode_translation.csv').values}
integer_map = {codepoint: i for i, codepoint in enumerate(unicode_map.keys())}

# Read in the training data
data = pd.read_csv(DIR + '/train.csv')

# Check for null values
print(data.isnull().sum())

# Check for duplicates
print(data.duplicated().sum())

# Disaply sample data
data.head()

# %%
from tqdm import tqdm
import random as rand
import matplotlib as mpl

# def extract_data():
#    '''
#    Takes in training data, segments the characters, and returns a list of character bounding boxes with encoding 
#    '''

#    X = []
#    Y = []   

#    for image_encoding in tqdm(data[0:10].values):   
#       try: 
#          # Clean data to be individal images 
#          seg_encoding = image_encoding[1].split(" ")
#          seg_encoding = [seg_encoding[i:i+5] for i in range(0, len(seg_encoding), 5)]

#          # print(seg_encoding)

#          # Read in the image and convert to threshold
#          img = Image.open(DIR + "/train_images/" + image_encoding[0] + ".jpg").convert("RGBA")
   
#          for char, x, y, w, h in seg_encoding:
            
#             # Threshold each character and get encoding
#             x,y,w,h = int(x), int(y), int(w), int(h)
#             char = unicode_map[char]
#             cropped_img = img.crop((x-10, y-10, x+w+10, y+h+10))
#             resized_img = cropped_img.resize((256, 256))
#             resized_img = np.array(resized_img)
#             ret, im_th = cv2.threshold(cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY), 130, 255, cv2.THRESH_BINARY_INV)
            
#             # Append to X and Y
#             try:
#                X.append(im_th)
#             except Exception as E:
#                print(E)
#                print(f"{[char, x, y, w, h]} with image encoding {image_encoding[0]} failed to append to X")
#                pass

#             try:    
#                Y.append(char)
#             except Exception as E:
#                print(E)
#                X.pop()
#                print(f"{[char, x, y, w, h]} with image encoding {image_encoding[0]} failed to append to Y")
#                pass

#       except Exception as E:
#          print(E)

#    return X, Y


def display_sample(X, Y,  num_samples=20):
   '''
   Takes a random subsample of characters and displays them
   '''

   if num_samples > 200:
      print("Please enter a number less than 200")
      return
   
   indicies = rand.sample(range(0, len(X)), num_samples)
   X_samples = [X[i] for i in indicies]   
   Y_samples = [Y[i] for i in indicies]

   # Change font to allow Japaense characters
   plt.figure(figsize=(20,20))
   mpl.rcParams['font.family'] = 'Noto Sans CJK JP'

   for i, sample in enumerate(range(num_samples)):
      plt.subplot(4, 5, i+1)
      plt.title(Y_samples[i])
      plt.imshow(X_samples[i])


# X_train, Y_train = extract_data()
# display_sample(X_train, Y_train)
# print(len(X_train))


# %%
import random

def extract_data(img_encoding, img_size):
   '''
   Takes in an image_file & its char encoding, returning all the characters & labels for that image 
   '''

   X = []
   Y = []   

   try:
      # Clean data to be individal images
      seg_encoding = img_encoding[1].split(" ")
      seg_encoding = [seg_encoding[i:i+5] for i in range(0, len(seg_encoding), 5)]
      
      # Shuffle encoding to ignore ordering
      random.shuffle(seg_encoding)

      # Read in the image and convert to threshold
      img = Image.open(DIR + "/train_images/" + img_encoding[0] + ".jpg").convert("RGBA")

      # Loop through each character and get encoding
      for char, x, y, w, h in seg_encoding:         
         x,y,w,h = int(x), int(y), int(w), int(h)
         # use integer_map instead of unique_code map for SCCE
         char = integer_map[char]
         cropped_img = img.crop((x-10, y-10, x+w+10, y+h+10))
         resized_img = cropped_img.resize((img_size, img_size))
         resized_img = np.array(resized_img)
         ret, im_th = cv2.threshold(cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY), 130, 255, cv2.THRESH_BINARY_INV)

         # Append images and labels to X and Y
         try:
            X.append(im_th)
         except Exception as E:
            print(E)
            print(f"{[char, x, y, w, h]} with image encoding {img_encoding[0]} failed to append to X")
            pass

         try:    
            Y.append(char)
         except Exception as E:
            print(E)
            X.pop()
            print(f"{[char, x, y, w, h]} with image encoding {img_encoding[0]} failed to append to Y")
            pass
         
   except Exception as E:
      print(E)
   
   X = np.asarray(X, dtype='float32')
   Y = np.asarray(Y, dtype='int32')

   return X, Y


def data_generator(data, batch_size, img_size):
   '''Creates a generator that returns a batch of data'''
   
   # Calculate total number of batches
   total_characters = sum([len(encoding[1]) // 5 for encoding in data.values])
   total_batches = total_characters // batch_size

   # Create random indicies for image_data
   indicies = rand.sample(range(0, len(data)), len(data))
   
   X_batch, Y_batch = [], []
   
  
   while True:
      mnspt_idx = 0

      # Create batch
      for batch_num in range(total_batches):
         

         while(mnspt_idx < len(data) and len(X_batch) < batch_size):   
            # Extract next manuscript
            X_temp, Y_temp = extract_data(data.values[indicies[mnspt_idx]], img_size)
            
            # print(f'X_temp type: {type(X_temp)} at {data.values[indicies[mnspt_idx]]}')

            if(len(X_batch) == 0):
               X_batch = X_temp
               Y_batch = Y_temp
            
            else:
               X_batch = np.concatenate((X_batch, X_temp), axis=0)
               Y_batch = np.concatenate((Y_batch, Y_temp), axis=0)
               
            # Increment manuscript
            mnspt_idx = mnspt_idx + 1
            # print(f'MNSPT_IDX: {mnspt_idx}')
         
         print(f'End: {X_batch.shape}')
        
         # Seperate batch & remaining characters 
         X_temp = X_temp[batch_size:]
         Y_temp = Y_temp[batch_size:]
         
         X_result = X_batch[:batch_size]
         Y_result = Y_batch[:batch_size]
      
         X_batch = X_temp
         Y_batch = Y_temp
         
         print(f'Start: {X_batch.shape}')

         # print("X_batch dtype: ", X_batch.dtype, " Y_batch dtype: ", Y_batch.dtype)

         yield X_result, Y_result

# %%
   # def collect_training(dataPath, trainImagePath):
   #    imageNames = os.listdir(os.path.join(DIR + trainImagePath))
   #    df = pd.read_csv(os.path.join(DIR + dataPath))
      
   #    # Check if there is a directory called \"training_images\" if there is, empty it otherwise make one,
   #    training_dir = os.path.join(DIR + "/segmented_training_images")
   #    if os.path.exists(training_dir):
   #       for filename in os.listdir(training_dir):
   #          file_path = os.path.join(training_dir, filename)
   #          try:
   #             os.remove(file_path)
   #          except Exception as e:
   #             print('Failed to delete %s. Reason: %s' % (file_path, e))
   #          else:
   #             os.makedirs(training_dir)

   #          print(imageNames)
   #          print(df.head())
   #          for ID in df['image_id']:
   #             print(DIR + trainImagePath + ID)
   #             img = cv2.imread(DIR + trainImagePath + ID + ".jpg")
   #             chars = df[df['image_id'] == ID]['labels'].tolist()[0].split(" ")
   #             rects = [chars[i:i+5] for i in range(0, len(chars), 5)]

   #             for i, rect in enumerate(rects):
   #                # Save a file for each character
   #                char, x, y, w, h = rect
   #                char_img = img[int(y):int(y)+int(h), int(x):int(x)+int(w)]
   #                cv2.imwrite(os.path.join(training_dir + " ", f"{ID}_{i}_{char}.jpg"), char_img)
                              
   #    # FILEPATH: /home/harris/Projects/AI-Stuff/AI-foundations/Competitions/Kuzushiji-Recognition-Comp/clean-data.ipynb
   #    collect_training("train.csv", "train_images")


# %%
from sklearn.model_selection import train_test_split

# Segmenting data to lower training cost
SEGMENT_DATA = len(data)
BATCH_SIZE = 256
IMAGE_SIZE = 256

indicies = rand.sample(range(0, len(data)), SEGMENT_DATA)

# Splitting training, validation and test data
train_indices, test_indices = train_test_split(indicies, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

# Creating generators for training and validation data
training_gen = data_generator(data.iloc[train_indices], BATCH_SIZE, IMAGE_SIZE)
validation_gen = data_generator(data.iloc[val_indices], BATCH_SIZE, IMAGE_SIZE)

# Calculate the number of steps per epoch
train_steps = sum([len(data.iloc[i]['labels']) // 5 for i in train_indices]) // BATCH_SIZE
validation_steps = sum([len(data.iloc[i]['labels']) // 5 for i in val_indices]) // BATCH_SIZE

# %%
# Creating model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = len(unicode_map.keys())

inputs = keras.Input(shape=(256, 256, 1))
x = layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(256, 256, 1))(inputs)
x = layers.MaxPooling2D((2,2))(x)
# x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
# x = layers.MaxPooling2D((2,2))(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation='relu')(x)# Calculate the number of steps per epoch

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# %%
x_batch, y_batch = next(training_gen)
print(x_batch.shape)  # This should show something like (batch_size, height, width, channels)

x_batch, y_batch = next(validation_gen)
print(x_batch.shape)

# %%
  # Replace with your generator and its parameters
total_batches = 10  # Total number of batches
batch_to_start_inspection = total_batches - 5  # Start inspection 5 batches before the end

for i in range(total_batches):
    try:
        X_batch, Y_batch = next(training_gen)
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        break

# %%
# Fit and compile model
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_gen,
    steps_per_epoch=train_steps,
    validation_data=validation_gen,
    validation_steps=validation_steps,
    epochs=10
)


