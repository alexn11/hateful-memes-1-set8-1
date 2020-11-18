import os

import numpy as np
import pandas
import PIL
import tensorflow as tf


# i rather not allow pickle but will for simplicity

def extract_features(images_data, source_dir, dest_dir, batch_size = 16, allow_pickle = True):
  
  inception_v3 = tf.keras.applications.InceptionV3(weights = 'imagenet')
  inception_v3 = tf.keras.Model(inception_v3.input, inception_v3.layers[-2].output)
  model_input_shape = list(inception_v3.input.shape)
  
  nb_images = len(images_data)
  
  output_ids = nb_images * [ None, ]
  output_file_names = nb_images * [ None, ]
  
  nb_batches = nb_images // batch_size
  last_batch_size = nb_images - nb_batches * batch_size
  if(last_batch_size > 0):
    nb_batches += 1
  else:
    last_batch_size = batch_size
  
  image_index = 0
  
  for batch in range(nb_batches):
    
    current_batch_size = batch_size if(batch < nb_batches - 1) else last_batch_size
    
    batch_input = np.zeros([current_batch_size, ] + model_input_shape[1:])
    start_index = image_index
    
    for index_in_batch in range(current_batch_size):
      image_data = images_data.loc[image_index]
      image_id = image_data['id']
      image_path = os.path.join(source_dir, image_data['img'])
      image = PIL.Image.open(image_path)
      image = image.resize(model_input_shape[1:3], PIL.Image.ANTIALIAS)
      image_array = np.array(image)
      if(image_array.shape[2] == 4): # remove alpha channel if exists
        image_array = image_array[:,:,:3]
      batch_input[index_in_batch,:] = tf.keras.applications.inception_v3.preprocess_input(image_array)
      output_ids[image_index] = image_id
      image_index += 1
    
    batch_output = np.array(inception_v3(batch_input))
    
    for index_in_batch in range(current_batch_size):
      current_image_index = start_index + index_in_batch
      output_file_name = f'{output_ids[current_image_index]:05d}.npy'
      output_file_names[current_image_index] = output_file_name
      np.save(os.path.join(dest_dir, output_file_name), batch_output[index_in_batch, :], allow_pickle = allow_pickle, fix_imports = False)
  
  output_data = pandas.DataFrame()
  output_data['id'] = output_ids
  output_data['file_name'] = output_file_names
  
  return output_data








































