
import tensorflow as tf

class MemeClassifierModel(tf.keras.Model):
  
  def __init__(self,
               image_features_dimension,
               text_features_dimension,
               return_logits = False,
               dropout = 0.1,
               has_preprocessing_layers = True,
               nb_hidden_layers = 0):
    
    super(MemeClassifierModel, self).__init__()
    
    self.image_input_size = image_features_dimension
    self.text_input_size = text_features_dimension
    self.return_logits = return_logits
    self.split_sizes = [ self.image_input_size, self.text_input_size ]
    self.has_preprocessing_layers = has_preprocessing_layers
    self.nb_hidden_layers = nb_hidden_layers
    self.hidden_size = self.image_input_size + self.text_input_size
    
    if(self.has_preprocessing_layers):
      self.image_feature_layer = tf.keras.layers.Dense(self.image_input_size, activation = 'relu', name = 'image_preprocessing_layer')
      self.text_feature_layer = tf.keras.layers.Dense(self.text_input_size, activation = 'relu', name = 'text_preprocessing_layer')
    
    self.feature_concatenator = tf.keras.layers.Concatenate(name = 'image_text_concat')
    
    self.feature_dropout = tf.keras.layers.Dropout(0.1, name = 'feature_dropout')
    
    self.hidden_layers = []
    for i in range(self.nb_hidden_layers):
      self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_size, activation = 'relu', name = f'hidden_layer_{i}'))
      self.hidden_layers.append(tf.keras.layers.Dropout(0.1, name = f'hidden_layer_dropout_{i}'))
    
    if(not self.return_logits):
      self.classifier_output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'classifier')
    else:
      self.classifier_output = tf.keras.layers.Dense(1, name = 'classifier')
  
  def call(self, inputs, training):
    
    if(self.has_preprocessing_layers):
      image_features, text_features = tf.split(inputs, self.split_sizes, axis = 1, name = 'text_image_split')
      x_image = self.image_feature_layer(image_features)
      x_text = self.text_feature_layer(text_features)
      x = self.feature_concatenator([x_image, x_text])
    else:
      x = inputs
    
    if(training):
      x = self.feature_dropout(x)
    
    for i in range(self.nb_hidden_layers):
      x = self.hidden_layers[2 * i](x)
      if(training): # dropout layer
        x = self.hidden_layers[2 * i + 1](x)
    
    output = self.classifier_output(x)
    return output

