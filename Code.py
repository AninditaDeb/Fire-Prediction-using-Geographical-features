# install and import Google Earth Engine libraries
!pip install geemap
import geemap
import ee

# Authenticate and Initialize GEE developper account
# Go to signup.earthengine.google.com to register an account
ee.Authenticate()
ee.Initialize()

# Declare new empty map
Map = geemap.Map()

# Select date range to retrieve images from
date_filt = ee.Filter.date('2020-06-01', '2020-07-01')

# Create a dataset consiting of MODIS bands, median is taken over selected filter
ds_modis = ee.ImageCollection("MODIS/006/MCD43A4") \
    .filter(date_filt)\
    .select(['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2',
              'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4',
              'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6',
              'Nadir_Reflectance_Band7'])\
    .median()

# Select bands that correspond to RGB to see results on a map
trueColor = ds_modis.select(['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band4',
                            'Nadir_Reflectance_Band3'])

# Color parameters suggested by GEE
modis_vis = {
  'min': 0,
  'max': 4000.0,
  'gamma': 1.4,
}

# Add to map as a new layer
Map.addLayer(trueColor, modis_vis, 'MODIS')

# Display Map
Map
#####################################Model ############################################################################################################################
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout, MaxPooling2D, BatchNormalization

def multiclass_unet_model(N_CLASSES,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS):
  # Initialize inputs to model
  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

  # Contracting path
  c1 = BatchNormalization()(inputs)
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
  c1 = Dropout(0.1)(c1)
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
  c1 = BatchNormalization()(c1)
  p1 = MaxPooling2D((2, 2))(c1)

  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2 = Dropout(0.1)(c2)
  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
  c2 = BatchNormalization()(c2)
  p2 = MaxPooling2D((2, 2))(c2)
  
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
  c3 = BatchNormalization()(c3)
  p3 = MaxPooling2D((2, 2))(c3)
  
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
  c4 = BatchNormalization()(c4)
  p4 = MaxPooling2D(pool_size=(2, 2))(c4)
  
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5 = Dropout(0.3)(c5)
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
  c5 = BatchNormalization()(c5)

  # Expanding path 
  u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6 = Dropout(0.2)(c6)
  c6 = BatchNormalization()(c6)
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
  
  u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7 = Dropout(0.2)(c7)
  c7 = BatchNormalization()(c7)
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
  
  u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8 = Dropout(0.1)(c8)
  c8 = BatchNormalization()(c8)
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
  
  u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9 = Dropout(0.1)(c9)
  c9 = BatchNormalization()(c9)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
  
  # Define outputs with softmax activation
  outputs = Conv2D(N_CLASSES, (1, 1), activation='softmax')(c9)

  # Assemble as keras Model
  model = Model(inputs=[inputs], outputs=[outputs])

  return model

