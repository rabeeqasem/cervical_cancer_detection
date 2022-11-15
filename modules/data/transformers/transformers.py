import albumentations as A
import tensorflow as tf

# get autotune variable for parallel processing
try:
    AUTOTUNE = tf.data.AUTOTUNE
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class GeneralTransformer():
    def __init__(self):
        self.transforms_applied = {}

#    def define_resize(self):
#        self.goal_aspect_ratio = 1.5
#        self.longest_side = 250
#        self.other_side = self.longest_side/self.goal_aspect_ratio
#        self.resize = A.Resize(self.other_side, self.longest_side,p=1)
#        self.transforms_applied['resize'] = (self.other_side,self.longest_side,'p=1')

    def define_resize(self):
        self.shortest_side = 124
        self.resize = A.SmallestMaxSize(self.shortest_side, p=1)
        self.transforms_applied['resize.SmallestMaxSize'] = (self.shortest_side, 'p=1')
    
    def define_flip_rotate(self):
        self.flip_rotate = A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)])

    def define_transform_pipeline(self):
        #TO DO: define transforms based on list of strings passed into pipeline eg 'resize','flip_rotate','greyscale' etc
        self.transform_pipeline = A.Compose([
            #self.flip_rotate,
            self.resize
            ])

    def create_transform_pipeline(self):
        self.define_resize()
        #self.define_flip_rotate()
        self.define_transform_pipeline()
        print('transform pipeline defined')
        return self.transform_pipeline



# Testing 
def transform_images_in_parallel(train_paths, train_labels, transform_pipeline):
    def parse_image(filename):
        image = tf.io.read_file(filename) #read filename
        image = tf.io.decode_image(image, channels=3) #detect image type and decode as appropiate
        np_image = image.numpy() #convert to np array , drop channel dim as color dim not needed
        aug_image = transform_pipeline(image=np_image)["image"] #apply transformations/augmentations on image
        aug_image = tf.cast(aug_image, tf.float32) #convert type to match with wrapper function
        return aug_image

    #wrap parse image function as tensorflow function
    def tf_parse_image(input):
        y = tf.numpy_function(parse_image, [input],tf.float32) 
        return y 

    print('starting image transformation')
    filenames_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    print('file names parsed')
    images_ds = filenames_ds.map(lambda x: tf_parse_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print('image ds created')
    print('finished image transformation')
    return images_ds

if __name__ == "__main__" :
    gt = GeneralTransformer()
    gt.create_transform_pipeline()
    print('all ran ok')
