# USAGE
# python extract_features.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16, DenseNet121
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from imutils import paths
import numpy as np
import pickle
import os,argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=".", help='Root directory of data')
args = parser.parse_args()

# load the network and initialize the label encoder
print("[INFO] loading network...")
model_res = ResNet50(weights="imagenet", include_top=False)
model_inc = InceptionV3(weights="imagenet", include_top=False)
model_vgg = VGG16(weights="imagenet", include_top=False)
model_den = DenseNet121(weights="imagenet", include_top=False)

models = [model_res, model_inc, model_vgg, model_den]
model_output_dirs = ["ResNet50_output", "Inceptionv3_output", "VGG16_output", "Densenet121_output"]

data_dirs = ["CRIC_data"]

csv_fh_dict = {}
data_paths_dict = {}
for data_dir in data_dirs:
	data_paths_dict[data_dir] = []
	csv_fh_dict[data_dir] = []
	for model_output_dir in model_output_dirs:
		if not os.path.exists(model_output_dir):
			os.makedirs(model_output_dir) # make the output directories if they do not exist

		csv_path = os.path.sep.join(["{}".format(model_output_dir), "{}.csv".format(data_dir)])
		csv = open(csv_path, "w")
		csv_fh_dict[data_dir].append(csv)

le = None

# loop over the data splits
for data_dir in data_dirs:
	print("[INFO] processing '{} split'...".format(data_dir))

	# grab all image paths in the current data_dir
	p = os.path.sep.join([args.root, data_dir])
	imagePaths = list(paths.list_images(p))

	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	#random.shuffle(imagePaths)
	labels = [p.split(os.path.sep)[-2] for p in imagePaths]

	# if the label encoder is None, create it
	if le is None:
		le = LabelEncoder()
		le.fit(labels)
		assert(len(le.classes_) == 6) # 6 for CRIC, 4 for Mendeley data, 7 for Herlev

	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(imagePaths), 32)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(imagePaths) / float(32)))))
		batchPaths = imagePaths[i:i + 32]
		batchLabels = le.transform(labels[i:i + 32])
		batchImages = []
		batchImagesPaths = []

		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility
			# while ensuring the image is resized to 224x224 pixels
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)

			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)
			batchImagesPaths.append(imagePath)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		batchImages = np.vstack(batchImages)
		for i, model in enumerate(models):
			features = model.predict(batchImages, batch_size=32)
			features = features.reshape((features.shape[0], 
										 features.shape[1]*features.shape[2]*features.shape[3]))

			# loop over the class labels and extracted features
			for (imgpath, label, vec) in zip(batchImagesPaths, batchLabels, features):
				# construct a row that exists of the class label and
				# extracted features
				vec = ",".join([str(v) for v in vec])
				csv_fh_dict[data_dir][i].write("{},{}\n".format(label, vec))
				if i == 0:
					data_paths_dict[data_dir].append(imgpath)

# close the CSV file
for data_dir in data_dirs:
	for i in range(len(models)):
		csv_fh_dict[data_dir][i].close()
           
# serialize the label encoder to disk
f = open("le.cpickle", "wb")
f.write(pickle.dumps(le))
f.close()

with open("data_paths.cpickle", "wb") as f:
	pickle.dump(data_paths_dict, f, pickle.HIGHEST_PROTOCOL)

#print(features.size)
