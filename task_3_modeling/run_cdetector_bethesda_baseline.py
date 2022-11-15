import os
import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import sys
sys.path.append("../")
from modules.data.dataset import CervData
from task_3_modeling.efficientnet_classifier import CervEffNet

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", type=str)
parser.add_argument("-output_dir", type=str)
parser.add_argument("-random_seed", type=int, default=7)
parser.add_argument("-base_model_size", type=str, default="medium", help="small, medium, large")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
RANDOM_SEED = args.random_seed
BASE_MODEL_SIZE = args.base_model_size


if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)


def label_clean_up(data_dir):
	"""
	Uses train_labels.csv and test_labels.csv created by Mohamed Dwedar.
	Assumes this file is in the same directory as the images.
	To transform the labels from multi-label to multi-class I took the
	"most severe" finding for each image as the label. For example,
	if an image has labels 2,3, and 4, I use 4 as the label in the
	multi-class setup.
	"""
	labels = []
	for split in ('train', 'test'):
		df = pd.read_csv(os.path.join(data_dir, f"{split}_labels.csv"))
		df.drop_duplicates(inplace=True)
		df = df.groupby('image_id').max()['category_id'].reset_index()
		assert len(df) == df['image_id'].nunique(), "Error: duplicates"
		df['relative_path'] = df['image_id'].apply(lambda x: f"{split}/{x}")
		df['label'] = df['category_id'].copy()
		df['split'] = split
		df.drop(columns=['image_id', 'category_id'], inplace=True)
		labels.append(df.copy())

	labels = pd.concat(labels, ignore_index=True)
	labels.to_csv(os.path.join(data_dir, 'labels.csv'), index=None)


def load_base_model(base_model_size):
	if base_model_size == 'small':
		from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
		weights = EfficientNet_V2_S_Weights.DEFAULT
		base_model = efficientnet_v2_s(weights=weights)
		base_model = base_model.to("cuda")
	elif base_model_size == 'medium':
		from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
		weights = EfficientNet_V2_M_Weights.DEFAULT
		base_model = efficientnet_v2_m(weights=weights)
		base_model = base_model.to("cuda")
	elif base_model_size == 'large':
		from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
		weights = EfficientNet_V2_L_Weights.DEFAULT
		base_model = efficientnet_v2_l(weights=weights)
		base_model = base_model.to("cuda")
	else:
		raise NotImplementedError("base_model_size must be 'small', 'medium' or 'large'")
	preprocess = weights.transforms()
	return base_model, preprocess


def create_train_test_split():
	df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

	df = df.sample(
		frac=1, replace=False, random_state=RANDOM_SEED
	).copy()

	df_train = df[df['split'] == 'train'].copy()
	df_test = df[df['split'] == 'test'].copy()
	df_train, df_val = train_test_split(df_train, train_size=0.8, random_state=RANDOM_SEED)

	print(f"Train size:      {len(df_train)}")
	print(f"Validation size: {len(df_val)}")
	print(f"Test size:       {len(df_test)}")

	return df_train, df_val, df_test


def pytorch_training_loop(
		model, optimizer, train_data, val_data, num_labels,
		start_epoch, total_epochs, batch_size, model_save_path,
		scheduler=None, best_val_score=0, patience=5, shuffle_data=False
	):
	epoch = start_epoch
	n_train = train_data.num_examples

	non_improvement = 0

	for epoch in range(epoch, total_epochs):

		epoch_loss = 0
		epoch_p = 0
		epoch_r = 0
		epoch_f = 0
		epoch_test_steps = 0

		print("Training epoch {}".format(epoch + 1))
		if non_improvement > 0:
			print("Early stopping {}/{}".format(non_improvement, patience))

		model.train()

		if shuffle_data:
			train_data.shuffle(random_seed=epoch)

		pbar = tqdm(range(0, n_train, batch_size), unit="steps", desc="Training")

		# Training loop
		for i in pbar:
			end_ix = min(i + batch_size, n_train)
			start_ix = end_ix - batch_size
			step_count = int(start_ix / batch_size)

			input_dict = train_data.get_batch(start_ix, end_ix, return_labels=True)

			optimizer.zero_grad()

			model_output = model(**input_dict)
			loss = model_output['loss']

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer.step()

			if step_count % 10 == 0:
				probabilities = model_output[
					'probabilities'].detach().cpu().numpy()
				predictions = (probabilities > 0.5).astype(np.int32)
				p, r, f, _ = precision_recall_fscore_support(
					input_dict['targets'].cpu().detach().numpy(),
					predictions,
					average='weighted',
					zero_division=0
				)
				epoch_test_steps += 1
				epoch_loss += loss.detach().item()
				epoch_p += p
				epoch_r += r
				epoch_f += f
				pbar.set_postfix_str(
					"Loss {:.4f}, Precision {:.4f}, Recall {:.4f}, F1 {:.4f}".format(
						epoch_loss / epoch_test_steps,
						epoch_p / epoch_test_steps,
						epoch_r / epoch_test_steps,
						epoch_f / epoch_test_steps
					)
				)

		# Validation loop
		val_probs, val_targs = pytorch_testing_loop(
			model, val_data, num_labels, batch_size
		)
		val_predictions = (val_probs > 0.5).astype(np.int32)
		val_p, val_r, val_f, _ = precision_recall_fscore_support(
			val_targs,
			val_predictions,
			average='weighted',
			zero_division=0
		)
		if val_f >= best_val_score:
			best_val_score = val_f
			non_improvement = 0
			torch.save(model.state_dict(), model_save_path)
		else:
			non_improvement += 1
		print('Precision (val): {:.4f}, Recall (val): {:.4f}, F1 (val): {:.4f} | Best validation F1: {:.4f}\n'.format(
			val_p, val_r, val_f, best_val_score)
		)

		if non_improvement == patience:
			return best_val_score

		if scheduler is not None:
			scheduler.step(val_f)

	return best_val_score


def pytorch_testing_loop(model, test_data, num_labels, batch_size):
	n_test = test_data.num_examples

	# Testing loop
	test_probs = np.zeros((n_test, num_labels))
	test_targs = np.zeros((n_test, num_labels))
	model.eval()

	pbar = tqdm(range(0, n_test, batch_size), unit="steps", desc="Testing")
	for i in pbar:
		end_ix = min(i + batch_size, n_test)
		start_ix = end_ix - batch_size

		input_dict = test_data.get_batch(start_ix, end_ix, return_labels=True)

		with torch.no_grad():
			model_output = model(**input_dict)

		_probs = model_output['probabilities'].detach().cpu().numpy()
		test_probs[start_ix:end_ix, :] = _probs
		test_targs[start_ix:end_ix, :] = input_dict['targets'].detach().cpu().numpy()

	return test_probs, test_targs


def calculate_f1(model, test_data, average="weighted"):
	average = 'weighted'

	test_probs, test_targs = pytorch_testing_loop(
		model=model,
		test_data=test_data,
		num_labels=test_data.num_labels,
		batch_size=8
	)
	test_predictions = (test_probs > 0.5).astype(np.int32)

	# Precision, recall, F1
	test_p, test_r, test_f, _ = precision_recall_fscore_support(
		test_targs,
		test_predictions,
		average=average,
		zero_division=0
	)

	# Accuracy
	pred_index = np.argmax(test_predictions, axis=1)
	target_index = np.argmax(test_targs, axis=1)
	correct = [p == y for p, y in zip(pred_index, target_index)]
	test_acc = np.mean(correct)

	# Confusion matrix
	test_conf_mat = confusion_matrix(np.argmax(test_targs, axis=1), np.argmax(test_predictions, axis=1))

	# F1 per label
	ind2label = {v: k for k, v in test_data.label2ind.items()}
	f1_per_label = {}
	for i in range(test_targs.shape[1]):
		p, r, f1, _ = precision_recall_fscore_support(
			test_targs[:, i],
			test_predictions[:, i],
			average='binary',
			zero_division=0
		)
		f1_per_label[ind2label[i]] = f1

	return test_p, test_r, test_f, test_acc, test_conf_mat, f1_per_label


# Clean up the labels and convert to multi-class
label_clean_up(DATA_DIR)

# Load the base model
base_model, preprocess = load_base_model(BASE_MODEL_SIZE)

# Prepare the data
df_train, df_val, df_test = create_train_test_split()

train_data = CervData(
	data_dir=DATA_DIR,
	df=df_train,
	label2ind=None,
	shuffle=True,
	random_seed=RANDOM_SEED,
	preprocessor=preprocess,
	multi_label=False,
	convert_to_grayscale=False,
	device='cuda'
)

val_data = CervData(
	data_dir=DATA_DIR,
	df=df_val,
	label2ind=train_data.label2ind,
	shuffle=True,
	random_seed=RANDOM_SEED,
	preprocessor=preprocess,
	multi_label=False,
	convert_to_grayscale=False,
	device='cuda'
)

test_data = CervData(
	data_dir=DATA_DIR,
	df=df_test,
	label2ind=train_data.label2ind,
	shuffle=True,
	random_seed=RANDOM_SEED,
	preprocessor=preprocess,
	multi_label=False,
	multi_label_sep=';',
	convert_to_grayscale=False,
	device='cuda'
)

# Let's look at the label distribution
datasets = {
	'train': train_data,
	'val': val_data,
	'test': test_data
}
for split, data in datasets.items():
	tmp = data._get_batch_labels(0, data.num_examples)
	counts = tmp.sum(axis=0).cpu().numpy()
	tmp = pd.DataFrame(
		data={
			'Label': train_data.label2ind.keys(),
			'count': counts,
		}
	)
	tmp['proportion'] = tmp['count'] / data.num_examples
	print(tmp.set_index('Label'))
	tmp.set_index('Label').to_csv(
		os.path.join(DATA_DIR, f"label_distribution_{split}.csv")
	)

# Construct the classifier
model = CervEffNet(
	base_model,
	num_classes=6,
	freeze_top_n=6,
	multi_label=False
)
model = model.to("cuda")

optimizer = torch.optim.Adam(
	model.parameters(),
	weight_decay=0,
	lr=3e-04
)

val_score = pytorch_training_loop(
	model=model,
	optimizer=optimizer,
	train_data=train_data,
	val_data=val_data,
	num_labels=train_data.num_labels,
	start_epoch=0,
	total_epochs=30,
	batch_size=8,
	model_save_path=os.path.join(DATA_DIR, 'model.pt'),
	scheduler=None,
	best_val_score=0,
	patience=5,
	shuffle_data=False,
)

# Test metrics
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'model.pt')))
average = "weighted"
test_p, test_r, test_f, test_acc, test_conf_mat, f1_per_label = calculate_f1(
	model, test_data, average=average
)
print("Weighted metrics")
print(f"Precision ({average}) (test) {test_p:.4f}")
print(f"Recall ({average}) (test)    {test_r:.4f}")
print(f"F1 ({average}) (test)        {test_f:.4f}")
print(f"Test accuracy:               {test_acc * 100:.2f}%")
with open(os.path.join(OUTPUT_DIR, "weighted_metrics.txt"), "w") as f:
	f.write("Weighted metrics\n")
	f.write(f"Precision ({average}) (test) {test_p:.4f}\n")
	f.write(f"Recall ({average}) (test)    {test_r:.4f}\n")
	f.write(f"F1 ({average}) (test)        {test_f:.4f}\n")
	f.write(f"Test accuracy:               {test_acc * 100:.2f}%")

average = "macro"
test_p, test_r, test_f, test_acc, _, _ = calculate_f1(
	model, test_data, average=average
)
print("Weighted metrics")
print(f"Precision ({average}) (test) {test_p:.4f}")
print(f"Recall ({average}) (test)    {test_r:.4f}")
print(f"F1 ({average}) (test)        {test_f:.4f}")
print(f"Test accuracy:               {test_acc * 100:.2f}%")
with open(os.path.join(OUTPUT_DIR, "macro_metrics.txt"), "w") as f:
	f.write("Weighted metrics\n")
	f.write(f"Precision ({average}) (test) {test_p:.4f}\n")
	f.write(f"Recall ({average}) (test)    {test_r:.4f}\n")
	f.write(f"F1 ({average}) (test)        {test_f:.4f}\n")
	f.write(f"Test accuracy:               {test_acc * 100:.2f}%")

print("Confusion matrix")
print(train_data.label2ind)
print(test_conf_mat)
with open(os.path.join(OUTPUT_DIR, "confusion_matrix.txt"), "w") as f:
	for i in test_conf_mat.shape[0]:
		for j in test_conf_mat.shape[1]:
			if j > 0:
				f.write("\t")
			f.write(test_conf_mat[i, j])
		f.write("\n")

print("F1 per label")
with open(os.path.join(OUTPUT_DIR, "f1_per_label.txt"), "w") as f:
	for k, v in f1_per_label.items():
		print(k, v)
		f.write(f"{k}\t{v}\n")
