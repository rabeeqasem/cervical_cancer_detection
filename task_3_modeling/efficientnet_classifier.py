import torch
from torch import nn


class CervEffNet(nn.Module):
	def __init__(self, base_model, num_classes, freeze_top_n=0, multi_label=False):
		super(CervEffNet, self).__init__()
		self.features = base_model.features
		self.avgpool = base_model.avgpool
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.2, inplace=True),
			nn.Linear(
				in_features=1280,
				out_features=num_classes,
				bias=True
			)
		)

		torch.nn.init.xavier_uniform_(self.classifier[1].weight)
		torch.nn.init.constant_(self.classifier[1].bias, 0)

		if freeze_top_n > 0:
			self.freeze_top(freeze_top_n)

		if multi_label:
			self.prob_func = nn.Sigmoid()
			self.loss_func = nn.BCEWithLogitsLoss()
		else:
			self.prob_func = nn.Softmax(dim=1)
			self.loss_func = nn.CrossEntropyLoss()

	def freeze_top(self, top_n=1):
		for param in self.features[:top_n].parameters():
			param.requires_grad = False

	def forward(self, x, targets=None):
		features = self.avgpool(
			self.features(x)
		)
		out = self.classifier(features.squeeze())

		outputs = {
			'pooled_output': out,
			'probabilities': self.prob_func(out)
		}

		if targets is not None:
			outputs['loss'] = self.loss_func(out, targets)

		return outputs
