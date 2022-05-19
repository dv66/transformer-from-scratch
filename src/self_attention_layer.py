import torch 
from torch import nn 



class SelfAttentionLayer(nn.Module):

	def __init__(self):
		self.w_q = None 
		self.w_k = None 
		self.w_v = None 


	def get_self_attention_score(self, p, others, p_dim):
		score = 0
		for o in others:
			score += torch.dot(o, p) / torch.sqrt(p_dim)

		return torch.softmax(score, -1)

	def forward(self, x):
		query = self.w_q * x
		key = self.w_k * x
		value = self.w_v * x
		attention_scores = torch.tensor(
				[get_self_attention_score(_, x, _.size[0]) for _ in x]
			)
		attention_vectors = torch.mul(value, attention_scores)

		return attention_vectors








		