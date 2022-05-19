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



	def dummy_forward(self, input_vectors):
		attention_vectors = []
		for x in input_vectors:
			query_vector = self.w_q * x
			key_vector = self.w_k * x
			value_vector = self.w_v * x
			attention_scores = torch.tensor(
					[get_self_attention_score(x, _, _.size[0]) for _ in input_vectors]
				)
			attention_vectors.append(torch.mul(value, attention_scores))

		return torch.tensor(attention_vectors)













