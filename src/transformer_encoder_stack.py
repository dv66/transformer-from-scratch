from transformer_encoder import TransformerEncoder
import torch

class TransformerEncoderStack(nn.Module):

	def __init__(self, n_encoders):
		self.n_encoders = n_encoders
		self.encoders = torch.nn.ModuleList([TransformerEncoder() for _ in range(n_encoders)])

	
