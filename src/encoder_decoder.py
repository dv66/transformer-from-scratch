



class TransformerEncoderDecoder(nn.Module):
	
	def __init__(self, encoder, decoder, source_embedding, target_embedding):
		super(TransformerEncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.source_embedding = source_embedding
		self.target_embedding = target_embedding

	def encode(self, source):
		return self.encoder(self.source_embedding(source))

	def decode(self, encoded_vector):
		return self.decoder(encoded_vector)

	def forward(self, source):
		return self.decode(self.encode(source))





