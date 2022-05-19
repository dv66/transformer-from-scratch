from self_attention_layer import SelfAttentionLayer
from feed_forward_layer import FeedForwardLayer


class TransformerEncoder:

	def __init__(self):
		self.self_attention_layer = SelfAttentionLayer()
		self.feed_forward_layer = FeedForwardLayer()

	