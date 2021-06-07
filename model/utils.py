__author__ = "Yu Du"
__Email__ = "yu.du@clinchoice.com"
__date__ = "March 3,2020"
################################################################################################################################################################################################################################################
import tensorflow as tf
import numpy as np
import re

class callback_(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('val_acc') is not None and logs.get('val_acc')>0.95):
			print('Reached 95% validation accuracy so cancelling training')
			self.model.stop_training=True

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	print(K.get_value(precision))
	print(K.get_value(recall))
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def read_emb_vecs(vocab_file, vector_file):
	"""
	This function takes 2 files as input and return the embeddings word_to_index, index_to_word and word_to_vec_map
	:param vocab_file vocabulary
	:param vector_file each line corresponds to the vector of the word from vocab
	"""
	with open(vocab_file, 'r', encoding="utf8") as f:
		with open(vector_file, 'r', encoding="utf8") as f2:
			words = set()
			word_to_vec_map = {}
			for line, line2 in zip(f, f2):
				line = line.strip().split()
				line2 = line2.rstrip('\n').split("\t")[:400]
				curr_word = re.sub(r'[^a-zA-Z]',' ', line[0])
				curr_word = curr_word.lower()
				# curr_word = line[0].lower()
				words.add(curr_word)
				word_to_vec_map[curr_word] = np.array(line2, dtype=np.float64)

			i = 1
			words_to_index = {}
			index_to_words = {}
			for w in sorted(words):
				words_to_index[w] = i
				index_to_words[i] = w
				i = i + 1
	return words_to_index, index_to_words, word_to_vec_map


def plotresult(hist, title, outputfile):
	acc = hist.history['acc']
	val_acc = hist.history['val_acc']

	epochs = len(acc)
	plt.plot(range(epochs), acc, marker='.', label='acc')
	plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
	plt.legend(loc='best')
	plt.grid()
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.title('Training/Validation: '+ title)
	plt.savefig('images/'+outputfile)
	plt.show()


##################################################################################################################################
#                                                 DO NOT USE CODE BELOW THIS LINE                                                #
##################################################################################################################################
# def save_emb_vecs(file_name):
# 	import pickle
# 	with open(file_name, 'wb') as handle:
#     	pickle.dump(word_to_vec_map, handle)



# def load_emb_vecs(file_name):
#     with open(file_name, 'rb') as handle:
#         file = pickle.load(handle)
#     return file
##################################################################################################################################
#                                                                                                                                #
##################################################################################################################################
