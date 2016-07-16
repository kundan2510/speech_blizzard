from models import *
from layers import GRU, Embedding, Softmax, FC, Concat, WrapperLayer, pixelConv
from training_stats import save_cost
from generic_utils import *

import theano
import theano.tensor as T
import theano.ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

import numpy

from wav_utils import write_audio_file
from dataset import feed_data

import pickle
import time
import os
import argparse
import sys
import logging




parser = argparse.ArgumentParser(description='Generating single acoustic sample at a time for blizzard')

parser.add_argument('-o','--output_folder', required=True, help='Base output folder for storing generated audio, parameters and training costs')
parser.add_argument('-tdata','--train_data_folder', required=True, help="A folder path containing flac files")
parser.add_argument('-bs', '--batch_size', default= 128, type=int, help="Number of examples to consider in a batch")
parser.add_argument('-tbptt', '--truncated_back_prop_time', default= 64, type=int, help="number of time-steps for backpropagating errors for recurrent cunnections")
parser.add_argument('-n_gru','--number_of_gru', default=1, type=int)
parser.add_argument('-ql', '--q_level', default=256, type=int)
parser.add_argument('-br','--bitrate', default=16000, type=int)
# parser.add_argument('-dim','--gru_dimension', default=512, type=int) # dim is width_dim x depth_dim
parser.add_argument('-oi','--other_info', default='')
parser.add_argument('-pm','--print_mode', default='epoch')
parser.add_argument('-si','--stop_iters', default=100000, type=int)
parser.add_argument('-pi','--print_iters', default=1000, type=int)
parser.add_argument('-pe','--print_epoch', default=1, type=int)
parser.add_argument('-se','--stop_epoch', default=100, type=int)
parser.add_argument('-pt','--print_time', default=30*60, type=int)
parser.add_argument('-st','--stop_time', default=4*24*60*60, type=int)
parser.add_argument('-gc','--grad_clip', default=1, type=numpy.float32)
parser.add_argument('-ns','--num_gen_samples', default=10, type=int)
parser.add_argument('-lgs','--length_gen_sample', default=3, type=int, help="Give length of generated samples in seconds")
parser.add_argument('-pre','--pre_trained_model', default=None, help="Path to pre-trained model")
parser.add_argument('-np','--num_pixelCNN_layer', default=4, type=int, help="Number of pixel CNN layers to use")
parser.add_argument('-wd','--width_dim', default=16, type=int, help="Number of groups of dimensions which are groupwise conditioned")
parser.add_argument('-dep','--depth_dim', default=32, type=int, help="Length of a single group of dimension in which everything is considered independent pf each other.")
parser.add_argument('-n_files','--num_files', default=10000, type=int, help="Number of flac files to use.")


args = parser.parse_args()


TRAIN_DATA_FOLDER = args.train_data_folder

BATCH_SIZE = args.batch_size
NUM_SAMPLES_IN_TBPTT = args.truncated_back_prop_time # How many audio samples to include in each truncated BPTT pass
N_GRUS = args.number_of_gru # How many GRUs to stack in the model
WIDTH = args.width_dim
DEPTH = args.depth_dim

DIM = WIDTH*DEPTH # GRU dimension
Q_LEVELS = args.q_level # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = args.grad_clip # Elementwise grad clip threshold
BITRATE = args.bitrate
RESET_AFTER_SAMPLES = args.reset_after # reset hidden state after these many samples
NUM_PIXEL_CNN_LAYER = args.num_pixelCNN_layer


TRAIN_MODE = args.print_mode #one of 'epoch', 'time' or 'iters'
PRINT_ITERS = args.print_iters # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = args.stop_iters # Stop after this many iterations
PRINT_TIME = args.print_time # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = args.stop_time # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
PRINT_EPOCH = args.print_epoch
STOP_EPOCH = args.stop_epoch
NUM_SAMPLES_TO_GEN = args.num_gen_samples

LENGTH_GEN = args.length_gen_sample*BITRATE
OTHER_INFO = args.other_info #info that is specific to model, will be used to create folder name

BASE_OUTPUT_FOLDER = args.output_folder # it will contain folders for params, generated audio and plots

FOLDER_TO_SAVE_PARAMS = '/{}_params/'.format(OTHER_INFO) # it should be relative to BASE_OUTPUT_FOLDER
FOLDER_TO_SAVE_GENERATED_AUDIO = '/{}_gen_audio/'.format(OTHER_INFO) # it should be relative to BASE_OUTPUT_FOLDER
FOLDER_TO_SAVE_STATS = '/{}_stats/'.format(OTHER_INFO) # it should be relative to BASE_OUTPUT_FOLDER




create_folder_if_not_there(BASE_OUTPUT_FOLDER + FOLDER_TO_SAVE_GENERATED_AUDIO)
create_folder_if_not_there(BASE_OUTPUT_FOLDER + FOLDER_TO_SAVE_PARAMS)
create_folder_if_not_there(BASE_OUTPUT_FOLDER + FOLDER_TO_SAVE_STATS)

save(args, BASE_OUTPUT_FOLDER + FOLDER_TO_SAVE_STATS + 'args.pkl')

LOG_FILENAME = BASE_OUTPUT_FOLDER +'/train.log'

logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(BASE_OUTPUT_FOLDER + FOLDER_TO_SAVE_STATS +'other_log_{}.log'.format(args.other_info))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



def create_output_node(model=None, input_sequences=None, num_gru=None, old_h0s=None, reset=None, num_pixelCNN_layer = None):
	assert(model is not None)
	assert(input_sequences is not None)
	assert(num_gru is not None)
	assert(old_h0s is not None)
	assert(reset is not None)
	assert(num_pixelCNN_layer is not None)

	new_h0s = T.zeros_like(old_h0s)
	h0s = theano.ifelse.ifelse(reset, new_h0s, old_h0s)

	level_embedding_layer = Embedding(Q_LEVELS, DIM, input_sequences, name = model.name+"Embedding.Q_LEVELS")
	model.add_layer(level_embedding_layer)

	model.add_layer(category_embedding_layer)

	last_layer = embedding_layer

	prev_out = embedding_layer.output()
	last_layer = WrapperLayer(prev_out.reshape(prev_out.shape[0], prev_out.shape[1], WIDTH, DEPTH))

	pixel_CNN = pixelConv(
		last_layer, 
		DEPTH, 
		DEPTH,
		name = model.name + ".pxCNN",
		num_layers = 3
	)

	prev_out = pixel_CNN.output()
	last_layer = WrapperLayer(prev_out.reshape((prev_out.shape[0], prev_out.shape[1], -1)))

	last_hidden_list = []

	for i in range(num_gru):
		gru_layer = GRU(DIM, DIM, last_layer, s0 = h0s[i,:,:], name = model.name+"GRU_{}".format(i))
		last_hidden_list.append(gru_layer.output()[:,-1])
		model.add_layer(gru_layer)
		last_layer = gru_layer

	fc1 = FC(DIM, Q_LEVELS, last_layer, name = model.name+"FullyConnected")
	model.add_layer(fc1)

	softmax = Softmax(fc1, name= model.name+"Softmax")
	model.add_layer(softmax)

	return softmax.output(), T.stack(last_hidden_list, axis = 0)


def generate_samples(generate_fn, base_name):

	samples = numpy.zeros((NUM_SAMPLES_TO_GEN_PER_CLASS, LENGTH_GEN+3), dtype='int32')
	samples[:,:2] = Q_LEVELS//2

	last_hidden = numpy.zeros((N_GRUS, NUM_SAMPLES_TO_GEN_PER_CLASS, DIM), dtype='float32')
	reset = floatX(0)

	for i in range(LENGTH_GEN-1):
		temp, last_hidden = generate_fn(samples[:,max(0,i-3*NUM_PIXEL_CNN_LAYER):i+3], category_index_gen[:,max(0,i-3*NUM_PIXEL_CNN_LAYER):i+2], last_hidden, reset)
		samples[:,i+2] = temp[:,-1]

	for i in range(len(samples)):
		write_audio_file("{}_{}".format(base_name,i), BITRATE, samples[i,2:LENGTH_GEN+2])


model = Model(name = "Blizzard.Speech.Model")

sequences   = T.imatrix('sequences')
h0s          = T.tensor3('h0s')
reset       = T.iscalar('reset')

input_sequences = sequences[:, :-1]
target_sequences = sequences[:, 1:]

output_probab, last_hidden = create_output_node(
	model = model,
	input_sequences = input_sequences,
	num_gru = N_GRUS,
	reset = reset,
	old_h0s = h0s,
	num_pixelCNN_layer = NUM_PIXEL_CNN_LAYER
	)

next_samples = sample_from_softmax(output_probab)

cost = T.nnet.categorical_crossentropy(
	output_probab.reshape((-1,Q_LEVELS)),
	target_sequences.flatten()
	).mean()

# reporting NLL in bits
cost = cost * floatX(1.44269504089)

params = model.get_params()

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function([sequences, category_index, h0s,reset], [cost, last_hidden], updates = updates)

valid_fn = theano.function([sequences, category_index, h0s,reset], [cost, last_hidden])

generate_fn = theano.function([sequences, category_index, h0s,reset], [next_samples, last_hidden])


if args.pre_trained_model is not None:
	model.load_params(args.pre_trained_model)

logger.info("Training!")

total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0

h0 = numpy.zeros((N_GRUS, BATCH_SIZE, DIM), dtype='float32')
costs = []

train_data_feeder = feed_data(
	TRAIN_DATA_FOLDER,
	N_FILES,
	BATCH_SIZE, 
	NUM_SAMPLES_IN_TBPTT, 
	1, 
	Q_LEVELS, 
	Q_LEVELS//2
	)
logger.info("Created train_data_feeder")

total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
last_print_epoch = 0

cost_dict = {'validation':[], 'iters':[]}


for seqs, act_ind, reset, epoch in train_data_feeder:

	start_time = time.time()
	cost, h0 = train_fn(seqs,act_ind, h0, reset)
	total_time += time.time() - start_time

	total_iters += 1

	costs.append(cost)

	if total_iters % 100 == 0:
		logger.info("Training, epoch : {}, iters {}, train cost : {}".format(epoch, total_iters, numpy.mean(costs)))


	if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
	    (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME) or \
	    (TRAIN_MODE=='epoch' and epoch-last_print_epoch >= PRINT_EPOCH):

		print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}\ttime per iter:{}".format(
		    epoch,
		    total_iters,
		    numpy.mean(costs),
		    total_time,
		    total_time / total_iters
		)

		base_name = "epoch_{}_iters_{}_cost_{}".format(epoch, total_iters, numpy.mean(costs))

		cost_dict['training'].append(numpy.mean(costs))
		cost_dict['iters'].append(total_iters)

		save_cost(cost_dict, BASE_OUTPUT_FOLDER+FOLDER_TO_SAVE_STATS+base_name)
		generate_samples(generate_fn, BASE_OUTPUT_FOLDER+FOLDER_TO_SAVE_GENERATED_AUDIO+base_name)
		model.save_params(BASE_OUTPUT_FOLDER+FOLDER_TO_SAVE_PARAMS+base_name+'.pkl')

		costs = []
		last_print_time = total_time
		last_print_iters = total_iters
		last_print_epoch = epoch

		if (TRAIN_MODE=='iters' and total_iters >= STOP_ITERS) or \
		    (TRAIN_MODE=='time' and total_time >= STOP_TIME) or \
		    (TRAIN_MODE=='epoch' and epoch >= STOP_EPOCH):

		    print "Done!"
		    sys.exit()

