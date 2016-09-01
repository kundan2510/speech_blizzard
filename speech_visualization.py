import matplotlib
matplotlib.use('Agg')


import numpy
import matplotlib.pyplot as plt
import os


from wav_utils import read_wav

from sys import argv

EPS = 0.0001

def normalise_data(data):
	data = data.astype('float32')
	data -= data.min()
	data += EPS
	data /= data.max()
	data -= 0.5
	return data


frame_rate, sample_width, nframes, audio_array = read_wav(argv[1])

data = normalise_data(audio_array)

assert(len(data) >= 0.5*frame_rate)

# plt.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off') 

plt.figure(figsize=(6, 12), dpi=320)


SCALES = 6
NUM_POINTS_TO_PLOT = 2000
GROWTH_RATE = 2
data = data[1000:]
# print data[:100]
gr = [2,3,4]
n_ps = [64, 64*4, 64*64]
for GROWTH_RATE in gr:
	for NUM_POINTS_TO_PLOT in n_ps:
		for j in range(SCALES):
			plt.subplot(SCALES,1,j+1)
			plt.ylim(-0.5, 0.5)
			down_scale = GROWTH_RATE**j
			for i in range(NUM_POINTS_TO_PLOT//down_scale):
				plt.plot(i*down_scale + (down_scale//2), numpy.min(data[i*down_scale: (i+1)*down_scale]), 'b.')
				plt.plot(i*down_scale + (down_scale//2), numpy.max(data[i*down_scale: (i+1)*down_scale]), 'r.')
				plt.plot(i*down_scale + (down_scale//2), numpy.mean(data[i*down_scale: (i+1)*down_scale]), 'g.')
		plt.savefig("{}_{}_{}.png".format(argv[1], NUM_POINTS_TO_PLOT, GROWTH_RATE))
		print "Done {}_{}_{}.png".format(argv[1], NUM_POINTS_TO_PLOT, GROWTH_RATE)
		plt.clf()




