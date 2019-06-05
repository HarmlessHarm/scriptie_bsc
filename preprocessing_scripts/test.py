from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import numpy as np
import pprint

with ND2Reader('test_file.nd2') as images:
	images.bundle_axes = 'czyx'
	print(images.frame_shape)
	for c, channel_data in enumerate(images[0]):
		for z, frame in enumerate(channel_data):
			print('channel: {}, z: {}'.format(c, z))
			if z == 15:
				plt.figure()
				plt.imshow(frame)

	plt.show()