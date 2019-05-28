from nd2reader import ND2Reader
import numpy as np

class ImageContainer(object):
	"""docstring for ImageContainer"""
	def __init__(self, filename):
		self.filename = filename
		images = self.read_file()
		
		# for i in images:
		# 	print(np.sum(i))
		# print(np.sum(next(images)))
		# print(np.sum(next(images)))
		# print(np.sum(next(images)))
		
	def read_file(self):
		with ND2Reader(self.filename) as images:
			print(images.sizes)
			# if '' in images.sizes.keys():
			# 	images.bundle_axes = 'zyx'
			# 	images.iter_axes = 'v'
			# else:
			images.bundle_axes = 'zyx'

			images.default_coords['c'] = 1
			# images.default_coords['v'] = 1
			images.iter_axes = 'v'

			for i in images:
				print(np.sum(i))

if __name__ == '__main__':
	testfile = '../data/test_stack.nd2'
	# testfile = '../data/test_file.nd2'
	images = ImageContainer(testfile)

