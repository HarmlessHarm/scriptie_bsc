from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from nd2reader import ND2Reader

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1)
            if self.ind > self.slices-1: self.ind = self.slices-1
        else:
            self.ind = (self.ind - 1)
            if self.ind < 0: self.ind = 0
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


fig, ax = plt.subplots(1, 1)

X = np.random.rand(20, 20, 40)
with ND2Reader('test_file.nd2') as images:
    images.bundle_axes = 'czyx'
    X = images[0][1].T


tracker = IndexTracker(ax, X)


fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()