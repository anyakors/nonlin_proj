import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import os

# Plots the result of bnn_main.py from files x1 and x2

def f(x):
    return np.sin(x)

def array_for(x):
    return np.array([f(xi) for xi in x])

def colorline(
    x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=0.7):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


x = np.genfromtxt('./x1',dtype='float')
y = np.genfromtxt('./x2',dtype='float')

fig, ax = plt.subplots()

path = mpath.Path(np.column_stack([x, y]))
verts = path.interpolated(steps=3).vertices
x, y = verts[:, 0], verts[:, 1]

z = abs(np.sin(1.6*y[::2]+0.4*x[::2])) #let's make them a bit more transparent by skipping every other value
colorline(x[::2], y[::2], z, cmap=plt.get_cmap('Spectral'), linewidth=1)
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())

#fig = plt.gcf()
#fig.set_size_inches(13.5, 13.5)
#fig.savefig('test.png', dpi=300)
plt.show()
