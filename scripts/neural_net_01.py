from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# General matplotlib setup
mpl.rcParams['figure.dpi'] = 300


class NeuralNet:

	def __init__(self, weights, biases, activations):
		"""Initializes attributes of the NeuralNet object. The architecture 
		of the network is implicitly specified with the weights and biases
		list strcuture.

		Paramaters
		----------
		weights : list
			A list of lists containing the weights per layer of the network.
		biases : list
			A list of lists containing the biases per layer of the network.
		activations : list
			A list with the activaiton functions per layer of the network.
		"""

		self.w = np.array([np.asarray(w) for w in weights])
		self.b = np.array([np.asarray(b) for b in biases])
		self.a = np.array(activations)


	@staticmethod
	def apply_layer(y_in, w, b, activation='linear'):
		"""Calculates the forward-pass from a layer into another. This 
		function acts over input batches.

		Parameters
		----------
		y_in : array-like, shape (batch size, # neurons in)
			An array-like containing the inputs of the layer (in batches).
		w : ndarray, shape (# neurons in, # neurons out)
			A ndarray with the weights associated to the input.
		b : ndarray, shape (# neurons out)ndarray
			A ndarray containing the bias to be added.
		activation : str
			The activation function to be used. The options are: 'sigmoid',
			'step' (or 'jump'), 'linear', 'relu' (or 'reLU'). Be default, 
			this parameter is set to 'linear'.

		Returns
		-------
		y_out : ndarray, shape (batch size, # neurons out)
			A ndarray containing the output of the layer.
		"""

		z = np.dot(y_in, w) + b

		if activation == 'sigmoid':
			return 1. / (1. + np.exp(-z))
		elif activation == 'jump' or 'step':
			return np.array(z > 0, dtype='float')
		elif activation == 'linear':
			return z
		elif activation == 'relu' or 'reLU':
			return (z > 0) * z


	def feedforward(self, y_in):
		"""Calculates the forward-pass between all the layers.

		Parameters
		----------
		y_in : array-like, shape (batch size, # neurons in)
			An array-like object containing the inputs of the layer (in
			batches).

		Returns
		-------
		y_out : ndarray, shape (batch size, # neurons out)
			A ndarray containing the output of the layer.
		"""

		# Transpose weights
		_w_t = [w.T for w in self.w]

		y_out = y_in
		for k in range(len(self.b)):
			y_out = self.apply_layer(y_out, _w_t[k], self.b[k], self.a[k])

		return y_out


	@classmethod
	def apply_net(cls, y_in, weights, biases, activations):
		"""Calculates the forward-pass between all the layers.

		Parameters
		----------
		y_in : array-like, shape (batch size, # neurons in)
			An array-like object containing the inputs of the layer (in
			batches).

		Returns
		-------
		y_out : ndarray, shape (batch size, # neurons out)
			A ndarray containing the output of the layer
		"""

		nn = cls(weights, biases, activations)
		y_out = nn.feedforward(y_in)

		return y_out

	@staticmethod
	def plot_connection(ax, x, y, w, linewidth=3., vmax=1.,
						col_a=[0.000, 0.588, 0.533], 
						col_b=[0.247, 0.317, 0.709]):
		"""Utility function to plot the connection between neurons.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes where the connection will be plotted.
		x : list
			A list containind the x-coordinates to plot the conncetion.
		y : list
			A list containind the y-coordinates to plot the conncetion.
		w : float
			The value of the weight to be plotted.
		linewidth : float
			The linewith that will be used for plotting. By default it 
			is set to 3.0.
		vmax : float
			The maximum value that will be used to normalize the alpha value.
			By default it is set to 1.0.
		color_a : list
			The list containing the RGB values normalized to 255 of the 
			activated connection. By default it is set to Teal 500 (#009688).
		color_b : list
			The list containing the RGB values normalized to 255 of the 
			activated connection. By default it is set to Indigo 500 (#3f51b5).
		"""

		t = np.linspace(0, 1, 20)

		if w > 0:
			color = col_a
		else:
			color = col_b

		ax.plot(
			x[0] + (3 * t ** 2 - 2 * t ** 3) * (x[1] - x[0]),
			y[0] + t * (y[1] - y[0]),
			alpha=abs(w) / vmax,
			color=color,
			linewidth=linewidth
		)

		return

	@staticmethod
	def plot_neuron(ax, x, y, b, size=100., vmax=1.,
					col_a=[0.000, 0.588, 0.533], col_b=[0.247, 0.317, 0.709]):
		"""Utility function to plot neurons over their conncetions.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes where the connection will be plotted.
		x : list
			A list containind the x-coordinates to plot the conncetion.
		y : list
			A list containind the y-coordinates to plot the conncetion.
		b : float
			The value of the bias to be plotted.
		size : float
			The size of the neuron. By default it is set to 100.0.
		vmax : float
			The maximum value that will be used to normalize the alpha value.
			By default it is set to 1.0.
		color_a : list
			The list containing the RGB values normalized to 255 of the 
			activated neuron. By default it is set to Teal 500 (#009688).
		color_b : list
			The list containing the RGB values normalized to 255 of the 
			activated neuron. By default it is set to Indigo 500 (#3f51b5).
		"""

		if b > 0:
			color = np.atleast_2d(col_a)
		else:
			color = np.atleast_2d(col_b)

		ax.scatter(
			[x], [y], 
			marker='o', 
			c=color, 
			alpha=abs(b) / vmax, 
			s=size, 
			zorder=10
		)

		return


	def visualize(self, M=200, y0_range=[-1, 1], y1_range=[-1, 1], size=400.,
				  linewidth=5., figsize=(16, 8), colormap='viridis',
				  col_a=[0.000, 0.588, 0.533], col_b=[0.247, 0.317, 0.709],
				  display=True):
		"""Visualize a neural network with 2 input neurons and 1 output 
		neuron.

		Parameters
		---------
		M : int
			The grid size to be used (M x M).
		y0_range : list
			The range of y0 neuron values (horizontal axis).
		y1_range : list
			The range of y1 neuron values (vertical axis).
		size : float
			The size of the neuron. By default it is set to 400.0.
		linewidth : float
			The linewith that will be used for plotting. By default it 
			is set to 5.0.
		figsize : tuple
			A tuple with the dimensions of the plot. By default it is set 
			to (16, 8).
		colormap : str
			The name of the matplotlib colormap to be used. By default it is
			set to 'viridis'.
		color_a : list
			The list containing the RGB values normalized to 255 of the 
			activated connection. By default it is set to Teal 500 (#009688).
		color_b : list
			The list containing the RGB values normalized to 255 of the 
			activated connection. By default it is set to Indigo 500 (#3f51b5).
		display : bool
			A plot that calls plt.show(). By default it is set to True.
		"""

		# Transpose weights
		_w_t = [w.T for w in self.w]

		# Generate grid...
		x = np.linspace(y0_range[0], y0_range[1], M)
		y = np.linspace(y1_range[0], y1_range[1], M)
		y0, y1 = np.meshgrid(x, y)

		# ...and evaluate network on it
		y_in = zip(y0.flatten(), y1.flatten())
		y_in = np.array(list(y_in))
		y_out = self.feedforward(y_in)

		# Generate base plot
		fig, ax = plt.subplots(1, 2, figsize=figsize)
		
		# Set positions of neurons on plot
		pos_x, pos_y = [[-0.5, 0.5]], [[0, 0]]
		vmax_w, vmax_b = 0.0, 0.0  # For maximum weigth and bias
		for b_i in range(len(self.b)):
			n_neurons = len(self.b[b_i])
			pos_x.append(np.array(range(n_neurons)) - 0.5 * (n_neurons - 1))
			pos_y.append(np.full(n_neurons, b_i + 1))
			vmax_w = np.maximum(vmax_w, np.max(np.abs(self.w[b_i])))
			vmax_b = np.maximum(vmax_b, np.max(np.abs(self.b[b_i])))

		# Plot connections
		for b_i in range(len(self.b)):
			for n_i in range(len(pos_x[b_i])):
				for n_j in range(len(pos_x[b_i + 1])):
					self.plot_connection(
						ax=ax[0],
						x=[pos_x[b_i][n_i], pos_x[b_i + 1][n_j]],
						y=[pos_y[b_i][n_i], pos_y[b_i + 1][n_j]],
						w=_w_t[b_i][n_i, n_j],
						linewidth=linewidth, vmax=vmax_w,
						col_a=col_a, col_b=col_b
					)
		
		# Plot neurons
		for x_i in range(len(pos_x[0])): # Input neurons (have no bias!)
			self.plot_neuron(
				ax=ax[0],
				x=pos_x[0][x_i],
				y=pos_y[0][x_i],
				b=vmax_b,
				size=size, vmax=vmax_b,
				col_a=col_a, col_b=col_b
			)
		for b_i in range(len(self.b)): # all other neurons
			for x_i in range(len(pos_x[b_i + 1])):
				self.plot_neuron(
					ax=ax[0],
					x=pos_x[b_i + 1][x_i],
					y=pos_y[b_i + 1][x_i],
					b=self.b[b_i][x_i],
					size=size, vmax=vmax_b,
					col_a=col_a, col_b=col_b
				)

		# Clean plot with the architecture
		ax[0].axis('off')

		# Plot visualization
		img = ax[1].imshow(
			np.reshape(y_out,[M, M]), origin='lower',
			extent=[y0_range[0], y0_range[1], y1_range[0],y1_range[1]],
			cmap=colormap
		)
		ax[1].set_xlabel(r'$y_0$')
		ax[1].set_ylabel(r'$y_1$')

		axins1 = inset_axes(
			ax[1],
			width="40%",  # Width is 50% of parent_bbox width
			height="5%",  # Height is 5%
			loc='upper right'
		)

		imgmin = np.min(y_out)
		imgmax = np.max(y_out)
		color_bar = fig.colorbar(
			img, cax=axins1, orientation="horizontal",
			ticks=np.linspace(imgmin, imgmax, 3)
		)
		cbxtick_obj = plt.getp(color_bar.ax.axes, 'xticklabels')
		plt.setp(cbxtick_obj, color="white")
		axins1.xaxis.set_ticks_position("bottom")

		if display:
			plt.show()

		return fig, ax