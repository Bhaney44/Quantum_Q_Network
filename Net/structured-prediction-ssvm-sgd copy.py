
# coding: utf-8

# # Introduction to structured prediction
# 
# In this example, we will introduce one category of **supervised learning**, which depends on solving optimization problems. It is called **structured prediction** or structured learning. 
# 
# Structured prediction is a subject which lies within the field of machine learning, which is itself a subfield of
# computer science. Structured prediction is an active field of research in machine learning.
# Crucial to its effectiveness is the crafting of features used to design a classifier.
# It is generally understood that the ability to capture contextual information significantly improves the prediction quality.
# Contextual information can be modeled using higher order features. However, the more complex the features, the harder the learning and prediction tasks.
# For this reason, most classifiers used in practice are designed using simple features that focus on single aspects of the input.
# In this regard, the fact that D-Wave quantum annealer can natively process pair-wise interactions significantly increases the computational power necessary to handle higher order features, thus potentially rendering better prediction results unattainable by software-only solutions.
# 
# The conceptual workflow of this notebook can be illustrated as follows:
# 
# <img width='800px' src='images/training_flows_ssvm.png'>
# 
# Normally, one starts with preprocessing the raw data:
# 

# # Preprocessing the data (feature engineering)
# The scene dataset has 2407 images of natural scenery from 6 categories, including _beach_, _sunset_, _fall foliage_, _field_, _mountain_ and _urban_. Each image has been processed into a vector of 294 dimensions.
# 
# Each image goes through 3 steps:
#     1. Each image is converted from RGB space to LUV space.
#     2. Each converted image is divided into 7-by-7 blocks of 3 channels.
#     3. Both the first order (mean) and second order (variance) moments from each block are extracted.
# 
# The reduced dimension of each image is now $49$ blocks $\times$ 2 moments $\times$ 3 channels = 294.
# 
# There are six possible labels for the images:
# 
# |              |   $\bf{y}$  |
# |--------------|:------:|
# |        Beach | 100000 |
# |       Sunset | 010000 |
# | Fall Foliage | 001000 |
# |        Field | 000100 |
# |     Mountain | 000010 |
# |        Urban | 000001 |
# 
# Note that for each data element, multiple labels may be turned on, e.g. an image that has both 'Beach' and 'Sunset' will have a label vector $[110000]$.
# 
# Let's do an exercise on this:

# ## Converting an image to a low-dimensional representation
# ### Loading a natural image

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('images/beach-sunset-mountain.jpg')
plt.imshow(img)


# ### Converting from RGB colorspace to CIE LUV colorspace
# 
# Images can live in different colorspaces. Every colorspace has its advantages depending on the tasks. RGB may be good for viewing, but a colorspace that involves spatial information is better to distinguish sceneries. For this task, we use images from a colorspace called **CIE 1976 (L*, u*, v*)** or **LUV** for short. Converting images from RGB to LUV colorspace will [Boutell et. al. 2004](https://www.rose-hulman.edu/~boutell/publications/boutell04PRmultilabel.pdf)
#     1. remove the nonlinear dependencies among RGB values,
#     2. have better uniformities among LUV values, and
#     3. have less complexity in its mapping.

# In[ ]:


# import skimage package
from skimage.color import rgb2luv

img_luv = rgb2luv(img)
plt.imshow(img_luv)
img_luv.shape


# ### Dividing images into 7x7 blocks and calculating first and second order moment for each block

# In[ ]:


(h, w, c) = img_luv.shape
# note that the image size might not be divisible by 7, 
# let us remove the extra pixles in the last row/column
num_blocks_w = 7 # number of blocks in width
num_blocks_h = 7 # numbe of blocks in height
block_w = w/num_blocks_w
block_h = h/num_blocks_h
img_blocks = [[[] for _ in xrange(num_blocks_w)] for _ in xrange(num_blocks_h)]
f, ax = plt.subplots(num_blocks_h,num_blocks_w)

# here we show a brute-force way of dividing the images, for demo purpose
for i in range(num_blocks_h):
    for j in range(num_blocks_w):
        img_block = img_luv[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, :]
        img_blocks[i][j]= img_block
        ax[i,j].imshow(img_block)
        ax[i,j].axis('off')


# ### Extracting the first and second order moment from each image block (block by block and channel by channel)

# In[ ]:


for i in range(num_blocks_h):
    for j in range(num_blocks_w):
        img_blocks[i][j] = [np.mean(img_blocks[i][j], axis=(0,1)), np.var(img_blocks[i][j], axis=(0,1))]

print np.array(img_blocks).shape
img_reduced = np.array(img_blocks).flatten()
print img_reduced.shape


# Now we processed the image into a reduced vector of $7 \times 7\times2\times3=294$ dimensions.
# 
# After we have preprocessed all the images into reduced vectors, we could also normalize the vector with in the range of $[0, 1]$, which is normally required by many ML models.

# # Loading preprocessed data
# From the steps shown above, we know that preprocessing of the data requires a lot of domain expertise. It can also be very time consuming. Fortunately, the **scene dataset** has been preprocessed and made public on https://sourceforge.net/projects/meka/files/Datasets/Train-test%20Splits/, along with many other datasets. For researchers who would like to focus more on the development of ML models, they can directly work on these datasets.
# 
# In this example, the scene dataset is in the standard [arff](http://www.cs.waikato.ac.nz/ml/weka/arff.html) format. 
# 
# Each dataset has an associated number of attributes, $n$, and a number of labels $L$. Each input consists of $n$ attributes $\bf{x}\in\mathbb{R}^n$ and the corresponding set of labels is encoded as $\bf{y}\in\{0,1\}^L$, where $y_i$ indicates whether label $i$ is ON (1) or OFF (0).
# 
# A snippet of the arff file is:
# ```arff
# @relation MultiLabelData
# 
# @attribute Att1 numeric
# @attribute Att2 numeric
# ...
# @attribute Att294 numeric
# @attribute Beach {0,1}
# @attribute Sunset {0,1}
# @attribute FallFoliage {0,1}
# @attribute Field {0,1}
# @attribute Mountain {0,1}
# @attribute Urban {0,1}
# 
# @data
# 0.646467,0.666435, ..., 0.029709,1,0,0,0,1,0
# ...
# ```
# 

# In[ ]:


from dwave_sp import localdata

# download the scene dataset from sourceforge.net

localdata.download_scene_files(localdata_basedir='/tmp/')

# load the scene recognition problem's training data, and divide into a training and testing iportion.
train_data, test_data, train_labels, test_labels = localdata.load_scene_split(localdata_basedir='/tmp/')
print("\tDataset loaded with {} training points, and {} testing points.".format(
        train_data.shape[0], test_data.shape[0]))

# there are totally 6 labels in scene dataset
number_of_labels = 6


# # Defining the Feature Mapping
# Let us denote the vector we obtained from previous step as $\bf{x}$ and the label vector as $\bf{y}$. The relationship between $\bf{x}$ and $\bf{y}$ can be depicted by the following figure:
# <img src='images/ssvm.png'>
# This is the 'structure' in structured prediction. Here we define the relationships between labels and how the input data will be mapped to those relationships.
# 
# We use features $\Psi(\bf{x, y})$ that can be either dependent or independent of $\bf{x}$, and linear or quadratic. Below, we create the matrix representation of all four combinations. You can choose to skip the cells for some of the features to observe the effects on the classifier.
# 
# The weighted sum of all features $\Psi(\bf{x, y})$ defines the _compatibility function_, given by
# $$f(\bf{x,y}) = <\bf{w, \Psi(\bf{x, y})}>,$$
# an optimization of which will yield the best parameters of the model, _i.e._ the weight $\bf{w}$. 
# 
# Notice that we have used **feature** engineering in the previous section to denote the how to convert an image from a high-dimensional space into a low-dimensional vector. The **feature** we talk about in this section can be considered as an extension to that, meaning that the **input-space feature** we have obtained will be combined with the **labels** either linearly or quadratically to form features that our ML model can directly work with. 

# Next we define both the linear features and quadratic features
# 1. Linear features
#     * input-independent: $\Psi(\bf{x, y}) = y_i$ for $1\leq i\leq T$.
#     * input-dependent: $\Psi(\bf{x, y}) = x_ky_i$ for $1\leq i\leq T$ and $1\leq k\leq n$.
# 2. Quadratic features
#     * input-independent: $\Psi(\bf{x, y}) = y_iy_j$ for $1\leq i < j\leq T$.
#     * input-dependent: $\Psi(\bf{x, y}) = x_ky_iy_j$ for $1\leq i < j \leq T$ and $1\leq k\leq n$.
#     
# The functions are defined in `features.py` under the directory of `dwave_sp/tools` 

# In[ ]:


from dwave_sp.tools import features as feature_fun

features = None
features = feature_fun.independent_linear(number_of_labels) # linear input-independent
features += feature_fun.dependent_linear(train_data.shape[1], number_of_labels) # linear input-dependent
features += feature_fun.independent_quadratic(number_of_labels) # quadratic input-independent
features += feature_fun.dependent_quadratic(train_data.shape[1], number_of_labels) # quadratic input-dependent

print len(features)


# # Building compatibility function 
# 
# If $\bf{y} \subseteq \{0, 1\}\times \{0, 1\} ... \{0, 1\}$, the compatibility function becomes a QUBO problem.
# 
# Here we show how to build a QUBO problem with the feature functions:

# In[ ]:


def build_q(w, features):
    """
    Using the given weight vector generate a QUBO that will assign labels to given features when the QUBO is solved.

    We apply the weights (1, F) to the feature vector (1, F).

    Then taking the dot product with the qubo matrix (L^2, F), leaving us with an L^2 long vector.
    We reshape then L^2 vector to a (L, L) matrix, which is our qubo.

    :param w: Weights for all the possible contributions to the QUBO
    :param features: Feature vector describing object to be classified.
    :return: QUBO to solve to classify x
    """

    temp = np.multiply(w, features)
    temp = self.to_qubo.dot(temp.T)
    return temp.reshape((self.label_count, self.label_count))


# ## Connecting to the QPU
# 
# Here we set up a connection to the D-Wave system.  For a discussion of the different parameters the QPU accepts, see the [Developer Guide for Python](https://cloud.dwavesys.com/qubist/docs/dev_guide_python/) (login required).
# 
# We will load the configuration information from file. You will need to create this file, in the same directory containing this notebook, to continue running it. Here is an example:
# 
# ```json
# {
#     "L": 4,
#     "M": 16,
#     "N": 16,
#     "params": {
#         "answer_mode": "histogram",
#         "auto_scale": true,
#         "num_reads": 500
#     },
#     "use_subdivision": true,
#     "solver": "mock_qp_C12_subgraph",
#     "token": "",
#     "url": ""
# }
# ```
# 
# Note that the token and url fields are empty, and need to be filled.

# In[ ]:


import json
from dwave_sp.tools.utils import unicode_to_ascii

with open('sp_configs_scene_example.json') as config_file:
    # load the data from the json file. Since the json module loads all strings as unicode strings, we'll convert 
    # the result to ascii using the object_hook.
    hw_config = json.load(config_file, object_hook=unicode_to_ascii)

# define SAPI-related stuff, we will overwrite the ones defined in the JSON file
hw_config['url'] = 'https://cloud.dwavesys.com/sapi'
hw_config['token'] = ''
hw_config['solver'] = ''


# ## Embedding the Problem
# 
# As per the [Structured Prediction Reference Example](./documentation/index.html) document, we will be minimizing the regularized objective function
# 
# $$
# F(w)=\frac{1}{d}\sum_{i=1}^{d}\left[\underset{y\in\{0,1\}^{L}}{max}\left\{\Delta(y,y_{i})+\langle w,\Psi(x_{i},y)-\Psi(x_{i},y_{i})\rangle\right\}\right]+\lambda\frac{||w||^{2}}{2}
# $$
# 
# with subgradient,
# 
# $$
# \partial_{w}F(w)=\frac{1}{d}\sum_{i=1}^{d}\left[\Psi(x_{i},\hat{y})-\Psi(x_{i},y_{i})\right]+\lambda w
# $$
# 
# where $\hat{y}_i$ is the maximizer for $\underset{\mathbf{y}\in\{0,1\}^{L}}{max}\left\{\Delta(\mathbf{y},\mathbf{y_{i}})+\langle\mathbf{w},\mathbf{\Psi}(\mathbf{x_{i}},\mathbf{y})-\mathbf{\Psi}(\mathbf{x_{i}},\mathbf{y_{i}})\rangle\right\}$
# 
# This gradient can be written as a QUBO and submitted to the D-Wave QPU.
# 
# As described in the [Developer Guide for Python](https://cloud.dwavesys.com/qubist/docs/dev_guide_python/intro/), the D-Wave system's QPU solves a class of Ising/QUBO problems, whose primal graph is minor embeddable into a particular graph called Chimera. Therefore, to submit a problem to the D-Wave QC, we must find such a minor embedding.
# 
# Here are some methods to help with this.

# ### Sub-dividing the Chimera graph
# 
# It is possible to divide the Chimera graph representing a particular solver into several subgraphs, and submit multiple QUBOs at the same time, speeding up the learning iteration.

# # Defining the QUBO solver

# In[ ]:


from dwave_sp import qubo_solvers
# Here we find as many complete graphs (cliques) on 6 vertices as possible. 
# Each clique will represent a 'sub-solver'

embedding = qubo_solvers.find_multiembeddings.simple_find_k6(hw_config)
print(len(embedding))


# This call finds `len(embedding)` subgraphs on the QPU that can accommodate our problem. We can therefore send that many learning examples to the QPU simultaneously.

# In[ ]:


# We now define the solver.
solver = qubo_solvers.MultiEmbeddingSolver(config=hw_config,
                                           variables=number_of_labels,
                                           problem_bundle=len(embedding),
                                           embedding=embedding,
                                           workers=len(embedding))


# Since we are using the QPU to solve an optimization problem, we set its preprocessing mode to `optimization`.

# In[ ]:


solver.set_postprocessing("optimization")


# # Defining the classifier
# 
# The `solver` object will be used to define a classifier object.

# In[ ]:


from dwave_sp.classifier import Classifier

classifier = Classifier(solver=solver,
                        features=features,
                        training_data=train_data,
                        training_labels=train_labels,
                        verbose=True)


# # Defining the optimizer: SGD
# 
# The vector $\vec{w}$ in the objective function is called a *weight* vector, and it is this vector that we wish to find. This is where supervised learning comes in. Given a set of training examples $(x_1,y_1), ... (x_m, y_m)$ for which the mapping is known, training $\phi$ means adjusting the weights in $\vec{w}$ so that $\phi(\vec{w}^T\vec{x_i}) = y_i$ for all $1\leq i\leq m$. 
# 
# Observe that it may not always be possible to do this for **all** training examples. Instead, we define a *cost* function $C(\phi(z_i), y_i)$ which measures how far we are from the correct answer, and minimize the accumulated cost over all training examples. Much of machine learning is focused on defining cost functions that provide a good measure of accuracy and can be optimized with relative ease.
# 
# Concretely, the learning algorithm behaves as follows:
#     
# 1. Initialize $\vec{w}$ to zero or small random numbers
# 2. For each training example $(\vec{x}_i,y_i)$,
#     1. compute $\hat{y_i}=\phi(\vec{w}^T\cdot\vec{x_i})$
#     1. for each weight $w_j$ in $\vec{w}$
#         1. calculate $\Delta w_j=\eta \cdot (y_i - \hat{y_i})x_j$
#         1. set the new weight $w_j \longleftarrow w_j + \Delta w_j$
#         
# Here, $\eta$ is known as the learning rate, and affects how quickly we get to $C$'s global minimum.
# 
# Next, let's define an SGD optimizer:

# In[ ]:


from dwave_sp.sgd import stochastic_gradient_descent as sgd
import math
steps = 4
sgd_solver = sgd(steps, lambda iteration: max(1 / math.sqrt(iteration + 1), 0.01))


# # Putting everything together to define an SSVM
# 
# 
# <img src='images/linear-discriminant.jpg'>
# 
# Structured SVM (SSVM) is an extension of SVM. Because the objective function $F(w)$ in SSVM is not differentiable (due to the fact that the `max` term inside the the function may not be unique), the SGD algorithm we mentioned previously should be called the subgradient descent (a generalization of gradients for convex, but not necessarily smooth objectives).
# 
# The subgradient of $F(w)$ w.r.t $w$ is given by:
# $$
#  \partial_w F(\bf{w}) = \partial_w R(\bf{w}) + \partial_w \big(  \lambda 
# \frac{||\bf{w}||^2}{2} \big)= \frac{1}{d}\ \sum_{i=1}^d \big[ {\bf{\Psi}(\bf{x}_i,\bf{\hat 
# y}_i)- 
# \bf{\Psi}(\bf{x}_i,\bf{y}_i)}\big] + \lambda \bf{w}\ ,
# $$
# where for each $i=1,\ldots,d$,
# $$
#  \bf{\hat y}_i = \arg\max_{\bf{y}\in\{0,1\}^L} \ \Delta(\bf{y},\bf{y}_i)+ 
# <\bf{w}, \bf{\Psi}(\bf{x}_i,\bf{y})- \bf{\Psi}(\bf{x}_i,\bf{y}_i)>
# $$
# which can be found by solving a QUBO.

# In[ ]:


from dwave_sp.ssvm import SSVM

trainer = SSVM(classifier, sgd_solver, batch_size=100, regularization_strength=0.005, weights_checked=2)


# # Starting training
# 
# <div class="alert alert-warning" role="alert" style="margin: 10px"> 
# __CAUTION__: The following code will run on the D-Wave quantum computer. It may take a long time under heavy usage. 
# </div>

# In[ ]:


# Uncommment the following line to run on the D-Wave quantum computer

# trainer.fit()


# # Plotting objective functions

# In[ ]:


plt.plot(classifier.obj_vals[1:])
plt.xlabel('Iterations')
plt.ylabel('Objective')


# # Loading trained model

# In[ ]:


classifier.w = np.load('classifier_w.npy')


# # Testing training performance
# Let's compare the result of training and testing. First, it predicts data labels using trained model. Then, it measures the error percentages of incorrect labels by counting every label individually. The `measure` function allows for partially correct labeling of a data vector and it is opposed by `score` function which requires all labels to be correct for a data vector.

# In[ ]:


# The following run may take some time on the QPU
classifier.measure(train_data, train_labels)


# In[ ]:


# The following run may take some time on the QPU
classifier.measure(test_data, test_labels)


# To compare the results, use independent SVMs to train model using different regularizer values

# In[ ]:


from dwave_sp.predict_linear import *
predict_linear(train_data, train_labels, test_data, test_labels, number_of_labels, penalty_parameter=0.1)
predict_linear(train_data, train_labels, test_data, test_labels, number_of_labels, penalty_parameter=1)
predict_linear(train_data, train_labels, test_data, test_labels, number_of_labels, penalty_parameter=10)


# Now we have finished training an SSVM model for structured prediction. 
# 
# # Documentation
# See the [documentation](./documentation/index.html) for more information.

# In[ ]:


# shutdown the kernel
get_ipython().run_line_magic('reset', '-f')

