GreenLearning
=============

GreenLearning is a deep learning library based on Tensorflow for learning `Green's functions <https://en.wikipedia.org/wiki/Green%27s_function>`_ associated with `partial differential operators <https://en.wikipedia.org/wiki/Differential_operator>`_.

|laplace|

.. centered:: *Exact and learned Green's function of the Laplace operator.*

.. |laplace| image:: images/laplace.png
   :width: 100%

The library is maintained by `Nicolas Boullé <https://people.maths.ox.ac.uk/boulle/>`_. If you are interested in using it, do not hesitate to get in contact at ``boulle@maths.ox.ac.uk``.

Features
--------

- GreenLearning learns Green's functions and homogeneous solutions associated with scalar and systems of linearized partial differential equations in 1D and 2D with deep learning.
- `Rational neural networks <https://proceedings.neurips.cc/paper/2020/file/a3f390d88e4c41f2747bfa2f1b5f87db-Paper.pdf>`_ are implemented and used to increase the accuracy of the learned Green's functions.
- GreenLearning requires no hyperparameter tuning to successfully learn Green's functions.
- The neural networks can be created and trained easily with a few lines of code.
- It is simple to generate the training datasets with MATLAB scripts.

Guide
-----

See the following sections to learn how to install and use the GreenLearning library.

.. toctree::
  :maxdepth: 2

  guide/installation
  guide/guide
  guide/gallery
  guide/citation

Code Documentation
------------------

This section contains the documentation of the classes and functions in the package.

.. toctree::
  :maxdepth: 2

  modules/modules
