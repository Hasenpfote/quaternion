|License| |Build Status|

Quaternion
==========

About
-----

This package provides a class for manipulating quaternion objects.

Compatibility
-------------

-  Python 3.x

Installation
------------

::

   python setup.py install

Usage
-----

::

   >>> from quaternion import Quaternion
   >>> q1 = Quaternion([1., 2., 3., 4.])
   >>> q2 = Quaternion([5., 6., 7., 8.])
   >>> q1 * q2
   Quaternion([-60.  12.  30.  24.])

Please refer to `the
reference <https://hasenpfote.github.io/quaternion/>`__ for the details.

License
-------

This software is released under the MIT License, see LICENSE.

.. |License| image:: https://img.shields.io/badge/license-MIT-brightgreen.svg
   :target: LICENSE
.. |Build Status| image:: https://travis-ci.com/Hasenpfote/quaternion.svg?branch=master
   :target: https://travis-ci.com/Hasenpfote/quaternion
