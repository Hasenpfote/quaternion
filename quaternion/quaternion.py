#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
import numpy as np


class Quaternion:
    '''The Quaternion class represents 3D rotations and orientations.

     Attributes:
         _components: the four components of the Quaternion.
    '''
    _SIZE = 4
    _SCALAR_PART = 0
    _VECTOR_PART = slice(1, 4)

    def __init__(self, object=None, *, copy=True):
        '''Initialize the Quaternion.

        Args:
            object: The object can be specified as follows.

                - Object like an array of length 4.

                - `None`.

                - A self class.
            copy: True if copy an object of ndarray, otherwise False.

        Examples:
            >>> q = Quaternion(numpy.array([1., 2., 3., 4.]))
            >>> q = Quaternion([1., 2., 3., 4.])
            >>> q = Quaternion((1., 2., 3., 4.))
            >>> q = Quaternion()
            >>> q = Quaternion(q)

            >>> arr = numpy.array([1., 2., 3., 4.])
            >>> q = Quaternion(arr, copy=True)
            >>> id(q._components) == id(arr) # False
            >>> q = Quaternion(arr, copy=False)
            >>> id(q._components) == id(arr) # True
        '''
        if object is None:
            self._components = np.empty(self._SIZE)
        elif isinstance(object, self.__class__):
            self._components = object._components.copy()
        elif isinstance(object, np.ndarray):
            if object.shape != (self._SIZE,):
                raise ValueError
            # Force the type to float64.
            self._components = np.array(object, dtype=np.float64, copy=copy)
        elif isinstance(object, (list, tuple)):
            if len(object) != self._SIZE:
                raise ValueError
            self._components = np.array(object)
        else:
            raise TypeError
        assert self._components.dtype == np.float64, 'dtype must be float64.'

    def __add__(self, other):
        '''Returns the Quaternion addition.

        Args:
            other: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = q + r
        '''
        return self.__class__(self._components + other._components, copy=False)

    def __iadd__(self, other):
        '''Add a Quaternion `other` to the Quaternion.

        Args:
            other: A Quaternion.

        Examples:
            >>> p += q
        '''
        self._components += other._components
        return self

    def __sub__(self, other):
        '''Returns the Quaternion subtraction.

        Args:
            other: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = q - r
        '''
        return self.__class__(self._components - other._components, copy=False)

    def __isub__(self, other):
        '''Subtract a Quaternion `other` from the Quaternion.

        Args:
            other: A Quaternion.

        Examples:
            >>> p -= q
        '''
        self._components -= other._components
        return self

    def __mul__(self, other):
        '''Returns the Quaternion multiplication or scalar multiplication.

        Args:
            other: A Quaternion or scalar.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = q * r
            >>> p = q * 2.
        '''
        if isinstance(other, self.__class__):
            return self.__class__(self._multiplication_for_ndarray(self._components, other._components), copy=False)
        else:
            return self.__class__(self._components * other, copy=False)

    def __imul__(self, other):
        '''Multiply the Quaternion by a Quaternion `other`.

        Args:
            other: A Quaternion.

        Examples:
            >>> p *= q
            >>> p *= 2.
        '''
        if isinstance(other, self.__class__):
            self._components[:] = self._multiplication_for_ndarray(self._components, other._components)
        else:
            self._components *= other
        return self

    def __rmul__(self, other):
        '''Returns the scalar multiplication.

        Args:
            other: A scalar.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = 2. * q
        '''
        return self.__class__(self._components * other, copy=False)

    def __truediv__(self, other):
        '''Returns the scalar division.

        Args:
            other: A scalar.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = q / 2.
        '''
        return self.__class__(self._components / other, copy=False)

    def __itruediv__(self, other):
        '''Divide the Quaternion by a scalar `other`.

        Args:
            other: A Quaternion.

        Examples:
            >>> q /= 2.
        '''
        self._components[:] = self._components / other
        return self

    def __pos__(self):
        '''Returns the positive of the Quaternion.

        Examples:
            >>> p = +q
        '''
        return self.__class__(+self._components, copy=False)

    def __neg__(self):
        '''Returns the negative of the Quaternion.

        Examples:
            >>> p = -q
        '''
        return self.__class__(-self._components, copy=False)

    def __repr__(self):
        '''Returns the components of the Quaternion as a string.

        Examples:
            >>> repr(q)
        '''
        return '{0}({1})'.format(
            self.__class__.__name__,
            self.__str__()
        )

    def __str__(self):
        '''Returns the components of the Quaternion as a string.

        Examples:
            >>> str(q)
        '''
        return '[{0:.8} {1:.8} {2:.8} {3:.8}]'.format(
            self._components[0],
            self._components[1],
            self._components[2],
            self._components[3]
        )

    @property
    def components(self):
        '''Returns the four components of the Quaternion.

        Examples:
            >>> c = q.components
        '''
        return self._components

    @components.setter
    def components(self, c):
        '''Sets the four components of the Quaternion.

        Args:
            c: The object like an array of length 4.

        Examples:
            >>> q.components = numpy.array([1., 2., 3., 4.])
            >>> q.components = [1., 2., 3., 4.]
            >>> q.components = (1., 2., 3., 4.)
        '''
        self._components[:] = c

    @property
    def parts(self):
        '''Returns the scalar and vector part of the Quaternion.

        Examples:
            >>> s, v = q.parts
        '''
        return self._components[self._SCALAR_PART], self._components[self._VECTOR_PART]

    @property
    def scalar_part(self):
        '''Returns the scalar part of the Quaternion.

        Examples:
            >>> s = q.scalar_part
        '''
        return self._components[self._SCALAR_PART]

    @scalar_part.setter
    def scalar_part(self, scalar):
        '''Sets the scalar part of the Quaternion.

        Args:
            scalar: A scalar value.

        Examples:
            >>> q.scalar_part = 2.
        '''
        self._components[self._SCALAR_PART] = scalar

    @property
    def vector_part(self):
        '''Returns the vector part of the Quaternion.

        Examples:
            >>> v = q.vector_part
        '''
        return self._components[self._VECTOR_PART]

    @vector_part.setter
    def vector_part(self, vector):
        '''Sets the vector part of the Quaternion.

        Args:
            vector: The object like an array of length 3.

        Examples:
            >>> q.vector_part = numpy.array([1., 2., 3.])
            >>> q.vector_part = [1., 2., 3.]
            >>> q.vector_part = (1., 2., 3.)
        '''
        self._components[self._VECTOR_PART] = vector

    def is_zero(self, *, atol=1e-08):
        '''Check if the Quaternion is zero.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the quaternion is zero, False otherwise.

        Examples:
            >>> q.is_zero()
        '''
        return np.allclose(0., self._components, rtol=0., atol=atol)

    def is_identity(self, *, atol=1e-08):
        '''Check if the Quaternion is identity.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the Quaternion is identity, False otherwise.

        Examples:
            >>> q.is_identity()
        '''
        s, v = self.parts
        return np.isclose(1., s, rtol=0., atol=atol) and np.allclose(0., v, rtol=0., atol=atol)

    def is_unit(self, *, atol=1e-08):
        '''Check if the Quaternion is unit.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the norm of the Quaternion is 1, False otherwise.

        Examples:
            >>> q.is_unit()
        '''
        return np.isclose(1., self.norm(), rtol=0., atol=atol)

    def is_real(self, *, atol=1e-08):
        '''Check if the quaternion is purely real.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the vector part of the Quaternion is zero, False otherwise.

        Examples:
            >>> q.is_real()
        '''
        _, v = self.parts
        return np.allclose(0., v, rtol=0., atol=atol)

    def is_pure(self, *, atol=1e-08):
        '''Check if the Quaternion is purely imaginary.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the scalar part of the Quaternion is zero, False otherwise.

        Examples:
            >>> q.is_pure()
        '''
        s, _ = self.parts
        return np.isclose(0., s, rtol=0., atol=atol)

    def norm_squared(self):
        '''Returns the squared norm of the Quaternion.

        Examples:
            >>> ns = q.norm_squared()
        '''
        return np.dot(self._components, self._components)

    def norm(self):
        '''Returns the norm of the Quaternion.

        Examples:
            >>> n = q.norm()
        '''
        return np.linalg.norm(self._components)

    def to_axis_angle(self):
        '''Returns the axis vector and the angle (in radians) of the rotation represented by the Quaternion.

        Examples:
            >>> axis, angle = q.to_axis_angle()
        '''
        s, v = self.parts
        n = np.linalg.norm(v)
        if np.isclose(0., n, rtol=0., atol=1e-09):
            return np.array([1., 0., 0.]), 0.

        rcp_n = 1. / n
        return v * rcp_n, 2. * math.atan2(n, s)

    def to_rotation_matrix(self):
        '''Returns the 3 by 3 rotation matrix associated with the Quaternion.

        Examples:
            >>> m = q.to_rotation_matrix()
        '''
        s, v = self.parts
        a = v * v
        b = v * np.take(v, (1, 2, 0))
        c = s * v

        return np.array([
            [1. - 2. * (a[1] + a[2]), 2. * (b[0] + c[2]), 2. * (b[2] - c[1])],
            [2. * (b[0] - c[2]), 1. - 2. * (a[0] + a[2]), 2. * (b[1] + c[0])],
            [2. * (b[2] + c[1]), 2. * (b[1] - c[0]), 1. - 2. * (a[0] + a[1])]
        ])

    @classmethod
    def equals(cls, q1, q2, *, rtol=1e-05, atol=1e-08):
        '''Whether two Quaternions are approximately equal.

        Args:
            q1: A Quaternion.
            q2: A Quaternion.
            rtol: The relative tolerance to use.
            atol: The absolute tolerance to use.

        Returns:
            True if both Quaternions are approximately equal, False otherwise.

        Examples:
            >>> Quaternion.equals(p, q)
        '''
        return np.allclose(q1._components, q2._components, rtol=rtol, atol=atol)

    @classmethod
    def are_same_rotation(cls, q1, q2, *, atol=1e-08):
        '''Check if both Quaternions are represent the same rotation.

        Args:
            q1: An unit Quaternion.
            q2: An unit Quaternion.
            atol: The absolute tolerance to use.

        Returns:
            True if both Quaternions are represent the same rotation, False otherwise.

        Examples:
            >>> Quaternion.are_same_rotation(p, q)
        '''
        return np.isclose(1., abs(cls.dot(q1, q2)), rtol=0., atol=atol)

    @classmethod
    def identity(cls):
        '''Returns the identity Quaternion.

        Examples:
            >>> q = Quaternion.identity()
        '''
        return cls(cls._make_ndarray_by_parts(1., [0., 0., 0.]), copy=False)

    @classmethod
    def inverse(cls, q):
        '''Returns the inverse Quaternion.

        Args:
            q: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.inverse(q)
        '''
        rcp_nsq = 1. / q.norm_squared()
        s, v = q.parts
        return cls(cls._make_ndarray_by_parts(s * rcp_nsq, -v * rcp_nsq), copy=False)

    @classmethod
    def conjugate(cls, q):
        '''Returns the conjugate Quaternion.

        Args:
            q: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.conjugate(q)
        '''
        s, v = q.parts
        return cls(cls._make_ndarray_by_parts(s, -v), copy=False)

    @classmethod
    def dot(cls, q1, q2):
        '''Returns the dot product of two Quaternions.

        Args:
            q1: A Quaternion.
            q2: A Quaternion.

        Returns:
            The resulting scalar value.

        Examples:
            >>> Quaternion.dot(p, q)
        '''
        return np.dot(q1._components, q2._components)

    @classmethod
    def ln(cls, q):
        '''Returns the logarithm of a Quaternion.

        Args:
            q: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.ln(q)
        '''
        s, v = q.parts
        vn = np.linalg.norm(v)
        phi = math.atan2(vn, s)
        n = q.norm()
        if np.isclose(0., vn, rtol=0., atol=1e-06):
            coef = 1. / (np.sinc(phi) * n)
        else:
            coef = phi / vn

        return cls(cls._make_ndarray_by_parts(math.log(n), coef * v), copy=False)

    @classmethod
    def ln_u(cls, q):
        '''Returns the logarithm of a unit Quaternion.

        Args:
            q: A unit Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.ln_u(q)
        '''
        s, v = q.parts
        vn = np.linalg.norm(v)
        phi = math.atan2(vn, s)
        n = q.norm()
        if np.isclose(0., vn, rtol=0., atol=1e-06):
            rcp_sinc = 1. / np.sinc(phi)
        else:
            rcp_sinc = phi * n / vn

        return cls(cls._make_ndarray_by_parts(0., rcp_sinc * v), copy=False)

    @classmethod
    def exp(cls, q):
        '''Returns the exponential of a Quaternion.

        Args:
            q: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.exp(q)
        '''
        s, v = q.parts
        vn = np.linalg.norm(v)
        if np.isclose(0., vn, rtol=0., atol=1e-07):
            sinc = np.sinc(vn)
        else:
            sinc = math.sin(vn) / vn

        exp_s = math.exp(s)
        return cls(cls._make_ndarray_by_parts(exp_s * math.cos(vn), exp_s * sinc * v), copy=False)

    @classmethod
    def exp_p(cls, q):
        '''Returns the exponential of a purely imaginary Quaternion.

        Args:
            q: A purely imaginary Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.exp_p(q)
        '''
        _, v = q.parts
        vn = np.linalg.norm(v)
        if np.isclose(0., vn, rtol=0., atol=1e-07):
            sinc = np.sinc(vn)
        else:
            sinc = math.sin(vn) / vn

        return cls(cls._make_ndarray_by_parts(math.cos(vn), sinc * v), copy=False)

    @classmethod
    def pow(cls, q, exponent):
        '''Returns the power of a Quaternion specified by an "exponent" parameter.

        Args:
            q: A Quaternion.
            exponent: An exponent value.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.pow(q, 2.)
        '''
        return cls.exp(cls.ln(q) * exponent)

    @classmethod
    def pow_u(cls, q, exponent):
        '''Returns the power of an unit Quaternion specified by an "exponent" parameter.

        Args:
            q: An unit Quaternion.
            exponent: an exponent value.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.pow_u(q, 2.)
        '''
        return cls.exp_p(cls.ln_u(q) * exponent)

    @classmethod
    def normalize(cls, q):
        '''Returns the normalized version of a Quaternion.

        Args:
            q: A Quaternion.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> p = Quaternion.normalize(q)
        '''
        return q / q.norm()

    @classmethod
    def from_axis_angle(cls, axis, angle):
        '''Returns the Quaternion as a rotation of axis and angle.

        Args:
            axis: Need to be an unit vector.
            angle: Angle in radians.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> axis = numpy.array([1., 0., 0.]) # or axis = [1., 0., 0.] or axis = (1., 0., 0.)
            >>> angle = math.radians(60.)
            >>> q = Quaternion.from_axis_angle(axis, angle)
        '''
        _axis = np.array(axis, dtype=np.float64, copy=False) # Force the type to float64.
        half_angle = angle * 0.5
        return cls(cls._make_ndarray_by_parts(math.cos(half_angle), math.sin(half_angle) * _axis), copy=False)

    @classmethod
    def from_rotation_matrix(cls, matrix):
        '''Returns the Quaternion from a 3 by 3 rotation matrix.

        Args:
            matrix: A 3 by 3 rotation matrix.

        Returns:
            The resulting Quaternion.

        Examples:
            >>> m = numpy.identity(3)
            >>> q = Quaternion.from_rotation_matrix(m)
        '''
        _matrix = np.array(matrix, dtype=np.float64, copy=False) # Force the type to float64.
        assert _matrix.shape == (3, 3), 'matrix must be a 3 by 3 matrix.'

        tr = 1. + np.trace(_matrix)
        if tr >= 1.:
            # When the absolute value of w is maximum.
            w = math.sqrt(tr) * 0.5
            rcp_4w = 1. / (4. * w)  # 1/4|w|
            x = (_matrix[1, 2] - _matrix[2, 1]) * rcp_4w  # 4wx/4|w|
            y = (_matrix[2, 0] - _matrix[0, 2]) * rcp_4w  # 4wy/4|w|
            z = (_matrix[0, 1] - _matrix[1, 0]) * rcp_4w  # 4wz/4|w|
        elif (_matrix[0, 0] > _matrix[1, 1]) and (_matrix[0, 0] > _matrix[2, 2]):
            # When the absolute value of x is maximum.
            x = math.sqrt(_matrix[0, 0] - _matrix[1, 1] - _matrix[2, 2] + 1.) * 0.5
            rcp_4x = 1. / (4. * x)  # 1/4|x|
            y = (_matrix[1, 0] + _matrix[0, 1]) * rcp_4x  # 4xy/4|x|
            z = (_matrix[2, 0] + _matrix[0, 2]) * rcp_4x  # 4xz/4|x|
            w = (_matrix[1, 2] - _matrix[2, 1]) * rcp_4x  # 4wx/4|x|
        elif (_matrix[1, 1] > _matrix[2, 2]):
            # When the absolute value of y is maximum.
            y = math.sqrt(_matrix[1, 1] - _matrix[2, 2] - _matrix[0, 0] + 1.) * 0.5
            rcp_4y = 1. / (4. * y)  # 1/4|y|
            z = (_matrix[1, 2] + _matrix[2, 1]) * rcp_4y  # 4yz/4|y|
            w = (_matrix[2, 0] - _matrix[0, 2]) * rcp_4y  # 4wy/4|y|
            x = (_matrix[1, 0] + _matrix[0, 1]) * rcp_4y  # 4xy/4|y|
        else:
            # When the absolute value of z is maximum.
            z = math.sqrt(_matrix[2, 2] - _matrix[0, 0] - _matrix[1, 1] + 1.) * 0.5
            rcp_4z = 1. / (4. * z)  # 1/4|z|
            x = (_matrix[2, 0] + _matrix[0, 2]) * rcp_4z  # 4xz/4|z|
            y = (_matrix[1, 2] + _matrix[2, 1]) * rcp_4z  # 4yz/4|z|
            w = (_matrix[0, 1] - _matrix[1, 0]) * rcp_4z  # 4wz/4|z|

        return cls(cls._make_ndarray_by_parts(w, [x, y, z]), copy=False)

    @classmethod
    def rotate(cls, q, v):
        '''Returns the image of a vector by a Quaternion rotation.

        Args:
            q: A Quaternion representing rotation.
            v: The object like an array of length 3.

        Returns:
            The resulting vector.

        Examples:
            >>> v = numpy.array([1., 2., 3.]) # or v = [1., 2., 3.] or v = (1., 2., 3.)
            >>> v = Quaternion.rotate(q, v)
        '''
        qv = cls(cls._make_ndarray_by_parts(0., v), copy=False)
        r = q * qv * cls.conjugate(q)
        return r.vector_part

    @classmethod
    def rotation_shortest_arc(cls, v1, v2):
        # Force the type to float64.
        _v1 = np.array(v1, dtype=np.float64, copy=False)
        _v2 = np.array(v2, dtype=np.float64, copy=False)
        d = np.dot(_v1, _v2)
        s = math.sqrt((1. + d) * 2.)
        c = cls._cross_for_ndarray(_v1, _v2)
        return cls(cls._make_ndarray_by_parts(s * 0.5, c / s), copy=False)

    @classmethod
    def rotational_difference(cls, q1, q2):
        return cls.conjugate(q1) * q2

    @classmethod
    def random(cls):
        '''Returns a random unit Quaternion.

        Returns:
            The resulting Quaternion.

        See Also:
            http://planning.cs.uiuc.edu/node198.html
        '''
        u1 = random.uniform(0., 1.)
        r1 = math.sqrt(1. - u1)
        r2 = math.sqrt(u1)
        t1 = 2. * math.pi * random.uniform(0., 1.) # u2
        t2 = 2. * math.pi * random.uniform(0., 1.) # u3
        return cls(cls._make_ndarray_by_parts(r2 * math.cos(t2),
                                              [r1 * math.sin(t1),
                                               r1 * math.cos(t1),
                                               r2 * math.sin(t2)]), copy=False)

    @classmethod
    def lerp(cls, q1, q2, t):
        '''Returns the linear interpolation of Quaternions q1 and q2, at time t.

        Args:
            q1: A Quaternion.
            q2: A Quaternion.
            t: should range in [0,1].

        Returns:
            The resulting Quaternion is between q1 and q2.(result is q1 when t=0 and q2 for t=1)

        Examples:
            >>> p = Quaternion.lerp(q, r, t)
        '''
        return cls(q1._components + t * (q2._components - q1._components), copy=False)

    @classmethod
    def slerp(cls, q1, q2, t, *, allow_flip=True):
        '''Returns the spherical linear interpolation of Quaternions q1 and q2, at time t.

        Args:
            q1: An unit Quaternion.
            q2: An unit Quaternion.
            t: should range in [0,1].
            allow_flip: True if interpolate along a shortest arc, False otherwise.

        Returns:
            The resulting Quaternion is between q1 and q2.(result is q1 when t=0 and q2 for t=1)

        Examples:
            >>> p = Quaternion.slerp(q, r, t)
        '''
        flipped = False
        cos_t = cls.dot(q1, q2)
        if allow_flip and (cos_t < 0.):
            # interpolate along the shortest arc.
            flipped = True
            cos_t = -cos_t

        if np.isclose(1., abs(cos_t), rtol=0., atol=1e-12):
            coef1 = 1. - t
            coef2 = t
        else:
            theta = math.acos(cos_t)
            cosec = 1. / math.sin(theta)
            coef1 = math.sin(theta * (1. - t)) * cosec
            coef2 = math.sin(theta * t) * cosec

        if flipped:
            coef2 = -coef2

        return cls(coef1 * q1._components + coef2 * q2._components, copy=False)

    @classmethod
    def squad(cls, p, q, a, b, t):
        '''Returns the spherical quadrangle interpolation of the two Quaternions p and q, at time t, using tangents a and b.

        Use :func:`squad_tangent() <quaternion.Quaternion.squad_tangent>` to define the Quaternion tangents a and b.

        Args:
            p: An unit Quaternion.
            q: An unit Quaternion.
            a: A tangent Quaternion.
            b: A tangent Quaternion.
            t: should range in [0,1].

        Returns:
            The resulting Quaternion is between p and q.(result is p when t=0 and q for t=1)

        Examples:
            >>> q_arr = [q0, ..., qn, ..., qN]
            >>> a = squad_tangent(q_arr[n-1], q_arr[n], q_arr[n+1])
            >>> b = squad_tangent(q_arr[n], q_arr[n+1], q_arr[n+2])
            >>> q = squad(q_arr[n], q_arr[n+1], a, b, t)
        '''
        return cls.slerp(cls.slerp(p, q, t, allow_flip=False), cls.slerp(a, b, t, allow_flip=False), 2. * t * (1. - t), allow_flip=False)

    @classmethod
    def squad_tangent(cls, before, center, after):
        '''Tangent Quaternion for "center", defined by "before" and "after" Quaternions.

        Useful for smooth spline interpolation of Quaternion with :func:`squad() <quaternion.Quaternion.squad>` and :func:`slerp() <quaternion.Quaternion.slerp>`.

        Args:
            before: An unit Quaternion.
            center: An unit Quaternion.
            after: An unit Quaternion.

        Returns:
            The resulting tangent Quaternion.
        '''
        conj_cur = cls.conjugate(center)
        return center * cls.exp_p(-0.25 * (cls.ln_u(conj_cur * before) + cls.ln_u(conj_cur * after)))

    @classmethod
    def _make_ndarray_by_parts(cls, scalar_part, vector_part):
        c = np.empty(cls._SIZE)
        c[cls._SCALAR_PART] = scalar_part
        c[cls._VECTOR_PART] = vector_part
        return c

    @classmethod
    def _multiplication_for_ndarray(cls, q1, q2):
        s1, v1 = q1[cls._SCALAR_PART], q1[cls._VECTOR_PART]
        s2, v2 = q2[cls._SCALAR_PART], q2[cls._VECTOR_PART]
        return cls._make_ndarray_by_parts(s1 * s2 - np.dot(v1, v2), s1 * v2 + s2 * v1 + cls._cross_for_ndarray(v1, v2))

    @classmethod
    def _cross_for_ndarray(cls, v1, v2):
        c = np.empty(3)
        c[0] = v1[1] * v2[2] - v1[2] * v2[1]
        c[1] = v1[2] * v2[0] - v1[0] * v2[2]
        c[2] = v1[0] * v2[1] - v1[1] * v2[0]
        return c
