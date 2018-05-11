#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
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
        if object is None:
            self._components = np.empty(self._SIZE)
        elif isinstance(object, self.__class__):
            self._components = object._components.copy()
        elif isinstance(object, np.ndarray):
            if object.shape != (self._SIZE,):
                raise ValueError
            self._components = np.array(object, copy=copy)
        elif isinstance(object, (list, tuple)):
            if len(object) != self._SIZE:
                raise ValueError
            self._components = np.array(object)
        else:
            raise TypeError

    def __add__(self, other):
        return self.__class__(self._components + other._components, copy=False)

    def __iadd__(self, other):
        self._components += other._components
        return self

    def __sub__(self, other):
        return self.__class__(self._components - other._components, copy=False)

    def __isub__(self, other):
        self._components -= other._components
        return self

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._multiplication_for_ndarray(self._components, other._components), copy=False)
        else:
            return self.__class__(self._components * other, copy=False)

    def __imul__(self, other):
        if isinstance(other, self.__class__):
            self._components[:] = self._multiplication_for_ndarray(self._components, other._components)
        else:
            self._components *= other
        return self

    def __rmul__(self, other):
        return self.__class__(self._components * other, copy=False)

    def __truediv__(self, other):
        return self.__class__(self._components / other, copy=False)

    def __itruediv__(self, other):
        self._components[:] = self._components / other
        return self

    def __pos__(self):
        return self.__class__(+self._components, copy=False)

    def __neg__(self):
        return self.__class__(-self._components, copy=False)

    def __str__(self):
        return np.array_str(self._components)

    @property
    def components(self):
        '''Returns the four components of the Quaternion.'''
        return self._components

    @components.setter
    def components(self, c):
        '''Sets the four components of the Quaternion.'''
        self._components[:] = c

    @property
    def parts(self):
        '''Returns the scalar and vector part of the Quaternion.'''
        return self._components[self._SCALAR_PART], self._components[self._VECTOR_PART]

    @property
    def scalar_part(self):
        '''Returns the scalar part of the Quaternion.'''
        return self._components[self._SCALAR_PART]

    @scalar_part.setter
    def scalar_part(self, scalar):
        '''Sets the scalar part of the Quaternion.'''
        self._components[self._SCALAR_PART] = scalar

    @property
    def vector_part(self):
        '''Returns the vector part of the Quaternion.'''
        return self._components[self._VECTOR_PART]

    @vector_part.setter
    def vector_part(self, vector):
        '''Sets the vector part of the Quaternion.'''
        self._components[self._VECTOR_PART] = vector

    def is_zero(self, *, atol=1e-08):
        '''Check if the Quaternion is zero.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the quaternion is zero, False otherwise.
        '''
        return np.allclose(0., self._components, rtol=0., atol=atol)

    def is_identity(self, *, atol=1e-08):
        '''Check if the Quaternion is identity.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the Quaternion is identity, False otherwise.
        '''
        s, v = self.parts
        return np.isclose(1., s, rtol=0., atol=atol) and np.allclose(0., v, rtol=0., atol=atol)

    def is_unit(self, *, atol=1e-08):
        '''Check if the Quaternion is unit.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the norm of the Quaternion is 1, False otherwise.
        '''
        return np.isclose(1., self.norm(), rtol=0., atol=atol)

    def is_real(self, *, atol=1e-08):
        '''Check if the quaternion is purely real.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the vector part of the Quaternion is zero, False otherwise.
        '''
        _, v = self.parts
        return np.allclose(0., v, rtol=0., atol=atol)

    def is_pure(self, *, atol=1e-08):
        '''Check if the Quaternion is purely imaginary.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            True if the scalar part of the Quaternion is zero, False otherwise.
        '''
        s, _ = self.parts
        return np.isclose(0., s, rtol=0., atol=atol)

    def norm_squared(self):
        '''Returns the squared norm of the Quaternion.'''
        return np.dot(self._components, self._components)

    def norm(self):
        '''Returns the norm of the Quaternion.'''
        return np.linalg.norm(self._components)

    def to_axis_angle(self):
        '''Returns the axis vector and the angle (in radians) of the rotation represented by the Quaternion.'''
        s, v = self.parts
        n = np.linalg.norm(v)
        if np.isclose(0., n, rtol=0., atol=1e-09):
            return np.array([1., 0., 0.]), 0.

        rcp_n = 1. / n
        return v * rcp_n, 2. * math.atan2(n, s)

    def to_rotation_matrix(self):
        '''Returns the 3 by 3 rotation matrix associated with the Quaternion.'''
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
        return np.allclose(q1._components, q2._components, rtol=rtol, atol=atol)

    @classmethod
    def are_same_rotation(cls, q1, q2, *, atol=1e-08):
        '''Check if both Quaternions are represent the same rotation.

        Args:
            q1:
            q2:
            atol: The absolute tolerance to use.

        Returns:
            True if both Quaternions are represent the same rotation, False otherwise.
        '''
        return np.isclose(1., abs(cls.dot(q1, q2)), rtol=0., atol=atol)

    @classmethod
    def identity(cls):
        '''Returns the identity Quaternion.'''
        return cls(cls._make_ndarray_by_parts(1., [0., 0., 0.]), copy=False)

    @classmethod
    def inverse(cls, q):
        '''Returns the inverse Quaternion.'''
        rcp_nsq = 1. / q.norm_squared()
        s, v = q.parts
        return cls(cls._make_ndarray_by_parts(s * rcp_nsq, -v * rcp_nsq), copy=False)

    @classmethod
    def conjugate(cls, q):
        '''Returns the conjugate Quaternion.'''
        s, v = q.parts
        return cls(cls._make_ndarray_by_parts(s, -v), copy=False)

    @classmethod
    def dot(cls, q1, q2):
        '''Returns the dot product of two Quaternions.'''
        return np.dot(q1._components, q2._components)

    @classmethod
    def ln(cls, q):
        '''Returns the logarithm of a Quaternion.'''
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
        '''Returns the logarithm of a unit Quaternion.'''
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
        '''Returns the exponential of a Quaternion.'''
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
        '''Returns the exponential of a purely imaginary Quaternion.'''
        _, v = q.parts
        vn = np.linalg.norm(v)
        if np.isclose(0., vn, rtol=0., atol=1e-07):
            sinc = np.sinc(vn)
        else:
            sinc = math.sin(vn) / vn

        return cls(cls._make_ndarray_by_parts(math.cos(vn), sinc * v), copy=False)

    @classmethod
    def pow(cls, q, exponent):
        '''Returns the power of a Quaternion specified by an "exponent" parameter.'''
        return cls.exp(cls.ln(q) * exponent)

    @classmethod
    def pow_u(cls, q, exponent):
        '''Returns the power of an unit Quaternion specified by an "exponent" parameter.'''
        return cls.exp_p(cls.ln_u(q) * exponent)

    @classmethod
    def normalize(cls, q):
        '''Returns the normalized version of a Quaternion.'''
        return q / q.norm()

    @classmethod
    def from_axis_angle(cls, axis, angle):
        '''Returns the Quaternion as a rotation of axis and angle.

        Args:
            axis: Need to be an unit vector.
            angle: Angle in radians.

        Returns:
            The resulting Quaternion.
        '''
        half_angle = angle * 0.5
        return cls(cls._make_ndarray_by_parts(math.cos(half_angle), math.sin(half_angle) * axis), copy=False)

    @classmethod
    def from_rotation_matrix(cls, matrix):
        '''Returns thr Quaternion from a 3 by 3 rotation matrix.'''
        assert matrix.shape == (3, 3), 'matrix must be a 3 by 3 matrix.'

        tr = 1. + np.trace(matrix)
        if tr >= 1.:
            # When the absolute value of w is maximum.
            w = math.sqrt(tr) * 0.5
            rcp_4w = 1. / (4. * w)  # 1/4|w|
            x = (matrix[1, 2] - matrix[2, 1]) * rcp_4w  # 4wx/4|w|
            y = (matrix[2, 0] - matrix[0, 2]) * rcp_4w  # 4wy/4|w|
            z = (matrix[0, 1] - matrix[1, 0]) * rcp_4w  # 4wz/4|w|
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            # When the absolute value of x is maximum.
            x = math.sqrt(matrix[0, 0] - matrix[1, 1] - matrix[2, 2] + 1.) * 0.5
            rcp_4x = 1. / (4. * x)  # 1/4|x|
            y = (matrix[1, 0] + matrix[0, 1]) * rcp_4x  # 4xy/4|x|
            z = (matrix[2, 0] + matrix[0, 2]) * rcp_4x  # 4xz/4|x|
            w = (matrix[1, 2] - matrix[2, 1]) * rcp_4x  # 4wx/4|x|
        elif (matrix[1, 1] > matrix[2, 2]):
            # When the absolute value of y is maximum.
            y = math.sqrt(matrix[1, 1] - matrix[2, 2] - matrix[0, 0] + 1.) * 0.5
            rcp_4y = 1. / (4. * y)  # 1/4|y|
            z = (matrix[1, 2] + matrix[2, 1]) * rcp_4y  # 4yz/4|y|
            w = (matrix[2, 0] - matrix[0, 2]) * rcp_4y  # 4wy/4|y|
            x = (matrix[1, 0] + matrix[0, 1]) * rcp_4y  # 4xy/4|y|
        else:
            # When the absolute value of z is maximum.
            z = math.sqrt(matrix[2, 2] - matrix[0, 0] - matrix[1, 1] + 1.) * 0.5
            rcp_4z = 1. / (4. * z)  # 1/4|z|
            x = (matrix[2, 0] + matrix[0, 2]) * rcp_4z  # 4xz/4|z|
            y = (matrix[1, 2] + matrix[2, 1]) * rcp_4z  # 4yz/4|z|
            w = (matrix[0, 1] - matrix[1, 0]) * rcp_4z  # 4wz/4|z|

        return cls(cls._make_ndarray_by_parts(w, [x, y, z]), copy=False)

    @classmethod
    def rotate(cls, q, v):
        '''Returns the image of a Vector by a Quaternion rotation.'''
        qv = cls(cls._make_ndarray_by_parts(0., v), copy=False)
        r = q * qv * cls.conjugate(q)
        return r.vector_part

    @classmethod
    def rotation_shortest_arc(cls, v1, v2):
        d = np.dot(v1, v2)
        s = math.sqrt((1. + d) * 2.)
        c = cls._cross_for_ndarray(v1, v2)
        return cls(cls._make_ndarray_by_parts(s * 0.5, c / s), copy=False)

    @classmethod
    def rotational_difference(cls, q1, q2):
        return cls.conjugate(q1) * q2

    @classmethod
    def lerp(cls, q1, q2, t):
        '''Returns the linear interpolation of Quaternions q1 and q2, at time t.

        Args:
            q1:
            q2:
            t: should range in [0,1].

        Returns:
            The resulting Quaternion is between q1 and q2.(result is q1 when t=0 and q2 for t=1)
        '''
        return cls(q1._components + t * (q2._components - q1._components), copy=False)

    @classmethod
    def slerp(cls, q1, q2, t, *, allow_flip=True):
        '''Returns the spherical linear interpolation of Quaternions q1 and q2, at time t.

        Args:
            q1:
            q2:
            t: should range in [0,1].
            allow_flip: True if interpolate along a shortest arc, False otherwise.

        Returns:
            The resulting Quaternion is between q1 and q2.(result is q1 when t=0 and q2 for t=1)
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
            p:
            q:
            a:
            b:
            t: should range in [0,1].

        Returns:
            The resulting Quaternion is between p and q.(result is p when t=0 and q for t=1)
        '''
        return cls.slerp(cls.slerp(p, q, t, allow_flip=False), cls.slerp(a, b, t, allow_flip=False), 2. * t * (1. - t), allow_flip=False)

    @classmethod
    def squad_tangent(cls, before, center, after):
        '''Tangent Quaternion for "center", defined by "before" and "after" Quaternions.

        Useful for smooth spline interpolation of Quaternion with :func:`squad() <quaternion.Quaternion.squad>` and :func:`slerp() <quaternion.Quaternion.slerp>`.

        Args:
            before:
            center:
            after:

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
