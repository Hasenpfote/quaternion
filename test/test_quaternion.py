#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from unittest import TestCase
import math
import numpy as np
from ..quaternion import Quaternion


class TestQuaternion(TestCase):

    @classmethod
    def are_same_rotation(cls, q1, q2, *, atol=1e-08):
        return np.isclose(1., abs(np.dot(q1, q2)), rtol=0., atol=atol)

    def test_create_from_self(self):
        expected = Quaternion([1., 2., 3., 4.])
        actual = Quaternion(expected)
        self.assertTrue(np.allclose(actual._components, expected._components))
        self.assertTrue(id(actual._components) != id(expected._components))

    def test_create_from_ndarray(self):
        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion(expected)
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) != id(expected))

    def test_create_from_invalid_ndarray(self):
        with self.assertRaises(ValueError):
            q = Quaternion(np.array([1., 2., 3., 4., 5]))

    def test_create_from_ndarray_without_copy(self):
        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion(expected, copy=False)
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == id(expected))

    def test_create_from_list(self):
        expected = [1., 2., 3., 4.]
        actual = Quaternion(expected)
        self.assertTrue(np.allclose(actual._components, expected))

    def test_create_from_invalid_list(self):
        with self.assertRaises(ValueError):
            q = Quaternion([1., 2., 3., 4., 5])

    def test_create_from_tuple(self):
        expected = (1., 2., 3., 4.)
        actual = Quaternion(expected)
        self.assertTrue(np.allclose(actual._components, expected))

    def test_create_from_invalid_tuple(self):
        with self.assertRaises(ValueError):
            q = Quaternion((1., 2., 3., 4., 5))

    def test_create_from_invalid_type(self):
        with self.assertRaises(TypeError):
            q = Quaternion('1., 2., 3., 4.')

    def test_add(self):
        expected = np.array([3., 5., 7., 9.])
        actual = Quaternion([1., 2., 3., 4.]) + Quaternion([2., 3., 4., 5.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_iadd(self):
        expected = np.array([3., 5., 7., 9.])
        actual = Quaternion([1., 2., 3., 4.])
        components_id = id(actual._components)
        actual += Quaternion([2., 3., 4., 5.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_sub(self):
        expected = np.array([1., 1., 1., 1.])
        actual = Quaternion([2., 3., 4., 5.]) - Quaternion([1., 2., 3., 4.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_isub(self):
        expected = np.array([1., 1., 1., 1.])
        actual = Quaternion([2., 3., 4., 5.])
        components_id = id(actual._components)
        actual -= Quaternion([1., 2., 3., 4.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_mul(self):
        expected = np.array([-36., 6., 12., 12.])
        actual = Quaternion([1., 2., 3., 4.]) * Quaternion([2., 3., 4., 5.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_mul_with_scalar(self):
        expected = np.array([2., 4., 6., 8.])
        actual = Quaternion([1., 2., 3., 4.]) * 2.
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_imul(self):
        expected = np.array([-36., 6., 12., 12.])
        actual = Quaternion([1., 2., 3., 4.])
        components_id = id(actual._components)
        actual *= Quaternion([2., 3., 4., 5.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_imul_with_scalar(self):
        expected = np.array([2., 4., 6., 8.])
        actual = Quaternion([1., 2., 3., 4.])
        components_id = id(actual._components)
        actual *= 2.
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_rmul(self):
        expected = np.array([2., 4., 6., 8.])
        actual = 2. * Quaternion([1., 2., 3., 4.])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_div(self):
        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion([2., 4., 6., 8.]) / 2.
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_idiv(self):
        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion([2., 4., 6., 8.])
        components_id = id(actual._components)
        actual /= 2.
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_pos(self):
        expected = np.array([-1., -2., -3., -4.])
        q = Quaternion([-1., -2., -3., -4.])
        actual = +q
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual.components) != id(q._components))

    def test_neg(self):
        expected = np.array([ 1.,  2.,  3.,  4.])
        q = Quaternion([-1., -2., -3., -4.])
        actual = -q
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))
        self.assertTrue(id(actual.components) != id(q._components))

    def test_str(self):
        q = Quaternion([1., 2., 3., 4.])
        s = q.__str__()
        self.assertTrue(isinstance(s, str))
        self.assertTrue(s == '[1. 2. 3. 4.]')

    def test_components_with_ndarray(self):
        actual = Quaternion()
        components_id = id(actual.components)
        expected = np.array([1., 2., 3., 4.])
        actual.components = expected
        self.assertTrue(isinstance(actual.components, np.ndarray))
        self.assertTrue(np.allclose(actual.components, expected))
        self.assertTrue(id(actual.components) == components_id)

    def test_components_with_invalid_ndarray(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.components = np.array([1., 2., 3., 4., 5.])

    def test_components_with_list(self):
        actual = Quaternion()
        components_id = id(actual.components)
        expected = [1., 2., 3., 4.]
        actual.components = expected
        self.assertTrue(isinstance(actual.components, np.ndarray))
        self.assertTrue(np.allclose(actual.components, expected))
        self.assertTrue(id(actual.components == components_id))

    def test_components_with_invalid_list(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.components = [1., 2., 3., 4., 5.]

    def test_components_with_tuple(self):
        actual = Quaternion()
        components_id = id(actual.components)
        expected = (1., 2., 3., 4.)
        actual.components = expected
        self.assertTrue(isinstance(actual.components, np.ndarray))
        self.assertTrue(np.allclose(actual.components, expected))
        self.assertTrue(id(actual.components) == components_id)

    def test_components_with_invalid_tuple(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.components = (1., 2., 3., 4., 5.)

    def test_components_with_invalid_type(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.components = '1., 2., 3., 4., 5.'

    def test_parts(self):
        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion([1., 2., 3., 4.])
        s, v = actual.parts
        self.assertTrue(isinstance(v, np.ndarray))
        self.assertTrue(np.isclose(s, expected[Quaternion._SCALAR_PART]))
        self.assertTrue(np.allclose(v, expected[Quaternion._VECTOR_PART]))

    def test_scalar_part(self):
        expected = 1.
        actual = Quaternion()
        components_id = id(actual._components)
        actual.scalar_part = expected
        self.assertTrue(np.isclose(actual.scalar_part, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_scalar_part_with_invalid_type(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.scalar_part = [1., 2.]

    def test_vector_part_with_ndarray(self):
        actual = Quaternion()
        components_id = id(actual._components)
        expected = np.array([2., 3., 4.])
        actual.vector_part = expected
        self.assertTrue(isinstance(actual.vector_part, np.ndarray))
        self.assertTrue(np.allclose(actual.vector_part, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_vector_part_with_invalid_ndarray(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.vector_part = np.array([1., 2.])

    def test_vector_part_with_list(self):
        actual = Quaternion()
        components_id = id(actual._components)
        expected = [2., 3., 4.]
        actual.vector_part = expected
        self.assertTrue(isinstance(actual.vector_part, np.ndarray))
        self.assertTrue(np.allclose(actual.vector_part, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_vector_part_with_invalid_list(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.vector_part = [1., 2.]

    def test_vector_part_with_tuple(self):
        actual = Quaternion()
        components_id = id(actual._components)
        expected = (2., 3., 4.)
        actual.vector_part = expected
        self.assertTrue(isinstance(actual.vector_part, np.ndarray))
        self.assertTrue(np.allclose(actual.vector_part, expected))
        self.assertTrue(id(actual._components) == components_id)

    def test_vector_part_with_invalid_tuple(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.vector_part = (1., 2.)

    def test_vector_part_with_invalid_type(self):
        with self.assertRaises(ValueError):
            actual = Quaternion()
            actual.vector_part = '1., 2., 3.'

    def test_is_zero(self):
        q = Quaternion([0., 0., 0., 0.])
        ret = q.is_zero()
        self.assertTrue(isinstance(ret, bool))
        self.assertTrue(ret)

        q = Quaternion([0., 0., 0., 1.])
        self.assertFalse(q.is_zero())

    def test_is_identity(self):
        q = Quaternion([1., 0., 0., 0.])
        ret = q.is_identity()
        self.assertTrue(isinstance(ret, bool))
        self.assertTrue(ret)

        q = Quaternion([0., 0., 0., 1.])
        self.assertFalse(q.is_identity())

    def test_is_unit(self):
        q = Quaternion([1., 0., 0., 0.])
        ret = q.is_unit()
        self.assertTrue(isinstance(ret, np.bool_))
        self.assertTrue(ret)

        q = Quaternion([1., 2., 3., 4.])
        self.assertFalse(q.is_unit())

    def test_is_real(self):
        q = Quaternion([1., 0., 0., 0.])
        ret = q.is_real()
        self.assertTrue(isinstance(ret, bool))
        self.assertTrue(ret)

        q = Quaternion([0., 2., 3., 4.])
        self.assertFalse(q.is_real())

    def test_is_pure(self):
        q = Quaternion([0., 1., 2., 3.])
        ret = q.is_pure()
        self.assertTrue(isinstance(ret, np.bool_))
        self.assertTrue(ret)

        q = Quaternion([1., 2., 3., 4.])
        self.assertFalse(q.is_pure())

    def test_norm_squared(self):
        expected = 4.
        actual = Quaternion([1., 1., 1., 1.]).norm_squared()
        self.assertTrue(np.isclose(actual, expected))

    def test_norm(self):
        expected = 2.
        actual = Quaternion([1., 1., 1., 1.]).norm()
        self.assertTrue(np.isclose(actual, expected))

    def test_to_axis_angle(self):
        expected = np.array([1., 0., 0.]), 1.047197551196598
        q = Quaternion([8.660254037844387e-001, 4.999999999999999e-001, 0., 0.])  # axis=[1,0,0] deg=60
        actual = q.to_axis_angle()
        self.assertTrue(isinstance(actual[0], np.ndarray))
        self.assertTrue(actual[0].shape == (3,))
        self.assertTrue(np.allclose(actual[0], expected[0]))
        self.assertTrue(np.isclose(actual[1], expected[1]))

    def test_to_axis_angle_with_identity(self):
        expected = np.array([1., 0., 0.]), 0.
        q = Quaternion([1., 0., 0., 0.])
        actual = q.to_axis_angle()
        self.assertTrue(isinstance(actual[0], np.ndarray))
        self.assertTrue(actual[0].shape == (3,))
        self.assertTrue(np.allclose(actual[0], expected[0]))
        self.assertTrue(np.isclose(actual[1], expected[1]))

    def test_to_rotation_matrix(self):
        angle = math.radians(60.)
        s, c = math.sin(angle), math.cos(angle)
        expected = np.identity(3)
        expected[1, 1:3] = c, s
        expected[2, 1:3] = -s, c
        q = Quaternion([0.8660254037844387, 0.5, 0., 0.])  # axis=[1,0,0] deg=60
        actual = q.to_rotation_matrix()
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.shape == (3, 3))
        self.assertTrue(np.allclose(actual, expected))

    def test_to_rotation_matrix_with_identity(self):
        expected = np.identity(3)
        q = Quaternion([1., 0., 0., 0.])
        actual = q.to_rotation_matrix()
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.shape == (3, 3))
        self.assertTrue(np.allclose(actual, expected))

    def test_equals(self):
        q1 = Quaternion([1., 2., 3., 4.])
        q2 = Quaternion([1., 2., 3., 4.])
        check = Quaternion.equals(q1, q2)
        self.assertTrue(isinstance(check, bool))
        self.assertTrue(check)

        q1 = Quaternion([1000.00, 2000.00, 3000.00, 4000.00])
        q2 = Quaternion([1000.01, 2000.01, 3000.01, 4000.01])
        check = Quaternion.equals(q1, q2)
        self.assertTrue(isinstance(check, bool))
        self.assertTrue(check)

        q2 = Quaternion([1000.10, 2000.10, 3000.10, 4000.10])
        check = Quaternion.equals(q1, q2)
        self.assertTrue(isinstance(check, bool))
        self.assertFalse(check)

    def test_are_same_rotation(self):
        q1 = Quaternion([ 1., 0., 0., 0.])
        q2 = Quaternion([-1., 0., 0., 0.]) # q2 = -q1
        check = Quaternion.are_same_rotation(q1, q2)
        self.assertTrue(isinstance(check, np.bool_))
        self.assertTrue(check)

        q1 = Quaternion([ 0.8660254037844387,  0.5, 0., 0.]) # axis=[ 1,0,0] deg=60
        q2 = Quaternion([-0.8660254037844387, -0.5, 0., 0.]) # axis=[-1,0,0] deg=300
        check = Quaternion.are_same_rotation(q1, q2)
        self.assertTrue(isinstance(check, np.bool_))
        self.assertTrue(check)

        q1 = Quaternion([0.9659258262890683, 0.25881904510252074, 0., 0.]) # axis=[1,0,0] deg=30
        q2 = Quaternion([0.7071067811865476, 0.7071067811865476, 0., 0.]) # axis=[1,0,0] deg=90
        check = Quaternion.are_same_rotation(q1, q2)
        self.assertTrue(isinstance(check, np.bool_))
        self.assertFalse(check)

    def test_identity(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.identity()
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_inverse(self):
        expected = Quaternion([1., 2., 3., 4.])
        actual = Quaternion.inverse(expected)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertFalse(np.allclose(actual._components, expected._components))
        actual = Quaternion.inverse(actual)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected._components))

    def test_conjugate(self):
        expected = np.array([1., -2., -3., -4.])
        actual = Quaternion.conjugate(Quaternion([1., 2., 3., 4.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_dot(self):
        expected = 4.
        q = Quaternion([1., 1., 1., 1.])
        actual = Quaternion.dot(q, q)
        self.assertTrue(np.isclose(actual, expected))

    def test_ln(self):
        expected = np.array([1.700598690831078, 5.151902926640850e-001, 7.727854389961275e-001, 1.030380585328170])
        actual = Quaternion.ln(Quaternion([1., 2., 3., 4.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_ln_with_identity(self):
        expected = np.array([0., 0., 0., 0.])
        actual = Quaternion.ln(Quaternion([1., 0., 0., 0.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_ln_u(self):
        expected = np.array([-1.110223024625157e-016, 5.151902926640851e-001, 7.727854389961275e-001, 1.030380585328170])
        q = Quaternion([1.825741858350554e-001, 3.651483716701107e-001, 5.477225575051661e-001, 7.302967433402214e-001]) # normalize([1,2,3,4])
        actual = Quaternion.ln_u(q)
        np.allclose(actual._components, expected)

    def test_ln_u_with_identity(self):
        expected = np.array([0., 0., 0., 0.])
        actual = Quaternion.ln_u(Quaternion([1., 0., 0., 0.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_exp(self):
        expected = np.array([1.693922723683299, -7.895596245415587e-001, -1.184339436812338, -1.579119249083117])
        actual = Quaternion.exp(Quaternion([1., 2., 3., 4.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_exp_with_zero(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.exp(Quaternion([0., 0., 0., 0.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_exp_p(self):
        expected = np.array([6.231593449762197e-001, -2.904627534478825e-001, -4.356941301718237e-001, -5.809255068957649e-001])
        q = Quaternion([0., 2., 3., 4.])
        actual = Quaternion.ln_u(q)
        np.allclose(actual._components, expected)

    def test_exp_p_with_zero(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.exp_p(Quaternion([0., 0., 0., 0.]))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_pow(self):
        expected = np.array([-28., 4., 6., 8.])
        actual = Quaternion.pow(Quaternion([1., 2., 3., 4.]), 2.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_pow_with_identity(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.pow(Quaternion([1., 0., 0., 0.]), 2.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_pow_u(self):
        expected = np.array([-9.333333333333331e-001, 1.333333333333333e-001, 1.999999999999999e-001, 2.666666666666666e-001])
        actual = Quaternion.pow_u(Quaternion([1.825741858350554e-001, 3.651483716701107e-001, 5.477225575051661e-001, 7.302967433402214e-001]), 2.) # normalize([1,2,3,4])
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_pow_u_with_identity(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.pow_u(Quaternion([1., 0., 0., 0.]), 2.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_normalize(self):
        c = np.array([1., 2., 3., 4.])
        q = Quaternion.normalize(Quaternion(c))
        self.assertTrue(isinstance(q, Quaternion))
        self.assertTrue(np.isclose(np.linalg.norm(q._components), 1.))
        self.assertTrue(np.allclose(q._components, c / np.linalg.norm(c)))

    def test_from_axis_angle(self):
        expected = np.array([8.660254037844387e-001, 4.999999999999999e-001, 0., 0.])
        actual = Quaternion.from_axis_angle(np.array([1., 0., 0.]), math.radians(60.))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_from_axis_angle_with_identity(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.from_axis_angle(np.array([1., 0., 0.]), 0.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_from_rotation_matrix(self):
        # When the absolute value of w is maximum.
        expected = np.array([0.5, 0.8660254037844386, 0., 0.])  # axis=[1,0,0] deg=120
        m = np.array([[1., 0., 0.],
                      [0., -0.5, 0.8660254037844387],
                      [0., -0.8660254037844387, -0.5]])
        actual = Quaternion.from_rotation_matrix(m)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # When the absolute value of x is maximum.
        expected = np.array([0.42261826174069944, -0.9063077870366499, 0., 0.])  # axis=[-1,0,0] deg=130
        m = np.array([[1., 0., 0.],
                      [0., -0.6427876096865394, -0.766044443118978],
                      [0., 0.766044443118978, -0.6427876096865394]])
        actual = Quaternion.from_rotation_matrix(m)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # When the absolute value of y is maximum.
        expected = np.array([0.42261826174069944, 0., -0.9063077870366499, 0.])  # axis=[0,-1,0] deg=130
        m = np.array([[-0.6427876096865394, 0., 0.766044443118978],
                      [0., 1., 0.],
                      [-0.766044443118978, 0., -0.6427876096865394]])
        actual = Quaternion.from_rotation_matrix(m)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # When the absolute value of z is maximum.
        expected = np.array([0.42261826174069944, 0., 0., -0.9063077870366499])  # axis=[0,0,-1] deg=130
        m = np.array([[-0.6427876096865394, -0.766044443118978, 0.],
                      [0.766044443118978, -0.6427876096865394, 0.],
                      [0., 0., 1.]])
        actual = Quaternion.from_rotation_matrix(m)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_from_rotation_matrix_with_identity(self):
        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.from_rotation_matrix(np.identity(3))
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_rotate(self):
        expected = np.array([0., 0., 1.])
        q = Quaternion([0.7071067811865476, 0.7071067811865476, 0., 0.])  # axis=[1,0,0] deg=90
        actual = Quaternion.rotate(q, np.array([0., 1., 0.]))
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.shape, (3,))
        self.assertTrue(np.allclose(actual, expected))

    def test_rotate_with_identity(self):
        expected = np.array([1., 0., 0.])
        q = Quaternion(np.array([1., 0., 0., 0.]))
        actual = Quaternion.rotate(q, np.array([1., 0., 0.]))
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.shape, (3,))
        self.assertTrue(np.allclose(actual, expected))

    def test_rotation_shortest_arc(self):
        expected = np.array([0.7071067811865476, 0., 0., 0.7071067811865476]) # axis=[0,0,1] deg=90
        v1 = [1., 0., 0.]
        v2 = [0., 1., 0.]
        actual = Quaternion.rotation_shortest_arc(v1, v2)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_rotational_difference(self):
        expected = np.array([0.8660254037844387, 0.5, 0., 0.]) # axis=[1,0,0] deg=60
        q1 = Quaternion([0.9659258262890683, 0.25881904510252074, 0., 0.]) # axis=[1,0,0] deg=30
        q2 = Quaternion([0.7071067811865476, 0.7071067811865476, 0., 0.]) # axis=[1,0,0] deg=90
        actual = Quaternion.rotational_difference(q1, q2)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_lerp(self):
        expected = np.array([1.25, 2.25, 3.25, 4.25])
        q1 = Quaternion([1., 2., 3., 4.])
        q2 = Quaternion([2., 3., 4., 5.])
        actual = Quaternion.lerp(q1, q2, 0.25)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_lerp_at_boundary(self):
        q1 = Quaternion([1., 2., 3., 4.])
        q2 = Quaternion([2., 3., 4., 5.])

        expected = np.array([1., 2., 3., 4.])
        actual = Quaternion.lerp(q1, q2, 0.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

        expected = np.array([2., 3., 4., 5.])
        actual = Quaternion.lerp(q1, q2, 1.)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_lerp_with_close_ones(self):
        expected = np.array([1., 2., 3., 4.])
        q1 = Quaternion([1., 2., 3., 4.])
        q2 = Quaternion([1., 2., 3., 4.])
        actual = Quaternion.lerp(q1, q2, 0.25)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(np.allclose(actual._components, expected))

    def test_slerp(self):
        # interpolate along the great circle arc.
        expected = np.array([0.8660254037844387, 0., 0., 0.5])  # axis=[0,0,1] deg=60
        q1 = Quaternion([1., 0., 0., 0.])  # e.g. axis=[0,0,1] deg=0
        q2 = Quaternion([-0.5, 0., 0., 0.8660254037844387])  # axis=[0,0,1] deg=240
        actual = Quaternion.slerp(q1, q2, 0.25, allow_flip=False)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # interpolate along the shortest arc.
        expected = np.array([0.9659258262890683, 0., 0., -0.25881904510252074])  # axis=[0,0,-1] deg=30
        q1 = Quaternion([1., 0., 0., 0.])  # e.g. axis=[0,0,1] deg=0
        q2 = Quaternion([-0.5, 0., 0., 0.8660254037844387])  # axis=[0,0,1] deg=240
        actual = Quaternion.slerp(q1, q2, 0.25, allow_flip=True)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_slerp_at_boundary(self):
        # interpolate along the great circle arc.
        q1 = Quaternion([1., 0., 0., 0.])  # e.g. axis=[0,0,1] deg=0
        q2 = Quaternion([-0.5, 0., 0., 0.8660254037844387])  # axis=[0,0,1] deg=240

        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.slerp(q1, q2, 0., allow_flip=False)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        expected = np.array([-0.5, 0., 0., 0.8660254037844387])
        actual = Quaternion.slerp(q1, q2, 1., allow_flip=False)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # interpolate along the shortest arc.
        q1 = Quaternion([1., 0., 0., 0.])  # e.g. axis=[0,0,1] deg=0
        q2 = Quaternion([-0.5, 0., 0., 0.8660254037844387])  # axis=[0,0,1] deg=240

        expected = np.array([1., 0., 0., 0.])
        actual = Quaternion.slerp(q1, q2, 0., allow_flip=True)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        expected = np.array([0.5, 0., 0., -0.8660254037844386]) # expected = -q2
        actual = Quaternion.slerp(q1, q2, 1., allow_flip=True)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_slerp_with_close_ones(self):
        # interpolate along the great circle arc.
        expected = np.array([1., 0., 0., 0.])
        q1 = Quaternion([1., 0., 0., 0.])
        q2 = Quaternion([1., 0., 0., 0.])
        actual = Quaternion.slerp(q1, q2, 0.25, allow_flip=False)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

        # interpolate along the shortest arc.
        expected = np.array([1., 0., 0., 0.])
        q1 = Quaternion([1., 0., 0., 0.])
        q2 = Quaternion([1., 0., 0., 0.])
        actual = Quaternion.slerp(q1, q2, 0.25, allow_flip=True)
        self.assertTrue(isinstance(actual, Quaternion))
        self.assertTrue(self.are_same_rotation(actual._components, expected))

    def test_squad(self):
        q1 = Quaternion([1., 2., 3., 4.])
        q2 = Quaternion([1., 2., 3., 4.])
        t1 = Quaternion([1., 2., 3., 4.])
        t2 = Quaternion([1., 2., 3., 4.])
        q = Quaternion.squad(q1, q2, t1, t2, 0.5)
        self.assertTrue(isinstance(q, Quaternion))

        self.fail()

    def test_squad_tangent(self):
        p = Quaternion([1., 2., 3., 4.])
        c = Quaternion([1., 2., 3., 4.])
        n = Quaternion([1., 2., 3., 4.])
        q = Quaternion.squad_tangent(p, c, n)
        self.assertTrue(isinstance(q, Quaternion))

        self.fail()

    def test__make_ndarray_by_parts_with_ndarray(self):
        s, v = 1., np.array([2., 3., 4.])
        q = Quaternion._make_ndarray_by_parts(s, v)
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertTrue(np.allclose(q[Quaternion._SCALAR_PART], s))
        self.assertTrue(np.allclose(q[Quaternion._VECTOR_PART], v))

    def test__make_ndarray_by_parts_with_list(self):
        s, v = 1., [2., 3., 4.]
        q = Quaternion._make_ndarray_by_parts(s, v)
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertTrue(np.allclose(q[Quaternion._SCALAR_PART], s))
        self.assertTrue(np.allclose(q[Quaternion._VECTOR_PART], v))

    def test__make_ndarray_by_parts_with_tuple(self):
        s, v = 1., (2., 3., 4.)
        q = Quaternion._make_ndarray_by_parts(s, v)
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertTrue(np.allclose(q[Quaternion._SCALAR_PART], s))
        self.assertTrue(np.allclose(q[Quaternion._VECTOR_PART], v))

    def test__multiplication_for_ndarray(self):
        expected = np.array([1., 0., 0., 0.])
        p = np.array([1., 1., 1., 1.])
        q = np.array([0.25, -0.25, -0.25, -0.25]) # inverse of p.

        actual = Quaternion._multiplication_for_ndarray(q, p)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(np.allclose(actual, expected))

        actual = Quaternion._multiplication_for_ndarray(p, q)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(np.allclose(actual, expected))

    def test__cross_for_ndarray(self):
        v1 = np.array([-2., 3.,-4.])
        v2 = np.array([ 5.,-6., 7.])
        expected = np.cross(v1, v2)
        actual = Quaternion._cross_for_ndarray(v1, v2)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(np.allclose(actual, expected))
