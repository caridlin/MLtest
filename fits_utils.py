# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export

def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError(
        "'image' (shape %s) must be three-dimensional." % image.shape)
  if require_static and not image_shape.is_fully_defined():
    raise ValueError("'image' (shape %s) must be fully defined." % image_shape)
  if any(x == 0 for x in image_shape):
    raise ValueError("all dims of 'image.shape' must be > 0: %s" % image_shape)
  if not image_shape.is_fully_defined():
    return [
        check_ops.assert_positive(
            array_ops.shape(image),
            ["all dims of 'image.shape' "
             'must be > 0.'])
    ]
  else:
    return []


def _Assert3DImage(image):
  """Assert that we are working with a properly shaped image.

    Performs the check statically if possible (i.e. if the shape
    is statically known). Otherwise adds a control dependency
    to an assert op that checks the dynamic shape.

    Args:
      image: 3-D Tensor of shape [height, width, channels]

    Raises:
      ValueError: if `image.shape` is not a 3-vector.

    Returns:
      If the shape of `image` could be verified statically, `image` is
      returned unchanged, otherwise there will be a control dependency
      added that asserts the correct dynamic shape.
    """
  return control_flow_ops.with_dependencies(
      _Check3DImage(image, require_static=False), image)

# @tf_export('image.per_image_norm')
"""
Adapted from tf.train.image.per_image_standardization
"""
def per_image_norm(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The standardized image with same shape as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  with ops.name_scope(None, 'per_image_norm', [image]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    image = _Assert3DImage(image)
    num_pixels = math_ops.reduce_prod(array_ops.shape(image))

    image = math_ops.cast(image, dtype=dtypes.float32)
    image_mean = math_ops.reduce_mean(image)

    variance = (
        math_ops.reduce_mean(math_ops.square(image)) -
        math_ops.square(image_mean))
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance)

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    pixel_value_scale = math_ops.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean

    image = math_ops.subtract(image, pixel_value_offset)
    image = math_ops.div(image, pixel_value_scale, name=scope)
    image = math_ops.subtract(image, -0.5)
    return image
