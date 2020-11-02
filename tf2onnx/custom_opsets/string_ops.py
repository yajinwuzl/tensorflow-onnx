# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for onnx string domain. """
import logging
from onnx import TensorProto
from tf2onnx import constants
from tf2onnx.handler import tf_op
from tf2onnx import utils

logger = logging.getLogger(__name__)

@tf_op("StaticRegexReplace", domain=constants.STRING_OPS_DOMAIN)
@tf_op("StringJoin", domain=constants.STRING_OPS_DOMAIN)
@tf_op("SparseToDense", domain=constants.STRING_OPS_DOMAIN)
#@tf_op("DynamicPartition", domain=constants.STRING_OPS_DOMAIN)
@tf_op("ParallelDynamicStitch", domain=constants.STRING_OPS_DOMAIN)
@tf_op("SegmentSum", domain=constants.STRING_OPS_DOMAIN)
@tf_op("StringSplit", domain=constants.STRING_OPS_DOMAIN)
@tf_op("SparseFillEmptyRows", domain=constants.STRING_OPS_DOMAIN)
@tf_op("StringToHashBucketFast", domain=constants.STRING_OPS_DOMAIN)
class HashTable:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.STRING_OPS_DOMAIN