from __future__ import absolute_import, division, print_function

import os
import absl.flags

FLAGS = absl.flags.FLAGS

def create_flags():
    # Importer
    # ========

    f = absl.flags

    flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedded vector (default: 128)")
    flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
    flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
    flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    flags.DEFINE_boolean('use_allow_growth', True, 'use Allow Growth flag which will allocate only required amount of GPU memory and prevent full allocation of available GPU memory')


#    # Register validators for paths which require a file to be specified
#
#    f.register_validator('alphabet_config_path',
#                         os.path.isfile,
#                         message='The file pointed to by --alphabet_config_path must exist and be readable.')
#
#    f.register_validator('one_shot_infer',
#                         lambda value: not value or os.path.isfile(value),
#                         message='The file pointed to by --one_shot_infer must exist and be readable.')
