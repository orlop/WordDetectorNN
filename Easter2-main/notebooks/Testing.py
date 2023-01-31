import os, sys
sys.path.append(os.path.join("..","src"))

from predict import test_on_iam

checkpoint_path = "../weights/saved_checkpoint.hdf5"

test_on_iam(show=False, partition="validation", checkpoint=checkpoint_path, uncased=True)

test_on_iam(show=False, partition="test", checkpoint=checkpoint_path, uncased=True)