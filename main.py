import argparse
import os

from Models.unet128 import *
from Models.unet256 import *
from Models.unet512 import *
from Models.unet1024 import *
from Models.preprocessing import *


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', default='train', help='Phase: Can be preprocess, train, test, or submission')
  parser.add_argument('--data_location', default='input', help='Directory to save the tfrecords file in')
  parser.add_argument('--delete_old', default='True', help='Should we keep the old results and tensorboard files')
  parser.add_argument('--model', default='unet128', help='Model to use: Can be unet128, unet256, unet512, or unet1024')

  args = parser.parse_args()

  configs = {
    'TF_RECORDS_TRAIN': os.path.join(args.data_location, 'unet_train.tfrecords'),
    'TF_RECORDS_TEST': os.path.join(args.data_location, 'unet_test.tfrecords'),
    'TF_RECORDS_META': os.path.join(args.data_location, "unet.meta"),
    'INPUT_SIZE': int(args.model[4:])
  }

  # Preprocess training data
  if args.phase == 'preprocess':
    prepare_train_and_test_data(configs, args, configs['INPUT_SIZE'])

  # Train the model
  elif args.phase == 'train':
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.train()

  # Test the model
  elif args.phase == 'test':
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.test()

  # Create Submission
  elif args.phase == "submission":
    stats = get_stats(configs)
    model = create_model(configs, args, stats)
    model.create_submission()

  else:
    print("No valid phase selected")

def create_model(configs, args, stats):
  # Select the model you want to use
  if args.model == "unet128":
    model = Unet128(args, stats, configs['TF_RECORDS_TRAIN'])

  elif args.model == "unet256":
    model = Unet256(args, stats, configs['TF_RECORDS_TRAIN'])

  elif args.model == "unet512":
    model = Unet512(args, stats, configs['TF_RECORDS_TRAIN'])

  elif args.model == "unet1024":
    model = Unet1024(args, stats, configs['TF_RECORDS_TRAIN'])

  return model

if __name__=="__main__":
  main(sys.argv)
