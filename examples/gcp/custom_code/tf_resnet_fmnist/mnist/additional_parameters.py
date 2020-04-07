#--stream-logs --runtime-version 1.10 \
#--job-dir=$GCS_JOB_DIR \
--package-path=trainer \
--module-name trainer.task \
#--region $REGION -- \
--package-path=trainer
--package-path=trainer
--module-name trainer.task
--train-file=gs://yjkim-dataset/images/fashion-mnist/train-images-idx3-ubyte.gz
--train-labels=gs://yjkim-dataset/images/fashion-mnist/train-labels-idx1-ubyte.gz
--test-file=gs://yjkim-dataset/images/fashion-mnist/t10k-labels-idx1-ubyte.gz
--test-labels-file=gs://yjkim-dataset/images/fashion-mnist/t10k-images-idx3-ubyte.gz




parser.add_argument(
    '--job-dir',
    type=str,
    required=True,
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train-file',
    type=str,
    required=True,
    help='Training file local or GCS')
  parser.add_argument(
    '--train-labels-file',
    type=str,
    required=True,
    help='Training labels file local or GCS')
  parser.add_argument(
    '--test-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--test-labels-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--num-epochs',
    type=float,
    default=5,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch-size',
    default=128,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning-rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  return parser.parse_args()