from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--inference_set', type=int, default=0, help='use either 0: vbd clean_trainset, 1: first half, or 2: second half')
        parser.add_argument('--random_inference', type=int, default=0, help='use either 0: sorted inference or 1: random inference')

        # To avoid cropping, the load_size should be the same as crop_size
        self.isTrain = False
        return parser
