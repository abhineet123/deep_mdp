import xgboost as xgb
import numpy as np

from models.model_base import ModelBase
from utilities import CustomLogger


class XGB(ModelBase):
    """
    :type _params: XGB.Params
    :type labels: np.ndarray
    :type accuracies: np.ndarray
    :type probabilities: np.ndarray
    """

    class Params(ModelBase.Params):
        def __init__(self):
            super(XGB.Params, self).__init__()

            self.max_depth = 2
            self.eta = 1
            self.objective = 'binary:logistic'
            self.nthread = 4
            self.eval_metric = 'auc'
            self.num_round = 10

            self.help = {
                'implementation': 'XGB implementation to use',
                'verbose': 'Show detailed diagnostic messages'
            }

    def __init__(self, params, logger, input_size, name=''):
        """
        :type params: XGB.Params
        :type logger: CustomLogger
        :rtype: None
        """
        ModelBase.__init__(self, params, logger, [input_size, ], name, 'XGB', n_classes=2)

        self._params = params

        self.bst = None
        self.predictions = None
        self.probabilities = None
        self._input_shape = [input_size, ]

        self.xgb_params = {
            'max_depth': self._params.max_depth,
            'eta': self._params.eta,
            'silent': not self._params.verbose,
            'objective': self._params.objective,
            'nthread': self._params.nthread,
            'eval_metric': self._params.eval_metric,
        }
        self.is_trained = False

    def train(self, labels, features, options=''):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :type options: str
        :rtype: None
        """
        dtrain = xgb.DMatrix(features, label=labels)
        self.bst = xgb.train(self.xgb_params, dtrain, self._params.num_round)

        self.is_trained = True

    def predict(self, labels, features, gt_labels=None):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :type gt_labels: np.ndarray | None
        :rtype: None
        """

        dtest = xgb.DMatrix(features)
        self.probabilities = self.bst.predict_proba(dtest)
        self.predictions = np.argmax(self.probabilities, axis=1)

        print()

    def save(self, fname):
        """
        :type fname: str
        :rtype: None
        """
        self.bst.save_model(fname)

    def load(self, fname):
        """
        :type fname: str
        :rtype: None
        """
        self.bst = xgb.Booster(self.xgb_params)
        self.bst.load_model(fname)

    def set(self, obj):
        """
        :type obj: XGB
        :rtype: None
        """
        self.bst = obj.bst
