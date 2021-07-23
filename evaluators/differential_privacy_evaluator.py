from evaluators.base import Evaluator


class DifferentialPrivacyEvaluator(Evaluator):
    """
    DifferentialPrivacyEvaluator evaluates the strength of the generated
    data's privacy against different attacks.

    """

    def __init__(self, data_path, config_file):
        super().__init__(data_path=data_path, config_file=config_file)
