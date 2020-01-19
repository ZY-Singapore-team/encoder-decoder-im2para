import os
from pprint import pprint
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import sys
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice

if __name__ == "__main__":
    prediction_path = sys.argv[1]
    ground_turth_path = sys.argv[2]
    with open(prediction_path, 'r') as f:
        predictions = [line.strip() for line in f]

    with open(ground_turth_path, 'r') as f:
        ground_turth = [line.strip() for line in f]

    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge()
    }

    gts = {}
    res = {}
    if len(predictions) == len(ground_turth):
        for ind, value in enumerate(predictions):
            # print(value)
            res[ind] = [value]

        for ind, value in enumerate(ground_turth):
            gts[ind] = [value]
    else:
        Min_Len = min(len(predictions), len(ground_turth))
        for ind in range(Min_Len):
            res[ind] = [predictions[ind]]
            gts[ind] = [ground_turth[ind]]

    # param gts: Dictionary of reference sentences (id, sentence)
    # param res: Dictionary of hypothesis sentences (id, sentence)

    print('samples: {} / {}'.format(len(res.keys()), len(gts.keys())))

    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    pprint(scores)
