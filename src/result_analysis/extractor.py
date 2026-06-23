import re
import numpy as np


def extract_prediction(prediction, op_map):
    op_val = list(op_map.keys())
    op_pattern = "|".join(map(re.escape, map(str, op_val)))

    prep_prediction = ("\\boxed{" + prediction + "}").lower()
    matches = re.findall(r"boxed\{(" + op_pattern + r")\}", prep_prediction)
    if len(matches) > 0:
        return op_map[matches[-1]]
    else:
        return -1

def extract_predictions(predictions, op_map):
    n = None

    if isinstance(predictions, (np.ndarray, list)) and isinstance(predictions[0], (np.ndarray, list)):    
        n = len(predictions[0])
        predictions = [x for a in predictions for x in a]

    output = [extract_prediction(x, op_map) for x in predictions]

    if n:
        output = np.array(output).reshape(-1, n)

    return output