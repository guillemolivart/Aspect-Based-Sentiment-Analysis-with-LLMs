import json
import sys


def flatten(value, prefix=""):
    if isinstance(value, (str, int, float, bool)):
        return [prefix + ":" + str(value)]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten(item, prefix))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for key, item in value.items():
            flattened.extend(flatten(item, prefix + f".{key}"))
        return flattened
    return []


def counts(prediction, gold):
    pred = flatten(prediction)
    expected = flatten(gold)
    ok = sum(1 for item in pred if item in expected)
    return len(pred), len(expected), ok


def prf(predicted, expected, ok):
    precision = 100.0 * ok / predicted if predicted else 0.0
    recall = 100.0 * ok / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} predictions.json")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as predictions_fd:
        examples = json.load(predictions_fd)

    predicted_total = expected_total = ok_total = 0
    precision_macro = recall_macro = f1_macro = 0.0

    for example in examples:
        predicted, expected, ok = counts(example.get("prediction", {}), example["gold"])
        predicted_total += predicted
        expected_total += expected
        ok_total += ok

        precision, recall, f1 = prf(predicted, expected, ok)
        print(f"{example['id']:35s}\t   {precision:5.1f} {recall:5.1f} {f1:5.1f}")

        precision_macro += precision
        recall_macro += recall
        f1_macro += f1

    precision_macro /= len(examples)
    recall_macro /= len(examples)
    f1_macro /= len(examples)
    precision_micro, recall_micro, f1_micro = prf(
        predicted_total, expected_total, ok_total
    )

    print("         P     R     F")
    print(f"M.avg  {precision_macro:5.1f} {recall_macro:5.1f} {f1_macro:5.1f}")
    print(f"m.avg  {precision_micro:5.1f} {recall_micro:5.1f} {f1_micro:5.1f}")


if __name__ == "__main__":
    main()
