"""Score new patients with the frozen model.

Inputs have the same layout as the training files (one clinical TSV, one
mutation TSV; outcome columns are ignored if present).

    .venv/bin/python scripts/predict.py \
        --clinical df_clinical_test.tsv --mutations mut_clinical_test.tsv \
        --out outputs/predictions_test.csv --cards outputs/cards --card-ids P123 P456

The CSV holds, per patient: risk_score (linear predictor, higher = worse, for
C-index), risk_2y and its range, the decision band, IPSS-R, IPSS-M, warnings.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mds_tool import features, tool  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--mutations", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "final_model.json"))
    ap.add_argument("--out", required=True, help="CSV of predictions")
    ap.add_argument("--cards", help="folder for HTML patient cards")
    ap.add_argument("--card-ids", nargs="*", help="patients to make cards for (default: first 5)")
    a = ap.parse_args(argv)

    clin, muts = features.load(a.clinical, a.mutations)
    m = tool.load_model(a.model)
    pred, feats, contrib = tool.predict_full(clin, muts, m)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pred.round(4).to_csv(a.out)
    print(f"{len(pred)} patients scored -> {a.out}")
    print(pred["band"].value_counts().to_string())
    if a.cards:
        ids = a.card_ids or list(pred.index[:5])
        for p in tool.write_cards(pred, feats, contrib, m, ids, a.cards):
            print("card:", p)


if __name__ == "__main__":
    main()
