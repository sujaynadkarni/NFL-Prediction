"""
Compare the current linear model against gradient-boosted trees (the XGBoost family).
Same features, same seasons, same scoring — train 2002–2017, choose on 2018–2023,
then a single honest run on the untouched 2024–25 test seasons.
Run: .venv/bin/python compare_models.py
"""
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
import nfl_model as nm

TRAIN, VALID, TEST = range(2002, 2018), range(2018, 2024), [2024, 2025]


def probs(pred, sigma):
    return np.array([nm.norm_cdf(x / sigma) for x in pred])


def report(name, y, pred_margin, p_home):
    mask = y != 0
    return {"model": name, "mae": float(np.mean(np.abs(y - pred_margin))) if pred_margin is not None else None,
            "acc": float(np.mean((p_home[mask] > 0.5) == (y[mask] > 0))),
            "brier": float(np.mean((p_home[mask] - (y[mask] > 0)) ** 2))}


def main():
    m = pickle.load(open(nm.MODEL_PATH, "rb"))
    feat, F = m["feat"], m["features"]
    played = feat[feat["played"]]
    tr, va, te = (played[played.season.isin(s)] for s in (TRAIN, VALID, TEST))
    full = played[played.season.isin(nm.TRAIN_SEASONS)]

    def run(fit_fn, name, train, test):
        pred, p = fit_fn(train, test)
        return report(name, test["result"].values, pred, p)

    def ridge(train, test):
        mdl = Ridge(alpha=1.0, fit_intercept=False).fit(train[F], train["result"])
        s = float(np.std(train["result"] - mdl.predict(train[F])))
        pred = mdl.predict(test[F])
        return pred, probs(pred, s)

    def gb_margin(train, test, **kw):
        mdl = HistGradientBoostingRegressor(random_state=42, **kw).fit(train[F], train["result"])
        s = float(np.std(train["result"] - mdl.predict(train[F])))
        pred = mdl.predict(test[F])
        return pred, probs(pred, s)

    def gb_win(train, test, **kw):
        t = train[train["result"] != 0]
        mdl = HistGradientBoostingClassifier(random_state=42, **kw).fit(t[F], (t["result"] > 0).astype(int))
        return None, mdl.predict_proba(test[F])[:, 1]

    settings = {"shallow":  dict(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0),
                "medium":   dict(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0),
                "deep":     dict(max_depth=6, learning_rate=0.1,  max_iter=500)}

    print("VALIDATION (train 2002–2017 → score 2018–2023) — used to pick the best settings")
    rows = [run(ridge, "ridge (current)", tr, va)]
    for k, kw in settings.items():
        rows.append(run(lambda a, b, kw=kw: gb_margin(a, b, **kw), f"boosted trees, margin ({k})", tr, va))
        rows.append(run(lambda a, b, kw=kw: gb_win(a, b, **kw), f"boosted trees, win/loss ({k})", tr, va))
    for r in sorted(rows, key=lambda r: r["brier"]):
        mae = f"{r['mae']:.2f}" if r["mae"] else "  — "
        print(f"  {r['model']:<32} Brier {r['brier']:.4f}   winners {r['acc']:.1%}   MAE {mae}")

    best = min([r for r in rows if r["model"] != "ridge (current)"], key=lambda r: r["brier"])["model"]
    kind, key = ("win", best.split("(")[1][:-1]) if "win/loss" in best else ("margin", best.split("(")[1][:-1])
    print(f"\nBest tree model on validation: {best}")

    print("\nTEST (train 2002–2023 → score 2024–2025) — the honest comparison")
    final = [run(ridge, "ridge (current)", full, te),
             run(lambda a, b: (gb_win if kind == "win" else gb_margin)(a, b, **settings[key]), best, full, te)]
    vt = te[te["spread_line"].notna()]
    final.append(report("Vegas spread", vt["result"].values, vt["spread_line"].values,
                        probs(vt["spread_line"].values, m["sigma"])))
    for r in final:
        mae = f"{r['mae']:.2f}" if r["mae"] else "  — "
        print(f"  {r['model']:<32} Brier {r['brier']:.4f}   winners {r['acc']:.1%}   MAE {mae}")


if __name__ == "__main__":
    main()
