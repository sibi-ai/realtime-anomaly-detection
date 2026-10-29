"""Train XGBoost / LightGBM ensemble for anomaly detection."""
import argparse, yaml, mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import structlog

log = structlog.get_logger()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_fold(X_train, y_train, X_val, y_val, cfg: dict) -> tuple:
    xgb = XGBClassifier(**cfg["xgboost"], eval_metric="aucpr", early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    lgb = LGBMClassifier(**cfg["lightgbm"])
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[])

    # Ensemble: average probabilities
    p_xgb = xgb.predict_proba(X_val)[:, 1]
    p_lgb = lgb.predict_proba(X_val)[:, 1]
    p_ens = 0.5 * p_xgb + 0.5 * p_lgb

    auc = roc_auc_score(y_val, p_ens)
    ap  = average_precision_score(y_val, p_ens)
    log.info("fold_metrics", auc=round(auc, 4), ap=round(ap, 4))
    return xgb, lgb, auc, ap


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    # Placeholder — replace with real data loading
    X = pd.DataFrame(np.random.randn(10_000, cfg["data"]["n_features"]))
    y = pd.Series((np.random.rand(10_000) > 0.95).astype(int))  # 5% anomaly rate

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps = [], []

    with mlflow.start_run(run_name="ensemble_training"):
        mlflow.log_params(cfg)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            log.info("training_fold", fold=fold)
            _, _, auc, ap = train_fold(
                X.iloc[tr], y.iloc[tr],
                X.iloc[va], y.iloc[va], cfg["model"]
            )
            aucs.append(auc); aps.append(ap)

        mlflow.log_metrics({"mean_auc": np.mean(aucs), "mean_ap": np.mean(aps)})
        log.info("training_complete", mean_auc=round(np.mean(aucs), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    main(parser.parse_args().config)
