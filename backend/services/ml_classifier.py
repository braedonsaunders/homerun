"""
ML False-Positive Classifier for Arbitrage Opportunities

Filters detected arbitrage opportunities by predicting which ones are truly
profitable vs false positives. Uses logistic regression trained via gradient
descent with numpy -- no sklearn dependency.

The model improves over time as OpportunityHistory accumulates resolved outcomes.
On cold start (no training data), returns neutral predictions (probability=0.5).
"""

import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, update, func, desc

from models.database import (
    OpportunityHistory,
    MLModelWeights,
    MLPredictionLog,
    AsyncSessionLocal,
)
from models.opportunity import ArbitrageOpportunity, StrategyType, MispricingType
from utils.logger import get_logger

logger = get_logger("ml_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All strategy types for one-hot encoding (stable ordering)
_STRATEGY_TYPES: list[str] = sorted([s.value for s in StrategyType])

# All mispricing types for one-hot encoding (stable ordering)
_MISPRICING_TYPES: list[str] = sorted([m.value for m in MispricingType])

# Thresholds for recommendation buckets
_EXECUTE_THRESHOLD = 0.65
_SKIP_THRESHOLD = 0.35

# Training hyper-parameters
_LEARNING_RATE = 0.05
_MAX_ITERATIONS = 500
_REGULARIZATION = 0.01  # L2 penalty
_CONVERGENCE_TOL = 1e-6
_MIN_TRAINING_SAMPLES = 10
_TEST_SPLIT_RATIO = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _build_feature_names() -> list[str]:
    """Return the ordered list of feature names used by the model."""
    names = [
        "roi_percent",
        "risk_score",
        "min_liquidity",
        "max_position_size",
        "num_markets",
        "avg_yes_price",
        "avg_no_price",
        "price_sum",
        "spread",
        "guaranteed_profit",
        "capture_ratio",
        "time_to_resolution_days",
        "num_risk_factors",
        "hour_of_day",
        "day_of_week",
    ]
    # One-hot: strategy types
    for s in _STRATEGY_TYPES:
        names.append(f"strategy_{s}")
    # One-hot: mispricing types
    for m in _MISPRICING_TYPES:
        names.append(f"mispricing_{m}")
    return names


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(opp: ArbitrageOpportunity) -> dict[str, float]:
    """
    Extract a flat feature dict from an ArbitrageOpportunity.

    Returns a dict mapping feature name -> float value.  All values are
    numeric; categorical fields are one-hot encoded.
    """
    # --- Scalar features ---------------------------------------------------
    features: dict[str, float] = {
        "roi_percent": float(opp.roi_percent),
        "risk_score": float(opp.risk_score),
        "min_liquidity": float(opp.min_liquidity),
        "max_position_size": float(opp.max_position_size),
    }

    # Market-derived
    markets = opp.markets or []
    num_markets = len(markets)
    features["num_markets"] = float(num_markets)

    yes_prices = [
        float(m.get("yes_price", 0.0))
        for m in markets
        if m.get("yes_price") is not None
    ]
    no_prices = [
        float(m.get("no_price", 0.0)) for m in markets if m.get("no_price") is not None
    ]

    features["avg_yes_price"] = float(np.mean(yes_prices)) if yes_prices else 0.0
    features["avg_no_price"] = float(np.mean(no_prices)) if no_prices else 0.0

    # price_sum: sum of the best prices across markets
    best_prices = []
    for m in markets:
        yp = float(m.get("yes_price", 0.0))
        np_ = float(m.get("no_price", 0.0))
        best_prices.append(max(yp, np_))
    price_sum = sum(best_prices)
    features["price_sum"] = price_sum
    features["spread"] = 1.0 - price_sum if best_prices else 0.0

    # Profit guarantee from Frank-Wolfe
    features["guaranteed_profit"] = (
        float(opp.guaranteed_profit) if opp.guaranteed_profit is not None else 0.0
    )
    features["capture_ratio"] = (
        float(opp.capture_ratio) if opp.capture_ratio is not None else 0.0
    )

    # Time to resolution
    if opp.resolution_date and opp.detected_at:
        delta = opp.resolution_date - opp.detected_at
        features["time_to_resolution_days"] = max(delta.total_seconds() / 86400.0, 0.0)
    else:
        features["time_to_resolution_days"] = -1.0  # sentinel for unknown

    # Risk factors count
    features["num_risk_factors"] = (
        float(len(opp.risk_factors)) if opp.risk_factors else 0.0
    )

    # Temporal
    now = opp.detected_at or datetime.utcnow()
    features["hour_of_day"] = float(now.hour)
    features["day_of_week"] = float(now.weekday())

    # --- One-hot: strategy -------------------------------------------------
    strategy_val = opp.strategy.value if opp.strategy else ""
    for s in _STRATEGY_TYPES:
        features[f"strategy_{s}"] = 1.0 if strategy_val == s else 0.0

    # --- One-hot: mispricing type ------------------------------------------
    mispricing_val = opp.mispricing_type.value if opp.mispricing_type else ""
    for m in _MISPRICING_TYPES:
        features[f"mispricing_{m}"] = 1.0 if mispricing_val == m else 0.0

    return features


def _extract_features_from_history(row: OpportunityHistory) -> dict[str, float]:
    """
    Extract features from an OpportunityHistory database row.

    The history table has a subset of the fields that ArbitrageOpportunity
    carries, so some features are derived from positions_data JSON.
    """
    features: dict[str, float] = {
        "roi_percent": float(row.expected_roi or 0.0),
        "risk_score": float(row.risk_score or 0.5),
        "min_liquidity": 0.0,
        "max_position_size": 0.0,
    }

    # Parse positions_data for market info
    positions = row.positions_data or {}
    markets = positions.get("markets", []) if isinstance(positions, dict) else []
    num_markets = len(markets)
    features["num_markets"] = float(num_markets)

    yes_prices = []
    no_prices = []
    best_prices = []
    for m in markets:
        if isinstance(m, dict):
            yp = float(m.get("yes_price", 0.0))
            np_ = float(m.get("no_price", 0.0))
            yes_prices.append(yp)
            no_prices.append(np_)
            best_prices.append(max(yp, np_))
            if "liquidity" in m:
                liq = float(m.get("liquidity", 0.0))
                if features["min_liquidity"] == 0.0 or liq < features["min_liquidity"]:
                    features["min_liquidity"] = liq

    features["avg_yes_price"] = float(np.mean(yes_prices)) if yes_prices else 0.0
    features["avg_no_price"] = float(np.mean(no_prices)) if no_prices else 0.0
    price_sum = sum(best_prices)
    features["price_sum"] = price_sum
    features["spread"] = 1.0 - price_sum if best_prices else 0.0

    # max_position_size from positions_data
    if isinstance(positions, dict):
        features["max_position_size"] = float(positions.get("max_position_size", 0.0))
        features["guaranteed_profit"] = float(positions.get("guaranteed_profit", 0.0))
        features["capture_ratio"] = float(positions.get("capture_ratio", 0.0))
        risk_factors = positions.get("risk_factors", [])
        features["num_risk_factors"] = (
            float(len(risk_factors)) if isinstance(risk_factors, list) else 0.0
        )
    else:
        features["guaranteed_profit"] = 0.0
        features["capture_ratio"] = 0.0
        features["num_risk_factors"] = 0.0

    # Time to resolution
    if row.resolution_date and row.detected_at:
        delta = row.resolution_date - row.detected_at
        features["time_to_resolution_days"] = max(delta.total_seconds() / 86400.0, 0.0)
    else:
        features["time_to_resolution_days"] = -1.0

    # Temporal from detected_at
    det = row.detected_at or datetime.utcnow()
    features["hour_of_day"] = float(det.hour)
    features["day_of_week"] = float(det.weekday())

    # One-hot: strategy
    strategy_val = row.strategy_type or ""
    for s in _STRATEGY_TYPES:
        features[f"strategy_{s}"] = 1.0 if strategy_val == s else 0.0

    # One-hot: mispricing type (may be in positions_data)
    mispricing_val = ""
    if isinstance(positions, dict):
        mispricing_val = positions.get("mispricing_type", "")
    for m in _MISPRICING_TYPES:
        features[f"mispricing_{m}"] = 1.0 if mispricing_val == m else 0.0

    return features


def _features_dict_to_array(
    feat_dict: dict[str, float], feature_names: list[str]
) -> np.ndarray:
    """Convert a feature dict to a numpy array following the canonical feature order."""
    return np.array(
        [feat_dict.get(name, 0.0) for name in feature_names], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Logistic Regression (numpy-only)
# ---------------------------------------------------------------------------


class _LogisticRegression:
    """
    Binary logistic regression trained via mini-batch gradient descent.

    Implements L2 regularization.  Feature standardization is done
    internally (stores mean/std for inference).
    """

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self.is_trained: bool = False
        self._train_metrics: dict = {}

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize model state to a JSON-safe dict."""
        return {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "metrics": self._train_metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_LogisticRegression":
        """Deserialize from a dict."""
        model = cls()
        model.weights = (
            np.array(d["weights"], dtype=np.float64)
            if d.get("weights") is not None
            else None
        )
        model.bias = float(d.get("bias", 0.0))
        model.mean = (
            np.array(d["mean"], dtype=np.float64) if d.get("mean") is not None else None
        )
        model.std = (
            np.array(d["std"], dtype=np.float64) if d.get("std") is not None else None
        )
        model.feature_names = d.get("feature_names", [])
        model.is_trained = d.get("is_trained", False)
        model._train_metrics = d.get("metrics", {})
        return model

    # -- training -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the model.

        Parameters
        ----------
        X : (n_samples, n_features) array of features
        y : (n_samples,) binary labels (0 or 1)

        Returns
        -------
        dict with train/test metrics
        """
        n_samples, n_features = X.shape

        # Standardize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < 1e-12] = 1.0
        X_norm = (X - self.mean) / self.std

        # Train/test split (deterministic based on sample count)
        n_test = max(1, int(n_samples * _TEST_SPLIT_RATIO))
        # Use last samples as test (they are chronologically latest)
        X_train, X_test = X_norm[:-n_test], X_norm[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        n_train = X_train.shape[0]

        # Initialize weights
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        # Gradient descent
        prev_loss = float("inf")
        for iteration in range(_MAX_ITERATIONS):
            # Forward pass
            z = X_train @ self.weights + self.bias
            predictions = _sigmoid(z)

            # Binary cross-entropy + L2 regularization
            eps = 1e-15
            loss = -np.mean(
                y_train * np.log(predictions + eps)
                + (1 - y_train) * np.log(1 - predictions + eps)
            ) + _REGULARIZATION * np.sum(self.weights**2)

            # Check convergence
            if abs(prev_loss - loss) < _CONVERGENCE_TOL:
                logger.info(
                    "Training converged",
                    iteration=iteration,
                    loss=round(loss, 6),
                )
                break
            prev_loss = loss

            # Gradients
            error = predictions - y_train
            dw = (X_train.T @ error) / n_train + 2 * _REGULARIZATION * self.weights
            db = np.mean(error)

            # Update
            self.weights -= _LEARNING_RATE * dw
            self.bias -= _LEARNING_RATE * db

        self.is_trained = True

        # Evaluate on test set
        metrics = self._evaluate(X_test, y_test)
        # Also compute train metrics
        train_metrics = self._evaluate(X_train, y_train)
        metrics["train_accuracy"] = train_metrics["accuracy"]
        metrics["train_samples"] = int(n_train)
        metrics["test_samples"] = int(n_test)
        metrics["total_samples"] = int(n_samples)
        metrics["iterations"] = (
            iteration + 1 if "iteration" in dir() else _MAX_ITERATIONS
        )
        self._train_metrics = metrics

        return metrics

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute classification metrics on a dataset."""
        if len(y) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        z = X @ self.weights + self.bias
        probs = _sigmoid(z)
        preds = (probs >= 0.5).astype(np.float64)

        tp = float(np.sum((preds == 1) & (y == 1)))
        fp = float(np.sum((preds == 1) & (y == 0)))
        fn = float(np.sum((preds == 0) & (y == 1)))
        tn = float(np.sum((preds == 0) & (y == 0)))

        accuracy = (tp + tn) / max(len(y), 1)
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # -- inference ----------------------------------------------------------

    def predict_proba(self, x: np.ndarray) -> float:
        """
        Predict probability that a single sample is truly profitable.

        Parameters
        ----------
        x : (n_features,) raw feature vector (will be standardized internally)

        Returns
        -------
        float in [0, 1]
        """
        if not self.is_trained or self.weights is None:
            return 0.5  # neutral when untrained
        x_norm = (x - self.mean) / self.std
        z = float(np.dot(self.weights, x_norm) + self.bias)
        return float(_sigmoid(np.array([z]))[0])

    def feature_importances(self) -> dict[str, float]:
        """
        Return a dict of feature_name -> importance score.

        Uses absolute weight magnitude (after standardization) as a
        simple proxy for importance.
        """
        if not self.is_trained or self.weights is None:
            return {}
        abs_w = np.abs(self.weights)
        total = float(np.sum(abs_w))
        if total < 1e-12:
            return {name: 0.0 for name in self.feature_names}
        importances = abs_w / total
        return {
            name: round(float(imp), 4)
            for name, imp in zip(self.feature_names, importances)
        }


# ---------------------------------------------------------------------------
# MLClassifier service (singleton)
# ---------------------------------------------------------------------------


class MLClassifier:
    """
    Machine-learning false-positive classifier for arbitrage opportunities.

    Singleton service that:
    - Extracts features from ArbitrageOpportunity objects
    - Trains a logistic regression model on OpportunityHistory data
    - Predicts whether a new opportunity is truly profitable
    - Persists model weights in the database
    - Logs predictions for auditing and future retraining
    """

    def __init__(self):
        self._model: _LogisticRegression = _LogisticRegression()
        self._feature_names: list[str] = _build_feature_names()
        self._model.feature_names = self._feature_names
        self._model_version: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Load the latest model weights from the database.

        If no model is found, the classifier starts in cold-start mode
        and returns neutral predictions until ``train_model`` is called.
        """
        if self._initialized:
            return

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(MLModelWeights)
                    .where(MLModelWeights.is_active == True)  # noqa: E712
                    .order_by(desc(MLModelWeights.created_at))
                    .limit(1)
                )
                row = result.scalar_one_or_none()

                if row is not None:
                    self._model = _LogisticRegression.from_dict(row.weights)
                    self._model.feature_names = self._feature_names
                    self._model_version = row.model_version
                    logger.info(
                        "ML model loaded from database",
                        version=row.model_version,
                        training_samples=row.training_samples,
                    )
                else:
                    logger.warning(
                        "No trained ML model found in database; operating in cold-start mode "
                        "(all predictions will return probability=0.5)"
                    )
        except Exception as exc:
            logger.error("Failed to load ML model from database", error=str(exc))

        self._initialized = True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def train_model(self) -> dict:
        """
        Train (or retrain) the classifier on historical opportunity data.

        Returns
        -------
        dict with training metrics or an error description.
        """
        logger.info("Starting ML model training")

        try:
            async with AsyncSessionLocal() as session:
                # Load resolved opportunities (ones where we know the outcome)
                result = await session.execute(
                    select(OpportunityHistory)
                    .where(OpportunityHistory.was_profitable.isnot(None))
                    .order_by(OpportunityHistory.detected_at.asc())
                )
                rows = result.scalars().all()

            n_rows = len(rows)
            logger.info("Loaded historical opportunities for training", count=n_rows)

            if n_rows < _MIN_TRAINING_SAMPLES:
                msg = (
                    f"Insufficient training data: {n_rows} samples "
                    f"(minimum {_MIN_TRAINING_SAMPLES} required). "
                    "Model remains in cold-start mode."
                )
                logger.warning(msg)
                return {
                    "status": "insufficient_data",
                    "samples": n_rows,
                    "message": msg,
                }

            # Build feature matrix
            X_list = []
            y_list = []
            for row in rows:
                feat_dict = _extract_features_from_history(row)
                feat_array = _features_dict_to_array(feat_dict, self._feature_names)
                X_list.append(feat_array)
                y_list.append(1.0 if row.was_profitable else 0.0)

            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)

            # Train
            metrics = self._model.fit(X, y)
            self._model_version += 1

            # Persist to database
            await self._save_model(metrics, n_rows)

            logger.info(
                "ML model training complete",
                version=self._model_version,
                accuracy=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                samples=n_rows,
            )

            return {
                "status": "trained",
                "model_version": self._model_version,
                "metrics": metrics,
            }

        except Exception as exc:
            logger.error("ML model training failed", error=str(exc))
            return {"status": "error", "message": str(exc)}

    async def _save_model(self, metrics: dict, training_samples: int) -> None:
        """Persist model weights to the database."""
        try:
            async with AsyncSessionLocal() as session:
                # Deactivate previous models
                await session.execute(
                    update(MLModelWeights)
                    .where(MLModelWeights.is_active == True)  # noqa: E712
                    .values(is_active=False)
                )

                # Insert new model
                model_row = MLModelWeights(
                    id=str(uuid.uuid4()),
                    model_version=self._model_version,
                    weights=self._model.to_dict(),
                    feature_names=self._feature_names,
                    metrics=metrics,
                    training_samples=training_samples,
                    is_active=True,
                )
                session.add(model_row)
                await session.commit()

            logger.info(
                "ML model weights saved to database",
                version=self._model_version,
            )
        except Exception as exc:
            logger.error("Failed to save ML model weights", error=str(exc))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict(self, opportunity: ArbitrageOpportunity) -> dict:
        """
        Predict whether an opportunity is truly profitable.

        Parameters
        ----------
        opportunity : ArbitrageOpportunity

        Returns
        -------
        dict with keys:
            probability  - float in [0, 1]
            recommendation - "execute" | "skip" | "review"
            confidence   - float in [0, 1]
            feature_importances - dict[str, float]
        """
        await self.initialize()

        feat_dict = extract_features(opportunity)
        feat_array = _features_dict_to_array(feat_dict, self._feature_names)

        probability = self._model.predict_proba(feat_array)

        # Recommendation
        if probability >= _EXECUTE_THRESHOLD:
            recommendation = "execute"
        elif probability <= _SKIP_THRESHOLD:
            recommendation = "skip"
        else:
            recommendation = "review"

        # Confidence: distance from the uncertain midpoint (0.5)
        confidence = round(abs(probability - 0.5) * 2.0, 4)

        importances = self._model.feature_importances()

        # Top features driving this particular prediction
        if self._model.is_trained and self._model.weights is not None:
            x_norm = (feat_array - self._model.mean) / self._model.std
            contributions = self._model.weights * x_norm
            abs_contrib = np.abs(contributions)
            total_contrib = float(np.sum(abs_contrib))
            if total_contrib > 1e-12:
                per_feature = {
                    name: round(float(c / total_contrib), 4)
                    for name, c in zip(self._feature_names, abs_contrib)
                }
            else:
                per_feature = importances
        else:
            per_feature = importances

        result = {
            "probability": round(probability, 4),
            "recommendation": recommendation,
            "confidence": confidence,
            "feature_importances": per_feature,
        }

        # Log prediction asynchronously
        await self._log_prediction(opportunity, feat_dict, result)

        return result

    async def should_execute(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Simple boolean gate: should this opportunity be executed?

        Returns True if the predicted probability of profitability is
        at or above the execute threshold.
        """
        result = await self.predict(opportunity)
        return result["probability"] >= _EXECUTE_THRESHOLD

    async def filter_opportunity(
        self, opportunity: ArbitrageOpportunity
    ) -> tuple[bool, str, dict]:
        """
        Integration point for auto_trader.

        Returns
        -------
        (should_trade, reason, prediction_details)
        """
        prediction = await self.predict(opportunity)
        should_trade = prediction["probability"] >= _EXECUTE_THRESHOLD

        if should_trade:
            reason = (
                f"ML classifier approves (prob={prediction['probability']:.2f}, "
                f"confidence={prediction['confidence']:.2f})"
            )
        elif prediction["recommendation"] == "review":
            reason = (
                f"ML classifier suggests review (prob={prediction['probability']:.2f}, "
                f"confidence={prediction['confidence']:.2f})"
            )
        else:
            reason = (
                f"ML classifier rejects (prob={prediction['probability']:.2f}, "
                f"confidence={prediction['confidence']:.2f})"
            )

        return should_trade, reason, prediction

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    async def _log_prediction(
        self, opp: ArbitrageOpportunity, features: dict, result: dict
    ) -> None:
        """Write a prediction record to the database for auditing."""
        try:
            async with AsyncSessionLocal() as session:
                log_entry = MLPredictionLog(
                    id=str(uuid.uuid4()),
                    opportunity_id=opp.id,
                    strategy_type=opp.strategy.value if opp.strategy else "unknown",
                    features=features,
                    probability=result["probability"],
                    recommendation=result["recommendation"],
                    confidence=result["confidence"],
                    model_version=self._model_version
                    if self._model.is_trained
                    else None,
                )
                session.add(log_entry)
                await session.commit()
        except Exception as exc:
            # Logging failures must not break the prediction path
            logger.error("Failed to log ML prediction", error=str(exc))

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def get_model_stats(self) -> dict:
        """
        Return current model statistics.

        Returns
        -------
        dict with model version, training metrics, feature importances,
        and training data size.
        """
        await self.initialize()

        stats: dict = {
            "is_trained": self._model.is_trained,
            "model_version": self._model_version,
            "feature_names": self._feature_names,
        }

        if self._model.is_trained:
            stats["metrics"] = self._model._train_metrics
            stats["feature_importances"] = self._model.feature_importances()
        else:
            stats["metrics"] = {}
            stats["feature_importances"] = {}
            stats["message"] = (
                "Model is in cold-start mode. Call train_model() to train."
            )

        # Count historical records available for training
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(func.count(OpportunityHistory.id)).where(
                        OpportunityHistory.was_profitable.isnot(None)
                    )
                )
                stats["available_training_samples"] = result.scalar() or 0
        except Exception:
            stats["available_training_samples"] = "unknown"

        return stats

    async def get_predictions_log(self, limit: int = 50) -> list[dict]:
        """
        Return recent predictions with their outcomes (if available).

        Parameters
        ----------
        limit : int
            Maximum number of records to return (default 50).

        Returns
        -------
        list of dicts, newest first.
        """
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(MLPredictionLog)
                    .order_by(desc(MLPredictionLog.predicted_at))
                    .limit(limit)
                )
                rows = result.scalars().all()

                return [
                    {
                        "id": row.id,
                        "opportunity_id": row.opportunity_id,
                        "strategy_type": row.strategy_type,
                        "probability": row.probability,
                        "recommendation": row.recommendation,
                        "confidence": row.confidence,
                        "model_version": row.model_version,
                        "predicted_at": row.predicted_at.isoformat()
                        if row.predicted_at
                        else None,
                        "actual_outcome": row.actual_outcome,
                        "actual_roi": row.actual_roi,
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Failed to fetch prediction log", error=str(exc))
            return []

    async def update_prediction_outcome(
        self, opportunity_id: str, was_profitable: bool, actual_roi: float
    ) -> None:
        """
        Update prediction log entries with actual outcomes for future analysis.
        """
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(MLPredictionLog)
                    .where(MLPredictionLog.opportunity_id == opportunity_id)
                    .values(actual_outcome=was_profitable, actual_roi=actual_roi)
                )
                await session.commit()
                logger.info(
                    "Updated prediction outcome",
                    opportunity_id=opportunity_id,
                    was_profitable=was_profitable,
                )
        except Exception as exc:
            logger.error("Failed to update prediction outcome", error=str(exc))


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

ml_classifier = MLClassifier()
