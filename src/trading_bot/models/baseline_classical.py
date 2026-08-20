"""Classical CPU baselines over the common Phase 7 batch/objective interfaces."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable
from importlib import import_module
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from trading_bot.config import ObjectiveConfig
from trading_bot.models.baseline_common import (
    BaselineComplexity,
    BaselineInferenceBenchmark,
    BaselineTargetNames,
    benchmark_tabular_inference,
    collect_tabular_features,
    collect_tabular_training_data,
    objective_target_name,
    scalar_target_values,
)
from trading_bot.training.contracts import TrainingBatch
from trading_bot.training.predictions import PredictionRecord


class ClassicalBaselineError(RuntimeError):
    """Raised when a classical baseline is used outside its fitted/configured contract."""


class ClassicalBaseline(ABC):
    """Common fit/predict/checkpoint surface for non-PyTorch Phase 7 baselines."""

    family: ClassVar[str]
    direction_model: ClassVar[bool] = False

    def __init__(self, objective: ObjectiveConfig, *, seed: int = 42) -> None:
        if seed < 0:
            raise ValueError("baseline seed must be non-negative")
        self.objective = objective
        self.seed = seed
        self._validate_objective()
        self._estimator = self._build_estimator()
        self._fitted = False

    def fit(
        self,
        batches: Iterable[TrainingBatch],
        *,
        targets: BaselineTargetNames,
    ) -> None:
        """Fit from the exact same identity-preserving TrainingBatch interface as neural models."""
        target_name = objective_target_name(self.objective, targets)
        features, target_values = collect_tabular_training_data(batches, target_name=target_name)
        if self.direction_model:
            if not bool(np.isin(target_values, np.asarray([0.0, 1.0])).all()):
                raise ValueError("logistic baseline direction targets must be encoded as 0/1")
            if len(np.unique(target_values)) < 2:
                raise ValueError("logistic baseline requires both direction classes")
        self._fit_estimator(features, target_values)
        self._fitted = True

    def predict_records(
        self,
        batches: Iterable[TrainingBatch],
        *,
        targets: BaselineTargetNames,
    ) -> tuple[PredictionRecord, ...]:
        """Emit the same durable logical prediction records consumed by Phase 6."""
        self._require_fitted()
        materialized = tuple(batches)
        features = collect_tabular_features(materialized)
        scores = self._predict_score(features)
        expected_rows = sum(batch.batch_size for batch in materialized)
        scores = _score_vector(scores, expected_rows)
        records: list[PredictionRecord] = []
        cursor = 0
        for batch in materialized:
            returns = scalar_target_values(batch, targets.return_target)
            timestamps = batch.timestamps_ns.detach().cpu().tolist()
            for index in range(batch.batch_size):
                score = float(scores[cursor])
                cursor += 1
                records.append(
                    PredictionRecord(
                        asset_id=batch.asset_ids[index],
                        timestamp_ns=int(timestamps[index]),
                        target=float(returns[index]),
                        expected_return=None if self.direction_model else score,
                        rank_score=score,
                        direction_probability=score if self.direction_model else None,
                    )
                )
        return tuple(records)

    def parameter_count(self) -> int:
        """Return a documented learned-scalar count for the fitted estimator."""
        self._require_fitted()
        return self._parameter_count()

    def complexity(self) -> BaselineComplexity:
        """Return learned-scalar count and exact serialized fitted-state byte size."""
        payload = self.checkpoint_payload()
        return BaselineComplexity(
            learned_scalar_count=self.parameter_count(),
            serialized_bytes=len(payload),
        )

    def benchmark_inference(
        self,
        batches: Iterable[TrainingBatch],
        *,
        warmup: int = 2,
        iterations: int = 10,
    ) -> BaselineInferenceBenchmark:
        self._require_fitted()
        features = collect_tabular_features(batches)
        return benchmark_tabular_inference(
            self._predict_score,
            features,
            warmup=warmup,
            iterations=iterations,
        )

    def checkpoint_payload(self) -> bytes:
        """Serialize fitted estimator state for the checksummed classical checkpoint bundle."""
        self._require_fitted()
        return pickle.dumps(self._estimator, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_checkpoint_payload(self, payload: bytes) -> None:
        """Restore only after the caller has verified the checkpoint checksum/identity."""
        if not payload:
            raise ClassicalBaselineError("classical checkpoint payload must not be empty")
        try:
            estimator = pickle.loads(payload)
        except (pickle.PickleError, AttributeError, EOFError, ImportError, IndexError) as exc:
            raise ClassicalBaselineError(
                "unable to deserialize classical baseline checkpoint"
            ) from exc
        self._estimator = estimator
        self._fitted = True
        self._validate_restored_estimator()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ClassicalBaselineError(f"{self.family} baseline is not fitted")

    def _fit_estimator(
        self,
        features: NDArray[np.float32],
        targets: NDArray[np.float64],
    ) -> None:
        fit = getattr(self._estimator, "fit", None)
        if not callable(fit):
            raise ClassicalBaselineError(f"{self.family} estimator does not expose fit")
        fit(features, targets)

    def _validate_restored_estimator(self) -> None:
        if not callable(getattr(self._estimator, "predict", None)):
            raise ClassicalBaselineError("restored classical estimator does not expose predict")

    @abstractmethod
    def _validate_objective(self) -> None: ...

    @abstractmethod
    def _build_estimator(self) -> Any: ...

    @abstractmethod
    def _predict_score(self, features: NDArray[np.float32]) -> NDArray[np.float64]: ...

    @abstractmethod
    def _parameter_count(self) -> int: ...


class _SklearnLinearRegressionBaseline(ClassicalBaseline):
    """Shared learned-state accounting for sklearn linear estimators."""

    def _predict_score(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        prediction = cast(Any, self._estimator).predict(features)
        return np.asarray(prediction, dtype=np.float64).reshape(-1)

    def _parameter_count(self) -> int:
        estimator = cast(Any, self._estimator)
        coefficient = np.asarray(estimator.coef_)
        intercept = np.asarray(estimator.intercept_)
        return int(coefficient.size + intercept.size)


class RidgeBaseline(_SklearnLinearRegressionBaseline):
    family = "ridge"

    def __init__(
        self,
        objective: ObjectiveConfig,
        *,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        if alpha < 0:
            raise ValueError("ridge alpha must be non-negative")
        self.alpha = alpha
        super().__init__(objective, seed=seed)

    def _validate_objective(self) -> None:
        _require_regression_objective(self.objective, losses={"mse"})

    def _build_estimator(self) -> Any:
        linear_model = import_module("sklearn.linear_model")
        return linear_model.Ridge(alpha=self.alpha)


class ElasticNetBaseline(_SklearnLinearRegressionBaseline):
    family = "elastic_net"

    def __init__(
        self,
        objective: ObjectiveConfig,
        *,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if alpha <= 0 or not 0 <= l1_ratio <= 1:
            raise ValueError("Elastic Net requires alpha > 0 and l1_ratio in [0, 1]")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        super().__init__(objective, seed=seed)

    def _validate_objective(self) -> None:
        _require_regression_objective(self.objective, losses={"mse"})

    def _build_estimator(self) -> Any:
        linear_model = import_module("sklearn.linear_model")
        return linear_model.ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=2_000,
            random_state=self.seed,
            selection="cyclic",
        )


class LogisticDirectionBaseline(ClassicalBaseline):
    family = "logistic_direction"
    direction_model = True

    def __init__(
        self,
        objective: ObjectiveConfig,
        *,
        regularization_c: float = 1.0,
        seed: int = 42,
    ) -> None:
        if regularization_c <= 0:
            raise ValueError("logistic regularization C must be positive")
        self.regularization_c = regularization_c
        super().__init__(objective, seed=seed)

    def _validate_objective(self) -> None:
        if self.objective.kind != "direction" or self.objective.loss != "bce":
            raise ValueError("logistic baseline requires direction objective with bce loss")

    def _build_estimator(self) -> Any:
        linear_model = import_module("sklearn.linear_model")
        return linear_model.LogisticRegression(
            C=self.regularization_c,
            max_iter=500,
            random_state=self.seed,
            solver="lbfgs",
        )

    def _predict_score(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        probabilities = np.asarray(
            cast(Any, self._estimator).predict_proba(features),
            dtype=np.float64,
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ClassicalBaselineError("logistic baseline returned invalid class probabilities")
        return probabilities[:, 1]

    def _parameter_count(self) -> int:
        estimator = cast(Any, self._estimator)
        coefficient = np.asarray(estimator.coef_)
        intercept = np.asarray(estimator.intercept_)
        return int(coefficient.size + intercept.size)


class LightGBMBaseline(ClassicalBaseline):
    family = "lightgbm"

    def __init__(
        self,
        objective: ObjectiveConfig,
        *,
        n_estimators: int = 32,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        _validate_tree_hyperparameters(n_estimators, max_depth, learning_rate)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        super().__init__(objective, seed=seed)

    def _validate_objective(self) -> None:
        _require_regression_objective(self.objective, losses={"mse", "huber"})

    def _build_estimator(self) -> Any:
        lightgbm = import_module("lightgbm")
        objective = "huber" if self.objective.loss == "huber" else "regression"
        return lightgbm.LGBMRegressor(
            objective=objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=min(2**self.max_depth, 31),
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

    def _predict_score(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        prediction = cast(Any, self._estimator).predict(features)
        return np.asarray(prediction, dtype=np.float64).reshape(-1)

    def _parameter_count(self) -> int:
        booster = cast(Any, self._estimator).booster_
        model = cast(dict[str, Any], booster.dump_model())
        tree_info = cast(list[dict[str, Any]], model.get("tree_info", []))
        return sum(_lightgbm_node_count(tree["tree_structure"]) for tree in tree_info)


class XGBoostBaseline(ClassicalBaseline):
    family = "xgboost"

    def __init__(
        self,
        objective: ObjectiveConfig,
        *,
        n_estimators: int = 32,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        _validate_tree_hyperparameters(n_estimators, max_depth, learning_rate)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        super().__init__(objective, seed=seed)

    def _validate_objective(self) -> None:
        _require_regression_objective(self.objective, losses={"mse", "huber"})

    def _build_estimator(self) -> Any:
        xgboost = import_module("xgboost")
        objective = (
            "reg:pseudohubererror" if self.objective.loss == "huber" else "reg:squarederror"
        )
        return xgboost.XGBRegressor(
            objective=objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    def _predict_score(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        prediction = cast(Any, self._estimator).predict(features)
        return np.asarray(prediction, dtype=np.float64).reshape(-1)

    def _parameter_count(self) -> int:
        booster = cast(Any, self._estimator).get_booster()
        trees = cast(list[str], booster.get_dump(dump_format="json"))
        return sum(_xgboost_node_count(cast(dict[str, Any], json.loads(tree))) for tree in trees)


def _score_vector(values: NDArray[np.float64], expected_rows: int) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (expected_rows,) or not bool(np.isfinite(result).all()):
        raise ClassicalBaselineError("classical baseline produced invalid prediction scores")
    return result


def _require_regression_objective(
    objective: ObjectiveConfig,
    *,
    losses: set[str],
) -> None:
    if objective.kind != "excess_return" or objective.loss not in losses:
        raise ValueError(
            "regression baseline requires excess_return objective with one of "
            f"{sorted(losses)} losses"
        )


def _validate_tree_hyperparameters(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> None:
    if n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
        raise ValueError("tree baseline hyperparameters must be positive")


def _lightgbm_node_count(node: dict[str, Any]) -> int:
    if "left_child" not in node or "right_child" not in node:
        return 1
    return (
        1
        + _lightgbm_node_count(cast(dict[str, Any], node["left_child"]))
        + _lightgbm_node_count(cast(dict[str, Any], node["right_child"]))
    )


def _xgboost_node_count(node: dict[str, Any]) -> int:
    children = node.get("children")
    if not isinstance(children, list):
        return 1
    return 1 + sum(_xgboost_node_count(cast(dict[str, Any], child)) for child in children)
