"""Genetic Algorithm for Binary Feature Selection in HeartRiskAI.
Performs parsimonious feature selection for cardiovascular risk prediction,
maximizing Stratified Cross-Validation ROC-AUC while penalizing subset size.
Enforces clinical mandatory features (age, sex) and guarantees strict
isolation of the hold-out test set.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    TRAIN_DATA_PATH,
    SELECTED_FEATURES_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    ALL_FEATURE_COLUMNS,
    MANDATORY_FEATURES,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data_preprocessing import enforce_mandatory_features

logger = logging.getLogger(__name__)


class GeneticAlgorithmFeatureSelector:
    """Genetic Algorithm for parsimonious binary feature selection.
    
    Attributes:
        feature_names: List of all candidate feature names (length N).
        mandatory_features: Features that must always be selected (e.g. age, male).
        pop_size: Number of chromosomes in each generation.
        n_generations: Number of evolutionary generations to iterate.
        crossover_rate: Probability of crossover between parent pairs.
        mutation_rate: Probability of bit-flip mutation per gene.
        tournament_size: Number of candidates in tournament selection.
        n_elites: Number of best chromosomes preserved intact to the next generation.
        parsimony_penalty: Weight (alpha) penalizing feature count in fitness:
            Fitness = Mean_CV_ROC_AUC - alpha * (selected_features / total_features).
        cv_folds: Number of stratified cross-validation folds.
        random_state: Random seed for deterministic reproducibility.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        mandatory_features: Optional[List[str]] = None,
        pop_size: int = 30,
        n_generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.08,
        tournament_size: int = 3,
        n_elites: int = 2,
        parsimony_penalty: float = 0.02,
        cv_folds: int = CV_FOLDS,
        random_state: int = RANDOM_STATE,
    ):
        self.feature_names = list(feature_names) if feature_names else list(ALL_FEATURE_COLUMNS)
        self.mandatory_features = list(mandatory_features) if mandatory_features else list(MANDATORY_FEATURES)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.n_elites = n_elites
        self.parsimony_penalty = parsimony_penalty
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.n_features = len(self.feature_names)

        # Precompute mandatory feature indices
        self.mandatory_indices = [
            self.feature_names.index(f)
            for f in self.mandatory_features
            if f in self.feature_names
        ]

        # Memoization cache for fitness evaluation: bitstring -> (fitness, roc_auc)
        self._fitness_cache: Dict[str, Tuple[float, float]] = {}

        # Optimization results storage
        self.history_: List[Dict[str, Any]] = []
        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_features_: List[str] = []
        self.best_fitness_: float = -float("inf")
        self.best_auc_: float = 0.0

    def chromosome_to_features(self, chromosome: np.ndarray) -> List[str]:
        """Converts binary chromosome into a list of feature names with mandatory enforcement."""
        selected = [
            self.feature_names[i]
            for i in range(self.n_features)
            if chromosome[i] == 1
        ]
        return enforce_mandatory_features(selected, self.mandatory_features)

    def features_to_chromosome(self, features: List[str]) -> np.ndarray:
        """Encodes a list of feature names into a binary chromosome."""
        feature_set = set(features)
        chrom = np.zeros(self.n_features, dtype=int)
        for i, f in enumerate(self.feature_names):
            if f in feature_set:
                chrom[i] = 1
        return chrom

    def repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """Enforces mandatory feature bits and guards against all-zero selections."""
        c = chromosome.copy()
        # Always set mandatory bits to 1
        for idx in self.mandatory_indices:
            c[idx] = 1

        # If still entirely zeros (e.g. no mandatory features existed), set random bit
        if np.sum(c) == 0:
            rand_idx = self.rng.randint(0, self.n_features)
            c[rand_idx] = 1

        return c

    def initialize_population(self) -> List[np.ndarray]:
        """Creates initial binary population with diverse feature density and all-ones baseline."""
        population = []

        # Seed individual 0: all features selected (standard clinical baseline)
        all_ones = np.ones(self.n_features, dtype=int)
        population.append(all_ones)

        # Seed individual 1: only mandatory features selected (minimal clinical baseline)
        mandatory_only = np.zeros(self.n_features, dtype=int)
        for idx in self.mandatory_indices:
            mandatory_only[idx] = 1
        population.append(mandatory_only)

        # Remaining individuals: random Bernoulli initialization with varying probabilities
        for i in range(2, self.pop_size):
            p_active = self.rng.uniform(0.3, 0.7)
            chrom = (self.rng.rand(self.n_features) < p_active).astype(int)
            chrom = self.repair_chromosome(chrom)
            population.append(chrom)

        return population

    def evaluate_chromosome(
        self,
        chromosome: np.ndarray,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[float, float]:
        """Computes fitness using Stratified K-Fold CV on training data with parsimony penalty.
        
        Fitness = Mean_CV_ROC_AUC - alpha * (selected_count / total_count)
        
        Returns:
            Tuple of (fitness, mean_cv_roc_auc).
        """
        repaired = self.repair_chromosome(chromosome)
        bitstring = "".join(str(b) for b in repaired)

        if bitstring in self._fitness_cache:
            return self._fitness_cache[bitstring]

        active_features = self.chromosome_to_features(repaired)
        if len(active_features) == 0:
            return -1.0, 0.0

        X_sub = X_train[active_features]
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=self.random_state,
        )

        try:
            scores = cross_val_score(clf, X_sub, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
            mean_auc = float(np.mean(scores))
        except Exception as e:
            logger.warning("Error evaluating chromosome %s: %s", bitstring, e)
            mean_auc = 0.5

        # Parsimony penalty: proportion of features selected out of all features
        sparsity_penalty = self.parsimony_penalty * (len(active_features) / self.n_features)
        fitness = mean_auc - sparsity_penalty

        self._fitness_cache[bitstring] = (fitness, mean_auc)
        return fitness, mean_auc

    def select_parent(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection: picks tournament_size candidates and returns the fittest."""
        candidates_idx = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best_candidate_idx = candidates_idx[np.argmax([fitness_scores[i] for i in candidates_idx])]
        return population[best_candidate_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs uniform crossover with probability crossover_rate."""
        if self.rng.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        mask = self.rng.rand(self.n_features) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        return self.repair_chromosome(child1), self.repair_chromosome(child2)

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Applies bit-flip mutation with probability mutation_rate per non-mandatory gene."""
        mutated = chromosome.copy()
        for i in range(self.n_features):
            if i in self.mandatory_indices:
                continue  # Never mutate mandatory features to 0
            if self.rng.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return self.repair_chromosome(mutated)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True,
    ) -> "GeneticAlgorithmFeatureSelector":
        """Executes Genetic Algorithm feature selection strictly on training data."""
        if TARGET_COLUMN in X_train.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' cannot be present in X_train.")

        # Reset cache and state
        self._fitness_cache.clear()
        self.history_ = []
        self.best_fitness_ = -float("inf")

        if verbose:
            print(f"Initializing GA Population: size={self.pop_size}, generations={self.n_generations}")
            print(f"Total features: {self.n_features}, Mandatory features: {self.mandatory_features}")

        population = self.initialize_population()

        for gen in range(self.n_generations):
            # Evaluate all individuals
            eval_results = [
                self.evaluate_chromosome(ind, X_train, y_train)
                for ind in population
            ]
            fitness_scores = [res[0] for res in eval_results]
            auc_scores = [res[1] for res in eval_results]

            # Track generation statistics
            gen_best_idx = int(np.argmax(fitness_scores))
            gen_best_fitness = float(fitness_scores[gen_best_idx])
            gen_best_auc = float(auc_scores[gen_best_idx])
            gen_best_chrom = population[gen_best_idx]
            gen_best_feats = self.chromosome_to_features(gen_best_chrom)
            gen_mean_fitness = float(np.mean(fitness_scores))
            gen_mean_auc = float(np.mean(auc_scores))

            # Update overall best
            if gen_best_fitness > self.best_fitness_:
                self.best_fitness_ = gen_best_fitness
                self.best_auc_ = gen_best_auc
                self.best_chromosome_ = gen_best_chrom.copy()
                self.best_features_ = list(gen_best_feats)

            self.history_.append({
                "generation": gen,
                "best_fitness": round(gen_best_fitness, 5),
                "mean_fitness": round(gen_mean_fitness, 5),
                "best_auc": round(gen_best_auc, 5),
                "mean_auc": round(gen_mean_auc, 5),
                "best_feature_count": len(gen_best_feats),
                "best_features": list(gen_best_feats),
            })

            if verbose and (gen % 5 == 0 or gen == self.n_generations - 1):
                print(
                    f"Gen {gen:02d} | Best Fitness: {gen_best_fitness:.5f} | "
                    f"Best AUC: {gen_best_auc:.4f} | "
                    f"Features: {len(gen_best_feats)}/{self.n_features} | "
                    f"Mean Fit: {gen_mean_fitness:.5f}"
                )

            # Build next generation with Elitism
            sorted_indices = np.argsort(fitness_scores)[::-1]
            next_generation = [population[i].copy() for i in sorted_indices[: self.n_elites]]

            # Reproduction loop
            while len(next_generation) < self.pop_size:
                p1 = self.select_parent(population, fitness_scores)
                p2 = self.select_parent(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                next_generation.append(c1)
                if len(next_generation) < self.pop_size:
                    next_generation.append(c2)

            population = next_generation

        if verbose:
            print("\nGA Feature Selection Complete.")
            print(f"Optimal Subset ({len(self.best_features_)} features): {self.best_features_}")
            print(f"Best Fitness: {self.best_fitness_:.5f} | Internal CV ROC-AUC: {self.best_auc_:.4f}")

        return self

    def plot_convergence(self, output_path: Optional[Path] = None) -> Path:
        """Generates academic convergence plot of fitness and feature count across generations."""
        if not self.history_:
            raise ValueError("Cannot plot convergence before fitting the algorithm.")

        if output_path is None:
            output_path = FIGURES_DIR / "ga_convergence.png"

        df_hist = pd.DataFrame(self.history_)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        # Subplot 1: Fitness Progression
        ax1.plot(df_hist["generation"], df_hist["best_fitness"], "o-", color="#2b5c8f", label="Best Fitness", linewidth=2)
        ax1.plot(df_hist["generation"], df_hist["mean_fitness"], "--", color="#6baed6", label="Mean Fitness", linewidth=1.5)
        ax1.plot(df_hist["generation"], df_hist["best_auc"], ":", color="#2ca02c", label="Best CV ROC-AUC", linewidth=1.5)
        ax1.set_ylabel("Fitness Score")
        ax1.set_title("Genetic Algorithm Convergence: Fitness & ROC-AUC", fontsize=12, pad=10)
        ax1.legend(loc="lower right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Subplot 2: Feature Count Reduction
        ax2.plot(df_hist["generation"], df_hist["best_feature_count"], "s-", color="#d95f02", label="Fittest Subset Size", linewidth=2)
        ax2.axhline(y=self.n_features, color="gray", linestyle="--", label=f"Full Feature Set ({self.n_features})")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of Features")
        ax2.set_ylim(0, self.n_features + 2)
        ax2.set_title("Feature Subset Complexity Reduction", fontsize=12, pad=10)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    def save_results(
        self,
        json_path: Path = SELECTED_FEATURES_PATH,
        history_csv_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Saves selected feature subset and optimization history to disk."""
        if not self.best_features_:
            raise ValueError("No results to save; fit the GA first.")

        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        history_csv = history_csv_path or (REPORTS_DIR / "ga_history.csv")
        history_csv.parent.mkdir(parents=True, exist_ok=True)

        # Save history DataFrame
        pd.DataFrame(self.history_).to_csv(history_csv, index=False)

        results_data = {
            "selected_features": self.best_features_,
            "mandatory_features": self.mandatory_features,
            "all_features": self.feature_names,
            "best_fitness": round(self.best_fitness_, 5),
            "best_cv_roc_auc": round(self.best_auc_, 5),
            "feature_count": len(self.best_features_),
            "total_feature_count": self.n_features,
            "reduction_percentage": round(
                ((self.n_features - len(self.best_features_)) / self.n_features) * 100, 2
            ),
            "hyperparameters": {
                "pop_size": self.pop_size,
                "n_generations": self.n_generations,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "tournament_size": self.tournament_size,
                "n_elites": self.n_elites,
                "parsimony_penalty": self.parsimony_penalty,
                "cv_folds": self.cv_folds,
                "random_state": self.random_state,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        return results_data


def compare_ga_vs_full_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: List[str],
    all_features: Optional[List[str]] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Compares GA-selected feature subset against full feature set under identical CV protocol."""
    all_feats = list(all_features) if all_features else list(ALL_FEATURE_COLUMNS)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=random_state)

    # 1. Full feature set evaluation
    scores_full = cross_val_score(clf, X_train[all_feats], y_train, cv=cv, scoring="roc_auc")
    full_mean_auc = float(np.mean(scores_full))
    full_std_auc = float(np.std(scores_full))

    # 2. GA-selected feature set evaluation
    scores_ga = cross_val_score(clf, X_train[selected_features], y_train, cv=cv, scoring="roc_auc")
    ga_mean_auc = float(np.mean(scores_ga))
    ga_std_auc = float(np.std(scores_ga))

    comparison = {
        "full_features": {
            "feature_count": len(all_feats),
            "features": all_feats,
            "mean_cv_roc_auc": round(full_mean_auc, 4),
            "std_cv_roc_auc": round(full_std_auc, 4),
        },
        "ga_selected_features": {
            "feature_count": len(selected_features),
            "features": selected_features,
            "mean_cv_roc_auc": round(ga_mean_auc, 4),
            "std_cv_roc_auc": round(ga_std_auc, 4),
        },
        "auc_difference": round(ga_mean_auc - full_mean_auc, 4),
        "feature_reduction": {
            "removed_features_count": len(all_feats) - len(selected_features),
            "percentage_reduction": round(
                ((len(all_feats) - len(selected_features)) / len(all_feats)) * 100, 2
            ),
        },
    }
    return comparison


def main():
    """CLI execution for Genetic Algorithm feature selection."""
    print("=" * 70)
    print("HeartRiskAI: Genetic Algorithm Feature Selection Engine")
    print("=" * 70)

    if not TRAIN_DATA_PATH.exists():
        print(f"[ERROR] Preprocessed train data missing at {TRAIN_DATA_PATH}. Run src.data_preprocessing first.")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    print(f"Loaded training data: {X_train.shape} features from {TRAIN_DATA_PATH.name}")

    ga = GeneticAlgorithmFeatureSelector(
        feature_names=list(X_train.columns),
        mandatory_features=MANDATORY_FEATURES,
        pop_size=30,
        n_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.08,
        parsimony_penalty=0.015,
        random_state=RANDOM_STATE,
    )

    ga.fit(X_train, y_train, verbose=True)
    ga.plot_convergence()
    results = ga.save_results()

    print("\n" + "=" * 70)
    print("Comparative Validation: Full Feature Set vs GA-Selected Subset")
    print("=" * 70)

    comparison = compare_ga_vs_full_features(
        X_train, y_train, selected_features=ga.best_features_, all_features=list(X_train.columns)
    )

    print(f"Full Feature Set (15 features):   ROC-AUC = {comparison['full_features']['mean_cv_roc_auc']} +/- {comparison['full_features']['std_cv_roc_auc']}")
    print(f"GA Selected Set ({comparison['ga_selected_features']['feature_count']} features):    ROC-AUC = {comparison['ga_selected_features']['mean_cv_roc_auc']} +/- {comparison['ga_selected_features']['std_cv_roc_auc']}")
    print(f"AUC Difference:                   {comparison['auc_difference']:+.4f}")
    print(f"Complexity Reduction:             {comparison['feature_reduction']['percentage_reduction']}% fewer features")
    print(f"Mandatory features verified:      {MANDATORY_FEATURES}")
    print(f"Saved artifacts to:               {SELECTED_FEATURES_PATH}")


if __name__ == "__main__":
    main()
