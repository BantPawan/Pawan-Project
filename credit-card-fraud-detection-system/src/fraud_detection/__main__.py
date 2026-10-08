"""Run the local fraud workflow from the project root.

The commands follow the same order as the learning notebooks:

1. ``prepare`` validates, joins and splits the raw data.
2. ``compare`` selects a candidate using training folds and validation.
3. ``study`` promotes a saved comparison into a detailed staged study.
4. ``finalize`` assesses that saved candidate without fitting it again.
5. ``predict`` scores raw rows with a saved pipeline or a registered model.

``train`` combines comparison and final assessment for a convenient local
run. Use ``python -m fraud_detection --help`` to see the available options.

Optional MLflow tracking is enabled with ``--track``. Local JSON, CSV and
joblib artifacts remain the source of truth; MLflow adds an index and a
registry on top of them.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .data import (
    PROJECT_ROOT,
    calculate_file_sha256,
    chronological_split,
    merge_transaction_identity,
    read_table,
)
from .data_preparation import (
    DataPreparation,
    load_prepared_data,
    validate_raw_manifest,
)
from .models import (
    generated_fixture,
    train_candidate,
    save_results,
    load_pipeline,
)
from .experiments import (
    compare_candidates,
    default_candidates,
    save_comparison,
    load_comparison,
    validate_comparison_contract,
    evaluate_selected_comparison,
)
from .predict import FraudPredictor


# ---------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------

def add_data_options(parser):
    """Give each independently runnable stage the same data boundaries."""
    parser.add_argument("--smoke", action="store_true", help="Generated fixture only")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--prepared-folder", type=Path, help="Consume checked preparation exports"
    )
    parser.add_argument(
        "--max-rows", type=int, help="Earliest N genuine rows; sampled scope"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--train-end", type=float)
    parser.add_argument("--validation-end", type=float)


def add_selection_options(parser):
    """Expose only the decisions selection can change on a checkpoint."""
    parser.add_argument(
        "--selection-method",
        choices=[
            "basic",
            "target_correlation",
            "random_forest",
            "lightgbm",
            "gradient_boosting",
            "permutation",
            "l1",
            "rfe",
            "shap",
            "consensus",
        ],
        default="basic",
    )
    parser.add_argument("--selection-max-features", type=int)
    parser.add_argument(
        "--selection-consensus-components",
        nargs="+",
        choices=[
            "target_correlation",
            "random_forest",
            "lightgbm",
            "gradient_boosting",
            "permutation",
            "l1",
            "rfe",
            "shap",
        ],
        default=("target_correlation", "random_forest", "l1"),
        help="Explicit ranking methods for a consensus selection recipe",
    )
    parser.add_argument("--preserve-features", nargs="*", default=[])


def add_feature_options(parser):
    """Describe a recipe that a complete pipeline can refit per fold."""
    add_selection_options(parser)
    parser.add_argument(
        "--encoding", choices=["auto", "legacy", "onehot"], default="auto"
    )
    parser.add_argument("--feature-groups", choices=["all", "core"], default="all")
    parser.add_argument("--n-clusters", type=int, default=0)
    parser.add_argument(
        "--raw-screen-strategy", choices=["all", "strategic"], default="all"
    )
    parser.add_argument("--top-v-features", type=int)


def add_tracking_options(parser, *, allow_register):
    """Optional MLflow tracking, and registration where it is meaningful."""
    parser.add_argument(
        "--track", action="store_true", help="Log this stage to MLflow"
    )
    if allow_register:
        parser.add_argument(
            "--register",
            action="store_true",
            help="Register the logged model and move it to Staging",
        )
    parser.add_argument(
        "--tracking-uri",
        help="Override the default SQLite tracking URI",
    )


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

def build_parser():
    """Describe each workflow stage and its explicit data/model choices."""
    parser = argparse.ArgumentParser(description="Local chronological fraud workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="Validate raw files, join, split and save v2 partitions"
    )
    add_data_options(prepare)

    study = commands.add_parser(
        "study",
        help="Shortlist, run detail CV and tune before freezing the winner",
    )
    study.add_argument("--comparison-folder", type=Path, required=True)
    study.add_argument("--detail-folds", type=int, default=5)
    study.add_argument("--shortlist-size", type=int, default=3)
    study.add_argument("--detail-estimators", type=int, default=160)
    study.add_argument("--no-tune", action="store_true")
    study.add_argument("--no-ensemble", action="store_true")
    study.add_argument("--output-dir", type=Path)

    train = commands.add_parser(
        "train",
        help="Compare four models, assess the winner, save and optionally track",
    )
    add_data_options(train)
    add_feature_options(train)
    train.add_argument("--cv-folds", type=int, default=3)
    train.add_argument(
        "--no-tune",
        action="store_true",
        help="Leave bounded tuning for the detailed study",
    )
    train.add_argument("--output-dir", type=Path)
    train.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost", "lightgbm"],
        help="Explicit single-model experiment; default compares all four",
    )
    add_tracking_options(train, allow_register=True)

    compare = commands.add_parser(
        "compare",
        help="Training CV/tuning and validation selection; no holdout scores",
    )
    add_data_options(compare)
    add_feature_options(compare)
    compare.add_argument("--cv-folds", type=int, default=3)
    compare.add_argument("--no-tune", action="store_true")
    compare.add_argument("--output-dir", type=Path)
    add_tracking_options(compare, allow_register=False)

    finalize = commands.add_parser(
        "finalize",
        help="Assess and optionally register the exact persisted comparison winner",
    )
    finalize.add_argument("--comparison-folder", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path)
    add_tracking_options(finalize, allow_register=True)

    predict = commands.add_parser(
        "predict", help="Score joined raw rows with a complete model"
    )
    model = predict.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path")
    model.add_argument("--model-uri")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--tracking-uri")

    server = commands.add_parser(
        "mlflow-server", help="Start loopback-only SQLite tracking UI"
    )
    server.add_argument("--port", type=int, default=5000)
    server.add_argument("--tracking-uri")

    return parser


# ---------------------------------------------------------------------
# Settings and recipe helpers
# ---------------------------------------------------------------------

def workflow_settings(args):
    """Record data-loading choices so later stages reconstruct the same rows."""
    settings = {
        name: getattr(args, name)
        for name in (
            "smoke",
            "max_rows",
            "seed",
            "train_fraction",
            "validation_fraction",
            "train_end",
            "validation_end",
        )
    }
    settings["raw_dir"] = str(args.raw_dir.resolve())

    prepared = getattr(args, "prepared_folder", None)
    settings["prepared_folder"] = str(prepared.resolve()) if prepared else None
    return settings


def candidate_feature_options(args, partitions):
    """Translate CLI options into the recipe each pipeline refits from raw.

    Every option comes directly from the command line: this build has no
    intermediate engineering or selection checkpoints, so the recipe can
    never drift from a saved stage. ``partitions`` is accepted to keep the
    call signature stable for future checkpoint support.
    """
    del partitions  # reserved for a future checkpoint contract
    return {
        "n_clusters": args.n_clusters,
        "encoding": args.encoding,
        "selection_method": args.selection_method,
        "selection_max_features": args.selection_max_features,
        "selection_consensus_components": tuple(args.selection_consensus_components),
        "feature_groups": args.feature_groups,
        "raw_screen_strategy": args.raw_screen_strategy,
        "top_v_features": args.top_v_features,
        "preserve_features": tuple(args.preserve_features),
    }


# ---------------------------------------------------------------------
# Partition loading
# ---------------------------------------------------------------------

def load_training_partitions(args):
    """Load raw rows and split them before any learned transformation.

    A later ``finalize`` call uses the stored settings to reconstruct the same
    partitions. File hashes and the sample manifest help detect changed
    inputs. Generated fixtures are kept separate from genuine sample and
    full-data scopes.
    """
    prepared_folder = getattr(args, "prepared_folder", None)

    if prepared_folder:
        if (
            args.max_rows is not None
            or args.train_end is not None
            or args.validation_end is not None
            or args.train_fraction != 0.6
            or args.validation_fraction != 0.2
        ):
            raise ValueError(
                "Prepared partitions already define row/time boundaries; "
                "do not resplit them"
            )

        prepared = load_prepared_data(prepared_folder)
        scope = prepared["manifest"]["data_scope"]

        if args.smoke and scope != "generated_fixture":
            raise ValueError(
                "A genuine preparation checkpoint cannot be used as smoke data"
            )

        hashes = {
            name: record["sha256"]
            for name, record in prepared["manifest"]["files"].items()
        }
        hashes["prepared_manifest"] = calculate_file_sha256(
            Path(prepared_folder) / "prepared_manifest.json"
        )

        return prepared["partitions"], scope, hashes, prepared["quality_report"]

    if args.smoke and args.max_rows is not None:
        raise ValueError(
            "--max-rows applies to real files; --smoke uses a fixed generated fixture"
        )

    source_files = {}
    quality_report = {}

    if args.smoke:
        pool = generated_fixture(seed=args.seed)
        scope = "generated_fixture"
    else:
        raw_directory = Path(args.raw_dir)
        transaction_path = raw_directory / "train_transaction.csv"
        identity_path = raw_directory / "train_identity.csv"
        paths = [transaction_path, identity_path]

        transactions = read_table(transaction_path)
        identity = read_table(identity_path)

        pool, quality_report = merge_transaction_identity(
            transactions, identity, labeled=True
        )
        source_files = {path.name: calculate_file_sha256(path) for path in paths}

        scope = "real_full"
        manifest = validate_raw_manifest(
            raw_directory, tuple(path.name for path in paths)
        )
        if manifest:
            scope = manifest["data_scope"]
            quality_report["source_manifest"] = manifest

        if args.max_rows is not None:
            if args.max_rows < 30:
                raise ValueError("--max-rows must be at least 30")

            pool = (
                pool.sort_values(["TransactionDT", "TransactionID"])
                .head(args.max_rows)
                .copy()
            )
            scope = "real_sample"

    partitions = chronological_split(
        pool,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    for name in ("train", "validation", "holdout"):
        getattr(partitions, name).attrs["data_scope"] = scope

    return partitions, scope, source_files, quality_report


# ---------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------

def save_workflow_result(
    result, partitions, args, scope, comparison=None, comparison_folder=None
):
    """Save the assessed pipeline, verify its reload, and record the evidence.

    Reports contain aggregate diagnostics and artifact identifiers. MLflow
    logging happens only after the saved pipeline reproduces its predictions.
    This function does not fit a model or change the threshold.
    """
    saved = save_results(result, args.output_dir)

    # Verify the saved artifact before logging or registration.
    reloaded_pipeline = load_pipeline(saved)
    saved_probabilities = reloaded_pipeline.predict_proba(result.validation_inputs)
    assessed_probabilities = result.pipeline.predict_proba(result.validation_inputs)
    np.testing.assert_allclose(saved_probabilities, assessed_probabilities)

    report = {
        "data_scope": scope,
        "pipeline_path": str(saved.resolve()),
        "artifact_id": result.metadata["artifact_id"],
        "threshold": result.threshold,
        "metrics": result.metrics,
        "saved_load_verified": True,
    }

    from .evaluation import save_evaluation_reports

    report["evaluation_reports"] = save_evaluation_reports(
        result, partitions, saved.parent, scope
    )

    if comparison is not None:
        report.update(
            winner_name=comparison.winner_name,
            selection_criterion=comparison.metadata["selection_criterion"],
            comparison_folder=str(comparison_folder.resolve()),
            comparison_table=comparison.table.to_dict(orient="records"),
            skipped_candidates=comparison.skipped,
        )

    if getattr(args, "track", False):
        from .tracking import log_result, DEFAULT_REGISTERED_MODEL

        register_name = (
            DEFAULT_REGISTERED_MODEL if getattr(args, "register", False) else None
        )
        register_stage = "Staging" if register_name else None

        report["tracking"] = {
            "run_id": log_result(
                result=result,
                partitions=partitions,
                saved_pipeline_path=saved,
                folder=saved.parent,
                comparison=comparison,
                data_scope=scope,
                tracking_uri=getattr(args, "tracking_uri", None),
                register_name=register_name,
                register_stage=register_stage,
            ),
            "registered_name": register_name,
            "registered_stage": register_stage,
        }

    folder = PROJECT_ROOT / "reports" / "generated"
    folder.mkdir(parents=True, exist_ok=True)

    report_path = folder / f"workflow_{result.metadata['artifact_id']}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


# ---------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------

def _run_prepare(args):
    if args.prepared_folder:
        raise ValueError(
            "prepare reads raw sources; later stages consume --prepared-folder"
        )

    if args.smoke and args.max_rows is not None:
        raise ValueError(
            "--max-rows applies to genuine sources, not smoke fixtures"
        )

    preparation = DataPreparation(raw_dir=args.raw_dir)

    if args.smoke:
        pool = generated_fixture(seed=args.seed)
        kaggle_test = generated_fixture(n_rows=80, seed=args.seed + 1).drop(
            columns="isFraud"
        )
        kaggle_test["TransactionID"] += 1_000_000
        scope = "generated_fixture"
        preparation.quality_report = {
            "data_scope": scope,
            "note": "Explicit generated fixture; no raw files read",
        }
    else:
        pool, kaggle_test = preparation.merge_datasets(*preparation.load_data())
        scope = preparation.data_scope

        if args.max_rows is not None:
            if args.max_rows < 30:
                raise ValueError("--max-rows must be at least 30")
            pool = (
                pool.sort_values(["TransactionDT", "TransactionID"])
                .head(args.max_rows)
                .copy()
            )
            scope = "real_sample"
            preparation.quality_report["data_scope"] = scope

    partitions = chronological_split(
        pool,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    preparation.save_data(pool, kaggle_test, partitions, data_scope=scope)

    print(
        json.dumps(
            {"splits": partitions.report, "outputs": preparation.output_paths},
            indent=2,
            default=str,
        )
    )
    return 0


def _run_predict(args):
    if args.model_uri:
        from .tracking import configure_tracking

        configure_tracking(args.tracking_uri)

    model_location = args.model_path or args.model_uri
    predictor = FraudPredictor(model_location).load_model()
    results = predictor.predict_batch(args.input, args.output)

    print(
        json.dumps(
            {"rows": len(results), "output": str(Path(args.output).resolve())}
        )
    )
    return 0


def _run_study(args):
    from .experiments import staged_model_study

    baseline = load_comparison(args.comparison_folder)
    if "workflow_settings" not in baseline.metadata:
        raise ValueError(
            "CLI study needs a comparison that recorded its data settings; "
            "run `compare` from the CLI first"
        )

    settings = argparse.Namespace(**baseline.metadata["workflow_settings"])
    partitions, scope, source_files, quality = load_training_partitions(settings)

    if source_files != baseline.metadata["source_file_sha256"]:
        raise ValueError("Source files changed since baseline comparison")

    study_result = staged_model_study(
        partitions,
        screening_comparison=baseline,
        seed=baseline.metadata["seed"],
        screening_folds=baseline.metadata["n_splits"],
        detail_folds=args.detail_folds,
        shortlist_size=args.shortlist_size,
        detail_estimators=args.detail_estimators,
        tune=not args.no_tune,
        data_scope=scope,
        include_ensemble=not args.no_ensemble,
    )
    study_result.metadata.update(
        workflow_settings=baseline.metadata["workflow_settings"],
        source_file_sha256=source_files,
        join_quality_report=quality,
        feature_recipe=baseline.metadata.get("feature_recipe"),
    )

    for result in study_result.results.values():
        result.metadata.update(
            source_file_sha256=source_files,
            join_quality_report=quality,
        )

    folder = save_comparison(study_result, args.output_dir)

    print(
        json.dumps(
            {
                "study_folder": str(folder.resolve()),
                "data_scope": scope,
                "winner_name": study_result.winner_name,
                "winner_artifact_id": study_result.winner.metadata["artifact_id"],
                "comparison_table": study_result.table.to_dict(orient="records"),
                "stage_budgets": study_result.metadata["stage_budgets"],
                "shortlist": study_result.metadata["shortlist"],
                "ensemble_policy": study_result.metadata["ensemble_policy"],
                "holdout_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


def _run_finalize(args):
    comparison = load_comparison(args.comparison_folder)
    if "workflow_settings" not in comparison.metadata:
        raise ValueError(
            "This handoff has no CLI data settings; rerun `compare` from the CLI"
        )

    settings = argparse.Namespace(**comparison.metadata["workflow_settings"])
    partitions, scope, source_files, _ = load_training_partitions(settings)

    if source_files != comparison.metadata["source_file_sha256"]:
        raise ValueError(
            "Source files changed since comparison; rerun the comparison"
        )

    validate_comparison_contract(comparison, partitions, scope)
    result = evaluate_selected_comparison(comparison, partitions)

    # Preserve the unassessed handoff and save an assessed snapshot. Reuse it
    # for repeated finalize calls to return cached holdout instead of rescoring.
    folder = save_comparison(comparison, args.output_dir)
    return save_workflow_result(
        result, partitions, args, scope, comparison, folder
    )


def _run_train_or_compare(args):
    partitions, scope, source_files, quality = load_training_partitions(args)
    feature_options = candidate_feature_options(args, partitions)

    if args.command == "train" and args.model:
        result = train_candidate(
            partitions,
            seed=args.seed,
            model=args.model,
            **feature_options,
        )
        result.metadata.update(
            data_scope=scope,
            source_file_sha256=source_files,
            join_quality_report=quality,
            selection_criterion="explicit single-model experiment",
        )
        return save_workflow_result(result, partitions, args, scope)

    candidates = default_candidates(smoke=scope == "generated_fixture")

    for config in candidates.values():
        options = dict(feature_options)
        if options["encoding"] == "auto":
            options.pop("encoding")
        config.update(options)

    comparison = compare_candidates(
        partitions,
        candidates,
        seed=args.seed,
        n_splits=args.cv_folds,
        tune=not args.no_tune,
        data_scope=scope,
    )
    comparison.metadata.update(
        workflow_settings=workflow_settings(args),
        source_file_sha256=source_files,
        join_quality_report=quality,
        feature_recipe=feature_options,
    )

    for result in comparison.results.values():
        result.metadata.update(
            source_file_sha256=source_files,
            join_quality_report=quality,
        )

    if args.command == "compare":
        folder = save_comparison(comparison, args.output_dir)

        if getattr(args, "track", False):
            from .tracking import log_comparison

            log_comparison(
                comparison,
                folder,
                tracking_uri=getattr(args, "tracking_uri", None),
            )

        print(
            json.dumps(
                {
                    "data_scope": scope,
                    "comparison_folder": str(folder.resolve()),
                    "winner_name": comparison.winner_name,
                    "selection_criterion": comparison.metadata["selection_criterion"],
                    "comparison_table": comparison.table.to_dict(orient="records"),
                    "skipped_candidates": comparison.skipped,
                    "holdout_evaluated": False,
                },
                indent=2,
            )
        )
        return 0

    result = evaluate_selected_comparison(comparison, partitions)
    folder = save_comparison(comparison, args.output_dir)
    return save_workflow_result(
        result, partitions, args, scope, comparison, folder
    )


def main(argv=None):
    """Dispatch one local stage and return its process exit code.

    Comparison uses validation to choose the model. Final assessment reloads
    the same fitted artifact and keeps losing candidates away from holdout
    metrics. An explicit ``--model`` is a single-family experiment rather
    than a comparison.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)

    if args.command == "prepare":
        return _run_prepare(args)

    if args.command == "mlflow-server":
        from .tracking import run_server

        return run_server(
            port=args.port,
            tracking_uri=getattr(args, "tracking_uri", None),
        )

    if args.command == "predict":
        return _run_predict(args)

    if args.command == "study":
        return _run_study(args)

    if getattr(args, "register", False) and not getattr(args, "track", False):
        raise ValueError("--register requires --track")

    if args.command == "finalize":
        return _run_finalize(args)

    return _run_train_or_compare(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Workflow blocked: {error}", file=sys.stderr)
        sys.exit(2)