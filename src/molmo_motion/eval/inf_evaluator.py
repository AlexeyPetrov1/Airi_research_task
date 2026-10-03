"""Class to evaluate models based on their generation outputs"""
import dataclasses
import itertools
import logging
import time
from collections import defaultdict
from typing import List, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics
from tqdm import tqdm

from molmo_motion._optional import wandb, WBValue
from .evaluators import HtmlTable, SavePredictions
from ..config import BaseConfig
from ..data.data_loader import DataLoaderConfig
from ..nn.beam_search import SamplingConfig, TopKSampler, TopPSampler, MultinomialSampler, \
    TopKTopPSampler, RepeatedNGramBlockingConstraint, RepetitionPenaltyConstraint, \
    FrequencyPenaltyConstraint
from ..torch_util import (
    get_global_rank,
    get_world_size,
    move_to_device, barrier,
)
from ..util import flatten_list

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InfEvaluator:
    """
    Evaluates the text outputs from a model on a task
    """
    metrics: List

    def __call__(self, predictions, example_metadata, tokenizer, device, step=None, **kwargs):
        inf_metrics = {}
        log.info("Computing metrics...")
        for metric in self.metrics:
            results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer, **kwargs)
            for k in results:
                if k in inf_metrics:
                    log.warning(f"Metric {k} had multiple values")
            inf_metrics.update(results)

        log.info("Aggregating metrics...")
        resolved_metrics = {}
        # sort so metrics are iterated on in the same order on all devices
        for k in sorted(inf_metrics):
            v = inf_metrics[k]
            if isinstance(v, float):
                # Trust the Evaluator writer to provide aggregated metrics
                resolved_metrics[k] = v
            elif isinstance(v, torchmetrics.Metric):
                resolved_metrics[k] = v.to(device).compute().item()
            elif isinstance(v, HtmlTable):
                # Special case, we aggregate table rows from all devices to ensure we can always
                # have enough rows to show even if each device only eval-ed a few examples
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    # HTML previews are a wandb-only artifact; skip without wandb
                    if WBValue is not None:
                        all_rows = flatten_list([x.rows for x in all_predictions])
                        resolved_metrics[k] = wandb.Html(HtmlTable(all_rows).get_html())
                else:
                    dist.gather_object(v, None)
            elif isinstance(v, List):
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    resolved_metrics[k] = []
                    for pred in all_predictions:
                        resolved_metrics[k] += pred
                else:
                    dist.gather_object(v, None)
            else:
                raise ValueError(f"Metric {v} not understood, must be aggregated between devices and of type float|List|HtmlTable|torchmetrics.Metric")

        # Some metrics need to some kind of more complex aggregation that cannot be done on the
        # individual worker, we hack those special cases in here
        for metric in self.metrics:
            if isinstance(metric, VSIBenchEval):
                resolved_metrics["object_rel_direction_accuracy"] = sum(
                    [
                        resolved_metrics[f"{k}_accuracy"]
                        for k in [
                            "object_rel_direction_easy",
                            "object_rel_direction_medium",
                            "object_rel_direction_hard",
                        ]
                    ]
                ) / 3.
                acc_types = [
                    "object_rel_direction",
                    "object_rel_distance",
                    "route_planning",
                    "obj_appearance_order",
                ]
                overall = [
                    resolved_metrics[f"{k}_MRA"]
                    for k in metric.NA_QUESTION_TYPES
                ]
                overall += [
                    resolved_metrics[f"{k}_accuracy"]
                    for k in acc_types
                ]
                resolved_metrics["overall"] = np.mean(overall)
            elif isinstance(metric, MulSetEval):
                resolved_metrics["overall"] = np.mean(
                    [
                        resolved_metrics[k]
                        for k in list(metric.TASKS.values())
                    ]
                )
            elif isinstance(metric, Ego3dBenchEval):
                # Compute RMSE
                for k in list(resolved_metrics.keys()):
                    if "MSE" in k:
                        resolved_metrics[k.replace("MSE", "RMSE")] = np.sqrt(resolved_metrics[k])
            elif isinstance(metric, (PointBenchEval,)):
                resolved_metrics["average"] = sum(resolved_metrics.get(cat) for cat in PointBenchEval.CATEGORIES) / len(PointBenchEval.CATEGORIES)
            elif isinstance(metric, (CountEval, PointCountEval, VixMoPointCountEval)):
                # Counting has a macro-score that should be computed once we have
                # scores from all devices
                counting_scores = {k: resolved_metrics[k] for
                                   k in list(resolved_metrics.keys()) if k.startswith("correct_")}
                resolved_metrics["per_category_average"] = np.mean(list(counting_scores.values()))
            elif isinstance(metric, MLVUGenEval):
                # MLVU has a macro-score that should be computed once we have
                # scores from all devices
                mlvu_sub_scene_scores = {k: resolved_metrics[k] for
                                         k in list(resolved_metrics.keys()) if k.startswith("sub_scene_")}
                resolved_metrics["sub_scene_total"] = np.sum(list(mlvu_sub_scene_scores.values()))
                mlvu_summary_scores = {k: resolved_metrics[k] for
                                      k in list(resolved_metrics.keys()) if k.startswith("summary_")}
                resolved_metrics["summary_total"] = np.sum(list(mlvu_summary_scores.values()))
                resolved_metrics["mlvu_gen_total"] = np.mean([resolved_metrics["sub_scene_total"], resolved_metrics["summary_total"]])
            elif isinstance(metric, TemporalBenchEval) and get_global_rank() == 0:
                type_video_scores = defaultdict(lambda: defaultdict(list))
                for cat, idx, score in zip(
                    resolved_metrics.pop("kind"),
                    resolved_metrics.pop("idx"),
                    resolved_metrics.pop("score"),
                ):
                    type_video_scores[cat][idx].append(score)
                total_score = 0
                for cat, video_scores in type_video_scores.items():
                    cat_score = sum(all(v) for v in video_scores.values())
                    resolved_metrics[cat] = cat_score / len(video_scores)
                    resolved_metrics[cat+"_count"] = len(video_scores)
                resolved_metrics["all"] = total_score / sum(len(x) for x in type_video_scores.values())
            elif isinstance(metric, VinogroundEval) and get_global_rank() == 0:
                score_matrix = torch.zeros((500, 4), dtype=torch.bool)
                for i in range(1000):
                    text_score = resolved_metrics["text"][i]
                    text_idx = resolved_metrics["text_idx"][i]
                    id, pos_neg = text_idx.split('_')
                    score_matrix[int(id), pos_neg=="neg"] = text_score

                    video_score = resolved_metrics["video"][i]
                    video_idx = resolved_metrics["video_idx"][i]
                    id, pos_neg = video_idx.split('_')
                    score_matrix[int(id), (pos_neg=="neg") + 2] = video_score
                resolved_metrics["text"] = (score_matrix[:, 0] & score_matrix[:, 1]).float().mean().item()
                resolved_metrics["video"] = (score_matrix[:, 2] & score_matrix[:, 3]).float().mean().item()
                resolved_metrics["group"] = (score_matrix[:, 0] & score_matrix[:, 1] & score_matrix[:, 2] & score_matrix[:, 3]).float().mean().item()
        return resolved_metrics


@dataclasses.dataclass
class EvaluatorConfig(BaseConfig):
    """Config for `Evaluator` objects that compute metrics"""

    n_to_log: int = 10
    """Num examples to log to console"""

    num_wandb_examples: int = 0
    """Num examples to log to Wandb as a HTML table"""

    save_predictions: Optional[str] = "_default"  # saves with default name to checkpoint dir
    """Where to save predictions files"""

    save_tokens: bool = False
    """If save predictions, should the tokens be saved"""

    egodex_3d_eval: bool = False
    """Compute trajectory-prediction metrics (per-dataset L2 / MSE) by
    parsing the model's `<tracks>` text output. The only inference metric
    shipped in the public release."""

    def build(self, default_save_dir=None) -> InfEvaluator:
        evaluators = []
        save_predictions = self.save_predictions
        if save_predictions == "_default":
            if default_save_dir is None:
                logging.info(
                    "save_predictions is \"_default\" but no default save "
                    "dir set so predictions will not be saved"
                )
            save_predictions = default_save_dir
        if save_predictions:
            evaluators.append(SavePredictions(
                save_predictions,
                log_examples=self.n_to_log,
                save_tokens=self.save_tokens,
            ))
        if self.egodex_3d_eval:
            from molmo_motion.eval.egodex_3d_evaluator import EgoDex3DEvaluator
            evaluators.append(EgoDex3DEvaluator())
        return InfEvaluator(evaluators)


@dataclasses.dataclass
class InfDatasetEvaluator:
    """Evaluates a model on a dataset"""
    label: str
    dataloader: Any
    evaluator: InfEvaluator
    n_steps: int
    max_new_tokens: int = 448
    console_log_interval: Optional[int] = None
    sampling_parameters: Optional[SamplingConfig] = None

    def run(self, model, device, autocast_precision, is_distributed, pbar=False, logger=None):
        eval_dataloader = self.dataloader
        eval_it = iter(eval_dataloader)
        n_steps = self.n_steps
        if n_steps is not None and 0 <= n_steps < len(self.dataloader):
            eval_it = itertools.islice(eval_it, 0, n_steps)
            total_steps = n_steps
        else:
            total_steps = len(eval_dataloader)

        constraints = []
        if self.sampling_parameters is None:
            sampler = None
        else:
            sampling = self.sampling_parameters
            if sampling.top_k is None and sampling.top_p == 1 and sampling.temperature == 0 and not sampling.ngram_size:
                sampler = None
            else:
                sampler = TopKTopPSampler(p=sampling.top_p, k=sampling.top_k, temperature=sampling.temperature)
            if sampling.ngram_size:
                constraints.append(RepeatedNGramBlockingConstraint(ngram_size=sampling.ngram_size))
            if sampling.repetition_penalty:
                constraints.append(RepetitionPenaltyConstraint(penalty=sampling.repetition_penalty))
            if sampling.frequency_penalty:
                constraints.append(FrequencyPenaltyConstraint(penalty=sampling.frequency_penalty))
        all_metadata = []
        predictions = defaultdict(list)
        done_init = False
        tok = model.config.build_tokenizer()
        pbar = pbar and get_global_rank() == 0

        # Per-batch JSONL save: one shard file per rank, appended after each batch.
        # Gives crash safety and lets us inspect partial progress mid-run.
        per_batch_shard = None
        for _m in getattr(self.evaluator, "metrics", []):
            if isinstance(_m, SavePredictions) and _m.output_dir:
                import os as _os
                _os.makedirs(_m.output_dir, exist_ok=True)
                _shard_path = _os.path.join(
                    _m.output_dir, f"batches_rank{get_global_rank()}.jsonl")
                per_batch_shard = open(_shard_path, "a")
                break

        for eval_step, batch in enumerate(tqdm(eval_it, total=total_steps, ncols=100, disable=not pbar)):
            if logger and eval_step % logger.log_interval == 0:
                logger.log_evaluation(self.label, eval_step, total_steps)
            if "metadata" in batch:
                batch_metadata = batch.pop("metadata")
            else:
                # Handle old-style data that used metadata/ prefix instead
                metadata = {k: batch.pop(k) for k in list(batch) if k.startswith("metadata/")}
                batch_metadata = []
                for i in range(len(batch["input_ids"])):
                    converted = {}
                    for k, v in metadata.items():
                        if isinstance(v[i], bytes):
                            converted[k] = v[i].decode("utf-8")
                        else:
                            converted[k] = v[i].tolist()
                    batch_metadata.append(converted)
            batch_inference = move_to_device(batch, device)
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    olmo_gen_output = model.generate(
                        batch=batch_inference,
                        max_steps=self.max_new_tokens,
                        sampler=sampler,
                        constraints=constraints,
                        is_distributed=is_distributed
                    )
            input_tokens = olmo_gen_output.token_ids[:, 0].detach().cpu().numpy()
            prompt_tokens = batch_inference["input_ids"].detach().cpu().numpy()
            prediction_text = [tok.decode(x[x >= 0]) for x in input_tokens]
            pred = {
                "predictions": input_tokens, # beam size of 1
                "prompts": prompt_tokens,
                "predictions_text": prediction_text,
                "prompts_text": [tok.decode(x[x >= 0]) for x in prompt_tokens],
            }
            if olmo_gen_output.token_target_ids is not None:
                points = []
                for text, point_indices, metadata in zip(prediction_text, olmo_gen_output.token_target_ids, batch_metadata):
                    points.append(model.config.token_ids_to_coordinates(text, point_indices, metadata))
                pred["points"] = points

            valid_ixs = [i for i, md in enumerate(batch_metadata) if md.get("valid", True)]
            all_metadata += [batch_metadata[i] for i in valid_ixs]
            for k, v in pred.items():
                for ix in valid_ixs:
                    predictions[k].append(v[ix])

            # Per-batch prediction dump (append-only JSONL for crash safety).
            if per_batch_shard is not None:
                import json as _json
                for ix in valid_ixs:
                    md = batch_metadata[ix]
                    row = {"step": eval_step, "rank": get_global_rank(),
                           "prediction": prediction_text[ix],
                           "prompt": pred["prompts_text"][ix]}
                    for k_md, v_md in md.items():
                        if isinstance(v_md, (str, int, float, bool, list, type(None))):
                            row[k_md] = v_md
                        elif isinstance(v_md, np.ndarray):
                            row[k_md] = v_md.tolist()
                    per_batch_shard.write(_json.dumps(row) + "\n")
                per_batch_shard.flush()

            # Log to console.
            if self.console_log_interval and not pbar:
                if eval_step + 1 == n_steps or (eval_step + 1) % self.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{total_steps}]")

        if per_batch_shard is not None:
            per_batch_shard.close()

        barrier()
        tokenizer = model.config.build_tokenizer()
        if logger:
            logger.log_evaluation(self.label, total_steps, total_steps)
        metrics = self.evaluator(predictions, all_metadata, tokenizer, device)
        return metrics


@dataclasses.dataclass
class InfDatasetEvaluatorConfig(BaseConfig):
    """Configuration for an inference evaluator"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig)
    """Evaluator to compute metrics from the generated outputs"""

    max_new_tokens: int = 448
    """Max number of tokens to generate"""

    device_batch_size: int = 4
    """Batch size"""

    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""

    include_image: bool = False
    """Include image in the metadata"""

    def build_dataset_evaluator(
        self,
        model_config,
        mesh,
        default_save_dir,
        device,
    ) -> InfDatasetEvaluator:
        assert mesh is None, "Mesh not supported for inference for now"
        global_batch_size = self.device_batch_size * get_world_size()
        if self.max_examples and self.max_examples > 0:
            max_steps = max(self.max_examples // global_batch_size, 1)
        elif self.subset_num_batches:
            max_steps = self.subset_num_batches
        else:
            max_steps = None

        eval_loader = self.data.build_eval_dataloader(
            model_config=model_config,
            batch_size=self.device_batch_size,
            mesh=mesh,
            for_inference=True,
            pad_batches=True,
            max_steps_for_padding=max_steps,
            include_image=self.include_image,
        )
        if self.max_examples is not None:
            num_batches = self.max_examples // self.device_batch_size*get_world_size()
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return InfDatasetEvaluator(
            label=self.label,
            dataloader=eval_loader,
            evaluator=self.evaluator.build(default_save_dir),
            n_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            console_log_interval=self.console_log_interval,
            sampling_parameters=self.sampling
        )
