"""Base evaluator classes used by the training loop.

Release version: contains only the small set of classes the trajectory
training loop reads from this module. All QA/captioning/clock/screen-spot
evaluators that previously lived here have been stripped — the only
trajectory-specific metric ships in :mod:`molmo_motion.eval.egodex_3d_evaluator`.
"""

import base64
import copy
import dataclasses
import io
import json
import logging
import os
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.distributed as dist
from PIL import Image
from torchmetrics import MeanMetric, SumMetric

from molmo_motion.html_utils import build_html_table, postprocess_prompt
from molmo_motion.io import write_file
from molmo_motion.torch_util import get_global_rank, get_world_size
from molmo_motion.util import flatten_list

log = logging.getLogger(__name__)


def mean_metric(v):
    metric = MeanMetric(nan_strategy="error")
    metric.update(np.mean(v) if len(v) > 0 else 0, len(v))
    return metric


def sum_metric(v):
    metric = SumMetric(nan_strategy="error")
    metric.update(np.sum(v) if len(v) > 0 else 0)
    return metric


@dataclasses.dataclass
class HtmlTable:
    """Returned by `gather_examples_as_html` as a special wandb metric."""
    rows: List[Dict[str, Any]]

    def get_html(self):
        return build_html_table(self.rows)


def gather_examples_as_html(n_examples, voc, metadatas, predictions) -> HtmlTable:
    """Minimal HTML table builder — used by `SavePredictions` to attach a
    small per-step prediction preview to wandb."""
    n = len(predictions["predictions"])
    if n_examples is not None:
        n = min(n, n_examples)
        n = (n + get_world_size() - 1) // get_world_size()
    rows = []
    new_tokens = predictions["predictions"]
    prompt_tokens = predictions["prompts"]
    for ix in range(n):
        prompt_text = postprocess_prompt(voc.decode(prompt_tokens[ix][prompt_tokens[ix] >= 0]))
        metadata = metadatas[ix]
        pred_seq = new_tokens[ix]
        pred_txt = voc.decode(pred_seq[pred_seq >= 0])

        image_src = None
        if "image_url" in metadata:
            image_src = metadata["image_url"]
        elif "image" in metadata and isinstance(metadata["image"], np.ndarray):
            img = Image.fromarray(metadata["image"])
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_src = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

        row: Dict[str, Any] = {}
        if image_src is not None:
            row["image"] = (
                f"<img style=\"max-height:500px;max-width:500px;\" src={image_src}>"
            )
        row["prompt"] = html_escape(prompt_text)
        row["prediction"] = html_escape(pred_txt)

        if "answer" in metadata:
            row["gt"] = html_escape(str(metadata["answer"]))
        elif "answers" in metadata:
            gt = metadata["answers"]
            row["gt"] = "<br>".join(html_escape(str(x)) for x in gt) if isinstance(gt, list) else html_escape(str(gt))

        if "display_in_eval" in metadata and metadata["display_in_eval"]:
            md = copy.deepcopy(metadata)
            md.pop("image", None)
            row["input_metadata"] = json.dumps(md)
        rows.append(row)
    return HtmlTable(rows)


class Evaluator:
    """Base evaluator interface — subclass and override `__call__`.

    Each evaluator receives the per-example `metadatas` (everything the
    dataset returned in `meta`) together with the model's
    `predictions` dict (which contains `predictions`, `prompts`, and any
    extra tensors the inference loop attached). It returns a dict of
    metric names → torchmetrics objects.
    """

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        raise NotImplementedError()


class SavePredictions(Evaluator):
    """Dump model predictions to a JSON file (and optionally wandb HTML)."""

    @staticmethod
    def get_file_name(step, process_index):
        filename = ""
        if step is not None:
            filename += f"step{step}-"
        if get_world_size() > 1 and process_index is not None:
            filename += f"shard{process_index}"
        filename += "predictions"
        return filename

    def __init__(self, output_dir, json=True, save_tokens=True,
                 log_examples=10, table=100):
        self.save_tokens = save_tokens
        self.output_dir = output_dir
        self.log_examples = log_examples
        self.json = json
        self.table = table

    def __call__(self, metadatas, predictions, tokenizer,
                 step=None, scores=None):
        if not self.output_dir.startswith("gs://"):
            if not os.path.exists(self.output_dir):
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        json_data = []

        n_no_eos = 0
        for tok in new_tokens:
            if not np.any(tok == tokenizer.eos_token_id):
                n_no_eos += 1
        if n_no_eos > 0:
            logging.warning(
                f"{n_no_eos}/{len(new_tokens)} ({n_no_eos / len(new_tokens):00.4f}) "
                f"examples have no EOS, your inference tokens might be too short"
            )

        for ex_ix, pred_seq in enumerate(new_tokens):
            text = tokenizer.decode(pred_seq[pred_seq >= 0])
            json_row: Dict[str, Any] = dict(prediction=text)
            if self.save_tokens:
                json_row["n_tokens"] = pred_seq.tolist()
            prompt_text = postprocess_prompt(tokenizer.decode(prompt_tokens[ex_ix][prompt_tokens[ex_ix] >= 0]))
            sep = " " if tokenizer.adds_space else ""
            json_row["prompt"] = prompt_text

            metadata = metadatas[ex_ix]
            if ex_ix < self.log_examples:
                log.info("*" * 30)
                if "example_id" in metadata:
                    log.info(metadata["example_id"])
                log.info(" ".join((prompt_text + sep + text.replace("\n", "\\n")).split()))
            json_row.update({k: v for k, v in metadata.items() if isinstance(v, (str, float, int))})
            json_data.append(json_row)

        metrics: Dict[str, Any] = {}
        if self.json:
            log.info("Save prediction JSON")
            if get_world_size() > 1:
                if get_global_rank() == 0:
                    all_predictions = [None] * get_world_size()
                    dist.gather_object(json_data, all_predictions)
                    json_data = flatten_list(all_predictions)
                else:
                    dist.gather_object(json_data, None)
            if get_global_rank() == 0:
                write_file(
                    self.output_dir,
                    self.get_file_name(step, None) + ".json",
                    json.dumps(json_data, indent=2),
                    save_overwrite=True,
                )
                log.info("done saving json")

        if self.table:
            metrics["prediction_table"] = gather_examples_as_html(
                self.table, tokenizer, metadatas, predictions
            )
        return metrics
