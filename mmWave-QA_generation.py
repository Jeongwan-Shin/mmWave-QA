#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from tqdm import tqdm


# Defaults
DEFAULT_ACTIONS_JSON = "/workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json"
DEFAULT_OUTPUT_ROOT = "/workspace/mmWave-QA/mmWaveQA_benchmark/all"


# -----------------------------
# Data model and dataset loader
# -----------------------------


@dataclass(frozen=True)
class ActionRecord:
    segment: str
    start: int
    end: int
    action: str
    body_part: Dict[str, bool]
    movement: bool
    info: Dict[str, object]

    @staticmethod
    def from_dict(entry: Dict[str, object]) -> "ActionRecord":
        info = entry.get("info", {}) or {}
        frames = entry.get("frames", {}) or {}
        segment = str(info.get("segment", "sequence_0"))
        start = int(frames.get("start", 0))
        end = int(frames.get("end", start))
        action = str(entry.get("action", ""))
        body_part = dict(entry.get("body_part", {}) or {})
        movement = bool(entry.get("movement", False))
        return ActionRecord(
            segment=segment,
            start=start,
            end=end,
            action=action,
            body_part=body_part,
            movement=movement,
            info=dict(info),
        )


class ActionsDataset:
    def __init__(self, json_path: str) -> None:
        self.json_path = json_path
        self.records: List[ActionRecord] = []

    def load(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parsed: List[ActionRecord] = []
        for entry in raw:
            try:
                parsed.append(ActionRecord.from_dict(entry))
            except Exception:
                continue
        # Sort globally by (segment, start)
        parsed.sort(key=lambda r: (r.segment, r.start, r.end))
        self.records = parsed

    def segments(self) -> List[str]:
        return sorted({r.segment for r in self.records})

    def by_segment(self) -> Dict[str, List[ActionRecord]]:
        by_seg: Dict[str, List[ActionRecord]] = {}
        for rec in self.records:
            by_seg.setdefault(rec.segment, []).append(rec)
        # Ensure deterministic order
        for seg in by_seg:
            by_seg[seg].sort(key=lambda r: (r.start, r.end))
        return by_seg

    def unique_actions(self, scope: Optional[Iterable[ActionRecord]] = None) -> List[str]:
        if scope is None:
            scope = self.records
        return sorted({r.action for r in scope if r.action})


# -----------------------------
# Base generator and utilities
# -----------------------------


class BaseGenerator:
    name: str = "base"
    question_category: str = "generic"

    def __init__(self, dataset: ActionsDataset, output_root: str, output_format: str = "files") -> None:
        self.dataset = dataset
        self.output_root = output_root
        self.output_dir = os.path.join(self.output_root, self.name)
        self.output_format = output_format  # "files" or "single"
        self._safe_re = re.compile(r"[^A-Za-z0-9_\-~]")

    def ensure_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def write_json(self, data: Dict[str, object], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def build_common_meta(self) -> Dict[str, object]:
        return {
            "benchmarkfrom": str(self.dataset.records[0].info.get("from", "mmbody")) if self.dataset.records else "mmbody",
        }

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        raise NotImplementedError

    def run(self) -> Tuple[int, int]:
        self.ensure_dirs()
        total_saved = 0
        total_segments = 0
        by_seg = self.dataset.by_segment()

        if self.output_format == "single":
            qa_pairs: List[Dict[str, object]] = []
            bench_from_set = set()
            qinp_list: List[str] = []
            qinp_seen = set()
            for segment, actions in tqdm(by_seg.items(), desc=f"{self.name}: segments"):
                total_segments += 1
                outputs = self.generate_for_segment(segment, actions)
                for _, payload in outputs:
                    bf = str(payload.get("benchmarkfrom", "")).strip()
                    if bf and bf not in bench_from_set:
                        bench_from_set.add(bf)
                    # Minimize duplication: remove generator-level fields
                    qa_item = dict(payload)
                    qa_item.pop("benchmarkfrom", None)
                    qa_item.pop("question_category", None)
                    # Track question_input at top-level, remove from items
                    qinp = qa_item.pop("question_input", None)
                    if isinstance(qinp, list):
                        for q in qinp:
                            if isinstance(q, str) and q not in qinp_seen:
                                qinp_seen.add(q)
                                qinp_list.append(q)
                    qa_pairs.append(qa_item)
                    total_saved += 1
            out_obj = {
                "task": self.name,
                "question_input": qinp_list,
                "QA-pairs": qa_pairs,
            }
            out_path = os.path.join(self.output_root, f"{self.name}.json")
            self.write_json(out_obj, out_path)
            print(f"[{self.name}] Saved single file with {total_saved} QA-pairs across {total_segments} segments. Out: {out_path}")
            return total_saved, total_segments

        # Default: per-item files grouped by segment
        for segment, actions in tqdm(by_seg.items(), desc=f"{self.name}: segments"):
            total_segments += 1
            outputs = self.generate_for_segment(segment, actions)
            seg_dir = os.path.join(self.output_dir, segment)
            os.makedirs(seg_dir, exist_ok=True)
            for fname, payload in outputs:
                self.write_json(payload, os.path.join(seg_dir, fname))
                total_saved += 1
        print(f"[{self.name}] Saved {total_saved} files across {total_segments} segments. Out: {self.output_dir}")
        return total_saved, total_segments

    # Utilities
    def safe(self, text: str) -> str:
        return self._safe_re.sub("_", text)

    @staticmethod
    def window_slices(items: List[ActionRecord], window: int) -> List[List[ActionRecord]]:
        if window <= 0:
            return []
        if len(items) < window:
            return []
        return [items[i : i + window] for i in range(0, len(items) - window + 1)]

    @staticmethod
    def choose_distractors(correct: str, pool: List[str], k: int) -> List[str]:
        candidates = [p for p in pool if p != correct]
        if len(candidates) <= k:
            return candidates
        return random.sample(candidates, k=k)


# -----------------------------
# Task generators
# -----------------------------


class ActionQAGenerator(BaseGenerator):
    name = "action_qa"
    question_category = "action_qa"

    def __init__(self, dataset: ActionsDataset, output_root: str, num_options: int = 4, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        self.num_options = max(2, int(num_options))
        self.q_templates: List[str] = [
            "Which action is occurring in this time span?",
            "Identify the pose for these frames.",
            "What is the action performed during this segment?",
            "Select the action for the given frame range.",
            "Which pose best describes this interval?",
        ]

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        unique_actions = self.dataset.unique_actions(actions)
        labels = [chr(ord("A") + i) for i in range(self.num_options)]
        for idx, rec in enumerate(actions):
            correct = rec.action
            distractors = self.choose_distractors(correct, unique_actions, k=self.num_options - 1)
            options = [correct] + distractors
            random.shuffle(options)
            options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
            payload = {
                **self.build_common_meta(),
                "segment": segment,
                "frames": {"start": rec.start, "end": rec.end},
                "question_category": self.question_category,
                "question_type": "4_options" if self.num_options == 4 else f"{self.num_options}_options",
                "question_input": self.q_templates,
                "options": options_labeled,
                "answer": correct,
                "action": rec.action,
                "body_part": rec.body_part,
                "movement": rec.movement,
                "info": rec.info,
            }
            fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}.json"
            outputs.append((fname, payload))
        return outputs


class MovementGenerator(BaseGenerator):
    name = "movement"
    question_category = "movement"

    def __init__(self, dataset: ActionsDataset, output_root: str, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        self.q_templates = [
            "Did the person walk or change location, indicating a shift of the whole body's center position?",
            "Has the person moved from one place to another rather than staying in the same spot?",
            "Is there evidence of locomotion, such as walking, where the person's body center travels across coordinates?",
            "Did the person's body centroid translate noticeably, consistent with walking or relocation?",
            "Has the person shifted their overall position across frames, indicating actual movement through space?"
        ]

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        for idx, rec in enumerate(actions):
            answer = "yes" if rec.movement else "no"
            payload = {
                **self.build_common_meta(),
                "info": rec.info,
                "frames": {"start": rec.start, "end": rec.end},
                "question_category": self.question_category,
                "question_type": "binary",
                "question_input": self.q_templates,
                "answer": answer,
                "action": rec.action,
                "movement": rec.movement,
            }
            fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}_move.json"
            outputs.append((fname, payload))
        return outputs


class CountGenerator(BaseGenerator):
    name = "count"
    question_category = "count"

    def __init__(self, dataset: ActionsDataset, output_root: str, comb_len: int = 3, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        # comb_len 파라미터는 무시하고 항상 1~4까지 모두 생성
        self.comb_len_range = [1, 2, 3, 4]
        self.q_templates = [
            "How many distinct actions appear in this segment?",
            "Count the unique poses shown in this sequence.",
            "What is the number of different actions in this window?",
            "How many unique action types are present here?",
            "What count of distinct poses occurs in this segment?",
            "Determine the number of unique actions within this sequence.",
        ]

    def _group_consecutive_actions(self, actions: List[ActionRecord]) -> List[List[ActionRecord]]:
        """연속된 같은 액션을 그룹으로 묶음"""
        if not actions:
            return []
        
        groups: List[List[ActionRecord]] = []
        current_group: List[ActionRecord] = [actions[0]]
        
        for i in range(1, len(actions)):
            # 이전 액션과 같으면 현재 그룹에 추가
            if actions[i].action == current_group[-1].action:
                current_group.append(actions[i])
            else:
                # 다른 액션이면 현재 그룹을 저장하고 새 그룹 시작
                groups.append(current_group)
                current_group = [actions[i]]
        
        # 마지막 그룹 추가
        groups.append(current_group)
        return groups

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        labels = ["A", "B", "C", "D"]

        # mmfi(source) 제외
        actions = [a for a in actions if str(a.info.get("from", "")).lower() != "mmfi"]
        if not actions:
            return outputs

        # 연속된 같은 액션을 그룹화
        action_groups = self._group_consecutive_actions(actions)
        
        # 각 comb_len에 대해 반복
        for comb_len in self.comb_len_range:
            # 그룹 개수가 comb_len보다 적으면 이 comb_len은 스킵
            if len(action_groups) < comb_len:
                continue
            
            # 그룹 기반으로 윈도우 생성
            for widx in range(len(action_groups) - comb_len + 1):
                window_groups = action_groups[widx:widx + comb_len]
                
                # 각 윈도우에서 모든 액션 레코드 수집
                all_records = []
                for group in window_groups:
                    all_records.extend(group)
                
                # 윈도우 내 액션 이름들 (그룹별로 하나씩만)
                window_actions = [group[0].action for group in window_groups]
                distinct = len(set(window_actions))
                
                # 시작/끝 프레임
                start = all_records[0].start
                end = all_records[-1].end
                
                # Options fixed to 1..4
                options = [f"{labels[i]}. {i+1}" for i in range(4)]
                correct_text = str(min(distinct, 4))
                payload = {
                    "info": all_records[0].info if all_records else {},
                    "frames": {"start": start, "end": end},
                    "window_actions": window_actions,
                    "comb_len": comb_len,
                    "question_category": self.question_category,
                    "question_type": "4_options",
                    "question_input": self.q_templates,
                    "options": options,
                    "answer": correct_text,
                }
                seq_name_join = "~".join(window_actions)
                fname = f"seq_{widx:04d}_comb{comb_len}_{self.safe(seq_name_join)}_count.json"
                outputs.append((fname, payload))
        return outputs


class TemporalGenerator(BaseGenerator):
    name = "temporal"
    question_category = "temporal"

    def __init__(self, dataset: ActionsDataset, output_root: str, comb_len: int = 3, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        self.comb_len = min(5, max(2, int(comb_len)))

    def _qa_previous(self, names: List[str]) -> Tuple[List[str], str]:
        k = len(names)
        anchor_idx = 1 if k <= 2 else random.randint(1, k - 1)
        anchor = names[anchor_idx]
        answer = names[anchor_idx - 1]
        q = [
            f"What action occurs immediately before {anchor}?",
            f"Which pose precedes {anchor}?",
            f"Right before {anchor}, what is the action?",
            f"Identify the pose directly before {anchor}.",
            f"Which action comes just prior to {anchor}?",
        ]
        return q, answer

    def _qa_present(self, names: List[str]) -> Tuple[List[str], str]:
        k = len(names)
        if k == 2:
            answer = names[0]
            q = [
                "What is the first action in this segment?",
                "Identify the initial pose in this sequence.",
                "Which action starts this segment?",
                "At the beginning of this segment, what is the action?",
            ]
            return q, answer
        if k == 3:
            answer = names[1]
            q = [
                "What is the middle action in this segment?",
                "Identify the center pose in this three-action sequence.",
                "Which action occurs between the first and the last?",
                "At the midpoint of this segment, what is the action?",
            ]
            return q, answer
        answer = names[-1]
        q = [
            "What is the last action in this segment?",
            "Identify the final pose in this sequence.",
            "Which action concludes this segment?",
            "At the end of this segment, what is the action?",
        ]
        return q, answer

    def _qa_next(self, names: List[str]) -> Tuple[List[str], str]:
        k = len(names)
        anchor_idx = 0 if k <= 2 else random.randint(0, k - 2)
        anchor = names[anchor_idx]
        answer = names[anchor_idx + 1]
        q = [
            f"What action occurs immediately after {anchor}?",
            f"Which pose follows {anchor}?",
            f"Right after {anchor}, what is the action?",
            f"Identify the pose that comes directly after {anchor}.",
            f"Which action happens just after {anchor}?",
        ]
        return q, answer

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        unique_actions = self.dataset.unique_actions(actions)
        windows = self.window_slices(actions, self.comb_len)
        labels = ["A", "B", "C", "D"]
        for widx, window in enumerate(windows):
            names = [r.action for r in window]
            start = window[0].start
            end = window[-1].end
            qa_variants = {
                "previous": self._qa_previous(names),
                "present": self._qa_present(names),
                "next": self._qa_next(names),
            }
            for qtype, (q_list, answer) in qa_variants.items():
                correct = answer
                distractors = self.choose_distractors(correct, unique_actions, k=3)
                options = [correct] + distractors
                random.shuffle(options)
                options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options[:4])]
                payload = {
                    **self.build_common_meta(),
                    "segment": segment,
                    "frames": {"start": start, "end": end},
                    "action_sequence": names,
                    "question_category": self.question_category,
                    "question_type": "4_options",
                    "question_subcontexts": qtype,
                    "question_input": q_list,
                    "options": options_labeled,
                    "answer": correct,
                    "info": window[0].info if window else {},
                }
                seq_name_join = "~".join(names)
                fname = f"seq_{widx:04d}_{self.safe(seq_name_join)}_{qtype}.json"
                outputs.append((fname, payload))
        return outputs


class BodyMoveGenerator(BaseGenerator):
    name = "body_move"
    question_category = "body_move"

    def __init__(self, dataset: ActionsDataset, output_root: str, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        self.parts_order = ["left arm", "right arm", "left leg", "right leg"]
        self.q_template = "Is the {part} involved in this action?"

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        for idx, rec in enumerate(actions):
            for part in self.parts_order:
                val = bool(rec.body_part.get(part, False))
                answer = "yes" if val else "no"
                payload = {
                    **self.build_common_meta(),
                    "segment": segment,
                    "frames": {"start": rec.start, "end": rec.end},
                    "question_category": self.question_category,
                    "question_type": "binary",
                    "question_input": [
                        self.q_template.format(part=part),
                        f"Does this segment include the {part}?",
                        f"Is {part} active in this interval?",
                    ],
                    "answer": answer,
                    "action": rec.action,
                    "body_part": rec.body_part,
                    "info": rec.info,
                }
                fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}_{self.safe(part)}.json"
                outputs.append((fname, payload))
        return outputs


# -----------------------------
# Task registry and CLI
# -----------------------------


GeneratorName = Literal["actionqa", "movement", "count", "temporal", "bodymove", "all"]


class TaskRegistry:
    def __init__(self) -> None:
        self._registry = {
            "actionqa": ActionQAGenerator,
            "movement": MovementGenerator,
            "count": CountGenerator,
            "temporal": TemporalGenerator,
            "bodymove": BodyMoveGenerator,
        }

    def names(self) -> List[str]:
        return list(self._registry.keys())

    def create(self, name: str, dataset: ActionsDataset, output_root: str, **kwargs) -> BaseGenerator:
        cls = self._registry[name]
        return cls(dataset=dataset, output_root=output_root, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mmWave QA tasks from mm_actions.json")
    parser.add_argument("--actions-json", type=str, default=DEFAULT_ACTIONS_JSON, help="Path to mm_actions.json")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task list: actionqa,movement,count,temporal,bodymove or 'all'",
    )
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count-comb-len", type=int, default=3)
    parser.add_argument("--temporal-comb-len", type=int, default=3)
    parser.add_argument("--options", type=int, default=4, help="Number of options for ActionQA (>=2)")
    parser.add_argument(
        "--output-format",
        type=str,
        default="files",
        choices=["files", "single"],
        help="Output as many files grouped by segment, or a single aggregated JSON file per task",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset = ActionsDataset(args.actions_json)
    dataset.load()

    registry = TaskRegistry()
    task_names = [t.strip().lower() for t in args.tasks.split(",")]
    if len(task_names) == 1 and task_names[0] == "all":
        task_names = registry.names()

    # Ensure output root exists
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, Tuple[int, int]]] = []
    for task in task_names:
        if task not in registry.names():
            print(f"[warn] Unknown task '{task}', skipping.")
            continue
        kwargs = {}
        if task == "count":
            kwargs["comb_len"] = args.count_comb_len
        if task == "temporal":
            kwargs["comb_len"] = args.temporal_comb_len
        if task == "actionqa":
            kwargs["num_options"] = max(2, int(args.options))
        # pass output format to all tasks
        kwargs["output_format"] = args.output_format

        generator = registry.create(task, dataset=dataset, output_root=args.output_root, **kwargs)
        res = generator.run()
        results.append((task, res))

    # Summary
    if results:
        print("\nTask summary:")
        for name, (saved, segs) in results:
            print(f"- {name}: saved={saved}, segments={segs}")


if __name__ == "__main__":
    main()


