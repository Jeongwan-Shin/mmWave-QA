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
            # For Temporal: organize templates by comb_len
            if hasattr(self, 'question_templates'):
                question_input_dict = self.question_templates
            # For BodyMove: organize templates by subcontexts
            elif hasattr(self, 'q_single_part') and hasattr(self, 'q_arm_or_leg') and hasattr(self, 'q_multiple_parts'):
                question_input_dict = {
                    "single_part": self.q_single_part,
                    "arm_or_leg": self.q_arm_or_leg,
                    "multiple_parts": self.q_multiple_parts,
                }
            else:
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
                    if not hasattr(self, 'question_templates') and not (hasattr(self, 'q_single_part') and hasattr(self, 'q_arm_or_leg')):
                        if isinstance(qinp, list):
                            for q in qinp:
                                if isinstance(q, str) and q not in qinp_seen:
                                    qinp_seen.add(q)
                                    qinp_list.append(q)
                    qa_pairs.append(qa_item)
                    total_saved += 1
                    
            out_obj = {
                "task": self.name,
                "question_input": question_input_dict if (hasattr(self, 'question_templates') or (hasattr(self, 'q_single_part') and hasattr(self, 'q_arm_or_leg'))) else qinp_list,
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
        # comb_len 파라미터는 무시하고 항상 2~3만 생성
        self.comb_len_range = [2, 3]
        
        # Question templates organized by comb_len and type
        self.question_templates = {
            2: {
                "previous": [
                    "What action occurs immediately before {anchor}?",
                    "Which pose precedes {anchor}?",
                    "Right before {anchor}, what is the action?",
                    "Identify the pose directly before {anchor}.",
                    "Which action comes just prior to {anchor}?",
                    "Immediately preceding {anchor}, which pose is observed?",
                ],
                "present": [
                    "What is the first action in this segment?",
                    "Identify the initial pose in this sequence.",
                    "Which action starts this segment?",
                    "What is the last action in this segment?",
                    "Identify the final pose in this sequence.",
                    "Which action concludes this segment?",
                ],
                "next": [
                    "What action occurs immediately after {anchor}?",
                    "Which pose follows {anchor}?",
                    "Right after {anchor}, what is the action?",
                    "Identify the pose that comes directly after {anchor}.",
                    "Which action happens just after {anchor}?",
                    "Immediately following {anchor}, which pose is observed?",
                ],
            },
            3: {
                "previous": [
                    "What action occurs immediately before {anchor}?",
                    "Which pose precedes {anchor}?",
                    "Right before {anchor}, what is the action?",
                    "Identify the pose directly before {anchor}.",
                    "Which action comes just prior to {anchor}?",
                    "Immediately preceding {anchor}, which pose is observed?",
                ],
                "present": [
                    "What is the first action in this segment?",
                    "Identify the initial pose in this sequence.",
                    "What is the middle action in this segment?",
                    "At the midpoint of this segment, what is the action?",
                    "What is the last action in this segment?",
                    "Identify the final pose in this sequence.",
                ],
                "next": [
                    "What action occurs immediately after {anchor}?",
                    "Which pose follows {anchor}?",
                    "Right after {anchor}, what is the action?",
                    "Identify the pose that comes directly after {anchor}.",
                    "Which action happens just after {anchor}?",
                    "Immediately following {anchor}, which pose is observed?",
                ],
            },
        }

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

    def _qa_previous(self, names: List[str], comb_len: int) -> Tuple[str, str]:
        # Ask about the action immediately before a randomly chosen anchor (index 1..k-1)
        k = len(names)
        anchor_idx = 1 if k <= 2 else random.randint(1, k - 1)
        anchor = names[anchor_idx]
        answer = names[anchor_idx - 1]
        return anchor, answer

    def _qa_present(self, names: List[str], comb_len: int) -> Tuple[Optional[str], str]:
        k = len(names)
        if k == 2:
            answer = names[0]
            return None, answer  # No anchor for present questions
        if k == 3:
            answer = names[1]
            return None, answer
        # k >= 4 (shouldn't happen but fallback)
        answer = names[-1]
        return None, answer

    def _qa_next(self, names: List[str], comb_len: int) -> Tuple[str, str]:
        # Ask about the action immediately after a randomly chosen anchor (index 0..k-2)
        k = len(names)
        anchor_idx = 0 if k <= 2 else random.randint(0, k - 2)
        anchor = names[anchor_idx]
        answer = names[anchor_idx + 1]
        return anchor, answer

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        labels = ["A", "B", "C", "D"]
        
        # mmfi(source) 제외
        actions = [a for a in actions if str(a.info.get("from", "")).lower() != "mmfi"]
        if not actions:
            return outputs
        
        # 연속된 같은 액션을 그룹화
        action_groups = self._group_consecutive_actions(actions)
        
        # 모든 unique actions (distractor pool용)
        unique_actions = list({r.action for r in actions})
        
        # 각 comb_len에 대해 반복 (2, 3만)
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
                names = [group[0].action for group in window_groups]
                start = all_records[0].start
                end = all_records[-1].end
                
                # Build three QA variants
                qa_variants = {
                    "previous": self._qa_previous(names, comb_len),
                    "present": self._qa_present(names, comb_len),
                    "next": self._qa_next(names, comb_len),
                }
                
                for qtype, (anchor, answer) in qa_variants.items():
                    correct = answer
                    distractors = self.choose_distractors(correct, unique_actions, k=3)
                    options = [correct] + distractors
                    random.shuffle(options)
                    options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options[:4])]
                    
                    # Generate all questions from templates and fill in {anchor} if needed
                    question_templates = self.question_templates[comb_len][qtype]
                    filled_questions = []
                    for question_template in question_templates:
                        filled_question = question_template
                        if anchor is not None:
                            filled_question = filled_question.replace("{anchor}", anchor)
                        filled_questions.append(filled_question)
                    
                    # Build payload with all questions as a list
                    payload = {
                        "info": all_records[0].info if all_records else {},
                        "frames": {"start": start, "end": end},
                        "action_sequence": names,
                        "comb_len": comb_len,
                        "question_category": self.question_category,
                        "question_type": "4_options",
                        "question_subcontexts": qtype,
                        "question": filled_questions,  # List of all filled questions
                        "question_input": self.question_templates[comb_len][qtype],  # Template with {anchor}
                        "options": options_labeled,
                        "answer": correct,
                    }
                    
                    # Add anchor if present (not for "present" type questions)
                    if anchor is not None:
                        payload["anchor"] = anchor
                    
                    seq_name_join = "~".join(names)
                    fname = f"seq_{widx:04d}_comb{comb_len}_{self.safe(seq_name_join)}_{qtype}.json"
                    outputs.append((fname, payload))
        return outputs


class BodyMoveGenerator(BaseGenerator):
    name = "body_move"
    question_category = "body_move"

    def __init__(self, dataset: ActionsDataset, output_root: str, output_format: str = "files") -> None:
        super().__init__(dataset, output_root, output_format=output_format)
        self.parts_order = ["left arm", "right arm", "left leg", "right leg"]
        
        # Question templates for each type
        self.q_single_part = [
            "Which single body part is primarily moving in this action?",
            "Identify the one body part that is active in this segment.",
            "Which body part is involved in this action?",
            "Select the body part that moves during this interval.",
            "What is the single moving body part in this pose?",
        ]
        
        self.q_arm_or_leg = [
            "Are the arms or legs primarily involved in this action?",
            "Which body region is active: arms or legs?",
            "Does this action primarily use arms or legs?",
            "Identify whether arms or legs are moving in this segment.",
            "Which limb type is involved: arms or legs?",
        ]
        
        self.q_multiple_parts = [
            "Which combination of body parts is moving in this action?",
            "Select all body parts that are active in this segment.",
            "Identify the body parts involved in this action.",
            "Which body parts move during this interval?",
            "What combination of limbs is used in this pose?",
        ]

    def _get_active_parts(self, rec: ActionRecord) -> List[str]:
        """Return list of active body parts."""
        return [part for part in self.parts_order if rec.body_part.get(part, False)]

    def _generate_single_part_qa(self, rec: ActionRecord, idx: int, segment: str) -> Optional[Tuple[str, Dict[str, object]]]:
        """Generate single-part question (5 options: 4 parts + none)."""
        active_parts = self._get_active_parts(rec)
        
        # Only generate if exactly 0 or 1 part is active
        if len(active_parts) > 1:
            return None
        
        # 5 options: right arm, left arm, right leg, left leg, none
        labels = ["A", "B", "C", "D", "E"]
        options = ["right arm", "left arm", "right leg", "left leg", "none"]
        options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
        
        if len(active_parts) == 0:
            answer = "none"
        else:
            answer = active_parts[0]
        
        payload = {
            "info": rec.info,
            "frames": {"start": rec.start, "end": rec.end},
            "question_category": self.question_category,
            "question_type": "5_options",
            "question_subcontexts": "single_part",
            "question": self.q_single_part,
            "question_input": self.q_single_part,
            "options": options_labeled,
            "answer": answer,
            "action": rec.action,
            "body_part": rec.body_part,
        }
        fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}_single.json"
        return (fname, payload)

    def _generate_arm_leg_binary_qa(self, rec: ActionRecord, idx: int, segment: str) -> Optional[Tuple[str, Dict[str, object]]]:
        """Generate binary question: arms only or legs only."""
        active_parts = self._get_active_parts(rec)
        
        # Cannot generate if no parts are active
        if len(active_parts) == 0:
            return None
        
        arms = ["left arm", "right arm"]
        legs = ["left leg", "right leg"]
        
        has_arms = any(part in arms for part in active_parts)
        has_legs = any(part in legs for part in active_parts)
        
        # Only generate if exclusively arms OR exclusively legs (not both, not none)
        if has_arms and has_legs:
            return None
        if not has_arms and not has_legs:
            return None
        
        if has_arms:
            answer = "arms"
        else:
            answer = "legs"
        
        # Binary options
        labels = ["A", "B"]
        options = ["arms", "legs"]
        options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
        
        payload = {
            "info": rec.info,
            "frames": {"start": rec.start, "end": rec.end},
            "question_category": self.question_category,
            "question_type": "binary",
            "question_subcontexts": "arm_or_leg",
            "question": self.q_arm_or_leg,
            "question_input": self.q_arm_or_leg,
            "options": options_labeled,
            "answer": answer,
            "action": rec.action,
            "body_part": rec.body_part,
        }
        fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}_armleg.json"
        return (fname, payload)

    def _generate_multiple_parts_qa(self, rec: ActionRecord, idx: int, segment: str, all_actions: List[ActionRecord]) -> Optional[Tuple[str, Dict[str, object]]]:
        """Generate multiple-choice question for actions with 2+ active parts."""
        active_parts = self._get_active_parts(rec)
        
        # Only generate if 2 or more parts are active
        if len(active_parts) < 2:
            return None
        
        # Correct answer: sorted active parts as a string
        correct = ", ".join(sorted(active_parts))
        
        # Generate distractors: other combinations from the dataset
        all_combinations = set()
        for other_rec in all_actions:
            other_active = self._get_active_parts(other_rec)
            if len(other_active) >= 2:
                combo = ", ".join(sorted(other_active))
                all_combinations.add(combo)
        
        # Remove correct answer from pool
        distractor_pool = [c for c in all_combinations if c != correct]
        
        # Always try to get exactly 3 distractors for 4 total options
        distractors = []
        if len(distractor_pool) >= 3:
            distractors = random.sample(distractor_pool, 3)
        else:
            # If not enough real combinations, add synthetic distractors
            distractors = distractor_pool[:]
            
            # Define possible synthetic combinations
            all_possible_combos = [
                "left arm, right arm",
                "left leg, right leg",
                "left arm, left leg",
                "left arm, right leg",
                "right arm, left leg",
                "right arm, right leg",
                "left arm, left leg, right arm",
                "left arm, left leg, right leg",
                "left arm, right arm, right leg",
                "left leg, right arm, right leg",
                "left arm, left leg, right arm, right leg",
            ]
            
            # Add synthetic ones that are not correct and not already in distractors
            for combo in all_possible_combos:
                if combo != correct and combo not in distractors:
                    distractors.append(combo)
                    if len(distractors) >= 3:
                        break
        
        # Ensure exactly 3 distractors
        distractors = distractors[:3]
        
        # Build options (always 4)
        options = [correct] + distractors
        random.shuffle(options)
        labels = ["A", "B", "C", "D"]
        options_labeled = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
        
        payload = {
            "info": rec.info,
            "frames": {"start": rec.start, "end": rec.end},
            "question_category": self.question_category,
            "question_type": "4_options",
            "question_subcontexts": "multiple_parts",
            "question": self.q_multiple_parts,
            "question_input": self.q_multiple_parts,
            "options": options_labeled,
            "answer": correct,
            "action": rec.action,
            "body_part": rec.body_part,
        }
        fname = f"{segment}_idx{idx:04d}_{self.safe(rec.action)}_multi.json"
        return (fname, payload)

    def generate_for_segment(self, segment: str, actions: List[ActionRecord]) -> List[Tuple[str, Dict[str, object]]]:
        outputs: List[Tuple[str, Dict[str, object]]] = []
        
        for idx, rec in enumerate(actions):
            # 1. Single part question (0 or 1 active part)
            qa1 = self._generate_single_part_qa(rec, idx, segment)
            if qa1:
                outputs.append(qa1)
            
            # 2. Binary arm/leg question (exclusive arms or legs)
            qa2 = self._generate_arm_leg_binary_qa(rec, idx, segment)
            if qa2:
                outputs.append(qa2)
            
            # 3. Multiple parts question (2+ active parts)
            qa3 = self._generate_multiple_parts_qa(rec, idx, segment, actions)
            if qa3:
                outputs.append(qa3)
        
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


