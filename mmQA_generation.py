#!/usr/bin/env python3
import os
import json
import glob
import csv
import re
import argparse
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional, List

LABELS_DIR = "/workspace/mmWave/data/mRI/mRI/dataset_release/raw_data/videolabels"
RADAR_DIR = "/workspace/mmWave/data/mRI/mRI/dataset_release/raw_data/radar"
OUTPUT_ROOT = "/workspace/mmWave/mmWaveQA_benchmark"



class BaseGenerator:
    def __init__(self, labels_dir: str = LABELS_DIR, radar_dir: str = RADAR_DIR, output_root: str = OUTPUT_ROOT):
        self.labels_dir = labels_dir
        self.radar_dir = radar_dir
        self.output_root = output_root
        # Default pose rename mapping; child classes may override
        self.pose_rename = {
            "pose_1": "left upper limb extension",
            "pose_2": "right upper limb extension",
            "pose_3": "both upper limb extension",
            "pose_4": "left front lunge",
            "pose_5": "right front lunge",
            "pose_6": "squat",
            "pose_7": "left side lunge",
            "pose_8": "right side lunge",
            "pose_9": "left limb extension",
            "pose_10": "right limb",
        }

    def load_all_ranges(self, labels_path: str) -> Dict[str, Tuple[int, int]]:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            labels = data.get("labels", {})
            out: Dict[str, Tuple[int, int]] = {}
            if isinstance(labels, dict):
                for k, v in labels.items():
                    if isinstance(v, list) and len(v) == 2:
                        try:
                            s, e = int(v[0]), int(v[1])
                            if e > s:
                                out[str(k)] = (s, e)
                        except Exception:
                            continue
            return out
        except Exception:
            return {}

    def find_radar_csv(self, subject_id: str) -> Optional[str]:
        pattern = os.path.join(self.radar_dir, f"{subject_id}_*.csv")
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches, key=len)[0]
        return None

    def subject_label(self, subject_id: str) -> str:
        m = re.match(r"subject(\d+)$", subject_id)
        if not m:
            return subject_id
        n = int(m.group(1))
        return f"subject_{n:02d}"

    def build_point_cloud_from_csv(self, csv_path: str, start: int, end: int) -> Dict[str, List[List[float]]]:
        frames: Dict[int, List[List[float]]] = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if not row or len(row) < 8:
                    continue
                try:
                    frame_idx = int(round(float(row[0])))
                except Exception:
                    continue
                if frame_idx < start or frame_idx >= end:
                    continue
                try:
                    x = float(row[2]); y = float(row[3]); z = float(row[4])
                    doppler = float(row[5]); intensity = float(row[6])
                except Exception:
                    continue
                frames.setdefault(frame_idx, []).append([x, y, z, doppler, intensity])
        pc: Dict[str, List[List[float]]] = {}
        for idx in sorted(frames.keys()):
            pc[f"frame_{idx:06d}"] = frames[idx]
        return pc


class MovementGenerator(BaseGenerator):
    def __init__(self, output_root: str = OUTPUT_ROOT):
        super().__init__(output_root=output_root)
        self.output_dir = os.path.join(self.output_root, "movement")
        self.question_input = [
            "Has the person’s center position shifted in coordinates?",
            "Did the person’s central point move based on their coordinates?",
            "Is there a displacement of the person’s center position?",
            "Has the person’s position centroid changed location?",
            "Did the person’s spatial center move across coordinates?",
            "Is the person’s central coordinate shifting over time?",
        ]

    def generate_subject_actions(self, subject_id: str) -> Dict[str, Dict]:
        labels_path = os.path.join(self.labels_dir, f"{subject_id}.json")
        ranges = self.load_all_ranges(labels_path)
        if not ranges:
            return {}
        csv_path = self.find_radar_csv(subject_id)
        if not csv_path:
            return {}
        subj_label = self.subject_label(subject_id)
        outputs: Dict[str, Dict] = {}
        for action_name, (start, end) in ranges.items():
            point_cloud = self.build_point_cloud_from_csv(csv_path, start, end)
            action_out_name = self.pose_rename.get(action_name, action_name)
            payload = {
                "benchmarkfrom": "mRI",
                "subject": subj_label,
                "action": action_out_name,
                "question_category": "movement",
                "question_subconcepts": "rgith left movement",
                "question_type": "binary",
                "question_input": self.question_input,
                "point_cloud": point_cloud,
            }
            outputs[action_out_name] = payload
        return outputs

    def run(self) -> Tuple[int, int]:
        os.makedirs(self.output_dir, exist_ok=True)
        label_files = sorted(glob.glob(os.path.join(self.labels_dir, "subject*.json")))
        saved = 0
        skipped = 0
        for lf in tqdm(label_files, desc="Movement: subjects"):
            base = os.path.basename(lf)
            subject_id = base.replace(".json", "")
            subject_folder = self.subject_label(subject_id)
            subj_out_dir = os.path.join(self.output_dir, subject_folder)
            os.makedirs(subj_out_dir, exist_ok=True)
            payloads = self.generate_subject_actions(subject_id)
            if not payloads:
                skipped += 1
                continue
            for action_name, payload in tqdm(payloads.items(), desc=f"{subject_folder}: actions", leave=False):
                action_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", action_name)
                out_path = os.path.join(subj_out_dir, f"{action_safe}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                saved += 1
        print(f"[Movement] Saved {saved} action files, skipped {skipped} subjects. Out: {self.output_dir}")
        return saved, skipped


class TemporalActionGenerator(BaseGenerator):
    def __init__(self, output_root: str = OUTPUT_ROOT, comb_len: int = 2):
        super().__init__(output_root=output_root)
        self.output_dir = os.path.join(self.output_root, "temporal_action")
        self.comb_len = max(2, int(comb_len))

    def _sorted_actions(self, labels_path: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Return list of (action_name, (start,end)) sorted by start frame."""
        ranges = self.load_all_ranges(labels_path)
        items = list(ranges.items())
        items.sort(key=lambda kv: kv[1][0])
        return items

    def _window_sequences(self, actions_sorted: List[Tuple[str, Tuple[int, int]]]) -> List[List[Tuple[str, Tuple[int, int]]]]:
        k = self.comb_len
        seqs: List[List[Tuple[str, Tuple[int, int]]]] = []
        for i in range(0, len(actions_sorted) - k + 1):
            seqs.append(actions_sorted[i:i+k])
        return seqs

    def _build_pc_for_window(self, csv_path: str, seq: List[Tuple[str, Tuple[int, int]]]) -> Dict[str, List[List[float]]]:
        start = seq[0][1][0]
        end = seq[-1][1][1]
        return self.build_point_cloud_from_csv(csv_path, start, end)

    def _qa_previous(self, seq_names: List[str]) -> Tuple[List[str], str]:
        # Ask about the action immediately before a randomly chosen anchor (index 1..k-1)
        k = len(seq_names)
        anchor_idx = 1 if k <= 2 else random.randint(1, k - 1)
        anchor_raw = seq_names[anchor_idx]
        anchor = self.pose_rename.get(anchor_raw, anchor_raw)
        answer = seq_names[anchor_idx - 1]
        q = [
            f"What action occurs immediately before {anchor}?",
            f"Which pose precedes {anchor}?",
            f"Right before {anchor}, what is the action?",
            f"Identify the pose directly before {anchor}.",
            f"Which action comes just prior to {anchor}?",
            f"Immediately preceding {anchor}, which pose is observed?",
        ]
        return q, answer

    def _qa_present(self, seq_names: List[str]) -> Tuple[List[str], str]:
        k = len(seq_names)
        if k == 2:
            answer = seq_names[0]
            q = [
                "What is the first action in this segment?",
                "Identify the initial pose in this sequence.",
                "Which action starts this segment?",
                "At the beginning of this segment, what is the action?",
                "Which pose appears first in this sequence?",
                "What is the starting pose for this segment?",
            ]
            return q, answer
        if k == 3:
            answer = seq_names[1]
            q = [
                "What is the middle action in this segment?",
                "Identify the center pose in this three-action sequence.",
                "Which action occurs between the first and the last?",
                "At the midpoint of this segment, what is the action?",
                "Which pose appears in the middle of the sequence?",
                "What is the second action in this three-step sequence?",
            ]
            return q, answer
        # k >= 4: ask for the last action
        answer = seq_names[-1]
        q = [
            "What is the last action in this segment?",
            "Identify the final pose in this sequence.",
            "Which action concludes this segment?",
            "At the end of this segment, what is the action?",
            "Which pose appears last in the sequence?",
            "What is the ending pose for this segment?",
        ]
        return q, answer

    def _qa_next(self, seq_names: List[str]) -> Tuple[List[str], str]:
        # Ask about the action immediately after a randomly chosen anchor (index 0..k-2)
        k = len(seq_names)
        anchor_idx = 0 if k <= 2 else random.randint(0, k - 2)
        anchor_raw = seq_names[anchor_idx]
        anchor = self.pose_rename.get(anchor_raw, anchor_raw)
        answer = seq_names[anchor_idx + 1]
        q = [
            f"What action occurs immediately after {anchor}?",
            f"Which pose follows {anchor}?",
            f"Right after {anchor}, what is the action?",
            f"Identify the pose that comes directly after {anchor}.",
            f"Which action happens just after {anchor}?",
            f"Immediately following {anchor}, which pose is observed?",
        ]
        return q, answer

    def run(self) -> Tuple[int, int]:
        os.makedirs(self.output_dir, exist_ok=True)
        comb_dir = os.path.join(self.output_dir, f"comb_{self.comb_len}")
        os.makedirs(comb_dir, exist_ok=True)

        label_files = sorted(glob.glob(os.path.join(self.labels_dir, "subject*.json")))
        saved = 0
        skipped = 0

        for lf in tqdm(label_files, desc=f"Temporal comb={self.comb_len}: subjects"):
            base = os.path.basename(lf)
            subject_id = base.replace(".json", "")
            subject_folder = self.subject_label(subject_id)
            subj_out_dir = os.path.join(comb_dir, subject_folder)
            os.makedirs(subj_out_dir, exist_ok=True)

            actions_sorted = self._sorted_actions(lf)
            if len(actions_sorted) < self.comb_len:
                skipped += 1
                continue
            csv_path = self.find_radar_csv(subject_id)
            if not csv_path:
                skipped += 1
                continue

            safe_re = re.compile(r"[^A-Za-z0-9_\-~]")
            seqs = self._window_sequences(actions_sorted)
            for seq_idx, seq in enumerate(tqdm(seqs, desc=f"{subject_folder}: windows", leave=False)):
                seq_names_raw = [name for name, _ in seq]
                # Use original names in questions; provide renamed in metadata
                seq_names_desc = [self.pose_rename.get(n, n) for n in seq_names_raw]
                pc = self._build_pc_for_window(csv_path, seq)

                # Build three QA variants
                qa_variants = {
                    "previous": self._qa_previous(seq_names_raw),
                    "present": self._qa_present(seq_names_raw),
                    "next": self._qa_next(seq_names_raw),
                }

                # Build pool of candidate option texts (descriptive names from all labels)
                full_pool_desc = list({self.pose_rename.get(n, n) for n, _ in actions_sorted})

                for qtype, (q_list, answer) in qa_variants.items():
                    # correct answer as descriptive text
                    correct_text = self.pose_rename.get(answer, answer)
                    # pick 3 random distractors
                    distractor_pool = [x for x in full_pool_desc if x != correct_text]
                    if len(distractor_pool) < 3:
                        # extend with any remaining pose descriptions to ensure enough options
                        extra = [v for v in self.pose_rename.values() if v not in distractor_pool and v != correct_text]
                        distractor_pool = list(dict.fromkeys(distractor_pool + extra))
                    sampled = random.sample(distractor_pool, k=min(3, len(distractor_pool)))
                    options = [correct_text] + sampled
                    random.shuffle(options)
                    # Label options as A./B./C./D. in order
                    labels = ["A", "B", "C", "D"]
                    options = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

                    # action_sequence map: raw label -> [start, end]
                    action_seq_map = {name: [rng[0], rng[1]] for name, rng in seq}

                    payload = {
                        "benchmarkfrom": "mRI",
                        "subject": subject_folder,
                        "action_sequence": action_seq_map,
                        "action_sequence_desc": seq_names_desc,
                        "question_category": "temporal_action",
                        "question_type": "4_options",
                        "question_subcontexts": qtype,
                        "question_input": q_list,
                        "options": options,
                        "answer": correct_text,
                        "point_cloud": pc,
                    }
                    seq_name_join = "~".join(seq_names_raw)
                    seq_name_safe = safe_re.sub("_", seq_name_join)
                    fname = f"seq_{seq_idx:04d}_{seq_name_safe}_{qtype}.json"
                    out_path = os.path.join(subj_out_dir, fname)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    saved += 1

        print(f"[Temporal] Saved {saved} files (comb={self.comb_len}), skipped {skipped} subjects. Out: {comb_dir}")
        return saved, skipped


class CountingGenerator(BaseGenerator):
    def __init__(self, output_root: str = OUTPUT_ROOT, comb_len: int = 2):
        super().__init__(output_root=output_root)
        self.output_dir = os.path.join(self.output_root, "counting")
        self.comb_len = max(2, int(comb_len))

    def _sorted_actions(self, labels_path: str) -> List[Tuple[str, Tuple[int, int]]]:
        ranges = self.load_all_ranges(labels_path)
        items = list(ranges.items())
        items.sort(key=lambda kv: kv[1][0])
        return items

    def _window_sequences(self, actions_sorted: List[Tuple[str, Tuple[int, int]]]) -> List[List[Tuple[str, Tuple[int, int]]]]:
        k = self.comb_len
        seqs: List[List[Tuple[str, Tuple[int, int]]]] = []
        for i in range(0, len(actions_sorted) - k + 1):
            seqs.append(actions_sorted[i:i+k])
        return seqs

    def _build_pc_for_window(self, csv_path: str, seq: List[Tuple[str, Tuple[int, int]]]) -> Dict[str, List[List[float]]]:
        start = seq[0][1][0]
        end = seq[-1][1][1]
        return self.build_point_cloud_from_csv(csv_path, start, end)

    def _qa_counting(self, seq_names: List[str]) -> Tuple[List[str], int]:
        # Count distinct actions in the window
        distinct = len(set(seq_names))
        q = [
            "How many distinct actions appear in this segment?",
            "Count the unique poses shown in this sequence.",
            "What is the number of different actions in this window?",
            "How many unique action types are present here?",
            "What count of distinct poses occurs in this segment?",
            "Determine the number of unique actions within this sequence.",
        ]
        return q, distinct

    def run(self) -> Tuple[int, int]:
        os.makedirs(self.output_dir, exist_ok=True)
        comb_dir = os.path.join(self.output_dir, f"comb_{self.comb_len}")
        os.makedirs(comb_dir, exist_ok=True)

        label_files = sorted(glob.glob(os.path.join(self.labels_dir, "subject*.json")))
        saved = 0
        skipped = 0

        for lf in tqdm(label_files, desc=f"Counting comb={self.comb_len}: subjects"):
            base = os.path.basename(lf)
            subject_id = base.replace(".json", "")
            subject_folder = self.subject_label(subject_id)
            subj_out_dir = os.path.join(comb_dir, subject_folder)
            os.makedirs(subj_out_dir, exist_ok=True)

            actions_sorted = self._sorted_actions(lf)
            if len(actions_sorted) < self.comb_len:
                skipped += 1
                continue
            csv_path = self.find_radar_csv(subject_id)
            if not csv_path:
                skipped += 1
                continue

            safe_re = re.compile(r"[^A-Za-z0-9_\-~]")
            seqs = self._window_sequences(actions_sorted)
            for seq_idx, seq in enumerate(tqdm(seqs, desc=f"{subject_folder}: windows", leave=False)):
                seq_names_raw = [name for name, _ in seq]
                seq_names_desc = [self.pose_rename.get(n, n) for n in seq_names_raw]
                pc = self._build_pc_for_window(csv_path, seq)

                q_list, distinct = self._qa_counting(seq_names_raw)

                # Options fixed A-D as 1..4
                options = ["A. 1", "B. 2", "C. 3", "D. 4"]
                # Correct answer text
                correct_text = f"{distinct}"

                # action_sequence map
                action_seq_map = {name: [rng[0], rng[1]] for name, rng in seq}

                payload = {
                    "benchmarkfrom": "mRI",
                    "subject": subject_folder,
                    "action_sequence": action_seq_map,
                    "action_sequence_desc": seq_names_desc,
                    "question_category": "counting",
                    "question_type": "4_options",
                    "question_subcontexts": "count_distinct",
                    "question_input": q_list,
                    "options": options,
                    "answer": correct_text,
                    "point_cloud": pc,
                }
                seq_name_join = "~".join(seq_names_raw)
                seq_name_safe = safe_re.sub("_", seq_name_join)
                fname = f"seq_{seq_idx:04d}_{seq_name_safe}_count.json"
                out_path = os.path.join(subj_out_dir, fname)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                saved += 1

        print(f"[Counting] Saved {saved} files (comb={self.comb_len}), skipped {skipped} subjects. Out: {comb_dir}")
        return saved, skipped

def main():
    parser = argparse.ArgumentParser(description="Generate mmWave QA benchmark (movement, temporal, counting)")
    parser.add_argument("--task", choices=["movement", "temporal", "counting", "pipeline"], default="movement")
    parser.add_argument("--labels-dir", type=str, default=LABELS_DIR)
    parser.add_argument("--radar-dir", type=str, default=RADAR_DIR)
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--comb-len", type=int, default=2, help="Temporal combination length (2, 3, or 4)")
    args = parser.parse_args()

    if args.task in ("movement", "pipeline"):
        mg = MovementGenerator(output_root=args.output_root)
        mg.labels_dir = args.labels_dir
        mg.radar_dir = args.radar_dir
        mg.run()

    if args.task in ("temporal", "pipeline"):
        tg = TemporalActionGenerator(output_root=args.output_root, comb_len=args.comb_len)
        tg.labels_dir = args.labels_dir
        tg.radar_dir = args.radar_dir
        tg.run()

    if args.task == "counting":
        cg = CountingGenerator(output_root=args.output_root, comb_len=args.comb_len)
        cg.labels_dir = args.labels_dir
        cg.radar_dir = args.radar_dir
        cg.run()


if __name__ == "__main__":
    main()


