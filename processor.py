#!/usr/bin/env python3
"""
Benchmark Response Processor

This script processes filtered_benchmark_responses.json through {model_name} and generates responses.
Input: results_pro/filtered_benchmark_responses.json
Output: results_pro/{model_name}_responses.json
"""

import os
import json
import logging
import numpy as np
import struct
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from openai import OpenAI
import time
import random
import argparse
from tqdm import tqdm

# ===== Configuration =====
INPUT_FILE = ""
OUTPUT_FILE = ""
OPENAI_API_KEY = ""
MODEL = ""  

# Data paths
MMBODY_BASE_PATH = ""
MMFI_BASE_PATH = ""
MRI_BASE_PATH = ""

# Ensure log directory exists
os.makedirs('log', exist_ok=True)

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/benchmark_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

UNIFORM_FRAMES = 16
OUTPUT_FILE = f"/workspace/mmWave/results_New/responses_frames_{UNIFORM_FRAMES}.json"
REQUEST_INTERVAL_SEC = 1.0
MAX_RETRIES = 5

class GPT5BenchmarkProcessor:
    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        uniform_frames: int = UNIFORM_FRAMES,
        request_interval: float = REQUEST_INTERVAL_SEC,
        enable_reasoning: bool = False,
    ):
        """Initialize the {model_name} benchmark processor."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.processed_count = 0
        self.error_count = 0
        self.uniform_frames = uniform_frames
        self.request_interval = request_interval
        self.enable_reasoning = enable_reasoning
        
    def get_instruction(self, data_source: str = "mmfi") -> str:
        """Get the instruction for {model_name} based on data source."""
        if data_source == "mmbody":
            return (
                "You are an expert in mmWave radar point cloud analysis. "
                "Given a sequence of mmWave radar point cloud frames, identify the human action being performed. "
                "Each point represents the 3D location, range velocity, amplitude, and energy power of a reflected wave of the corresponding point in the scene. "
                "Select exactly one option (A, B, C, or D) that best describes the action."
            )
        else:  # mmfi or mri
            return (
                "You are an expert in mmWave radar point cloud analysis. "
                "Given a sequence of mmWave radar point cloud frames, identify the human action being performed. "
                "Each frame contains points with coordinates [x, y, z, doppler, intensity]. "
                "Select exactly one option (A, B, C, or D) that best describes the action."
            )
    
    def _subsample_list(self, items: List[Any], max_count: int) -> List[Any]:
        """Uniformly subsample a list to at most max_count elements."""
        if max_count <= 0 or len(items) <= max_count:
            return items
        step = len(items) / float(max_count)
        indices = [int(i * step) for i in range(max_count)]
        # Ensure uniqueness and bounds
        indices = sorted(set(min(idx, len(items) - 1) for idx in indices))
        return [items[i] for i in indices]

    def _select_uniform_frames(self, frames: Dict[str, List[List[float]]], limit: int) -> Dict[str, List[List[float]]]:
        """Select up to 'limit' frames uniformly across the sequence (no point subsampling)."""
        if limit <= 0:
            return {}
        keys = sorted(frames.keys())
        if len(keys) <= limit:
            selected = keys
        else:
            step = len(keys) / float(limit)
            indices = [int(i * step) for i in range(limit)]
            # Ensure uniqueness and bounds
            indices = sorted(set(min(idx, len(keys) - 1) for idx in indices))
            selected = [keys[i] for i in indices]
        
        return {key: frames[key] for key in selected}

    def load_mmbody_point_cloud(self, segment: str, mode: str, start_frame: int, end_frame: int) -> Dict[str, List[List[float]]]:
        """Load point cloud data from mmBody dataset."""
        segment_path = os.path.join(MMBODY_BASE_PATH, mode, segment)
        
        if not os.path.exists(segment_path):
            logger.warning(f"mmBody path not found: {segment_path}")
            return {}
        
        frame_clouds = {}
        
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = os.path.join(segment_path, f"frame_{frame_idx}.npy")
            
            if os.path.exists(frame_file):
                try:
                    # Load numpy array
                    frame_data = np.load(frame_file)
                    
                    # Convert to list format and round to 2 decimal places
                    if frame_data.size > 0:
                        if frame_data.ndim == 2 and frame_data.shape[1] >= 4:
                            # mmBody filtered data has 6 columns: x, y, z, doppler, intensity, energy
                            if frame_data.shape[1] == 6:
                                points = [[round(float(x), 2), round(float(y), 2), round(float(z), 2), round(float(doppler), 2), round(float(intensity), 2)] 
                                        for x, y, z, doppler, intensity, energy in frame_data]
                            elif frame_data.shape[1] == 5:
                                points = [[round(float(x), 2), round(float(y), 2), round(float(z), 2), round(float(doppler), 2), round(float(intensity), 2)] 
                                        for x, y, z, doppler, intensity in frame_data]
                            else:  # 4 columns
                                points = [[round(float(x), 2), round(float(y), 2), round(float(z), 2), round(float(intensity), 2)] 
                                        for x, y, z, intensity in frame_data]
                        else:
                            points = []
                        
                        if points:
                            frame_clouds[str(frame_idx)] = points
                            
                except Exception as e:
                    logger.warning(f"Error loading mmBody frame {frame_idx}: {e}")
                    continue
        
        return frame_clouds

    def load_mmfi_point_cloud(self, segment: str, start_frame: int, end_frame: int) -> Dict[str, List[List[float]]]:
        """Load point cloud data from mmFi dataset."""
        # segment 형식: E01_S01_A02
        parts = segment.split('_')
        if len(parts) != 3:
            logger.warning(f"Invalid mmFi segment format: {segment}")
            return {}
        
        env, session, action = parts
        segment_path = os.path.join(MMFI_BASE_PATH, env, session, action, "mmwave")
        
        if not os.path.exists(segment_path):
            logger.warning(f"mmFi path not found: {segment_path}")
            return {}
        
        frame_clouds = {}
        
        for frame_idx in range(start_frame, end_frame + 1):
            # mmFi는 frame001.bin부터 시작 (1-indexed)
            frame_file = os.path.join(segment_path, f"frame{frame_idx+1:03d}.bin")
            
            if os.path.exists(frame_file):
                try:
                    with open(frame_file, 'rb') as f:
                        data = f.read()
                        
                        # 4개의 float64 값 (32바이트)씩 읽기
                        num_points = len(data) // 32
                        points = []
                        
                        for i in range(num_points):
                            offset = i * 32
                            # x, y, z, intensity (float64)
                            x, y, z, intensity = struct.unpack('dddd', data[offset:offset+32])
                            
                            # 비정상적인 값 필터링
                            if abs(x) > 100 or abs(y) > 100 or abs(z) > 100 or abs(intensity) > 100:
                                continue
                            
                            # 소수점 2자리로 제한
                            points.append([round(float(x), 2), round(float(y), 2), round(float(z), 2), round(float(intensity), 2)])
                        
                        if points:
                            frame_clouds[str(frame_idx)] = points
                            
                except Exception as e:
                    logger.warning(f"Error loading mmFi frame {frame_idx}: {e}")
                    continue
        
        return frame_clouds

    def load_mri_point_cloud(self, subject: str, start_frame: int, end_frame: int) -> Dict[str, List[List[float]]]:
        """Load point cloud data from mRI dataset."""
        # subject 형식: subject1, subject2, ...
        # CSV 파일 형식: subject1_0427.csv, subject2_0428.csv, ...
        
        # subject 번호에 따른 날짜 매핑
        subject_date_map = {
            'subject1': '0427', 'subject2': '0428', 'subject3': '0501', 'subject4': '0501',
            'subject5': '0502', 'subject6': '0503', 'subject7': '0503', 'subject8': '0503',
            'subject9': '0504', 'subject10': '0504', 'subject11': '0505', 'subject12': '0505',
            'subject13': '0505', 'subject14': '0505', 'subject15': '0508', 'subject16': '0508',
            'subject17': '0510', 'subject18': '0511', 'subject19': '0513', 'subject20': '0513'
        }
        
        if subject not in subject_date_map:
            logger.warning(f"Date mapping not found for {subject}")
            return {}
        
        date = subject_date_map[subject]
        csv_file = os.path.join(MRI_BASE_PATH, f"{subject}_{date}.csv")
        
        if not os.path.exists(csv_file):
            logger.warning(f"mRI CSV file not found: {csv_file}")
            return {}
        
        try:
            frames = {}
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # 헤더 건너뛰기
                _ = next(reader, None)
                
                for row in reader:
                    if not row or len(row) < 8:
                        continue
                    try:
                        frame_idx = int(round(float(row[0])))
                    except Exception:
                        continue
                    
                    # 프레임 범위 체크 (end_frame 포함, 범위 밖이면 건너뛰기)
                    if frame_idx < start_frame:
                        continue
                    if frame_idx > end_frame:
                        continue
                    
                    try:
                        # X, Y, Z, Intensity 추출 및 소수점 2자리로 반올림 (Doppler 제외)
                        x = round(float(row[2]), 2)
                        y = round(float(row[3]), 2)
                        z = round(float(row[4]), 2)
                        intensity = round(float(row[6]), 2)
                        
                        # 프레임별로 포인트 저장
                        frames.setdefault(frame_idx, []).append([x, y, z, intensity])
                        
                    except Exception:
                        continue
            
            # 딕셔너리 키를 문자열 형식으로 변환 (다른 로더들과 일관되게 순수 인덱스 문자열 사용)
            pc = {}
            for idx in sorted(frames.keys()):
                pc[str(idx)] = frames[idx]
            
            return pc
            
        except Exception as e:
            logger.warning(f"Error loading mRI CSV: {e}")
            return {}

    def load_point_cloud(self, info: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Load point cloud data based on dataset type."""
        data_source = info.get("from", "")
        start_frame = info.get("frames", {}).get("start", 0)
        end_frame = info.get("frames", {}).get("end", start_frame)
        
        if data_source == "mmbody":
            segment = info.get("segment", "")
            mode = info.get("mode", "train")
            return self.load_mmbody_point_cloud(segment, mode, start_frame, end_frame)
        
        elif data_source == "mmfi":
            segment = info.get("segment", "")
            return self.load_mmfi_point_cloud(segment, start_frame, end_frame)
        
        elif data_source == "mri":
            segment = info.get("segment", "")
            # segment 형식: subject16_pose_7 -> subject16 추출
            subject = segment.split('_')[0] if '_' in segment else segment
            return self.load_mri_point_cloud(subject, start_frame, end_frame)
        
        else:
            logger.warning(f"Unknown data source: {data_source}")
            return {}

    def _call_gpt5_with_retry(self, instruction: str, point_cloud: Dict[str, List[List[float]]], 
                            question: str, options: List[str], data_source: str = "mmfi") -> Tuple[str, str, str]:
        """Call GPT-5 with retry logic."""
        # Uniformly sample frames
        sampled_frames = self._select_uniform_frames(point_cloud, self.uniform_frames)
        
        if not sampled_frames:
            logger.warning("No frames available for processing")
            return "", "", "No frames available"
        
        # Format options with A, B, C, D labels
        opts_str = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])

        # Compose prompts (reasoning/non-reasoning)
        system_prompt = f"{instruction}"
        if self.enable_reasoning:
            user_content = (
                f"Point Cloud Data (frames with radar point coordinates):\n"
                f"{json.dumps(sampled_frames, ensure_ascii=False, separators=(',', ':'))}\n\n"
                f"Question: {question}\n"
                f"Options:\n{opts_str}\n\n"
                "Respond with ONLY a valid JSON object in this exact format (no markdown code blocks, no additional text):\n"
                "{\n  \"Rationale\": \"Brief reasoning for the selected action\",\n  \"Answer\": \"A\"\n}"
            )
        else:
            user_content = (
                f"Point Cloud Data (frames with radar point coordinates):\n"
                f"{json.dumps(sampled_frames, ensure_ascii=False, separators=(',', ':'))}\n\n"
                f"Question: {question}\n"
                f"Options:\n{opts_str}\n\n"
                f"Answer with exactly one option letter (A, B, C, or D).\n"
                f"Answer:"
            )
        
        backoff = 1.5
        delay = 2.0
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                #response = self.client.chat.completions.create(
                #    model=self.model,
                #    messages=[
                #        {
                #            "role": "system",
                #            "content": "You are an expert in mmWave radar point cloud analysis."
                #        },
                #        {
                #            "role": "user",
                #            "content": user_content
                #        }
                #    ],
                #    max_completion_tokens=100
                #)
                
                response = self.client.responses.create(
                    model = self.model,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_content}],
                        },
                    ],
                )
                
                gpt_response = response.output_text if hasattr(response, 'output_text') else str(response)
                logger.info(f"GPT-5 response: '{gpt_response}'")
                time.sleep(self.request_interval)
                
                # Parse response
                if self.enable_reasoning:
                    # Try JSON parsing for {"Rationale": "...", "Answer": "A"}
                    reasoning_text = ""
                    answer = ""
                    try:
                        cleaned = gpt_response.strip()
                        if cleaned.startswith("```json"):
                            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
                        elif cleaned.startswith("```"):
                            cleaned = cleaned.replace("```", "").strip()
                        parsed = json.loads(cleaned)
                        reasoning_text = str(parsed.get("Rationale", "")).strip()
                        answer_part = str(parsed.get("Answer", "")).strip().upper()
                        if answer_part in ["A", "B", "C", "D"]:
                            answer = answer_part
                        elif answer_part[:1] in ["A", "B", "C", "D"]:
                            answer = answer_part[:1]
                    except Exception:
                        reasoning_text = gpt_response
                        answer = ""
                    if not answer:
                        # Fallback to pattern-based extraction
                        import re
                        patterns = [r'Answer:\s*([ABCD])', r'Option\s*([ABCD])', r'\b([ABCD])\b']
                        for pattern in patterns:
                            m = re.search(pattern, gpt_response, re.IGNORECASE)
                            if m:
                                answer = m.group(1).upper()
                                break
                        if not answer:
                            logger.warning(f"No valid answer found in response: {gpt_response}")
                            answer = "A"
                    return answer, reasoning_text, ""
                else:
                    # Extract answer (A, B, C, or D) - more flexible parsing
                    answer = ""
                    import re
                    patterns = [
                        r'[ABCD]\)',  # A), B), C), D)
                        r'[ABCD]\.',  # A., B., C., D.
                        r'Answer:\s*([ABCD])',  # Answer: A
                        r'Option\s*([ABCD])',  # Option A
                        r'\b([ABCD])\b'  # Just A, B, C, D
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, gpt_response, re.IGNORECASE)
                        if matches:
                            answer = matches[0].upper()
                            break
                    if not answer:
                        logger.warning(f"No valid answer found in response: {gpt_response}")
                        answer = "A"  # Default fallback
                    return answer, gpt_response, ""
                
            except Exception as e:
                err_text = str(e)
                last_error = err_text
                is_rate_limit = ("rate limit" in err_text.lower()) or ("429" in err_text)
                
                if is_rate_limit:
                    wait_s = None
                    try:
                        import re
                        m = re.search(r"try again in ([0-9]+\.?[0-9]*)s", err_text, flags=re.IGNORECASE)
                        if m:
                            wait_s = float(m.group(1))
                    except Exception:
                        wait_s = None
                    sleep_s = wait_s if wait_s is not None else delay
                    logger.warning(f"Rate limit on attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    delay *= backoff
                    continue
                    
                logger.warning(f"{model_name} call failed (attempt {attempt}/{MAX_RETRIES}): {err_text}")
                time.sleep(delay)
                delay *= backoff
                continue
                
        # Exhausted retries
        return "", "", f"Failed to call GPT-5 after {MAX_RETRIES} attempts. Last error: {last_error}"

    def process_qa_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single QA entry."""
        try:
            info = entry.get("info", {})
            data_source = info.get("from", "")
            
            # Load point cloud data
            point_cloud = self.load_point_cloud(info)
            
            if not point_cloud:
                logger.warning(f"No point cloud data loaded for entry")
                return {
                    **entry,
                    "gpt5_response": "",
                    "gpt5_answer": "",
                    "gpt5_reasoning": "No point cloud data available",
                    "error": "No point cloud data"
                }
            
            # Get instruction and call GPT-5
            instruction = self.get_instruction(data_source)
            question = entry.get("question", "")
            options = entry.get("options", [])
            
            answer, reasoning, error = self._call_gpt5_with_retry(
                instruction, point_cloud, question, options, data_source
            )
            
            if error:
                self.error_count += 1
                logger.error(f"Error processing entry: {error}")
            
            result = {
                **entry,
                "gpt5_response": reasoning,
                "gpt5_answer": answer,
                "gpt5_reasoning": reasoning,
                "error": error
            }
            
            self.processed_count += 1
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing entry: {e}")
            return {
                **entry,
                "gpt5_response": "",
                "gpt5_answer": "",
                "gpt5_reasoning": "",
                "error": str(e)
            }

    def process_benchmark_file(self, input_file: str, output_file: str, max_entries: int = None, test_each_dataset: bool = False):
        """Process the benchmark file and generate GPT-5 responses."""
        logger.info(f"Loading benchmark data from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        if test_each_dataset:
            # Find one entry from each dataset
            selected_entries = []
            datasets_found = set()
            
            for entry in benchmark_data:
                source = entry.get('info', {}).get('from', '')
                if source not in datasets_found:
                    selected_entries.append(entry)
                    datasets_found.add(source)
                    logger.info(f"Selected {source} entry: {entry.get('info', {})}")
                    if len(datasets_found) == 3:  # mmbody, mmfi, mri
                        break
            
            benchmark_data = selected_entries
            logger.info(f"Processing {len(benchmark_data)} entries (one from each dataset)")
        elif max_entries:
            benchmark_data = benchmark_data[:max_entries]
            logger.info(f"Processing {max_entries} entries")
        
        results = []
        
        for i, entry in enumerate(tqdm(benchmark_data, desc="Processing entries")):
            logger.info(f"Processing entry {i+1}/{len(benchmark_data)}")
            
            result = self.process_qa_entry(entry)
            results.append(result)
            
            # Add delay between requests
            if i < len(benchmark_data) - 1:
                time.sleep(self.request_interval)
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Processed: {self.processed_count}, Errors: {self.error_count}")

def main():
    parser = argparse.ArgumentParser(description='Process benchmark data with {model_name}')
    parser.add_argument('--input', default=INPUT_FILE, help='Input benchmark file')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output file')
    parser.add_argument('--max-entries', type=int, help='Maximum number of entries to process')
    parser.add_argument('--test-each-dataset', action='store_true', help='Test one entry from each dataset (mmbody, mmfi, mri)')
    parser.add_argument('--uniform-frames', type=int, default=UNIFORM_FRAMES, help='Number of uniform frames')
    parser.add_argument('--request-interval', type=float, default=REQUEST_INTERVAL_SEC, help='Request interval in seconds')
    parser.add_argument('--enable-reasoning', action='store_true', default=False, help='Enable reasoning mode with rationale generation')
    
    args = parser.parse_args()
    
    processor = GPT5BenchmarkProcessor(
        api_key=OPENAI_API_KEY,
        model=MODEL,
        uniform_frames=args.uniform_frames,
        request_interval=args.request_interval,
        enable_reasoning=args.enable_reasoning,
    )
    
    processor.process_benchmark_file(
        input_file=args.input,
        output_file=args.output,
        max_entries=args.max_entries,
        test_each_dataset=args.test_each_dataset
    )

if __name__ == "__main__":
    main()
