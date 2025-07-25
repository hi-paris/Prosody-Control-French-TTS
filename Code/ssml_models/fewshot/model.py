#!/usr/bin/env python
"""
model.py – SSML prediction experiments (zero-shot & few-hot) with Ollama.
Uses YAML config, structured logging, ThreadPoolExecutor for LLM calls,
and multiprocessing for per‐model parallelism.
"""

from __future__ import annotations
import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import random
import re
import xml.sax.saxutils as saxutils
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ExperimentConfig:
    """All experiment knobs loaded from YAML."""
    # I/O
    data_path: str
    output_dir: str = "results"

    # Model / inference
    model_names: List[str] = field(default_factory=lambda: ["mistral"])
    temperature: float = None
    ollama_url: Optional[str] = None
    keep_alive: str = "5m"
    num_gpu: int = 3  # Number of GPUs to use (0 for CPU only)
    num_batch: int = 512  # Batch size for prompt processing
    num_ctx: int = 4096  # Context window size (smaller = faster)
    # mirostat: int = 0  # 0, 1, or 2 for different sampling strategies
    # repeat_penalty: float = 1.1  # Penalize repetition (1.0-1.5 range)

    # Evaluation
    num_samples: int = 20
    max_examples: int = 8
    mode: str = "both"             # zero-shot | few-shot | both
    break_position_threshold: int = 0
    prosody_position_threshold: int = 0

    # Parallelism
    worker_processes: Optional[int] = None  # per-model Pool
    parallel_requests: int = 4              # threads per process

    # Debug
    debug: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SSMLMetrics:
    total_tags: int = 0
    prosody_count: int = 0
    break_count: int = 0
    pitch_adjustments: List[float] = field(default_factory=list)
    rate_adjustments: List[float] = field(default_factory=list)
    volume_adjustments: List[float] = field(default_factory=list)
    break_times: List[int] = field(default_factory=list)
    tags_per_sentence: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Predictor Base & Zero-Shot / Few-Shot Implementations
# ─────────────────────────────────────────────────────────────────────────────
class SSMLPredictor:
    def __init__(self, cfg: ExperimentConfig, model_name: str):
        self.cfg = cfg
        self.model_name = model_name
        self.temperature = cfg.temperature

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "num_gpu": cfg.num_gpu,
            "num_ctx": cfg.num_ctx,
            "num_batch": cfg.num_batch
        }
        if cfg.ollama_url:
            kwargs["base_url"] = cfg.ollama_url
        if hasattr(OllamaLLM, "keep_alive"):
            kwargs["keep_alive"] = cfg.keep_alive

        self.llm = OllamaLLM(**kwargs)
        logger.debug("Initialized OllamaLLM with %s", kwargs)

    @staticmethod
    def _extract_metrics(ssml: str) -> SSMLMetrics:
        m = SSMLMetrics()
        m.total_tags = len(re.findall(r"<[^/][^>]*>", ssml))
        prosody_tags = re.findall(r"<prosody[^>]*>", ssml)
        break_tags = re.findall(r"<break[^>]*>", ssml)
        m.prosody_count = len(prosody_tags)
        m.break_count = len(break_tags)

        for tag in prosody_tags:
            for attr, store in (
                ("pitch", m.pitch_adjustments),
                ("rate", m.rate_adjustments),
                ("volume", m.volume_adjustments),
            ):
                if match := re.search(fr'{attr}="([^"]*)"', tag):
                    val = match.group(1).rstrip("%")
                    try:
                        store.append(float(val))
                    except ValueError:
                        pass

        for tag in break_tags:
            if match := re.search(r'time="([^"]*)"', tag):
                val = match.group(1).rstrip("ms")
                try:
                    m.break_times.append(int(val))
                except ValueError:
                    pass

        sentences = len(re.findall(r"[.!?]", ssml)) or 1
        m.tags_per_sentence = m.total_tags / sentences
        return m

    @staticmethod
    def _ensure_speak(ssml: str) -> str:
        if not ssml.startswith("<speak>"):
            ssml = f"<speak>\n{ssml}\n</speak>"
        return re.sub(r"</?speak>\s*</speak>", "</speak>", ssml, flags=re.S)

    def predict(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
class ZeroShotSSMLPredictor(SSMLPredictor):
    def __init__(self, cfg: ExperimentConfig, model_name: str):
        super().__init__(cfg, model_name)
        system = (
            "You are an expert in French Language for Text-to-Speech systems. "
            "Your task is to analyze the text and output parameters for generating very natural sounding speech. These will later be used in SSML (Speech Synthesis Markup Language). "
            "IMPORTANT: You must never change the input text content or generate new text. Use exactly the input text provided. "
            "Output valid JSON with `segments`[], each "
            "containing `text`, `prosody` {{pitch,rate,volume}}, `break_before` and `break_after`. "
            "The `text` field should contain the text to be spoken, i.e. the same as the input text. "
            "The `prosody` field should contain the prosody parameters for the text, each given as a signed percentage value with two decimals. These indicate the desired change in pitch, rate, and volume relative to the default baseline of the TTS system. Reasonable percentages are in the -10.00% to +10.00% range. "
            "The `break_before` and `break_after` fields should contain the pause a speaker should make before or after the text, in milliseconds. "
            "Give no other commentary."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Analyze this text and output parameters as JSON:\n\n{text}"),
        ])
        self.chain = self.prompt | self.llm

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from model responses with enhanced error handling and recovery."""
        # Remove thinking sections
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        
        # Strategy 1: Try to extract from code blocks
        if code_match := re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw):
            try:
                result = json.loads(code_match.group(1))
                # Handle list result by wrapping it in a dictionary
                if isinstance(result, list):
                    return {"segments": result}
                return result
            except json.JSONDecodeError:
                # If direct parsing fails, continue to other strategies
                payload = code_match.group(1)
        else:
            payload = raw.strip()
        
        # Strategy 2: Try to find the largest valid JSON object in the text
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
        matches = re.findall(json_pattern, payload)
        
        if matches:
            # Sort by length (largest first) to get the most complete JSON object
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    result = json.loads(match)
                    # Handle list result by wrapping it in a dictionary
                    if isinstance(result, list):
                        return {"segments": result}
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try to find valid JSON arrays in the text
        json_array_pattern = r'(\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\])'
        array_matches = re.findall(json_array_pattern, payload)
        
        if array_matches:
            # Sort by length (largest first)
            array_matches.sort(key=len, reverse=True)
            for match in array_matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, list):
                        return {"segments": result}
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Try to fix common JSON formatting issues and try again
        # Replace single quotes with double quotes for JSON properties
        fixed_payload = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', payload)
        # Replace trailing commas in arrays and objects
        fixed_payload = re.sub(r",(\s*[\]}])", r"\1", fixed_payload)
        
        try:
            result = json.loads(fixed_payload)
            # Handle list result by wrapping it in a dictionary
            if isinstance(result, list):
                return {"segments": result}
            return result
        except json.JSONDecodeError:
            pass
        
        # Default empty response
        return {"segments": []}

    def predict(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        raw = self.chain.invoke({"text": text})
        params = self._parse_json(raw)
        ssml = SSMLBuilder.build_from_params(params)
        metrics = self._extract_metrics(ssml)
        return {
            "input_text": text,
            "response": raw,
            "predicted_ssml": ssml,  # Renamed from 'ssml' to 'predicted_ssml'
            "ssml": ssml,            # Keep original field for backward compatibility
            "params": params,
            "metrics": asdict(metrics),
        }

# ─────────────────────────────────────────────────────────────────────────────
class FewShotSSMLPredictor(SSMLPredictor):
    def __init__(
        self,
        cfg: ExperimentConfig,
        model_name: str,
        max_examples: int,
        target_voice: Optional[str],
        train_data: List[Dict[str, Any]],
    ):
        super().__init__(cfg, model_name)
        self.max_examples = max_examples
        self.target_voice = target_voice
        self.examples = self._process_train_data(train_data)

        self.system_prompt = """
        You are an expert in SSML. I will give you examples of text→parameters JSON.
        Then predict for new text in the same JSON format, no commentary.
        """

    def _process_train_data(self, train_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        exs: List[Dict[str, Any]] = []
        for item in train_data:
            text = item.get("text", "").strip()
            seq = item.get("parsed_sequence", [])
            if not text or not isinstance(seq, list):
                continue
            params = self._convert_parsed_sequence_to_json(seq)
            exs.append({"text": text, "params": params, "voice": item.get("voice")})
        random.shuffle(exs)
        return exs[: self.max_examples]

    def _convert_parsed_sequence_to_json(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        segments: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for elem in sequence:
            t = elem.get("type")
            if t == "break":
                if current:
                    current["break_after"] = elem.get("time", "0ms")
                    segments.append(current)
                    current = None
                else:
                    current = {"break_before": elem.get("time", "0ms")}
            elif t == "text":
                if current and "text" not in current:
                    current["text"] = elem.get("text", "")
                    current["prosody"] = elem.get("prosody", {"pitch": "0%", "rate": "0%", "volume": "0%"})
                else:
                    current = {
                        "text": elem.get("text", ""),
                        "prosody": elem.get("prosody", {"pitch": "0%", "rate": "0%", "volume": "0%"}),
                        "break_before": "0ms",
                    }
        if current and "text" in current:
            current.setdefault("break_after", "0ms")
            segments.append(current)

        return {"segments": segments}

    def _build_prompt(self, text: str) -> ChatPromptTemplate:
        body = ""
        for i, ex in enumerate(self.examples, 1):
            j = json.dumps(ex["params"], indent=2)
            body += f"Example {i}:\nText: \"{ex['text']}\"\nParameters:\n```json\n{j}\n```\n\n"
        human = f"{body}Now predict for this text:\n\nText: \"{text}\"\n\nParameters:"
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human)
        ])

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from model responses with enhanced error handling and recovery."""
        # Remove thinking sections
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        
        # Strategy 1: Try to extract from code blocks
        if code_match := re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw):
            try:
                result = json.loads(code_match.group(1))
                # Handle list result by wrapping it in a dictionary
                if isinstance(result, list):
                    return {"segments": result}
                return result
            except json.JSONDecodeError:
                # If direct parsing fails, continue to other strategies
                payload = code_match.group(1)
        else:
            payload = raw.strip()
        
        # Strategy 2: Try to find the largest valid JSON object in the text
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
        matches = re.findall(json_pattern, payload)
        
        if matches:
            # Sort by length (largest first) to get the most complete JSON object
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    result = json.loads(match)
                    # Handle list result by wrapping it in a dictionary
                    if isinstance(result, list):
                        return {"segments": result}
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try to find valid JSON arrays in the text
        json_array_pattern = r'(\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\])'
        array_matches = re.findall(json_array_pattern, payload)
        
        if array_matches:
            # Sort by length (largest first)
            array_matches.sort(key=len, reverse=True)
            for match in array_matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, list):
                        return {"segments": result}
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Try to fix common JSON formatting issues and try again
        # Replace single quotes with double quotes for JSON properties
        fixed_payload = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', payload)
        # Replace trailing commas in arrays and objects
        fixed_payload = re.sub(r",(\s*[\]}])", r"\1", fixed_payload)
        
        try:
            result = json.loads(fixed_payload)
            # Handle list result by wrapping it in a dictionary
            if isinstance(result, list):
                return {"segments": result}
            return result
        except json.JSONDecodeError:
            pass
        
        # Default empty response
        return {"segments": []}
    
    def predict(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        if voice and self.target_voice and voice == self.target_voice:
            # filter out same-voice examples
            self.examples = [e for e in self.examples if e.get("voice") != voice]

        # Build prompt directly without template variables
        body = ""
        for i, ex in enumerate(self.examples, 1):
            j = json.dumps(ex["params"], indent=2)
            body += f"Example {i}:\nText: \"{ex['text']}\"\nParameters:\n```json\n{j}\n```\n\n"
        
        prompt_text = f"{body}Now predict for this text:\n\nText: \"{text}\"\n\nParameters:"
        
        # Use direct message objects instead of a template
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt_text)
        ]
        
        raw = self.llm.invoke(messages)
        params = self._parse_json(raw)
        ssml = SSMLBuilder.build_from_params(params)
        metrics = self._extract_metrics(ssml)
        
        return {
            "input_text": text,
            "response": raw,
            "predicted_ssml": ssml,  # Renamed from 'ssml' to 'predicted_ssml'
            "ssml": ssml,            # Keep original field for backward compatibility
            "params": params,
            "metrics": asdict(metrics),
            "examples_used": len(self.examples),
        }

# ─────────────────────────────────────────────────────────────────────────────
# SSML Builder
# ─────────────────────────────────────────────────────────────────────────────
class SSMLBuilder:
    @staticmethod
    def build_from_params(params: Dict[str, Any]) -> str:
        parts: List[str] = ["<speak>"]
        
        # Handle both dictionary with segments key and direct list of segments
        segments = params.get("segments", []) if isinstance(params, dict) else params if isinstance(params, list) else []
        
        for i, seg in enumerate(segments):
            # Skip non-dictionary segments
            if not isinstance(seg, dict):
                # Add warning log
                import logging
                logging.getLogger(__name__).warning(f"Skipping non-dictionary segment at index {i}: {type(seg)} - {seg}")
                continue
                
            # Now safely proceed with dictionary operations
            if bb := seg.get("break_before"):
                if bb != "0ms":
                    parts.append(f'<break time="{bb}"/>')
            txt = saxutils.escape(seg.get("text", ""))
            if p := seg.get("prosody"):
                parts.append(
                    f'<prosody pitch="{p.get("pitch","0%")}" '
                    f'rate="{p.get("rate","0%")}" '
                    f'volume="{p.get("volume","0%")}">{txt}</prosody>'
                )
            else:
                parts.append(txt)
            if ba := seg.get("break_after"):
                if ba != "0ms":
                    parts.append(f'<break time="{ba}"/>')
        parts.append("</speak>")
        return "\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Unified Evaluator (detailed metrics + thread-pool)
# ─────────────────────────────────────────────────────────────────────────────
class SSMLEvaluator:
    def __init__(self, samples: List[Dict[str, Any]], cfg: ExperimentConfig):
        self.samples = samples
        self.cfg = cfg

    def _extract_breaks(self, seq: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        out = []
        pos = 0
        for item in seq:
            if item.get("type") == "break":
                t = item.get("time", "0ms")
                try:
                    tm = int(t.rstrip("ms"))
                except:
                    tm = 0
                out.append({"position": pos, "time": tm})
            pos += 1
        return out

    def _extract_prosody(self, seq: List[Dict[str, Any]]) -> List[Dict[str, Union[int,float]]]:
        out = []
        pos = 0
        for item in seq:
            if item.get("type") == "text" and "prosody" in item:
                p = item["prosody"]
                try:
                    pitch = float(str(p.get("pitch","0%")).rstrip("%"))
                    rate  = float(str(p.get("rate","0%")).rstrip("%"))
                    vol   = float(str(p.get("volume","0%")).rstrip("%"))
                except:
                    pitch=rate=vol=0.0
                out.append({"position": pos, "pitch": pitch, "rate": rate, "volume": vol})
            pos += 1
        return out

    def _calculate_segment_averages(self, sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate segment-level averages for each prosody parameter and break times.
        
        Args:
            sequence: Parsed SSML sequence (list of elements)
            
        Returns:
            Dictionary with average values for pitch, rate, volume, and break_time
        """
        averages = {
            "pitch": [],
            "rate": [],
            "volume": [],
            "break_time": []
        }
        
        # Extract all values from the sequence
        for item in sequence:
            if item.get("type") == "text" and "prosody" in item:
                p = item.get("prosody", {})
                try:
                    pitch = float(str(p.get("pitch", "0%")).rstrip("%").lstrip("+"))
                    rate = float(str(p.get("rate", "0%")).rstrip("%").lstrip("+"))
                    volume = float(str(p.get("volume", "0%")).rstrip("%").lstrip("+"))
                    averages["pitch"].append(pitch)
                    averages["rate"].append(rate)
                    averages["volume"].append(volume)
                except (ValueError, TypeError):
                    pass
            
            elif item.get("type") == "break":
                try:
                    time_ms = int(str(item.get("time", "0ms")).rstrip("ms"))
                    if time_ms > 0:
                        averages["break_time"].append(time_ms)
                except (ValueError, TypeError):
                    pass
        
        # Calculate means (or use 0 if no values)
        result = {}
        for param, values in averages.items():
            result[param] = np.mean(values) if values else 0.0
            
        return result

    def _eval_sample(self, model: SSMLPredictor, sample: Dict[str, Any]) -> Dict[str, Any]:
        return model.predict(sample["text"], sample.get("voice"))

    def evaluate(self, model: SSMLPredictor, skip_sampling: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model using segment-level averaging approach.
        """
        # Use provided samples directly if skip_sampling is True
        pop = self.samples if skip_sampling else random.sample(self.samples, min(self.cfg.num_samples, len(self.samples)))
        results, counts = [], defaultdict(list)
        segment_metrics_list = []
        
        # Lists for segment-level metrics
        true_pitch, pred_pitch = [], []
        true_rate, pred_rate = [], []
        true_volume, pred_volume = [], []
        true_break_time, pred_break_time = [], []
        
        # Error lists for traditional metrics
        pitch_err, rate_err, vol_err, break_err = [], [], [], []
        
        # Break detection metrics
        tp = fp = fn = 0
        
        # Add counters for gold standard tags
        gold_total_tags = []
        gold_prosody_count = []
        gold_break_count = []
        gold_tags_per_sentence = []
        
        logger.info(
            "Evaluating %s on %d samples with segment-level averaging (threads=%d)",
            model.__class__.__name__,
            len(pop),
            self.cfg.parallel_requests
        )

        with ThreadPoolExecutor(max_workers=self.cfg.parallel_requests) as pool:
            futs = {pool.submit(self._eval_sample, model, s): s for s in pop}
            
            for fut in tqdm(as_completed(futs), total=len(pop)):
                pred = fut.result()
                sample = futs[fut]  # Get the original sample
                
                # Add ground truth to the result for later use
                pred["ground_truth"] = sample.get("parsed_sequence", [])
                
                # Extract and add gold SSML from the training data
                segment_id = sample.get("segment")
                voice_id = sample.get("voice")
                
                # Try to get gold SSML from training data
                if segment_id and voice_id:
                    # Load the original data once per evaluation run
                    if not hasattr(self, '_training_data'):
                        try:
                            with open(self.cfg.data_path, "r", encoding="utf-8") as f:
                                self._training_data = json.load(f)
                        except Exception as e:
                            logger.error(f"Failed to load training data: {e}")
                            self._training_data = {}
                    
                    # Look for voice data in EP01 or EP02
                    voice_data = self._training_data.get(f"{voice_id}_EP01", {}) or self._training_data.get(f"{voice_id}_EP02", {})
                    
                    # Extract the stripped_ssml if available
                    if "y" in voice_data and "stripped_ssml" in voice_data["y"] and segment_id in voice_data["y"]["stripped_ssml"]:
                        ssml_fragments = voice_data["y"]["stripped_ssml"][segment_id]
                        gold_ssml = "<speak>\n" + "\n".join(ssml_fragments) + "\n</speak>"
                        pred["gold_ssml"] = gold_ssml
            
                # If we couldn't get gold SSML from training data, generate it from parsed_sequence
                if "gold_ssml" not in pred and pred["ground_truth"]:
                    # Create a dummy FewShotSSMLPredictor to use its conversion method
                    dummy_config = ExperimentConfig(data_path=self.cfg.data_path)
                    dummy = FewShotSSMLPredictor(dummy_config, "", 0, None, [])
                    gold_params = dummy._convert_parsed_sequence_to_json(pred["ground_truth"])
                    gold_ssml = SSMLBuilder.build_from_params(gold_params)
                    pred["gold_ssml"] = gold_ssml
                
                results.append(pred)

                # Collect any simple per-sample metrics
                for k, v in pred["metrics"].items():
                    if isinstance(v, (int, float)):
                        counts[k].append(v)

                # Get reference parsed sequence from the sample (ground truth)
                ref_seq = sample.get("parsed_sequence", [])
                if not ref_seq:
                    logger.warning(f"No ground truth found for sample: {sample.get('text', '')[:50]}")
                    continue
                    
                # Calculate gold standard tag metrics
                g_prosody_count = sum(1 for item in ref_seq if item.get("type") == "text" and "prosody" in item)
                g_break_count = sum(1 for item in ref_seq if item.get("type") == "break")
                g_total_tags = g_prosody_count + g_break_count
                
                text = sample.get("text", "")
                sentences = len(re.findall(r"[.!?]", text)) or 1
                g_tags_per_sentence = g_total_tags / sentences
                
                # Store the gold metrics
                gold_total_tags.append(g_total_tags)
                gold_prosody_count.append(g_prosody_count)
                gold_break_count.append(g_break_count)
                gold_tags_per_sentence.append(g_tags_per_sentence)
                
                # Calculate segment-level averages for ground truth
                true_averages = self._calculate_segment_averages(ref_seq)
                
                # Calculate segment-level averages for prediction
                pred_params = pred.get("params", {})
                pred_segments = pred_params.get("segments", [])
                
                # Create a synthetic sequence from the predicted segments
                pred_seq = []
                for seg in pred_segments:
                    # Add break before if specified
                    if (bb := seg.get("break_before")) and bb != "0ms":
                        pred_seq.append({"type": "break", "time": bb})
                    
                    # Add text with prosody
                    if "text" in seg:
                        text_elem = {
                            "type": "text",
                            "text": seg["text"]
                        }
                        if "prosody" in seg:
                            text_elem["prosody"] = seg["prosody"]
                        pred_seq.append(text_elem)
                    
                    # Add break after if specified
                    if (ba := seg.get("break_after")) and ba != "0ms":
                        pred_seq.append({"type": "break", "time": ba})
                
                # Calculate averages for the prediction
                pred_averages = self._calculate_segment_averages(pred_seq)
                
                # Collect segment-level values for R² calculation
                true_pitch.append(true_averages["pitch"])
                pred_pitch.append(pred_averages["pitch"])
                true_rate.append(true_averages["rate"])
                pred_rate.append(pred_averages["rate"])
                true_volume.append(true_averages["volume"])
                pred_volume.append(pred_averages["volume"])
                
                # For break time, we need to handle the possibility of no breaks
                if true_averages["break_time"] > 0 or pred_averages["break_time"] > 0:
                    true_break_time.append(true_averages["break_time"])
                    pred_break_time.append(pred_averages["break_time"])
                    
                    # Calculate absolute error for this segment
                    break_err.append(abs(true_averages["break_time"] - pred_averages["break_time"]))
                    
                    # Count as TP if both have breaks, FP if only prediction has breaks
                    if true_averages["break_time"] > 0 and pred_averages["break_time"] > 0:
                        tp += 1
                    elif pred_averages["break_time"] > 0:
                        fp += 1
                    elif true_averages["break_time"] > 0:
                        fn += 1
                
                # Calculate errors for prosody parameters
                pitch_err.append(abs(true_averages["pitch"] - pred_averages["pitch"]))
                rate_err.append(abs(true_averages["rate"] - pred_averages["rate"]))
                vol_err.append(abs(true_averages["volume"] - pred_averages["volume"]))

                # ── Segment-level metrics ──
                segment_metrics = {
                    "segment_id": sample.get("segment", "unknown"),
                    "text": sample.get("text", "")[:50] + "...",
                    "errors": {
                        "pitch_error": abs(true_averages["pitch"] - pred_averages["pitch"]),
                        "rate_error": abs(true_averages["rate"] - pred_averages["rate"]),
                        "volume_error": abs(true_averages["volume"] - pred_averages["volume"]),
                        "break_time_error": abs(true_averages["break_time"] - pred_averages["break_time"]),
                    },
                    "true_values": true_averages,
                    "pred_values": pred_averages
                }

                # Add to a list of segment metricss
                segment_metrics_list.append(segment_metrics)

        # ── Aggregate counts ──
        agg = {f"{k}_mean": np.mean(v) for k, v in counts.items() if v}
        agg.update({f"{k}_std": np.std(v) for k, v in counts.items() if v})

        # ── Store values for R² calculation ──
        agg["true_pitch_values"] = true_pitch
        agg["true_rate_values"] = true_rate
        agg["true_volume_values"] = true_volume
        agg["true_break_time_values"] = true_break_time
        agg["pred_pitch_values"] = pred_pitch
        agg["pred_rate_values"] = pred_rate
        agg["pred_volume_values"] = pred_volume
        agg["pred_break_time_values"] = pred_break_time

        # Debug: Print collected values
        logger.info(f"Collected {len(true_pitch)} segment-level pitch values")
        logger.info(f"Collected {len(true_rate)} segment-level rate values")
        logger.info(f"Collected {len(true_volume)} segment-level volume values")
        logger.info(f"Collected {len(true_break_time)} segment-level break values")

        # ── Break Precision/Recall/F1 ──
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        agg.update({
            "break_precision": precision,
            "break_recall": recall,
            "break_f1": f1
        })

        # ── Error metrics ──
        agg["pitch_mae"] = np.mean(pitch_err) if pitch_err else 0
        agg["pitch_mse"] = np.mean(np.square(pitch_err)) if pitch_err else 0
        agg["rate_mae"] = np.mean(rate_err) if rate_err else 0
        agg["rate_mse"] = np.mean(np.square(rate_err)) if rate_err else 0
        agg["volume_mae"] = np.mean(vol_err) if vol_err else 0
        agg["volume_mse"] = np.mean(np.square(vol_err)) if vol_err else 0
        agg["break_time_mae"] = np.mean(break_err) if break_err else 0
        agg["break_time_mse"] = np.mean(np.square(break_err)) if break_err else 0

        # Aggregate gold metrics alongside existing metrics
        agg["gold_total_tags_mean"] = np.mean(gold_total_tags) if gold_total_tags else 0
        agg["gold_prosody_count_mean"] = np.mean(gold_prosody_count) if gold_prosody_count else 0
        agg["gold_break_count_mean"] = np.mean(gold_break_count) if gold_break_count else 0
        agg["gold_tags_per_sentence_mean"] = np.mean(gold_tags_per_sentence) if gold_tags_per_sentence else 0
        
        # Also add standard deviations
        agg["gold_total_tags_std"] = np.std(gold_total_tags) if gold_total_tags else 0
        agg["gold_prosody_count_std"] = np.std(gold_prosody_count) if gold_prosody_count else 0
        agg["gold_break_count_std"] = np.std(gold_break_count) if gold_break_count else 0
        agg["gold_tags_per_sentence_std"] = np.std(gold_tags_per_sentence) if gold_tags_per_sentence else 0

        return {
            "model_name": model.__class__.__name__,
            "num_samples": len(pop),
            "results": results,
            "metrics": agg,
            "segment_metrics": segment_metrics_list,  # Add this line
            "evaluation_approach": "segment_level_averaging"
        }

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_experiments_for_model(cfg: ExperimentConfig, model_name: str, samples: List[Dict[str, Any]], eval_samples: List[Dict[str, Any]]):
    # No need to load data or select samples again
    logger.info("=== Running experiments for model %s ===", model_name)
    
    # Zero-shot (use fixed eval samples)
    if cfg.mode in ("zero-shot", "both"):
        zs = ZeroShotSSMLPredictor(cfg, model_name)
        ez = SSMLEvaluator(eval_samples, cfg)
        # Update SSMLEvaluator.evaluate to not sample again
        res = ez.evaluate(zs, skip_sampling=True)
        path = os.path.join(cfg.output_dir, f"{model_name}_zero_shot.json")
        json.dump(res, open(path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
        logger.info("Zero-shot saved to %s", path)

    # Few-shot
    if cfg.mode in ("few-shot","both"):
        # Use the common eval_samples passed into the function
        # DO NOT resample here!
        
        # For training examples, use samples not in the evaluation set
        train_samples = [s for s in samples if s not in eval_samples]
        
        # If we don't have enough training samples, we can reuse some evaluation samples
        # (not ideal but better than having too few examples)
        if len(train_samples) < cfg.max_examples:
            logger.warning(f"Not enough training samples ({len(train_samples)}). Using some evaluation samples as examples.")
            additional_needed = cfg.max_examples - len(train_samples)
            train_samples.extend(random.sample(eval_samples, min(additional_needed, len(eval_samples))))
        
        # Create predictor with randomly selected training samples
        fs = FewShotSSMLPredictor(cfg, model_name, cfg.max_examples, None, train_samples)
        
        # Evaluate on our selected evaluation samples
        ez = SSMLEvaluator(eval_samples, cfg)
        res = ez.evaluate(fs)
        
        # Save a single result file
        path = os.path.join(cfg.output_dir, f"{model_name}_few_shot.json")
        json.dump(res, open(path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
        logger.info(f"Few-shot results saved to {path}")

# ─────────────────────────────────────────────────────────────────────────────
def generate_consolidated_html_comparison(output_dir: str, data_path: str):
    """
    Generate a single HTML file with a dropdown menu to view all comparisons.
    This function scans the output directory for all result files and creates
    a consolidated view.
    
    Args:
        output_dir: Directory containing the result JSON files
        data_path: Path to the original data file
    """
    # Find all JSON result files
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        logger.warning("No result files found in %s", output_dir)
        return
    
    logger.info("Generating consolidated HTML comparison for %d result files", len(json_files))
    
    # Group results by model and shot type
    results_data = {}
    for json_file in json_files:
        filename = os.path.basename(json_file)
        match = re.match(r"([^_]+)_([^_]+)(?:_([^.]+))?\.json", filename)
        if not match:
            continue
            
        model_name = match.group(1)
        shot_type = match.group(2)
        voice = match.group(3) if match.group(3) else "all"
        
        # Load the JSON data
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            key = f"{model_name}_{shot_type}_{voice}"
            display_name = f"{model_name.capitalize()} - {shot_type.replace('_', ' ').title()}" + (f" ({voice})" if voice != "all" else "")
            
            # Store the results and metadata
            results_data[key] = {
                "display_name": display_name,
                "data": data,
                "filename": filename
            }
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    # Generate the HTML file
    comp_path = os.path.join(output_dir, "ssml_comparison_viewer.html")
    
    with open(comp_path, "w", encoding="utf-8") as fh:
        # HTML header with CSS and JavaScript
        fh.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        fh.write("<style>\n")
        fh.write("  body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }\n")
        fh.write("  h2 { color: #333; text-align: center; margin-bottom: 10px; }\n")
        fh.write("  .header { background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n")
        fh.write("  .dropdown-container { display: flex; justify-content: center; gap: 10px; align-items: center; }\n")
        fh.write("  select { padding: 8px; font-size: 16px; border-radius: 4px; border: 1px solid #ddd; }\n")
        fh.write("  table { border-collapse: collapse; margin: 0 auto; }\n")
        fh.write("  th, td { border: 1px solid #ccc; vertical-align: top; padding: 12px; }\n")
        fh.write("  th { background: #f0f0f0; font-weight: bold; text-align: left; }\n")
        fh.write("  pre { margin: 0; font-family: 'Courier New', monospace; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }\n")
        fh.write("  .pitch { background-color: #ffeecc; padding: 2px; border-radius: 3px; }\n")
        fh.write("  .rate { background-color: #e6ffcc; padding: 2px; border-radius: 3px; }\n")
        fh.write("  .volume { background-color: #cce6ff; padding: 2px; border-radius: 3px; }\n")
        fh.write("  .break { background-color: #ffccf2; padding: 2px; border-radius: 3px; }\n")
        fh.write("  tr:nth-child(even) { background-color: #f9f9f9; }\n")
        fh.write("  .sample-index { font-weight: bold; background: #eee; text-align: center; }\n")
        fh.write("  .comparison-table { display: none; }\n")
        fh.write("  .active { display: block; }\n")
        fh.write("</style>\n")
        
        # JavaScript for dropdown functionality
        fh.write("<script>\n")
        fh.write("function showComparison(id) {\n")
        fh.write("  // Hide all comparisons\n")
        fh.write("  const tables = document.querySelectorAll('.comparison-table');\n")
        fh.write("  tables.forEach(table => table.classList.remove('active'));\n")
        fh.write("  \n")
        fh.write("  // Show selected comparison\n")
        fh.write("  const selected = document.getElementById(id);\n")
        fh.write("  if (selected) {\n")
        fh.write("    selected.classList.add('active');\n")
        fh.write("    document.title = 'SSML Comparison - ' + id.replace(/_/g, ' ');\n")
        fh.write("  }\n")
        fh.write("}\n")
        fh.write("</script>\n")
        fh.write("</head><body>\n")
        
        # Header with dropdown
        fh.write("<div class='header'>\n")
        fh.write("<h2>SSML Comparison Viewer</h2>\n")
        fh.write("<div class='dropdown-container'>\n")
        fh.write("<label for='comparison-selector'>Select Model/Configuration:</label>\n")
        fh.write("<select id='comparison-selector' onchange='showComparison(this.value)'>\n")
        
        # Add options to dropdown
        first_key = None
        for key, item in sorted(results_data.items(), key=lambda x: x[1]["display_name"]):
            if first_key is None:
                first_key = key
            fh.write(f"<option value='{key}'>{item['display_name']}</option>\n")
        
        fh.write("</select>\n")
        fh.write("</div>\n")  # End dropdown-container
        fh.write("</div>\n")  # End header
        
        # Function to format SSML with highlighting
        def format_ssml(ssml: str) -> str:
            # Add line breaks around tags for better readability
            formatted = re.sub(r'(<[^/][^>]*>)([^<]+)', r'\1\n\2\n', ssml)
            formatted = re.sub(r'(</[^>]+>)', r'\n\1\n', formatted)
            
            # Highlight special values
            formatted = re.sub(r'(pitch="[^"]*")', r'<span class="pitch">\1</span>', formatted)
            formatted = re.sub(r'(rate="[^"]*")', r'<span class="rate">\1</span>', formatted)
            formatted = re.sub(r'(volume="[^"]*")', r'<span class="volume">\1</span>', formatted)
            formatted = re.sub(r'(time="[^"]*")', r'<span class="break">\1</span>', formatted)
            
            return formatted
        
        # Create a dummy SSML builder for gold SSML
        # Fixed: Pass required data_path parameter to ExperimentConfig
        dummy_config = ExperimentConfig(data_path="dummy_path")
        dummy = FewShotSSMLPredictor(dummy_config, "", max_examples=0, target_voice=None, train_data=[])
        
        # Generate each comparison table
        for key, item in results_data.items():
            fh.write(f"<div id='{key}' class='comparison-table{' active' if key == first_key else ''}'>\n")
            fh.write(f"<h2>{item['display_name']}</h2>\n")
            fh.write("<table>\n<tr><th>Gold SSML</th><th>Predicted SSML</th></tr>\n")
            
            # Process results if available
            if "results" in item["data"]:
                results = item["data"]["results"]
                
                # Find the original samples to match with results
                with open(data_path, "r", encoding="utf-8") as f:
                    try:
                        all_samples = json.load(f)
                        # Extract samples from the raw data
                        samples_dict = {}
                        for key, sample_data in all_samples.items():
                            seq = sample_data.get("y", {}).get("parsed_sequence", [])
                            seg_groups = defaultdict(list)
                            for elem in seq:
                                seg_id = elem.get("segment")
                                if seg_id is not None:
                                    seg_groups[seg_id].append(elem)
                                    
                            for seg_id, elems in seg_groups.items():
                                text_items = [e for e in elems if e.get("type") == "text" and e.get("text")]
                                if not text_items:
                                    continue
                                
                                sample_text = " ".join(e["text"].strip() for e in text_items)
                                cleaned_seq = [{k: v for k, v in e.items() if k != "segment"} for e in elems]
                                samples_dict[sample_text] = {
                                    "text": sample_text, 
                                    "parsed_sequence": cleaned_seq,
                                    "voice": key.split("_")[0]
                                }
                    except Exception as e:
                        logger.error(f"Error loading samples: {e}")
                        samples_dict = {}
                
                # Generate comparison rows
                for i, result in enumerate(results):
                    input_text = result.get("input_text", "")
                    original_sample = samples_dict.get(input_text, None)
                    
                    if not original_sample:
                        continue
                        
                    # Get gold SSML
                    try:
                        gold_dict = dummy._convert_parsed_sequence_to_json(original_sample["parsed_sequence"])
                        gold_ssml = SSMLBuilder.build_from_params(gold_dict)
                    except Exception as e:
                        gold_ssml = f"<speak>\nError generating gold SSML: {e}\n</speak>"
                    
                    # Format SSMLs
                    formatted_gold = format_ssml(gold_ssml)
                    formatted_pred = format_ssml(result["ssml"])
                    
                    # Add sample
                    fh.write(f"<tr><td colspan='2' class='sample-index'>Sample {i+1} - \"{input_text[:50]}...\"</td></tr>\n")
                    fh.write("<tr>\n")
                    fh.write(f"  <td><pre>{formatted_gold}</pre></td>\n")
                    fh.write(f"  <td><pre>{formatted_pred}</pre></td>\n")
                    fh.write("</tr>\n")
            else:
                fh.write("<tr><td colspan='2'>No results available</td></tr>\n")
                
            fh.write("</table>\n")
            fh.write("</div>\n")
        
        fh.write("</body></html>\n")
    
    logger.info("Generated consolidated comparison viewer: %s", comp_path)
    return comp_path

# ─────────────────────────────────────────────────────────────────────────────

def process_samples_from_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process raw JSON data into a list of sample dictionaries for evaluation.
    
    Args:
        raw_data: Raw JSON data loaded from the data file
        
    Returns:
        List of processed sample dictionaries
    """
    samples = []
    
    # Extract samples from the raw data format
    for key, sample_data in raw_data.items():
        # Extract the parsed sequence from the raw data
        seq = sample_data.get("y", {}).get("parsed_sequence", [])
        seg_groups = defaultdict(list)
        
        # Group elements by segment ID
        for elem in seq:
            seg_id = elem.get("segment")
            if seg_id is not None:
                seg_groups[seg_id].append(elem)
        
        # Process each segment group
        for seg_id, elems in seg_groups.items():
            # Get all text elements
            text_items = [e for e in elems if e.get("type") == "text" and e.get("text")]
            if not text_items:
                continue
            
            # Join text fragments and clean up sequence
            sample_text = " ".join(e["text"].strip() for e in text_items)
            cleaned_seq = [{k: v for k, v in e.items() if k != "segment"} for e in elems]
            
            # Create sample dictionary
            samples.append({
                "text": sample_text,
                "parsed_sequence": cleaned_seq,
                "voice": key.split("_")[0],
                "segment": seg_id
            })
    
    logger.info(f"Processed {len(samples)} samples from input data")
    return samples


def main():
    p=argparse.ArgumentParser(description="SSML experiments")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument(
        "--html-only",
        action="store_true",
        help="Skip running experiments; only (re)generate the comparison HTML from existing JSON results"
    )
    args=p.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    if args.html_only:
        path = generate_consolidated_html_comparison(cfg.output_dir, cfg.data_path)
        print(f"HTML only mode: wrote {path}")
        return
    if cfg.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Loaded config: %s", cfg)

    procs = cfg.worker_processes or min(mp.cpu_count(), len(cfg.model_names))
    logger.info("Spawning %d process(es) for %d model(s)", procs, len(cfg.model_names))

    # Load data once
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Process samples once (moved from run_experiments_for_model)
    samples = process_samples_from_data(raw)
    
    # Select evaluation samples once for all models
    eval_samples = random.sample(samples, min(cfg.num_samples, len(samples)))
    
    # Update function signature and calls
    with mp.Pool(procs) as pool:
        pool.starmap(run_experiments_for_model, [(cfg, m, samples, eval_samples) for m in cfg.model_names])

    generate_consolidated_html_comparison(cfg.output_dir, cfg.data_path)

    logger.info("All experiments complete. Results in %s", cfg.output_dir)

if __name__=="__main__":
    main()