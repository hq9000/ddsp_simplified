from typing import Dict, Tuple, List, Optional

import numpy as np

from feature_names import MIDI_FEATURE_VELOCITY


class DatasetCompressor:
    def __init__(self):
        pass

    def compress_data(
            self,
            audio_data: np.ndarray,
            midi_features: Dict[str, np.ndarray],
            frame_rate: int,
            sample_rate: int,
            silence_tail_size_frames: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        regions_to_keep = self._get_regions_to_keep(midi_features[MIDI_FEATURE_VELOCITY], silence_tail_size_frames)
        compressed_audio_data = self._get_compressed_array(audio_data, regions_to_keep, int(sample_rate / frame_rate))

        compressed_midi_features = {}
        for name, data in midi_features.items():
            compressed_midi_features[name] = self._get_compressed_array(midi_features[name], regions_to_keep, 1)

        return compressed_audio_data, compressed_midi_features

    def _get_compressed_array(
            self,
            source_data: np.ndarray,
            regions_to_keep: List[Tuple[int, int]],
            scale: int
    ) -> np.ndarray:
        new_total_length = 0
        for region_to_keep in regions_to_keep:
            new_total_length = new_total_length + (region_to_keep[1] - region_to_keep[0]) * scale

        compressed = np.zeros(new_total_length, dtype=np.float32)
        cursor = 0
        for region_to_keep in regions_to_keep:
            start_in_source = region_to_keep[0] * scale
            end_in_source = region_to_keep[1] * scale

            start_in_compressed = cursor
            end_in_compressed = cursor + end_in_source - start_in_source

            compressed[start_in_compressed:end_in_compressed] = source_data[start_in_source:end_in_source]
            cursor = end_in_compressed

        return compressed

    def _get_regions_to_keep(self, velocities: np.ndarray, silence_tail_size_frames: int) -> List[Tuple[int, int]]:
        length_of_silence = 0

        regions_to_keep: List[Tuple[int, int]] = []
        region_to_keep_start_idx: Optional[int] = None

        for i, velocity in enumerate(velocities):
            if velocity == 0 and (i == 0 or velocities[i - 1] != 0):
                # this is a start of a silence region
                length_of_silence = 0
                continue

            if velocity != 0 and (velocities[i - 1] == 0 or i == 0):
                if region_to_keep_start_idx is None:
                    region_to_keep_start_idx = i
                    length_of_silence = 0

            if velocity == 0 and velocities[i - 1] == 0:
                length_of_silence = length_of_silence + 1
                if length_of_silence >= silence_tail_size_frames and region_to_keep_start_idx is not None:
                    regions_to_keep.append((region_to_keep_start_idx, i))
                    region_to_keep_start_idx = None

            if i == len(velocities) - 1:
                if region_to_keep_start_idx is not None:
                    regions_to_keep.append((region_to_keep_start_idx, i))

        return regions_to_keep
