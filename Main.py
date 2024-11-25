from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Patch

def load_ground_truth():
    return [
        (0.0, 5.009, "Gary"),
        (5.009, 24.004, "Ruben"),
        (25.005, 31.002, "Gary"),
        (31.004, 47.003, "Ruben"),
        (47.003, 48.004, "Gary"),
        (49.005, 88.000, "Ruben"),
        (88.002, 94.004, "Gary"),
        (94.009, 135.003, "Ruben"),
        (136.000, 142.000, "Gary"),
        (142.000, 200.004, "Ruben"),
        (200.007, 220.000, "Gary"),
        (220.008, 285.006, "Ruben"),
        (286.005, 300.001, "Gary"),
        (301.000, 306.000, "Ruben")
    ]

def perform_diarization(audio_path):
    print(f"Starting diarization at {time.strftime('%H:%M:%S')}")
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    window_size = 3 * 16000
    segments = []
    
    for i in range(0, len(wav), window_size):
        segment = wav[i:i + window_size]
        if len(segment) < window_size:
            continue
        embed = encoder.embed_utterance(segment)
        segments.append((i/16000, (i+len(segment))/16000, embed))
        progress = (i / len(wav)) * 100
        print(f"Progress: {progress:.1f}%", end='\r')
    
    print("\nClustering speakers...")
    results = []
    for start, end, embed in segments:
        similarity = np.dot(embed, segments[0][2])
        speaker = "SPEAKER_0" if similarity > 0.85 else "SPEAKER_1"
        results.append((start, end, speaker))
    
    return results

def calculate_diarization_error_rate(ground_truth, diarization_result, step_size=0.1):
    max_time = max(max(end for start, end, _ in ground_truth),
                  max(end for start, end, _ in diarization_result))
    
    times = np.arange(0, max_time, step_size)
    gt_labels = np.zeros(len(times))
    dr_labels = np.zeros(len(times))
    
    for start, end, speaker in ground_truth:
        start_idx = int(start / step_size)
        end_idx = int(end / step_size)
        gt_labels[start_idx:end_idx] = 1 if speaker == "Gary" else 2
    
    for start, end, speaker in diarization_result:
        start_idx = int(start / step_size)
        end_idx = int(end / step_size)
        dr_labels[start_idx:end_idx] = 1 if speaker == "SPEAKER_0" else 2
    
    total_frames = len(times)
    error_frames = np.sum(gt_labels != dr_labels)
    der = error_frames / total_frames
    
    gary_frames = np.sum(gt_labels == 1)
    ruben_frames = np.sum(gt_labels == 2)
    gary_correct = np.sum((gt_labels == 1) & (dr_labels == 1))
    ruben_correct = np.sum((gt_labels == 2) & (dr_labels == 2))
    
    print(f"\nDiarization Error Rate: {der:.2%}")
    print(f"Per-speaker Accuracy:")
    print(f"Gary: {gary_correct/gary_frames:.2%}")
    print(f"Ruben: {ruben_correct/ruben_frames:.2%}")
    
    return der

def visualize_results(ground_truth, diarization_result, der=None):
    plt.figure(figsize=(15, 5))
    
    for start, end, speaker in ground_truth:
        color = 'blue' if speaker == "Gary" else 'red'
        plt.barh(1, end-start, left=start, height=0.3, color=color, alpha=0.6)
    
    for start, end, speaker in diarization_result:
        color = 'blue' if speaker == "SPEAKER_0" else 'red'
        plt.barh(0, end-start, left=start, height=0.3, color=color, alpha=0.6)
    
    plt.yticks([0, 1], ['Diarization', 'Ground Truth'])
    plt.xlabel('Time (seconds)')
    
    title = 'Speaker Diarization Results'
    if der is not None:
        title += f' (DER: {der:.2%})'
    plt.title(title)
    
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Gary/Speaker_0'),
        Patch(facecolor='red', alpha=0.6, label='Ruben/Speaker_1')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('diarization_result.png')
    plt.close()

def main():
    audio_file = "processed_audio.wav"
    
    try:
        results = perform_diarization(audio_file)
        ground_truth = load_ground_truth()
        der = calculate_diarization_error_rate(ground_truth, results)
        visualize_results(ground_truth, results, der)
        print("\nVisualization saved as 'diarization_result.png'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()