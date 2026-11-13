#!/usr/bin/env python3
"""Test corrected precision/recall metrics for HARVEY."""

import numpy as np


def original_precision_recall(main_set, indices_bad, indices_clean, indices_noise):
    """Original buggy metric from official HARVEY."""
    if len(main_set) == 0:
        return 0.0, 0.0, 0
    compute_poi = lambda target_set: len([i for i in target_set if i in indices_bad]) / len(target_set)
    main_poi_rate = compute_poi(main_set)
    num_poi_main = int(len(main_set) * main_poi_rate)

    precision = 1 - num_poi_main / len(indices_bad)
    recall = (len(indices_bad) - num_poi_main) / len(main_set)
    return precision, recall, num_poi_main


def corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise):
    """Corrected precision/recall for selected benign set."""
    main_set = set(main_set)
    indices_bad = set(indices_bad)
    indices_clean = set(indices_clean)
    indices_noise = set(indices_noise)

    # Ground truth partition
    all_clean = indices_clean | indices_noise  # noise samples are not poisoned
    all_poisoned = indices_bad

    # Selected set intersections
    selected_clean = len(main_set & all_clean)
    selected_poisoned = len(main_set & all_poisoned)
    selected_noise = len(main_set & indices_noise)

    # Total ground truth
    total_clean = len(all_clean)
    total_poisoned = len(all_poisoned)

    # Precision: selected clean / all selected
    precision = selected_clean / len(main_set) if len(main_set) > 0 else 0.0

    # Recall: selected clean / all ground-truth clean
    recall = selected_clean / total_clean if total_clean > 0 else 0.0

    # Poison contamination rate
    contamination = selected_poisoned / len(main_set) if len(main_set) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'contamination_rate': contamination,
        'selected_size': len(main_set),
        'selected_clean': selected_clean,
        'selected_poisoned': selected_poisoned,
        'selected_noise': selected_noise,
        'total_clean': total_clean,
        'total_poisoned': total_poisoned
    }


def test_metrics():
    """Test with known synthetic data."""
    print("=" * 60)
    print("Test 1: Perfect selection (no poisoned samples in main_set)")
    print("=" * 60)
    # 100 total clean, 20 poisoned
    indices_clean = set(range(100))
    indices_bad = set(range(100, 120))
    indices_noise = set()
    main_set = list(range(80))  # Select 80 clean samples

    orig_p, orig_r, num_poi = original_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    corr = corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise)

    print(f"Original: precision={orig_p:.4f}, recall={orig_r:.4f}, num_poi_main={num_poi}")
    print(f"Corrected: {corr}")
    assert corr['precision'] == 1.0
    assert corr['recall'] == 0.8
    assert corr['contamination_rate'] == 0.0
    print("✓ PASS\n")

    print("=" * 60)
    print("Test 2: Some poisoned samples in main_set")
    print("=" * 60)
    main_set = list(range(80)) + [100, 101, 102]  # 80 clean + 3 poisoned
    orig_p, orig_r, num_poi = original_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    corr = corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise)

    print(f"Original: precision={orig_p:.4f}, recall={orig_r:.4f}, num_poi_main={num_poi}")
    print(f"Corrected: {corr}")
    assert corr['precision'] == 80/83
    assert corr['recall'] == 0.8
    assert abs(corr['contamination_rate'] - 3/83) < 1e-6
    print("✓ PASS\n")

    print("=" * 60)
    print("Test 3: With noise samples")
    print("=" * 60)
    indices_clean = set(range(80))
    indices_bad = set(range(80, 100))
    indices_noise = set(range(100, 110))
    main_set = list(range(70)) + [80, 81, 82] + [100, 101]  # 70 clean + 3 poisoned + 2 noise

    orig_p, orig_r, num_poi = original_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    corr = corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise)

    print(f"Original: precision={orig_p:.4f}, recall={orig_r:.4f}, num_poi_main={num_poi}")
    print(f"Corrected: {corr}")
    assert corr['selected_clean'] == 72  # 70 clean + 2 noise
    assert corr['selected_poisoned'] == 3
    assert corr['selected_noise'] == 2
    assert corr['precision'] == 72/75
    assert corr['recall'] == 72/90  # 90 total clean (80 clean + 10 noise)
    print("✓ PASS\n")

    print("=" * 60)
    print("Test 4: Empty main_set")
    print("=" * 60)
    main_set = []
    orig_p, orig_r, num_poi = original_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    corr = corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    print(f"Original: precision={orig_p:.4f}, recall={orig_r:.4f}")
    print(f"Corrected: {corr}")
    assert corr['precision'] == 0.0
    assert corr['recall'] == 0.0
    print("✓ PASS\n")

    print("=" * 60)
    print("Test 5: All clean selected (ideal case)")
    print("=" * 60)
    indices_clean = set(range(5000))
    indices_bad = set(range(5000, 5500))  # 500 poisoned
    indices_noise = set()
    main_set = list(range(4000))  # Select 4000 clean

    orig_p, orig_r, num_poi = original_precision_recall(main_set, indices_bad, indices_clean, indices_noise)
    corr = corrected_precision_recall(main_set, indices_bad, indices_clean, indices_noise)

    print(f"Original: precision={orig_p:.4f}, recall={orig_r:.4f}, num_poi_main={num_poi}")
    print(f"Corrected: {corr}")
    # Original formula is wrong here - let's check
    # num_poi_main = int(4000 * 0) = 0 (since no poisoned in main_set)
    # precision = 1 - 0/500 = 1.0
    # recall = (500 - 0)/4000 = 0.125
    # Corrected: precision = 4000/4000 = 1.0, recall = 4000/5000 = 0.8
    assert corr['precision'] == 1.0
    assert corr['recall'] == 0.8
    print("✓ PASS - Original recall was wrong (0.125 vs correct 0.8)\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_metrics()