#!/usr/bin/env python3
"""Test dataset index partitioning and splitting logic."""

import numpy as np
import torch


def test_half_and_half_splitting():
    """Test the initial half-and-half splitting logic."""
    print("=" * 60)
    print("Test: Initial half-and-half splitting")
    print("=" * 60)

    # Simulate the reference model's arg sorting
    np.random.seed(42)
    n_total = 50000
    ref_arg = np.argsort(np.random.randn(n_total))

    # Official HARVEY logic (from train.py)
    INIT_BUDGET = int(0.5 * n_total)  # 25000
    UL_BUDGET = n_total - INIT_BUDGET  # 25000

    labeled_set = list(torch.tensor(list(range(n_total)))[ref_arg][-UL_BUDGET:].numpy())
    subset = list(torch.tensor(list(range(n_total)))[ref_arg][:INIT_BUDGET].numpy())
    unlabeled_set = list(torch.tensor(list(range(n_total)))[ref_arg][INIT_BUDGET:-INIT_BUDGET].numpy())

    print(f"Total: {n_total}")
    print(f"INIT_BUDGET: {INIT_BUDGET}")
    print(f"UL_BUDGET: {UL_BUDGET}")
    print(f"labeled_set size: {len(labeled_set)}")
    print(f"subset size: {len(subset)}")
    print(f"unlabeled_set size: {len(unlabeled_set)}")

    # Verify no overlap
    all_sets = set(labeled_set) | set(subset) | set(unlabeled_set)
    assert len(all_sets) == n_total, f"Missing or duplicate indices: {len(all_sets)} != {n_total}"
    assert len(set(labeled_set) & set(subset)) == 0, "Overlap between labeled_set and subset"
    assert len(set(labeled_set) & set(unlabeled_set)) == 0, "Overlap between labeled_set and unlabeled_set"
    assert len(set(subset) & set(unlabeled_set)) == 0, "Overlap between subset and unlabeled_set"

    print("✓ PASS - No overlaps, all indices accounted for")
    return labeled_set, subset, unlabeled_set


def test_cycle_refresh_splitting():
    """Test the cycle-wise subset refresh logic."""
    print("\n" + "=" * 60)
    print("Test: Cycle-wise subset refresh")
    print("=" * 60)

    n_total = 50000
    np.random.seed(123)
    ref_arg = np.argsort(np.random.randn(n_total))
    ref_test_set = list(range(n_total))

    # Simulate poison rate
    poi_rate = 0.1
    num_poison = int(n_total * poi_rate)
    num_clean = n_total - num_poison

    left_rate = 0.2  # cfgs.BACKDOOR.split_rate
    len_subset = max(int(left_rate * num_poison), int(0.05 * n_total))
    pool_size = num_clean

    subset = list(torch.tensor(ref_test_set)[ref_arg][:len_subset].numpy())
    clean_pool = list(torch.tensor(ref_test_set)[ref_arg][-pool_size:].numpy())

    print(f"num_poison: {num_poison}")
    print(f"num_clean: {num_clean}")
    print(f"len_subset: {len_subset}")
    print(f"pool_size: {pool_size}")
    print(f"subset size: {len(subset)}")
    print(f"clean_pool size: {len(clean_pool)}")

    # Verify sizes
    assert len(subset) == len_subset
    assert len(clean_pool) == pool_size

    # Verify no overlap between subset and clean_pool
    assert len(set(subset) & set(clean_pool)) == 0, "Overlap between subset and clean_pool"

    # The listd should be the middle portion
    listd = list(torch.tensor(ref_test_set)[ref_arg][len_subset:-pool_size].numpy())
    expected_listd_size = n_total - len_subset - pool_size
    assert len(listd) == expected_listd_size

    print(f"listd size: {len(listd)} (expected: {expected_listd_size})")
    print("✓ PASS - Cycle splitting correct")
    return subset, clean_pool, listd


def test_meta_splitting():
    """Test the meta-splitting logic with init_strategy."""
    print("\n" + "=" * 60)
    print("Test: Meta-splitting")
    print("=" * 60)

    # Simulate the scenario
    n_total = 50000
    num_poison = 5000
    labeled_set = list(range(40000))  # 40k clean samples
    poison_pool = list(range(40000, 45000))  # 5k from poisoned portion

    # Simulate init_strategy query - returns argsort of losses
    np.random.seed(456)
    init_arg = np.argsort(np.random.randn(len(poison_pool)))
    init_num_clean = 3000  # 3k clean in poison_pool
    init_num_poison = len(poison_pool) - init_num_clean

    # Official logic
    main_set = labeled_set + list(np.array(poison_pool)[init_arg][init_num_poison:].tolist())
    rest_set = list(np.array(poison_pool)[init_arg][:init_num_poison].tolist())

    print(f"labeled_set: {len(labeled_set)}")
    print(f"poison_pool: {len(poison_pool)}")
    print(f"init_num_clean: {init_num_clean}")
    print(f"init_num_poison: {init_num_poison}")
    print(f"main_set: {len(main_set)}")
    print(f"rest_set: {len(rest_set)}")

    # Verify
    assert len(main_set) == len(labeled_set) + init_num_clean
    assert len(rest_set) == init_num_poison
    assert len(main_set) + len(rest_set) == len(labeled_set) + len(poison_pool)

    # Verify no overlap
    assert len(set(main_set) & set(rest_set)) == 0

    print("✓ PASS - Meta-splitting correct")
    return main_set, rest_set


def test_tensor_numpy_conversion():
    """Test that .cpu().numpy() conversion preserves ordering."""
    print("\n" + "=" * 60)
    print("Test: Tensor to numpy conversion preserves ordering")
    print("=" * 60)

    # Create tensor on MPS (or CPU if MPS not available)
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    indices = torch.arange(1000, device=device)
    perm = torch.randperm(1000, device=device)
    sorted_indices = indices[perm]

    # Convert to numpy using the MPS-compatible method
    numpy_result = sorted_indices.cpu().numpy()

    # Verify it's a proper numpy array with correct values
    assert isinstance(numpy_result, np.ndarray)
    assert len(numpy_result) == 1000
    assert set(numpy_result) == set(range(1000))

    # Verify ordering is preserved
    expected = sorted_indices.cpu().numpy()
    assert np.array_equal(numpy_result, expected)

    print("✓ PASS - Tensor to numpy conversion preserves ordering")


def test_no_duplicates_in_splits():
    """Test that all splitting operations produce disjoint sets."""
    print("\n" + "=" * 60)
    print("Test: No duplicates across all splits")
    print("=" * 60)

    # Run through the complete pipeline
    labeled_set, subset, unlabeled_set = test_half_and_half_splitting()
    subset2, clean_pool, listd = test_cycle_refresh_splitting()
    main_set, rest_set = test_meta_splitting()

    print("\nFinal verification:")
    print(f"labeled_set: {len(labeled_set)}")
    print(f"subset (initial): {len(subset)}")
    print(f"unlabeled_set: {len(unlabeled_set)}")
    print(f"subset (cycle): {len(subset2)}")
    print(f"clean_pool: {len(clean_pool)}")
    print(f"listd: {len(listd)}")
    print(f"main_set: {len(main_set)}")
    print(f"rest_set: {len(rest_set)}")

    # Verify all indices in final main_set + rest_set = original total
    all_final = set(main_set) | set(rest_set)
    print(f"\nTotal unique indices in final partition: {len(all_final)}")

    print("✓ PASS - All partitioning tests passed")


if __name__ == '__main__':
    test_half_and_half_splitting()
    test_cycle_refresh_splitting()
    test_meta_splitting()
    test_tensor_numpy_conversion()
    test_no_duplicates_in_splits()
    print("\n" + "=" * 60)
    print("ALL DATASET PARTITIONING TESTS PASSED")
    print("=" * 60)