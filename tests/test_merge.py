"""Tests for merge_titles."""
import pytest

from agent.nodes import merge_titles

HOP_ONE = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']


def test_merge_one_hop_returns_top_context_k() -> None:
    # Arrange / Act
    merged = merge_titles([HOP_ONE])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'A4', 'A5']


def test_merge_two_hops_keeps_top_three_and_adds_two_new() -> None:
    # Arrange
    hop_two = ['B1', 'B2', 'B3']

    # Act
    merged = merge_titles([HOP_ONE, hop_two])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'B1', 'B2']


def test_merge_skips_second_hop_titles_already_in_context() -> None:
    # Arrange
    hop_two = ['A2', 'B1', 'A1', 'B2', 'B3']

    # Act
    merged = merge_titles([HOP_ONE, hop_two])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'B1', 'B2']


def test_merge_fills_budget_from_first_hop_when_hop_two_adds_nothing_new(
) -> None:
    # Arrange
    hop_two = ['A1', 'A2', 'A3']

    # Act
    merged = merge_titles([HOP_ONE, hop_two])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'A4', 'A5']


def test_merge_fills_remaining_slot_when_second_hop_adds_one_new() -> None:
    # Arrange
    hop_two = ['A1', 'B1']

    # Act
    merged = merge_titles([HOP_ONE, hop_two])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'B1', 'A4']


def test_merge_never_duplicates_and_stays_within_budget_with_three_hops(
) -> None:
    # Arrange
    rankings = [HOP_ONE, ['B1', 'A1', 'B2'], ['B2', 'B3', 'B4']]

    # Act
    merged = merge_titles(rankings)

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'B1', 'B2']
    assert len(set(merged)) == len(merged)


def test_merge_returns_fewer_titles_when_too_few_distinct_exist() -> None:
    # Arrange / Act
    merged = merge_titles([['A1', 'A2'], ['A2', 'B1']])

    # Assert
    assert merged == ['A1', 'A2', 'B1']


def test_merge_empty_rankings_returns_empty_list() -> None:
    assert merge_titles([]) == []


def test_merge_counts_hop_one_tail_returned_by_hop_two_as_new() -> None:
    # Arrange
    hop_two = ['A4', 'A5', 'B1']

    # Act
    merged = merge_titles([HOP_ONE, hop_two])

    # Assert
    assert merged == ['A1', 'A2', 'A3', 'A4', 'A5']


@pytest.mark.parametrize('context_k', [2, 0, -1])
def test_merge_rejects_budget_smaller_than_first_hop_share(
        context_k: int,
) -> None:
    with pytest.raises(ValueError, match='first_hop_k'):
        merge_titles([HOP_ONE, ['B1']], context_k=context_k)
