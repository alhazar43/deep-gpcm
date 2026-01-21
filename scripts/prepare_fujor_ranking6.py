"""
Utility script to extract 6-category ranking and matrix items from the Fujor
dataset and save them in the standard ``new_format`` sequential layout used across
Deep-GPCM (e.g., assist2009_dkvmn).

Each ranking/matrix option is treated as an individual sub-item; responses are
polytomous with categories 0-5. Generated files:

    data/fujor_ranking6/fujor_ranking6_train.txt
    data/fujor_ranking6/fujor_ranking6_test.txt
    data/fujor_ranking6/metadata.json
    data/fujor_ranking6/subitem_mapping.json
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fujor"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "fujor_ranking6"
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


@dataclass(frozen=True)
class SubItem:
    """Descriptor for a sub-item derived from a ranking option."""

    index: int
    question_id: int
    question_type: str
    answer_id: int
    question_option: int
    answer_category: int


def _load_selected_pairs() -> Dict[Tuple[int, int], Dict[str, int]]:
    """Return metadata rows for ranking questions with 6 categories."""
    doc_path = DATA_DIR / "q28_doc_textless.csv"
    pairs: Dict[Tuple[int, int], Dict[str, int]] = {}
    allowed_types = {"Ranking", "Matrix"}

    with doc_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qtype = row.get("QuestionType")
            if qtype not in allowed_types:
                continue

            try:
                category = int(float(row["AnswerCategory"]))
            except (TypeError, ValueError):
                continue

            if category != 6:
                continue

            try:
                qid = int(float(row["QuestionId"]))
                aid = int(float(row["AnswerId"]))
            except (TypeError, ValueError):
                continue

            pairs[(qid, aid)] = {
                "question_id": qid,
                "question_type": qtype,
                "answer_id": aid,
                "question_option": int(float(row.get("QuestionOption", 0) or 0)),
                "answer_category": category,
            }

    if not pairs:
        raise ValueError("No 6-category ranking items found in Fujor metadata.")

    return pairs


def _load_response_table() -> List[List[str]]:
    """Read the response table as raw strings."""
    resp_path = DATA_DIR / "q28_resp_full.csv"
    with resp_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def _build_subitems(
    header_row: Sequence[str],
    answer_row: Sequence[str],
    pairs: Dict[Tuple[int, int], Dict[str, int]],
) -> Tuple[List[SubItem], Dict[int, int]]:
    """
    Determine which columns correspond to the desired sub-items.

    Returns:
        list of SubItem descriptors (ordered by column appearance)
        mapping from column index to sub-item index
    """
    subitems: List[SubItem] = []
    col_to_subitem: Dict[int, int] = {}

    for col_idx in range(1, len(header_row)):
        try:
            qid = int(float(header_row[col_idx]))
            aid = int(float(answer_row[col_idx]))
        except (TypeError, ValueError):
            continue

        key = (qid, aid)
        if key not in pairs:
            continue

        meta = pairs[key]
        sub_idx = len(subitems)
        col_to_subitem[col_idx] = sub_idx
        subitems.append(
            SubItem(
                index=sub_idx,
                question_id=meta["question_id"],
                question_type=meta["question_type"],
                answer_id=meta["answer_id"],
                question_option=meta["question_option"],
                answer_category=meta["answer_category"],
            )
        )

    if not subitems:
        raise ValueError("No matching response columns found for selected sub-items.")

    return subitems, col_to_subitem


def _parse_sequences(
    rows: Sequence[Sequence[str]],
    col_to_subitem: Dict[int, int],
) -> List[Dict[str, object]]:
    """Construct sequential data from the filtered columns."""
    sequences: List[Dict[str, object]] = []

    for row in rows:
        if not row:
            continue

        student_id = row[0]
        questions: List[int] = []
        responses: List[int] = []

        for col_idx, sub_idx in col_to_subitem.items():
            if col_idx >= len(row):
                continue

            value = row[col_idx].strip()
            if value == "":
                continue

            try:
                response_val = float(value)
            except ValueError:
                continue

            if math.isnan(response_val):
                continue

            int_val = int(round(response_val))
            questions.append(sub_idx)
            responses.append(int_val)

        if questions:
            sequences.append(
                {
                    "student_id": student_id,
                    "questions": questions,
                    "responses": responses,
                }
            )

    if not sequences:
        raise ValueError("All sequences are empty after filtering.")

    return sequences


def _write_split(filename: Path, data: Sequence[Dict[str, object]]) -> None:
    """Persist split data in new_format layout."""
    with filename.open("w", encoding="utf-8") as f:
        for seq in data:
            questions = seq["questions"]
            responses = seq["responses"]
            assert len(questions) == len(responses), "Mismatched sequence lengths"
            length = len(questions)
            f.write(f"{length}\n")
            f.write(",".join(str(q) for q in questions) + "\n")
            f.write(",".join(str(r) for r in responses) + "\n")


def _save_metadata(
    subitems: Sequence[SubItem],
    sequences: Sequence[Dict[str, object]],
    train_size: int,
) -> None:
    """Save dataset metadata and sub-item mapping."""
    seq_lengths = [len(seq["questions"]) for seq in sequences]
    metadata = {
        "dataset_name": "fujor_ranking6",
        "source": "fujor/q28_resp_full.csv",
        "n_students": len(sequences),
        "n_questions": len(subitems),
        "n_cats": 6,
        "response_type": "ordered_categorical",
        "format": "new_format",
        "description": "Fujor ranking and matrix items with 6 categories converted to sub-items.",
        "train_students": train_size,
        "test_students": len(sequences) - train_size,
        "seq_len_range": [min(seq_lengths), max(seq_lengths)],
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    mapping = [
        {
            "subitem_id": sub.index,
            "question_id": sub.question_id,
            "question_type": sub.question_type,
            "answer_id": sub.answer_id,
            "question_option": sub.question_option,
            "answer_category": sub.answer_category,
        }
        for sub in subitems
    ]
    (OUTPUT_DIR / "subitem_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = _load_selected_pairs()
    raw_rows = _load_response_table()

    if len(raw_rows) < 3:
        raise ValueError("Response table does not contain expected metadata rows.")

    header_row = raw_rows[0]
    answer_row = raw_rows[1]
    data_rows = raw_rows[2:]

    subitems, col_to_subitem = _build_subitems(header_row, answer_row, pairs)
    sequences = _parse_sequences(data_rows, col_to_subitem)

    random.shuffle(sequences)
    train_cutoff = int(len(sequences) * TRAIN_FRACTION)

    train_split = sequences[:train_cutoff]
    test_split = sequences[train_cutoff:]

    _write_split(OUTPUT_DIR / "fujor_ranking6_train.txt", train_split)
    _write_split(OUTPUT_DIR / "fujor_ranking6_test.txt", test_split)
    _save_metadata(subitems, sequences, len(train_split))

    print(f"Saved {len(train_split)} train and {len(test_split)} test sequences to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
