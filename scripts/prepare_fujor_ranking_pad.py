"""
Create a padded Fujor dataset where all ranking items are expanded to 9
sub-items and matrix items to 6 sub-items. Missing positions are filled with
the lowest category (0), ensuring consistent dimensionality per item.

Generated files:

    data/fujor_ranking_pad/fujor_ranking_pad_train.txt
    data/fujor_ranking_pad/fujor_ranking_pad_test.txt
    data/fujor_ranking_pad/metadata.json
    data/fujor_ranking_pad/subitem_mapping.json
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fujor"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "fujor_ranking_pad"
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

MAX_CATEGORIES = {
    "Ranking": 9,
    "Matrix": 6,
}

PAD_VALUE = 0  # Lowest category value


@dataclass
class ActualOption:
    subitem_id: int
    question_id: int
    question_type: str
    answer_id: int
    question_option: int
    answer_category: int
    column_index: int


@dataclass
class PaddingOption:
    subitem_id: int
    question_id: int
    question_type: str
    pad_position: int
    target_slots: int
    question_option: int
    answer_category: int = field(default=0)


@dataclass
class QuestionBundle:
    question_id: int
    question_type: str
    options: List[ActualOption]
    padding: List[PaddingOption]


def _load_metadata() -> List[Dict]:
    path = DATA_DIR / "q28_doc_textless.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_responses() -> List[List[str]]:
    path = DATA_DIR / "q28_resp_full.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def _build_column_map(header_row: Sequence[str], answer_row: Sequence[str]) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    for idx in range(1, len(header_row)):
        try:
            qid = int(float(header_row[idx]))
            aid = int(float(answer_row[idx]))
        except (TypeError, ValueError):
            continue
        mapping[(qid, aid)] = idx
    return mapping


def _build_question_bundles(metadata: List[Dict], column_map: Dict[Tuple[int, int], int]) -> Tuple[List[QuestionBundle], List[Dict]]:
    # Filter to ranking and matrix questions
    rows = [
        row for row in metadata
        if row.get("QuestionType") in MAX_CATEGORIES
    ]

    # Group rows by question id
    grouped: Dict[int, List[Dict]] = {}
    for row in rows:
        try:
            qid = int(float(row["QuestionId"]))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(qid, []).append(row)

    bundles: List[QuestionBundle] = []
    subitems_meta: List[Dict] = []
    next_subitem_id = 0

    for question_id in sorted(grouped):
        entries = grouped[question_id]
        qtype = entries[0].get("QuestionType")
        base_target = MAX_CATEGORIES[qtype]

        # Sort actual answers by AnswerId for deterministic ordering
        entries.sort(key=lambda e: float(e.get("AnswerId", 0)))

        options: List[ActualOption] = []
        for entry in entries:
            try:
                answer_id = int(float(entry["AnswerId"]))
                question_option = int(float(entry.get("QuestionOption", 0) or 0))
                answer_category = int(float(entry.get("AnswerCategory", 0) or 0))
            except (TypeError, ValueError):
                continue

            col_idx = column_map.get((question_id, answer_id))
            if col_idx is None:
                continue

            option = ActualOption(
                subitem_id=next_subitem_id,
                question_id=question_id,
                question_type=qtype,
                answer_id=answer_id,
                question_option=question_option,
                answer_category=answer_category,
                column_index=col_idx,
            )
            options.append(option)
            subitems_meta.append({
                "subitem_id": option.subitem_id,
                "question_id": option.question_id,
                "question_type": option.question_type,
                "answer_id": option.answer_id,
                "question_option": option.question_option,
                "answer_category": option.answer_category,
                "is_padding": False,
                "pad_position": None
            })
            next_subitem_id += 1

        target_slots = max(base_target, len(options))
        pad_needed = max(0, target_slots - len(options))
        padding: List[PaddingOption] = []
        for pad_pos in range(pad_needed):
            pad = PaddingOption(
                subitem_id=next_subitem_id,
                question_id=question_id,
                question_type=qtype,
                pad_position=pad_pos,
                target_slots=target_slots,
                question_option=target_slots
            )
            padding.append(pad)
            subitems_meta.append({
                "subitem_id": pad.subitem_id,
                "question_id": pad.question_id,
                "question_type": pad.question_type,
                "answer_id": None,
                "question_option": pad.question_option,
                "answer_category": None,
                "is_padding": True,
                "pad_position": pad.pad_position
            })
            next_subitem_id += 1

        if options:
            bundles.append(QuestionBundle(
                question_id=question_id,
                question_type=qtype,
                options=options,
                padding=padding
            ))

    return bundles, subitems_meta


def _parse_sequences(
    data_rows: Sequence[Sequence[str]],
    bundles: Sequence[QuestionBundle],
) -> List[Dict[str, object]]:
    sequences: List[Dict[str, object]] = []

    for row in data_rows:
        if not row:
            continue

        student_id = row[0]
        questions: List[int] = []
        responses: List[int] = []

        for bundle in bundles:
            question_has_response = False

            for option in bundle.options:
                if option.column_index >= len(row):
                    continue

                value = row[option.column_index].strip()
                if value == "":
                    continue

                try:
                    response_val = float(value)
                except ValueError:
                    continue

                if math.isnan(response_val) or math.isinf(response_val):
                    continue

                int_val = int(round(response_val))
                questions.append(option.subitem_id)
                responses.append(int_val)
                question_has_response = True

            if question_has_response:
                for pad in bundle.padding:
                    questions.append(pad.subitem_id)
                    responses.append(PAD_VALUE)

        if questions:
            sequences.append({
                "student_id": student_id,
                "questions": questions,
                "responses": responses,
            })

    if not sequences:
        raise ValueError("All sequences are empty after padding transformation.")

    return sequences


def _write_split(filename: Path, sequences: Sequence[Dict[str, object]]) -> None:
    with filename.open("w", encoding="utf-8") as f:
        for seq in sequences:
            q = seq["questions"]
            r = seq["responses"]
            assert len(q) == len(r)
            f.write(f"{len(q)}\n")
            f.write(",".join(str(x) for x in q) + "\n")
            f.write(",".join(str(x) for x in r) + "\n")


def _save_metadata(subitems: Sequence[Dict], sequences: Sequence[Dict], train_size: int) -> None:
    seq_lengths = [len(seq["questions"]) for seq in sequences]
    max_response = max(val for seq in sequences for val in seq["responses"])

    metadata = {
        "dataset_name": "fujor_ranking_pad",
        "source": "fujor/q28_resp_full.csv",
        "n_students": len(sequences),
        "n_questions": len(subitems),
        "n_cats": max_response + 1,
        "response_type": "ordered_categorical",
        "format": "new_format",
        "description": "Fujor ranking/matrix items padded to fixed slots (ranking=9, matrix=6).",
        "train_students": train_size,
        "test_students": len(sequences) - train_size,
        "seq_len_range": [min(seq_lengths), max(seq_lengths)],
    }

    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "subitem_mapping.json").write_text(json.dumps(subitems, indent=2), encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = _load_metadata()
    resp_rows = _load_responses()
    if len(resp_rows) < 3:
        raise ValueError("Response table does not contain expected metadata rows.")

    header_row = resp_rows[0]
    answer_row = resp_rows[1]
    data_rows = resp_rows[2:]

    column_map = _build_column_map(header_row, answer_row)
    bundles, subitems_meta = _build_question_bundles(metadata_rows, column_map)

    sequences = _parse_sequences(data_rows, bundles)

    random.shuffle(sequences)
    train_cutoff = int(len(sequences) * TRAIN_FRACTION)
    train_split = sequences[:train_cutoff]
    test_split = sequences[train_cutoff:]

    _write_split(OUTPUT_DIR / "fujor_ranking_pad_train.txt", train_split)
    _write_split(OUTPUT_DIR / "fujor_ranking_pad_test.txt", test_split)
    _save_metadata(subitems_meta, sequences, len(train_split))

    print(f"Saved {len(train_split)} train and {len(test_split)} test sequences to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
