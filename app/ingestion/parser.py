from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TURN_RE = re.compile(
    r"(?m)^(?P<timestamp>\d{2}:\d{2}:\d{2})[ \t]+"
    r"(?P<speaker>Speaker [A-Z])\s*$"
)
HEADER_RE = re.compile(
    r"(?m)^\s*(Meeting|Team|Meeting number|Speakers|Times|Source):\s*(.*?)\s*$"
)
COMBINED_SEPARATOR_RE = re.compile(r"(?m)^\s*===\s*.+?\.txt\s*===\s*$")


@dataclass(frozen=True)
class TranscriptTurn:
    timestamp: str
    speaker: str
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class ParsedTranscript:
    meeting_id: str
    team_id: str | None
    meeting_sequence: int | None
    meeting_suffix: str | None
    speakers: tuple[str, ...]
    source_filename: str
    original_text: str
    turns: tuple[TranscriptTurn, ...]


def _parse_meeting_number(value: str | None) -> tuple[int | None, str | None]:
    if not value:
        return None, None
    match = re.match(r"\s*(\d+)\s*(?:\(([a-zA-Z])\))?", value)
    if not match:
        return None, None
    sequence = int(match.group(1))
    suffix = match.group(2).lower() if match.group(2) else None
    return sequence, suffix


def parse_transcript(text: str, source_filename: str) -> ParsedTranscript:
    """
    Parse one original AMI meeting transcript.

    Turn offsets are offsets into `original_text`. Turn text is an exact source
    slice beginning at the timestamp/speaker line and ending immediately before
    the next turn header (or EOF).
    """
    if not text.strip():
        raise ValueError(f"{source_filename}: transcript is empty")

    # The supplied combined example is for format inspection only. Requiring
    # one meeting per file prevents accidental ingestion of that combined file.
    separators = list(COMBINED_SEPARATOR_RE.finditer(text))
    meeting_headers = list(re.finditer(r"(?m)^\s*Meeting:\s*\S+", text))
    if len(separators) > 0 or len(meeting_headers) > 1:
        raise ValueError(
            f"{source_filename}: appears to contain multiple combined transcripts; "
            "ingest the original one-meeting-per-file transcripts instead"
        )

    metadata = {key: value.strip() for key, value in HEADER_RE.findall(text)}
    meeting_id = metadata.get("Meeting") or Path(source_filename).stem
    team_id = metadata.get("Team") or None

    sequence, suffix = _parse_meeting_number(metadata.get("Meeting number"))

    # Conservative fallback: e.g. TS3012a -> team TS3012, suffix a.
    if team_id is None:
        fallback = re.match(r"^(.*\d+)([a-zA-Z])$", meeting_id)
        if fallback:
            team_id = fallback.group(1)
            if suffix is None:
                suffix = fallback.group(2).lower()

    if sequence is None and suffix is not None:
        sequence = ord(suffix) - ord("a") + 1

    matches = list(TURN_RE.finditer(text))
    if not matches:
        raise ValueError(
            f"{source_filename}: no timestamp/speaker turn headers found. "
            "Expected lines such as '00:00:02 Speaker A'."
        )

    turns: list[TranscriptTurn] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turn_text = text[start:end]
        turns.append(
            TranscriptTurn(
                timestamp=match.group("timestamp"),
                speaker=match.group("speaker"),
                char_start=start,
                char_end=end,
                text=turn_text,
            )
        )

    speaker_values = metadata.get("Speakers", "")
    speakers = tuple(
        item.strip() for item in speaker_values.split(",") if item.strip()
    )
    if not speakers:
        speakers = tuple(sorted({turn.speaker for turn in turns}))

    return ParsedTranscript(
        meeting_id=meeting_id,
        team_id=team_id,
        meeting_sequence=sequence,
        meeting_suffix=suffix,
        speakers=speakers,
        source_filename=Path(source_filename).name,
        original_text=text,
        turns=tuple(turns),
    )


def parse_transcript_file(path: str | Path) -> ParsedTranscript:
    path = Path(path)
    # newline="" preserves original CRLF/LF characters. The offsets and stored
    # chunk content therefore refer to the exact decoded source representation.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        text = handle.read()
    return parse_transcript(text, path.name)
