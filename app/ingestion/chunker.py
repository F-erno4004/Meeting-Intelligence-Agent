from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from .parser import ParsedTranscript, TranscriptTurn


@dataclass(frozen=True)
class TranscriptChunkData:
    chunk_index: int
    speaker: str
    content: str
    char_start: int
    char_end: int
    token_count: int


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # text-embedding-3-small uses the cl100k_base tokenizer.
        return tiktoken.get_encoding("cl100k_base")


def chunk_transcript(
    transcript: ParsedTranscript,
    max_tokens: int = 550,
    overlap_turns: int = 1,
    tokenizer_model: str = "text-embedding-3-small",
) -> list[TranscriptChunkData]:
    """
    Group complete transcript turns into token-bounded chunks.

    Chunk content is always an exact contiguous slice of original_text.
    A one-turn overlap helps preserve conversational context at chunk
    boundaries. A single unusually long turn is kept intact rather than
    silently splitting its source text.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_turns < 0:
        raise ValueError("overlap_turns cannot be negative")
    if not transcript.turns:
        return []

    encoding = _encoding_for_model(tokenizer_model)
    turns = transcript.turns
    turn_token_counts = [
        len(encoding.encode(turn.text)) for turn in turns
    ]

    chunks: list[TranscriptChunkData] = []
    start_index = 0

    while start_index < len(turns):
        end_index = start_index
        accumulated = 0

        while end_index < len(turns):
            next_count = turn_token_counts[end_index]
            if end_index > start_index and accumulated + next_count > max_tokens:
                break
            accumulated += next_count
            end_index += 1

            # Include at least one complete turn, even if it exceeds max_tokens.
            if accumulated >= max_tokens:
                break

        selected = turns[start_index:end_index]
        first_turn = selected[0]
        last_turn = selected[-1]

        char_start = first_turn.char_start
        char_end = last_turn.char_end
        content = transcript.original_text[char_start:char_end]

        speakers = {turn.speaker for turn in selected}
        speaker = next(iter(speakers)) if len(speakers) == 1 else "multiple"

        chunks.append(
            TranscriptChunkData(
                chunk_index=len(chunks),
                speaker=speaker,
                content=content,
                char_start=char_start,
                char_end=char_end,
                token_count=len(encoding.encode(content)),
            )
        )

        if end_index >= len(turns):
            break

        # Move forward while retaining the requested number of complete turns.
        next_start = max(start_index + 1, end_index - overlap_turns)
        start_index = next_start

    return chunks
