from datetime import datetime
from typing import Optional

from sqlalchemy import ( String, Text, Integer, ForeignKey, DateTime, CheckConstraint, Index, Computed, UniqueConstraint,func)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.memory.database import Base


# =====================================================
# 1. MEETINGS
# =====================================================

class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    meeting_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    team_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    meeting_sequence: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    source_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    transcript_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships

    chunks: Mapped[list["TranscriptChunk"]] = relationship(
        back_populates="meeting",
        cascade="all, delete-orphan",
    )

    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="meeting",
    )

    actions: Mapped[list["MeetingAction"]] = relationship(
        back_populates="meeting",
    )

    disagreements: Mapped[list["Disagreement"]] = relationship(
        back_populates="meeting",
    )


# =====================================================
# 2. TRANSCRIPT CHUNKS
# =====================================================

class TranscriptChunk(Base):
    __tablename__ = "transcript_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    speaker: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Position in the original transcript.
    char_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    char_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # text-embedding-3-small default dimensions = 1536
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536),
        nullable=True,
    )

    # PostgreSQL full-text search index.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', content)",
            persisted=True,
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    meeting: Mapped["Meeting"] = relationship(
        back_populates="chunks",
    )

    decision_evidence: Mapped[list["DecisionEvidence"]] = relationship(
        back_populates="chunk",
    )

    action_evidence: Mapped[list["ActionEvidence"]] = relationship(
        back_populates="chunk",
    )

    disagreement_evidence: Mapped[list["DisagreementEvidence"]] = relationship(
        back_populates="chunk",
    )

    __table_args__ = (
        UniqueConstraint(
            "meeting_id",
            "chunk_index",
            name="uq_meeting_chunk_index",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_chunk_index_nonnegative",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_chunk_char_start_nonnegative",
        ),
        CheckConstraint(
            "char_end IS NULL OR char_start IS NULL OR char_end >= char_start",
            name="ck_chunk_char_range",
        ),
        Index(
            "ix_transcript_chunks_search",
            "search_vector",
            postgresql_using="gin",
        ),
    )


# =====================================================
# 3. DECISIONS
# =====================================================

class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    team_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    topic: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    decision: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="proposed",
        index=True,
    )

    supersedes_decision_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("decisions.id", ondelete="SET NULL"),
        nullable=True,
    )

    meeting: Mapped["Meeting"] = relationship(
        back_populates="decisions",
    )

    supersedes: Mapped[Optional["Decision"]] = relationship(
        remote_side="Decision.id",
        foreign_keys=[supersedes_decision_id],
    )

    evidence: Mapped[list["DecisionEvidence"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('proposed', 'current', 'superseded', "
            "'rejected', 'unclear')",
            name="ck_decision_status",
        ),
    )


# =====================================================
# 4. DECISION EVIDENCE
# =====================================================

class DecisionEvidence(Base):
    __tablename__ = "decision_evidence"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("transcript_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence_role: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="supporting",
    )

    decision: Mapped["Decision"] = relationship(
        back_populates="evidence",
    )

    chunk: Mapped["TranscriptChunk"] = relationship(
        back_populates="decision_evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "evidence_role IN ('supporting', 'challenging', 'revision')",
            name="ck_decision_evidence_role",
        ),
    )


# =====================================================
# 5. MEETING ACTIONS
# =====================================================

class MeetingAction(Base):
    __tablename__ = "meeting_actions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    team_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    task: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    owner_speaker: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    owner_description: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="unclear",
        index=True,
    )

    due_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    meeting: Mapped["Meeting"] = relationship(
        back_populates="actions",
    )

    evidence: Mapped[list["ActionEvidence"]] = relationship(
        back_populates="action",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'in_progress', 'completed', "
            "'cancelled', 'unclear')",
            name="ck_action_status",
        ),
    )


# =====================================================
# 6. ACTION EVIDENCE
# =====================================================

class ActionEvidence(Base):
    __tablename__ = "action_evidence"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    action_id: Mapped[int] = mapped_column(
        ForeignKey("meeting_actions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("transcript_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    action: Mapped["MeetingAction"] = relationship(
        back_populates="evidence",
    )

    chunk: Mapped["TranscriptChunk"] = relationship(
        back_populates="action_evidence",
    )


# =====================================================
# 7. DISAGREEMENTS
# =====================================================

class Disagreement(Base):
    __tablename__ = "disagreements"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    team_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    topic: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    position_a: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    position_b: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    speaker_a: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    speaker_b: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    resolution_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="unclear",
        index=True,
    )

    meeting: Mapped["Meeting"] = relationship(
        back_populates="disagreements",
    )

    evidence: Mapped[list["DisagreementEvidence"]] = relationship(
        back_populates="disagreement",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "resolution_status IN ('open', 'resolved', 'unclear')",
            name="ck_disagreement_status",
        ),
    )


# =====================================================
# 8. DISAGREEMENT EVIDENCE
# =====================================================

class DisagreementEvidence(Base):
    __tablename__ = "disagreement_evidence"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    disagreement_id: Mapped[int] = mapped_column(
        ForeignKey("disagreements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("transcript_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    disagreement: Mapped["Disagreement"] = relationship(
        back_populates="evidence",
    )

    chunk: Mapped["TranscriptChunk"] = relationship(
        back_populates="disagreement_evidence",
    )
