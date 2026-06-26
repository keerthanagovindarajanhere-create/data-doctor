from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pandas as pd


@dataclass
class DatasetCertificate:
    certificate_id: str
    grade: str
    score: int
    risk: str
    ml_ready: bool
    summary: str


def _grade(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _risk(score: int) -> str:
    if score >= 90:
        return "Low"
    if score >= 75:
        return "Moderate"
    return "High"


def generate_certificate(
    dataframe: pd.DataFrame,
    readiness_score: int,
    findings: list,
    recommendations: list,
) -> DatasetCertificate:

    fingerprint = hashlib.sha256(
        dataframe.head(100).to_csv(index=False).encode("utf-8")
    ).hexdigest()[:10].upper()

    certificate_id = f"DD-{fingerprint}"

    high_findings = sum(
        1 for finding in findings
        if finding.get("Severity") == "High"
    )

    ml_ready = (
        readiness_score >= 75
        and high_findings <= 2
    )

    if ml_ready:
        summary = (
            "This dataset is suitable for machine learning after "
            "recommended preprocessing."
        )
    else:
        summary = (
            "This dataset requires additional preprocessing before "
            "reliable model training."
        )

    return DatasetCertificate(
        certificate_id=certificate_id,
        grade=_grade(readiness_score),
        score=readiness_score,
        risk=_risk(readiness_score),
        ml_ready=ml_ready,
        summary=summary,
    )