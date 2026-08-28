"""Step 28 (Sections 32-33): final scientific interpretation. Answers the
11 required questions strictly from computed results, classifies each as
SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED, and applies the Section 33
failure-interpretation rules rather than forcing a desired conclusion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Conclusion:
    question: str
    answer: str
    verdict: str  # SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
    evidence: dict


def classify_distillation_improvement(s0_metric: float, s1_metric: float, ci_low: float) -> Conclusion:
    """Q1: Did GenBio distillation improve the 3.7M CNN over ImageNet init?"""
    delta = s1_metric - s0_metric
    if ci_low > 0:
        verdict = "SUPPORTED"
    elif delta > 0:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    return Conclusion("Did GenBio distillation improve the 3.7M CNN over ImageNet init?",
                       f"S1 - S0 downstream metric delta = {delta:.4f} (95% CI lower bound {ci_low:.4f})",
                       verdict, {"s0": s0_metric, "s1": s1_metric, "delta": delta, "ci_low": ci_low})


def classify_representation_fidelity(mean_cosine: float) -> Conclusion:
    """Q2: How closely did the student reproduce GenBio representations?"""
    if mean_cosine >= 0.7:
        verdict, txt = "SUPPORTED", "high"
    elif mean_cosine >= 0.4:
        verdict, txt = "PARTIALLY_SUPPORTED", "moderate"
    else:
        verdict, txt = "NOT_SUPPORTED", "low"
    return Conclusion("How closely did the student reproduce GenBio representations on unseen CPTAC patients?",
                       f"mean test cosine similarity = {mean_cosine:.4f} ({txt} fidelity)", verdict,
                       {"mean_cosine": mean_cosine})


def classify_compression_retention(student_metric: float, genbio_metric: float) -> Conclusion:
    """Q3: how much performance retained after ~300x compression."""
    if genbio_metric == 0:
        retention = float("nan")
    else:
        retention = student_metric / genbio_metric
    if retention >= 0.9:
        verdict = "SUPPORTED"
    elif retention >= 0.6:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    return Conclusion("How much performance was retained after ~300x parameter compression?",
                       f"student/GenBio performance retention = {retention:.2%}", verdict,
                       {"student": student_metric, "genbio": genbio_metric, "retention": retention})


def classify_real_vs_shuffled(hypothesis_test: dict) -> Conclusion:
    """Q4: primary hypothesis (Section 25)."""
    ci_low = hypothesis_test["bootstrap"]["ci_low"]
    verdict = "SUPPORTED" if hypothesis_test["h1_supported_ci_above_zero"] else (
        "PARTIALLY_SUPPORTED" if hypothesis_test["bootstrap"]["mean_delta"] > 0 else "NOT_SUPPORTED")
    note = ""
    if verdict != "SUPPORTED":
        note = " -- current evidence does not show topology contributes beyond graph capacity" if ci_low <= 0 <= hypothesis_test["bootstrap"]["mean_delta"] * 2 else ""
    return Conclusion("Did the real graph outperform the degree-preserving shuffled graph?",
                       f"mean delta R = {hypothesis_test['bootstrap']['mean_delta']:.4f}, "
                       f"95% CI [{ci_low:.4f}, {hypothesis_test['bootstrap']['ci_high']:.4f}]{note}",
                       verdict, hypothesis_test)


def classify_multiscale_vs_local(delta_ci: dict) -> Conclusion:
    """Q5. Section 33: 'If local > multiscale, do not claim multiscale superiority.'"""
    verdict = "SUPPORTED" if delta_ci["ci_low"] > 0 else ("NOT_SUPPORTED" if delta_ci["mean_delta"] <= 0 else "PARTIALLY_SUPPORTED")
    return Conclusion("Did multiscale context outperform purely local context?",
                       f"mean delta R (multiscale-local) = {delta_ci['mean_delta']:.4f}, CI low {delta_ci['ci_low']:.4f}",
                       verdict, delta_ci)


def classify_morans_correlation(corr_result: dict) -> Conclusion:
    """Q6."""
    r, p = corr_result["pearson_r"], corr_result["p_value"]
    verdict = "SUPPORTED" if (r == r and p == p and r > 0 and p < 0.05) else (
        "PARTIALLY_SUPPORTED" if (r == r and r > 0) else "NOT_SUPPORTED")
    return Conclusion("Did graph improvement correlate with spatial autocorrelation of gene expression?",
                       f"Corr(delta_g, MoranI_g) = {r}, p = {p}", verdict, corr_result)


def build_final_report(conclusions: list[Conclusion]) -> str:
    lines = ["# Final Scientific Interpretation\n"]
    for i, c in enumerate(conclusions, 1):
        lines.append(f"## Q{i}. {c.question}\n\n**Verdict: {c.verdict}**\n\n{c.answer}\n")
    return "\n".join(lines)
