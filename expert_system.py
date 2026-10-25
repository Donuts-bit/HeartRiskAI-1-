"""Rule-Based Cardiovascular Risk Expert System for HeartRiskAI.
Evaluates structured patient biomarkers against established clinical guidelines:
- 2017 ACC/AHA High Blood Pressure Clinical Practice Guidelines
- NCEP ATP III / 2018 AHA/ACC Cholesterol Guidelines
- American Diabetes Association (ADA) 2024 Standards of Care
- CDC / WHO Tobacco & Obesity Cardiovascular Risk Assessment Standards

Strict Architectural Principles:
1. Pure symbolic rule-based reasoning (no machine learning, no arbitrary pseudo-probabilities).
2. Explicit IF-THEN rules with documented medical sources and rule identifiers.
3. Rigorous input validation and distinction between missing data and negative findings.
4. Independent execution from ML classifiers for decoupled modular testing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set, Tuple

from src.config import PHYSIOLOGICAL_BOUNDS, CLINICAL_THRESHOLDS


class SeverityLevel(Enum):
    """Clinical severity gradations for rule outcomes."""
    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ClinicalRule:
    """Explicit clinical rule with citation, logic, and educational guidance."""
    rule_id: str
    category: str
    name: str
    condition_description: str
    severity: SeverityLevel
    medical_source: str
    condition_fn: Callable[[Dict[str, Any]], bool]
    finding: str
    recommendation: str


@dataclass
class ExpertEvaluationResult:
    """Structured output from expert system evaluation."""
    patient_data: Dict[str, Any]
    missing_fields: List[str]
    invalid_fields: Dict[str, str]
    triggered_rules: List[Dict[str, Any]]
    identified_risk_factors: List[str]
    educational_guidance: List[str]
    qualitative_risk_tier: str
    has_critical_alerts: bool
    clinical_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_data": self.patient_data,
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
            "triggered_rules": self.triggered_rules,
            "identified_risk_factors": self.identified_risk_factors,
            "educational_guidance": self.educational_guidance,
            "qualitative_risk_tier": self.qualitative_risk_tier,
            "has_critical_alerts": self.has_critical_alerts,
            "clinical_disclaimer": self.clinical_disclaimer,
        }


class CardiovascularExpertSystem:
    """Clinical rule-based inference engine for cardiovascular risk factor identification."""

    DISCLAIMER = (
        "CLINICAL DISCLAIMER: This expert system is an educational decision-support "
        "tool intended for academic demonstration. Rule matches identify documented "
        "cardiovascular risk factors based on published clinical guidelines. They do "
        "not constitute a medical diagnosis, do not establish clinical prognosis, and "
        "do not predict imminent acute myocardial infarction (heart attack). Any health "
        "decisions must be evaluated by a licensed medical practitioner."
    )

    def __init__(self):
        self.rules: List[ClinicalRule] = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[ClinicalRule]:
        """Codifies evidence-based clinical rules with authoritative medical citations."""
        return [
            # -------------------------------------------------------------
            # HYPERTENSION RULES (2017 ACC/AHA Blood Pressure Guidelines)
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_BP_CRISIS",
                category="Blood Pressure",
                name="Hypertensive Crisis Alert",
                condition_description="sysBP >= 180 mmHg OR diaBP >= 120 mmHg",
                severity=SeverityLevel.CRITICAL,
                medical_source="2017 ACC/AHA Guideline for High Blood Pressure in Adults (Whelton et al., JACC 2018)",
                condition_fn=lambda p: (
                    p.get("sysBP") is not None and p["sysBP"] >= 180.0
                ) or (
                    p.get("diaBP") is not None and p["diaBP"] >= 120.0
                ),
                finding="Systolic BP >= 180 mmHg or Diastolic BP >= 120 mmHg indicates Hypertensive Crisis.",
                recommendation="CRITICAL: Immediate medical evaluation is required. Re-check BP promptly; if accompanied by chest pain, shortness of breath, or neurological deficits, seek emergency care.",
            ),
            ClinicalRule(
                rule_id="RULE_BP_STAGE2",
                category="Blood Pressure",
                name="Stage 2 Hypertension",
                condition_description="140 <= sysBP < 180 mmHg OR 90 <= diaBP < 120 mmHg",
                severity=SeverityLevel.HIGH,
                medical_source="2017 ACC/AHA Guideline for High Blood Pressure in Adults",
                condition_fn=lambda p: (
                    (p.get("sysBP") is not None and 140.0 <= p["sysBP"] < 180.0)
                    or (p.get("diaBP") is not None and 90.0 <= p["diaBP"] < 120.0)
                ),
                finding="Blood pressure meets criteria for Stage 2 Hypertension.",
                recommendation="Consult a physician regarding antihypertensive pharmacotherapy and intensive lifestyle modifications (DASH diet, sodium reduction < 1,500 mg/day, regular physical activity).",
            ),
            ClinicalRule(
                rule_id="RULE_BP_STAGE1",
                category="Blood Pressure",
                name="Stage 1 Hypertension",
                condition_description="(130 <= sysBP < 140 mmHg OR 80 <= diaBP < 90 mmHg) AND sysBP < 140 AND diaBP < 90",
                severity=SeverityLevel.MODERATE,
                medical_source="2017 ACC/AHA Guideline for High Blood Pressure in Adults",
                condition_fn=lambda p: (
                    p.get("sysBP") is not None
                    and p.get("diaBP") is not None
                    and (130.0 <= p["sysBP"] < 140.0 or 80.0 <= p["diaBP"] < 90.0)
                    and p["sysBP"] < 140.0
                    and p["diaBP"] < 90.0
                ),
                finding="Blood pressure meets criteria for Stage 1 Hypertension.",
                recommendation="Recommend non-pharmacological lifestyle interventions with a 3- to 6-month clinical re-assessment.",
            ),
            ClinicalRule(
                rule_id="RULE_BP_ELEVATED",
                category="Blood Pressure",
                name="Elevated Blood Pressure",
                condition_description="120 <= sysBP < 130 mmHg AND diaBP < 80 mmHg",
                severity=SeverityLevel.LOW,
                medical_source="2017 ACC/AHA Guideline for High Blood Pressure in Adults",
                condition_fn=lambda p: (
                    p.get("sysBP") is not None
                    and p.get("diaBP") is not None
                    and 120.0 <= p["sysBP"] < 130.0
                    and p["diaBP"] < 80.0
                ),
                finding="Systolic BP is elevated (120-129 mmHg) with normal diastolic BP (< 80 mmHg).",
                recommendation="Healthy dietary habits, weight management, and routine annual blood pressure monitoring.",
            ),
            ClinicalRule(
                rule_id="RULE_BP_MEDICATED",
                category="Blood Pressure",
                name="Active Antihypertensive Therapy",
                condition_description="BPMeds == 1",
                severity=SeverityLevel.MODERATE,
                medical_source="Framingham Heart Study Cardiovascular Risk Factors Profile (Wilson et al., Circulation 1998)",
                condition_fn=lambda p: p.get("BPMeds") == 1,
                finding="Patient is currently receiving pharmacological treatment for high blood pressure.",
                recommendation="Continue prescribed medication compliance and monitor blood pressure periodically to verify target control.",
            ),

            # -------------------------------------------------------------
            # DYSLIPIDEMIA RULES (NCEP ATP III / 2018 AHA/ACC Guidelines)
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_CHOL_HIGH",
                category="Cholesterol",
                name="High Total Cholesterol (Hypercholesterolemia)",
                condition_description="totChol >= 240 mg/dL",
                severity=SeverityLevel.HIGH,
                medical_source="Third Report of the NCEP Expert Panel (ATP III), JAMA 2001",
                condition_fn=lambda p: p.get("totChol") is not None and p["totChol"] >= 240.0,
                finding="Total serum cholesterol >= 240 mg/dL indicates hypercholesterolemia.",
                recommendation="Recommend comprehensive fasting lipid panel (LDL, HDL, triglycerides). Discuss dietary saturated fat restriction, soluble fiber increase, and potential statin therapy evaluation.",
            ),
            ClinicalRule(
                rule_id="RULE_CHOL_BORDERLINE",
                category="Cholesterol",
                name="Borderline High Total Cholesterol",
                condition_description="200 <= totChol < 240 mg/dL",
                severity=SeverityLevel.MODERATE,
                medical_source="NCEP ATP III Cholesterol Guidelines",
                condition_fn=lambda p: (
                    p.get("totChol") is not None and 200.0 <= p["totChol"] < 240.0
                ),
                finding="Total serum cholesterol is borderline high (200-239 mg/dL).",
                recommendation="Adopt heart-healthy Mediterranean or DASH dietary pattern and maintain aerobic exercise (>= 150 min/week).",
            ),

            # -------------------------------------------------------------
            # GLYCEMIA / DIABETES RULES (American Diabetes Association 2024)
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_DIABETES_DIAGNOSED",
                category="Glycemia",
                name="Diagnosed Diabetes Mellitus",
                condition_description="diabetes == 1",
                severity=SeverityLevel.HIGH,
                medical_source="ADA Standards of Care in Diabetes (Diabetes Care 2024)",
                condition_fn=lambda p: p.get("diabetes") == 1,
                finding="Documented history of diabetes mellitus (major independent CVD risk multiplier).",
                recommendation="Target glycemic control (HbA1c monitoring), regular cardiovascular screening, and renal function evaluations as recommended by ADA guidelines.",
            ),
            ClinicalRule(
                rule_id="RULE_GLUCOSE_DIABETIC_RANGE",
                category="Glycemia",
                name="Fasting Hyperglycemia (Diabetes Threshold)",
                condition_description="glucose >= 126 mg/dL AND diabetes == 0",
                severity=SeverityLevel.HIGH,
                medical_source="ADA Criteria for the Diagnosis of Diabetes (Diabetes Care 2024)",
                condition_fn=lambda p: (
                    p.get("glucose") is not None
                    and p["glucose"] >= 126.0
                    and p.get("diabetes", 0) != 1
                ),
                finding="Fasting blood glucose >= 126 mg/dL without prior diabetes diagnosis.",
                recommendation="Repeat fasting plasma glucose or measure HbA1c to confirm diabetes diagnosis. Consult a physician for early metabolic management.",
            ),
            ClinicalRule(
                rule_id="RULE_GLUCOSE_IMPAIRED",
                category="Glycemia",
                name="Impaired Fasting Glucose (Prediabetes)",
                condition_description="100 <= glucose < 126 mg/dL AND diabetes == 0",
                severity=SeverityLevel.MODERATE,
                medical_source="ADA Classification and Diagnosis of Diabetes (Diabetes Care 2024)",
                condition_fn=lambda p: (
                    p.get("glucose") is not None
                    and 100.0 <= p["glucose"] < 126.0
                    and p.get("diabetes", 0) != 1
                ),
                finding="Fasting glucose between 100-125 mg/dL indicates prediabetes / impaired fasting glucose.",
                recommendation="Lifestyle intervention targeting 5-7% weight reduction and moderate-intensity physical activity to prevent progression to overt type 2 diabetes.",
            ),

            # -------------------------------------------------------------
            # TOBACCO CONSUMPTION RULES (CDC / WHO Guidelines)
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_TOBACCO_HEAVY",
                category="Tobacco",
                name="Heavy Tobacco Smoking",
                condition_description="cigsPerDay >= 20",
                severity=SeverityLevel.HIGH,
                medical_source="CDC Office on Smoking and Health / US Surgeon General Report on Smoking Cessation",
                condition_fn=lambda p: p.get("cigsPerDay") is not None and p["cigsPerDay"] >= 20.0,
                finding="Smoking >= 20 cigarettes/day (1 pack/day or more) represents severe tobacco burden.",
                recommendation="Urgent smoking cessation counseling. Strongly consider nicotine replacement therapy (NRT) or pharmacotherapy (bupropion, varenicline). Smoking cessation reduces CVD risk by ~50% within 1-2 years.",
            ),
            ClinicalRule(
                rule_id="RULE_TOBACCO_ACTIVE",
                category="Tobacco",
                name="Active Tobacco Smoking",
                condition_description="(currentSmoker == 1 OR cigsPerDay > 0) AND cigsPerDay < 20",
                severity=SeverityLevel.MODERATE,
                medical_source="WHO Guidelines for Controlling and Monitoring the Tobacco Epidemic",
                condition_fn=lambda p: (
                    (p.get("currentSmoker") == 1 or (p.get("cigsPerDay") is not None and p["cigsPerDay"] > 0))
                    and (p.get("cigsPerDay") is None or p["cigsPerDay"] < 20.0)
                ),
                finding="Active cigarette smoking identified.",
                recommendation="Complete tobacco cessation is the single most impactful modifiable lifestyle intervention for cardiovascular prevention.",
            ),

            # -------------------------------------------------------------
            # OBESITY RULES (CDC / WHO BMI Classifications)
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_OBESITY_CLASS",
                category="Body Weight",
                name="Clinical Obesity (BMI >= 30)",
                condition_description="BMI >= 30.0 kg/m^2",
                severity=SeverityLevel.HIGH,
                medical_source="CDC Adult BMI Guidelines / WHO Obesity Classification",
                condition_fn=lambda p: p.get("BMI") is not None and p["BMI"] >= 30.0,
                finding="Body Mass Index >= 30.0 kg/m^2 indicates clinical obesity.",
                recommendation="Multidisciplinary weight management, caloric restriction, and regular physical activity to reduce cardiac workload and metabolic inflammation.",
            ),
            ClinicalRule(
                rule_id="RULE_OVERWEIGHT",
                category="Body Weight",
                name="Overweight Status (BMI 25-29.9)",
                condition_description="25.0 <= BMI < 30.0 kg/m^2",
                severity=SeverityLevel.LOW,
                medical_source="CDC Adult BMI Guidelines",
                condition_fn=lambda p: p.get("BMI") is not None and 25.0 <= p["BMI"] < 30.0,
                finding="Body Mass Index between 25.0-29.9 kg/m^2 indicates overweight status.",
                recommendation="Weight stabilization and dietary awareness to prevent progression to obesity.",
            ),

            # -------------------------------------------------------------
            # DEMOGRAPHIC & VASCULAR HISTORY RULES
            # -------------------------------------------------------------
            ClinicalRule(
                rule_id="RULE_AGE_SEX_RISK",
                category="Demographics",
                name="Age-Sex Baseline Cardiovascular Vulnerability",
                condition_description="(male == 1 AND age >= 45) OR (male == 0 AND age >= 55)",
                severity=SeverityLevel.MODERATE,
                medical_source="Framingham Risk Score Age Cutoffs (Grundy et al., Circulation 1999)",
                condition_fn=lambda p: (
                    p.get("age") is not None
                    and p.get("male") is not None
                    and ((p["male"] == 1 and p["age"] >= 45) or (p["male"] == 0 and p["age"] >= 55))
                ),
                finding="Age and biological sex place patient in standard cardiovascular risk screening window (Men >= 45, Women >= 55).",
                recommendation="Adhere to standard periodic preventive cardiovascular screening intervals.",
            ),
            ClinicalRule(
                rule_id="RULE_HISTORY_STROKE",
                category="Vascular History",
                name="Prior Cerebrovascular Event (Stroke)",
                condition_description="prevalentStroke == 1",
                severity=SeverityLevel.CRITICAL,
                medical_source="AHA/ASA Secondary Stroke Prevention Guidelines (Kleindorfer et al., Stroke 2021)",
                condition_fn=lambda p: p.get("prevalentStroke") == 1,
                finding="Prior stroke history indicates established vascular disease.",
                recommendation="Secondary cardiovascular prevention under specialist neurology/cardiology care (antiplatelet therapy, strict BP and lipid control).",
            ),
        ]

    def validate_inputs(self, raw_input: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
        """Validates input types, completeness, and physiological bounds.
        
        Distinguishes missing information (None) from explicit negative/normal findings.
        """
        validated: Dict[str, Any] = {}
        missing_fields: List[str] = []
        invalid_fields: Dict[str, str] = {}

        expected_numeric = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        expected_binary = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]

        # Check numeric features
        for key in expected_numeric:
            val = raw_input.get(key)
            if val is None or val == "":
                missing_fields.append(key)
                validated[key] = None
            else:
                try:
                    num_val = float(val)
                    bounds = PHYSIOLOGICAL_BOUNDS.get(key)
                    if bounds and (num_val < bounds["min"] or num_val > bounds["max"]):
                        invalid_fields[key] = f"Value {num_val} is outside plausible physiological range [{bounds['min']}, {bounds['max']}]."
                        validated[key] = None
                    else:
                        validated[key] = num_val
                except (ValueError, TypeError):
                    invalid_fields[key] = f"Cannot convert '{val}' to numeric float."
                    validated[key] = None

        # Check binary features
        for key in expected_binary:
            val = raw_input.get(key)
            if val is None or val == "":
                missing_fields.append(key)
                validated[key] = None
            else:
                try:
                    int_val = int(val)
                    if int_val not in (0, 1):
                        invalid_fields[key] = f"Binary feature '{key}' must be 0 or 1, found {val}."
                        validated[key] = None
                    else:
                        validated[key] = int_val
                except (ValueError, TypeError):
                    invalid_fields[key] = f"Binary feature '{key}' must be integer 0 or 1."
                    validated[key] = None

        return validated, missing_fields, invalid_fields

    def evaluate(self, patient_data: Dict[str, Any]) -> ExpertEvaluationResult:
        """Evaluates patient data against clinical knowledge base rules."""
        validated_data, missing_fields, invalid_fields = self.validate_inputs(patient_data)

        triggered_rules: List[Dict[str, Any]] = []
        findings: List[str] = []
        recommendations: List[str] = []
        severities: Set[SeverityLevel] = set()

        for rule in self.rules:
            try:
                if rule.condition_fn(validated_data):
                    triggered_rules.append({
                        "rule_id": rule.rule_id,
                        "category": rule.category,
                        "name": rule.name,
                        "severity": rule.severity.value,
                        "condition": rule.condition_description,
                        "medical_source": rule.medical_source,
                        "finding": rule.finding,
                        "recommendation": rule.recommendation,
                    })
                    findings.append(f"[{rule.category}] {rule.finding}")
                    recommendations.append(rule.recommendation)
                    severities.add(rule.severity)
            except Exception as e:
                logger.error("Error evaluating rule %s: %s", rule.rule_id, e)

        # Deduplicate recommendations while preserving order
        unique_recs = []
        seen_recs = set()
        for r in recommendations:
            if r not in seen_recs:
                unique_recs.append(r)
                seen_recs.add(r)

        # Add notes for missing fields to avoid assuming negative findings
        if missing_fields:
            missing_str = ", ".join(missing_fields)
            unique_recs.append(
                f"Information Note: The following clinical data points were not provided: [{missing_str}]. "
                "These risk factors could not be evaluated."
            )

        # Determine qualitative clinical risk profile
        has_critical = (SeverityLevel.CRITICAL in severities)
        if has_critical:
            risk_tier = "CRITICAL_RED_FLAG (Immediate Evaluation Warranted)"
        elif SeverityLevel.HIGH in severities:
            risk_tier = "HIGH_RISK_FACTORS_IDENTIFIED"
        elif SeverityLevel.MODERATE in severities:
            risk_tier = "MODERATE_RISK_FACTORS_IDENTIFIED"
        elif SeverityLevel.LOW in severities:
            risk_tier = "LOW_RISK_FACTORS_IDENTIFIED"
        else:
            risk_tier = "OPTIMAL_OR_UNASSESSED (No Guideline Thresholds Triggered)"

        return ExpertEvaluationResult(
            patient_data=validated_data,
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
            triggered_rules=triggered_rules,
            identified_risk_factors=findings,
            educational_guidance=unique_recs,
            qualitative_risk_tier=risk_tier,
            has_critical_alerts=has_critical,
            clinical_disclaimer=self.DISCLAIMER,
        )


def main():
    """Demonstrates standalone expert system inference."""
    print("=" * 70)
    print("HeartRiskAI: Clinical Expert System Standalone Demonstration")
    print("=" * 70)

    sample_patient = {
        "age": 58,
        "male": 1,
        "sysBP": 162,
        "diaBP": 98,
        "totChol": 248,
        "currentSmoker": 1,
        "cigsPerDay": 25,
        "glucose": 134,
        "diabetes": 0,
        "BPMeds": 1,
        "BMI": 31.4,
        "heartRate": 82,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
    }

    expert = CardiovascularExpertSystem()
    result = expert.evaluate(sample_patient)

    print(f"Qualitative Risk Tier: {result.qualitative_risk_tier}")
    print(f"Triggered Rules Count: {len(result.triggered_rules)}")
    print(f"Critical Alerts:       {result.has_critical_alerts}")
    print("\n--- Triggered Rules ---")
    for r in result.triggered_rules:
        print(f"[{r['rule_id']}] ({r['severity']}) {r['name']}")
        print(f"   Finding: {r['finding']}")
        print(f"   Source:  {r['medical_source']}")

    print("\n--- Educational Recommendations ---")
    for idx, rec in enumerate(result.educational_guidance, 1):
        print(f"{idx}. {rec}")

    print("\n" + "=" * 70)
    print(result.clinical_disclaimer)
    print("=" * 70)


if __name__ == "__main__":
    main()
