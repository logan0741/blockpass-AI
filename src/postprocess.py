import re
from statistics import median
from typing import Dict, Any, List
import json


def _normalize_text(text: str) -> Dict[str, Any]:
    raw_lines = [line.rstrip() for line in text.splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]
    joined = ", ".join(lines)
    cleaned = re.sub(r"\s+", " ", joined).strip()
    full_text = "\n".join(lines).strip()
    return {"cleaned": cleaned, "lines": lines, "full_text": full_text}


def _coerce_raw_text(raw_text: str) -> str:
    """Extract raw_text if the model returned a JSON string."""
    text = raw_text.strip()
    if not text.startswith("{"):
        return raw_text
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r'"raw_text"\s*:\s*"((?:\\\\.|[^"\\\\])*)"', text, re.S)
        if not match:
            marker = '"raw_text": "'
            idx = text.find(marker)
            if idx == -1:
                return raw_text
            extracted = text[idx + len(marker):]
            extracted = re.sub(r'"\s*}\s*$', "", extracted, flags=re.S)
            return extracted
        try:
            return json.loads(f"\"{match.group(1)}\"")
        except Exception:
            return match.group(1)
    if isinstance(parsed, dict) and isinstance(parsed.get("raw_text"), str):
        return parsed["raw_text"]
    return raw_text


def _detect_sections(lines: List[str]) -> Dict[str, str]:
    sections = {k: [] for k in ["header", "business", "period", "payment", "refund", "terms"]}
    keywords = {
        "header": ["계약서", "이용 계약", "수강 등록", "계약 내용", "이용약관"],
        "business": ["상호", "사업장", "시설명", "학원명", "주소", "대표", "연락처", "사업자등록"],
        "period": ["기간", "이용기간", "수강기간", "계약기간", "유효기간", "개월", "년", "일"],
        "payment": ["금액", "결제", "이용료", "수강료", "회비", "납부", "입금", "할부"],
        "refund": ["환불", "해지", "취소", "환급"],
        "terms": ["특약", "유의사항", "기타", "약관", "주의", "비고"],
    }
    base_weight = {k: 1 for k in sections}

    for line in lines:
        scores = {k: base_weight[k] for k in sections}
        for sec, keys in keywords.items():
            for kw in keys:
                if kw in line:
                    scores[sec] += 2
        chosen = max(scores.items(), key=lambda x: x[1])[0]
        sections[chosen].append(line)

    return {k: "\n".join(v).strip() if v else "" for k, v in sections.items()}


def _extract_business_name(lines: List[str]) -> str | None:
    for line in lines:
        m = re.search(r"기본 정보[:\s]*([^/]+)", line)
        if m:
            return m.group(1).strip()
        m = re.search(r"^(.+?)\\s*회원가입\\s*계약서", line)
        if m:
            return m.group(1).strip()
        m = re.search(r"(상호|사업장|시설명|학원명|업체명)[:\s]+(.+)", line)
        if m:
            return m.group(2).strip()
    return None


def _infer_service_type(text: str) -> str:
    if any(k in text for k in ["헬스", "휘트니스", "피트니스", "gym"]):
        return "gym"
    if any(k in text for k in ["독서실", "스터디", "study"]):
        return "study_room"
    if any(k in text for k in ["학원", "수강", "academy"]):
        return "academy"
    return "other"


def _parse_duration_days(text: str) -> int | None:
    m = re.search(r"(\d+)\s*일\s*이용", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*[-~]\s*(\d{4}[./-]\d{1,2}[./-]\d{1,2})", text)
    if m:
        try:
            start = [int(x) for x in re.split(r"[./-]", m.group(1))]
            end = [int(x) for x in re.split(r"[./-]", m.group(2))]
            from datetime import date
            return (date(end[0], end[1], end[2]) - date(start[0], start[1], start[2])).days + 1
        except Exception:
            pass
    m = re.search(r"(\d+)\s*년", text)
    if m:
        return int(m.group(1)) * 365
    m = re.search(r"(\d+)\s*개월", text)
    if m:
        return int(m.group(1)) * 30
    m = re.search(r"(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    return None


def _parse_amount_krw(text: str) -> int | None:
    amounts = [int(a.replace(",", "")) for a in re.findall(r"(\d[\d,]{2,})\s*원", text)]
    return max(amounts) if amounts else None


def _parse_eth_ratio(text: str) -> Dict[str, int | None]:
    m = re.search(r"비즈니스\s*(\d{1,3})\s*%", text)
    n = re.search(r"에스크로\s*(\d{1,3})\s*%", text)
    if m and n:
        return {"business": int(m.group(1)), "escrow": int(n.group(1))}
    m = re.search(r"(비율|ratio|ETH)\s*[:\s]*(\d{1,2})\s*[:/]\s*(\d{1,2})", text, re.IGNORECASE)
    if m:
        return {"business": int(m.group(2)), "escrow": int(m.group(3))}
    return {"business": None, "escrow": None}


def _parse_refund_rules(lines: List[str]) -> List[Dict[str, int]]:
    rules: List[Dict[str, int]] = []
    for line in lines:
        if not any(k in line for k in ["환불", "해지", "취소"]):
            continue
        days = None
        percent = None
        m = re.search(r"(\d+)\s*일", line)
        if m:
            days = int(m.group(1))
        m = re.search(r"(\d+)\s*개월", line)
        if m and days is None:
            days = int(m.group(1)) * 30
        m = re.search(r"(\d+)\s*%", line)
        if m:
            percent = int(m.group(1))
        if percent is None:
            if "전액" in line:
                percent = 100
            elif "불가" in line:
                percent = 0
        if days is not None and percent is not None:
            rules.append({"days": days, "percent": percent})
    return rules


def _parse_protection_days(text: str, refund_rules: List[Dict[str, int]]) -> int | None:
    m = re.search(r"보호\s*기간[:\s]*(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    if refund_rules:
        return refund_rules[0].get("days")
    return None


def build_output_schema(raw_text: str) -> Dict[str, Any]:
    normalized = _normalize_text(_coerce_raw_text(raw_text))
    cleaned = normalized["cleaned"]
    lines = normalized["lines"]
    full_text = normalized["full_text"]
    sections = _detect_sections(lines)
    business_name = _extract_business_name(lines)
    service_type = _infer_service_type(cleaned)
    refund_rules = _parse_refund_rules(lines)
    eth_ratio = _parse_eth_ratio(cleaned)

    return {
        "success": True if cleaned else False,
        "confidence": 0.2 if cleaned else 0.0,
        "business_name": business_name,
        "service_type": service_type,
        "duration_days": _parse_duration_days(cleaned),
        "amount_krw": _parse_amount_krw(cleaned),
        "eth_ratio_business": eth_ratio["business"],
        "eth_ratio_escrow": eth_ratio["escrow"],
        "protection_days": _parse_protection_days(cleaned, refund_rules),
        "refund_rules": refund_rules,
        "full_text": full_text,
        "sections": sections,
    }
