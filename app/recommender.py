"""
Smart recommendation engine that combines multiple search strategies
to find the most relevant SHL assessments.
"""
import re
from typing import List, Dict, Optional
from .catalog_loader import get_catalog, Catalog, KEY_TO_TYPE

# Popular/common assessment patterns for direct matching
COMMON_ASSESSMENTS = {
    # Personality
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "personality": "Occupational Personality Questionnaire OPQ32r",
    # Cognitive/Ability
    "verify g+": "SHL Verify Interactive G+",
    "verify g": "SHL Verify Interactive G+",
    "cognitive": "SHL Verify Interactive G+",
    "reasoning": "SHL Verify Interactive G+",
    # Numerical
    "numerical": "SHL Verify Interactive – Numerical Reasoning",
    "numerical reasoning": "SHL Verify Interactive – Numerical Reasoning",
    # Verbal
    "verbal": "SHL Verify Interactive – Verbal Reasoning",
    "verbal reasoning": "SHL Verify Interactive – Verbal Reasoning",
    # Safety
    "safety": "Dependability and Safety Instrument (DSI)",
    "dependability": "Dependability and Safety Instrument (DSI)",
    "dsi": "Dependability and Safety Instrument (DSI)",
    # SJT/Graduate
    "situational judgement": "Graduate Scenarios",
    "situational judgment": "Graduate Scenarios",
    "sjt": "Graduate Scenarios",
    "graduate": "Graduate Scenarios",
    # Skills assessment
    "skills assessment": "Global Skills Assessment",
    "gsa": "Global Skills Assessment",
    # SVAR
    "svar": "SVAR - Spoken English (US) (New)",
    "spoken english": "SVAR - Spoken English (US) (New)",
    "spoken language": "SVAR - Spoken English (US) (New)",
    # Sales
    "sales": "OPQ MQ Sales Report",
    # Live coding
    "live coding": "Smart Interview Live Coding",
    # Contact center
    "contact center": "Contact Center Call Simulation (New)",
    "contact centre": "Contact Center Call Simulation (New)",
    "call center": "Contact Center Call Simulation (New)",
    "call centre": "Contact Center Call Simulation (New)",
    "customer service": "Customer Service Phone Simulation",
}

    # Technology keyword to assessment name mapping (ordered by relevance for senior roles)
TECH_ASSESSMENTS = {
    "java": ["Core Java (Advanced Level) (New)", "Java 8 (New)"],
    "core java": ["Core Java (Advanced Level) (New)"],
    "java advanced": ["Core Java (Advanced Level) (New)"],
    "spring": ["Spring (New)"],
    "sql": ["SQL (New)"],
    "docker": ["Docker (New)"],
    "aws": ["Amazon Web Services (AWS) Development (New)"],
    "amazon web services": ["Amazon Web Services (AWS) Development (New)"],
    ".net": [".NET Framework 4.5", ".NET MVC (New)"],
    "c#": ["C# (New)"],
    "python": ["Python (New)"],
    "angular": ["Angular (New)"],
    "react": ["React (New)"],
    "rest api": ["RESTful Web Services (New)"],
    "restful": ["RESTful Web Services (New)"],
    "networking": ["Networking and Implementation (New)"],
    "linux": ["Linux Programming (General)"],
    "excel": ["MS Excel (New)"],
    "ms excel": ["MS Excel (New)"],
    "word": ["MS Word (New)"],
    "ms word": ["MS Word (New)"],
    "microsoft office": ["MS Excel (New)", "MS Word (New)"],
    "hipaa": ["HIPAA (Security)"],
    "medical terminology": ["Medical Terminology (New)"],
    "financial accounting": ["Financial Accounting (New)"],
    "finance": ["Financial Accounting (New)"],
    "accounting": ["Financial Accounting (New)"],
    "statistics": ["Basic Statistics (New)"],
    "safety knowledge": ["Workplace Health and Safety (New)"],
    "cloud computing": ["Cloud Computing (New)"],
    "devops": ["Docker (New)", "Amazon Web Services (AWS) Development (New)"],
    "rust": ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)"],
    "go": ["Smart Interview Live Coding", "Linux Programming (General)"],
    "kotlin": ["Smart Interview Live Coding"],
    "swift": ["Smart Interview Live Coding"],
}

# Role-based recommendation templates
ROLE_TEMPLATES = {
    "senior_leader": {
        "search_terms": "senior leadership executive personality OPQ",
        "always_include": ["Occupational Personality Questionnaire OPQ32r"],
        "optional": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
        "keywords": ["leadership", "executive", "cxo", "director", "c-suite", "senior leader", "vp", "vice president", "president", "ceo", "cto", "cfo", "coo"],
    },
    "developer": {
        "search_terms": "programming developer software knowledge skills",
        "always_include": ["SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
        "optional": [],
        "keywords": ["developer", "engineer", "programmer", "software", "backend", "frontend", "full-stack", "fullstack", "devops"],
    },
    "safety": {
        "search_terms": "safety dependability compliance manufacturing personality",
        "always_include": ["Dependability and Safety Instrument (DSI)", "Manufac. & Indust. - Safety & Dependability 8.0"],
        "optional": ["Workplace Health and Safety (New)"],
        "keywords": ["safety", "plant operator", "chemical", "manufacturing", "industrial", "reliability", "compliance", "procedure"],
    },
    "contact_center": {
        "search_terms": "contact center call centre customer service phone simulation spoken language SVAR",
        "always_include": ["Contact Center Call Simulation (New)", "Entry Level Customer Serv-Retail & Contact Center"],
        "optional": ["Customer Service Phone Simulation"],
        "keywords": ["contact center", "contact centre", "call center", "call centre", "customer service", "phone", "inbound", "outbound"],
    },
    "graduate": {
        "search_terms": "graduate entry level university cognitive personality situational judgement",
        "always_include": ["SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r", "Graduate Scenarios"],
        "optional": [],
        "keywords": ["graduate", "entry-level", "entry level", "trainee", "intern", "campus", "university", "student", "fresh"],
    },
    "finance": {
        "search_terms": "financial accounting numerical reasoning finance",
        "always_include": ["SHL Verify Interactive – Numerical Reasoning", "Financial Accounting (New)", "Occupational Personality Questionnaire OPQ32r"],
        "optional": ["Basic Statistics (New)", "Graduate Scenarios"],
        "keywords": ["finance", "financial", "accounting", "analyst", "banking", "investment", "bookkeeping"],
    },
    "admin": {
        "search_terms": "administrative assistant office excel word knowledge skills personality",
        "always_include": ["MS Excel (New)", "MS Word (New)", "Occupational Personality Questionnaire OPQ32r"],
        "optional": [],
        "keywords": ["admin", "assistant", "office", "clerical", "secretary", "receptionist"],
    },
    "sales": {
        "search_terms": "sales selling personality OPQ motivation skills assessment",
        "always_include": ["Global Skills Assessment", "Occupational Personality Questionnaire OPQ32r", "OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor"],
        "optional": ["Global Skills Development Report"],
        "keywords": ["sales", "selling", "account executive", "business development", "revenue", "quota", "sdr", "bdr", "re-skill"],
    },
    "healthcare": {
        "search_terms": "healthcare medical HIPAA safety personality",
        "always_include": ["HIPAA (Security)", "Dependability and Safety Instrument (DSI)", "Occupational Personality Questionnaire OPQ32r"],
        "optional": ["Medical Terminology (New)", "Microsoft Word 365 - Essentials (New)"],
        "keywords": ["healthcare", "medical", "hospital", "clinical", "nurse", "doctor", "patient", "hipaa"],
    },
}

# Clarification questions based on missing information
CLARIFICATION_QUESTIONS = {
    "role": "What role or position are you hiring for?",
    "seniority": "What is the seniority level — entry-level, mid-level, senior, or executive?",
    "measurement": "What do you want to measure — knowledge, personality, cognitive ability, or something else?",
    "language": "What language do the assessments need to be in?",
    "volume": "How many candidates are you screening?",
    "context": "Is this for selection (choosing between candidates) or development (growing existing employees)?",
}


def identify_role(query: str) -> Optional[str]:
    """Identify the role type from a user query.
    
    When multiple roles match, we score each role by:
    1. Number of keyword matches (more = more specific to this role)
    2. Total length of matching keywords (longer = more specific terms)
    3. Specificity bonus for domain-specific roles (finance, safety, healthcare)
       over generic ones (graduate, admin)
    """
    query_lower = query.lower()
    
    # Score each role
    role_scores = {}
    for role_type, template in ROLE_TEMPLATES.items():
        matched_kws = []
        for kw in template["keywords"]:
            if kw in query_lower:
                matched_kws.append(kw)
        if matched_kws:
            # Score = number of matching keywords * 10 + total keyword length
            role_scores[role_type] = len(matched_kws) * 10 + sum(len(kw) for kw in matched_kws)
    
    if not role_scores:
        return None
    
    # Specificity bonus: domain-specific roles beat generic ones
    specificity_bonus = {
        "finance": 15,      # very specific domain
        "safety": 15,       # very specific domain
        "healthcare": 15,   # very specific domain
        "contact_center": 12,  # specific domain
        "sales": 10,        # moderately specific
        "senior_leader": 10,  # moderately specific
        "developer": 8,     # moderately specific
        "admin": 0,         # generic
        "graduate": 0,     # generic/catch-all
    }
    
    for role in role_scores:
        role_scores[role] += specificity_bonus.get(role, 0)
    
    # Return the highest-scoring role
    best_role = max(role_scores, key=role_scores.get)
    return best_role


def extract_tech_keywords(query: str) -> List[str]:
    """Extract technology keywords from a query."""
    query_lower = query.lower()
    found = []
    for tech in TECH_ASSESSMENTS:
        if tech in query_lower:
            found.append(tech)
    # Sort by length (longer matches first to avoid partial matches)
    found.sort(key=len, reverse=True)
    # Remove substrings
    filtered = []
    for t in found:
        if not any(t != f and t in f for f in filtered):
            filtered.append(t)
    return filtered


def extract_job_description(text: str) -> dict:
    """Extract structured info from a job description text."""
    info = {
        "technologies": extract_tech_keywords(text),
        "role_type": identify_role(text),
        "seniority": None,
        "languages": [],
        "measurement_types": [],
    }

    text_lower = text.lower()

    # Detect seniority
    if any(w in text_lower for w in ["senior", "sr.", "lead", "principal", "staff", "5+ years", "10+ years", "15+ years"]):
        info["seniority"] = "senior"
    elif any(w in text_lower for w in ["mid-level", "mid level", "intermediate", "3+ years", "4+ years"]):
        info["seniority"] = "mid"
    elif any(w in text_lower for w in ["entry-level", "entry level", "junior", "jr.", "graduate", "intern", "trainee", "fresh"]):
        info["seniority"] = "entry"
    elif any(w in text_lower for w in ["executive", "cxo", "director", "vp", "c-suite", "ceo", "cto", "cfo"]):
        info["seniority"] = "executive"

    # Detect measurement types
    if any(w in text_lower for w in ["cognitive", "reasoning", "ability", "aptitude", "intelligence"]):
        info["measurement_types"].append("A")
    if any(w in text_lower for w in ["personality", "behaviour", "behavior", "behavioural", "behavioral", "fit"]):
        info["measurement_types"].append("P")
    if any(w in text_lower for w in ["knowledge", "skills", "technical", "proficiency"]):
        info["measurement_types"].append("K")
    if any(w in text_lower for w in ["situational", "judgement", "judgment", "scenarios", "sjt"]):
        info["measurement_types"].append("B")
    if any(w in text_lower for w in ["simulation", "practical", "hands-on"]):
        info["measurement_types"].append("S")

    # Detect language requirements
    lang_map = {
        "spanish": "Spanish",
        "french": "French",
        "german": "German",
        "portuguese": "Portuguese",
        "chinese": "Chinese",
        "japanese": "Japanese",
        "dutch": "Dutch",
        "italian": "Italian",
        "korean": "Korean",
        "hindi": "Hindi",
        "arabic": "Arabic",
    }
    for lang_kw, lang_name in lang_map.items():
        if lang_kw in text_lower:
            info["languages"].append(lang_name)

    return info


def get_recommendations(
    query: str,
    conversation_context: str = "",
    existing_shortlist: list = None,
    add_types: list = None,
    remove_names: list = None,
) -> List[Dict]:
    """
    Generate assessment recommendations based on query and context.
    
    Uses multiple strategies:
    1. Direct tech keyword matching
    2. Role-based templates
    3. TF-IDF semantic search
    4. Filtering and ranking
    """
    catalog = get_catalog()
    full_query = f"{query} {conversation_context}".strip()

    # Extract structured info
    info = extract_job_description(full_query)

    candidates = {}  # name -> item dict (dedup)

    # Strategy 1: Direct tech keyword matching (highest priority)
    tech_matched = {}  # url -> item (for tracking tech matches)
    for tech in info.get("technologies", []):
        if tech in TECH_ASSESSMENTS:
            for name in TECH_ASSESSMENTS[tech]:
                item = catalog.get_by_name(name)
                if item and item["url"] not in tech_matched:
                    tech_matched[item["url"]] = item

    # Add tech matches to candidates
    for url, item in tech_matched.items():
        candidates[url] = item

    # Strategy 2: Role-based templates
    role_type = info.get("role_type")
    if role_type and role_type in ROLE_TEMPLATES:
        template = ROLE_TEMPLATES[role_type]
        for name in template.get("always_include", []):
            item = catalog.get_by_name(name)
            if item:
                candidates[item["url"]] = item
        for name in template.get("optional", []):
            item = catalog.get_by_name(name)
            if item:
                candidates[item["url"]] = item

    # Strategy 3: TF-IDF search
    search_query_parts = [full_query]
    if role_type and role_type in ROLE_TEMPLATES:
        search_query_parts.append(ROLE_TEMPLATES[role_type]["search_terms"])
    search_query = " ".join(search_query_parts)

    search_results = catalog.search(search_query, top_k=20)
    for item in search_results:
        if item["url"] not in candidates:
            candidates[item["url"]] = item

    # Strategy 4: Add based on measurement types
    for mtype in info.get("measurement_types", []):
        type_results = catalog.search_by_filters(test_types=[mtype], top_k=10)
        for item in type_results:
            if item["url"] not in candidates:
                candidates[item["url"]] = item

    # Now rank and filter
    all_items = list(candidates.values())

    # Score each item based on relevance
    scored = []
    for item in all_items:
        score = 0.0
        item_name_lower = item["name"].lower()
        query_lower = full_query.lower()

        # Tech keyword match bonus (highest priority)
        is_tech_match = item["url"] in tech_matched
        if is_tech_match:
            score += 25.0  # Very strong boost for direct tech keyword match
        else:
            for tech in info.get("technologies", []):
                if tech in item_name_lower:
                    score += 3.0  # Lower bonus for incidental matches

        # Role template match bonus
        if role_type and role_type in ROLE_TEMPLATES:
            template = ROLE_TEMPLATES[role_type]
            if item["name"] in template.get("always_include", []):
                score += 8.0
            if item["name"] in template.get("optional", []):
                score += 5.0

        # Seniority match
        if info.get("seniority"):
            job_levels = item.get("job_levels", [])
            seniority_map = {
                "entry": ["Entry-Level", "Graduate"],
                "mid": ["Mid-Professional", "Front Line Manager"],
                "senior": ["Manager", "Mid-Professional", "Director"],
                "executive": ["Executive", "Director"],
            }
            target_levels = seniority_map.get(info["seniority"], [])
            for level in target_levels:
                if level in job_levels:
                    score += 2.0

        # Measurement type match
        if info.get("measurement_types"):
            item_types = set(item["test_type"].split(",")) if item["test_type"] else set()
            for mtype in info["measurement_types"]:
                if mtype in item_types:
                    score += 3.0

        # TF-IDF search score bonus
        if "score" in item:
            score += item["score"] * 5.0

        # Text overlap bonus
        query_words = set(query_lower.split())
        name_words = set(item_name_lower.split())
        desc_words = set(item.get("description", "").lower().split())
        overlap = len(query_words & (name_words | desc_words))
        score += overlap * 0.5

        # Penalty for items that don't match the query well
        # If item is "Entry Level" but query mentions "senior", penalize heavily
        if info.get("seniority") == "senior" or info.get("seniority") == "executive":
            if "entry level" in item_name_lower or "entry-level" in item_name_lower:
                score -= 30.0
        if info.get("seniority") == "entry":
            if "advanced" in item_name_lower or "senior" in item_name_lower:
                score -= 10.0

        # Penalty for solution/package items when we're looking for individual tests
        if "solution" in item_name_lower and not any(w in query_lower for w in ["solution", "package", "bundle"]):
            score -= 3.0

        # Bonus for popular/core assessments
        core_assessments = [
            "OPQ32r", "Verify Interactive G+", "DSI", "Graduate Scenarios",
            "Numerical Reasoning", "Verbal Reasoning",
        ]
        for core in core_assessments:
            if core.lower() in item_name_lower:
                score += 2.0

        scored.append((score, item))

    # Sort by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Handle existing shortlist modifications
    if existing_shortlist is not None:
        # Start with existing shortlist
        result_urls = {r["url"] for r in existing_shortlist}
        result_items = {r["url"]: catalog.get_by_url(r["url"]) or r for r in existing_shortlist}

        # Add new items
        if add_types:
            for mtype in add_types:
                for score, item in scored:
                    item_types = set(item["test_type"].split(",")) if item["test_type"] else set()
                    if mtype in item_types and item["url"] not in result_urls:
                        result_urls.add(item["url"])
                        result_items[item["url"]] = item

        # Remove items
        if remove_names:
            for name in remove_names:
                name_lower = name.lower()
                to_remove = []
                for url, item in result_items.items():
                    if name_lower in item.get("name", "").lower():
                        to_remove.append(url)
                for url in to_remove:
                    del result_items[url]
                    result_urls.discard(url)

        # Return modified shortlist
        return [item for item in result_items.values() if item]

    # Determine template items that should always be included (even if "report")
    template_always_names = set()
    template_optional_names = set()
    if role_type and role_type in ROLE_TEMPLATES:
        template_always_names = set(ROLE_TEMPLATES[role_type].get("always_include", []))
        template_optional_names = set(ROLE_TEMPLATES[role_type].get("optional", []))

    # Separate report vs main items, but keep template always_include items
    main_items = []
    report_items = []
    for score, item in scored:
        name_lower = item["name"].lower()
        is_template_item = item["name"] in template_always_names or item["name"] in template_optional_names
        is_report = any(w in name_lower for w in ["report", "profile report", "narrative report", "candidate report", "candidate plus"])
        is_leadership_report = "leadership report" in name_lower or "competency report" in name_lower
        # Keep template items and leadership reports; filter other reports
        if is_report and not is_leadership_report and not is_template_item:
            report_items.append(item)
        else:
            main_items.append(item)

    # If we're looking at leadership roles, also include leadership reports
    if role_type == "senior_leader":
        for score, item in scored:
            name_lower = item["name"].lower()
            if "leadership report" in name_lower or "competency report" in name_lower:
                if item not in main_items:
                    main_items.append(item)

    # Prefer main assessments, add reports only if specifically relevant
    result = main_items[:10]

    # If we have very few main items, supplement with reports
    if len(result) < 3:
        for item in report_items:
            if item not in result:
                result.append(item)
            if len(result) >= 10:
                break

    # Filter out clearly irrelevant items: if the item has no meaningful overlap
    # with the query and wasn't matched by tech/role/template, remove it
    query_lower = full_query.lower()
    # Build meaningful query keywords (skip stop words)
    stop_words = {"i", "a", "an", "the", "is", "are", "was", "were", "be", "been",
                  "do", "does", "did", "will", "would", "could", "should", "may",
                  "might", "shall", "can", "need", "we", "you", "they", "he", "she",
                  "it", "my", "your", "their", "our", "me", "him", "her", "us",
                  "this", "that", "these", "those", "for", "of", "in", "on", "at",
                  "to", "with", "from", "by", "as", "or", "and", "not", "no", "but",
                  "if", "so", "than", "too", "very", "also", "just", "about", "up",
                  "out", "how", "what", "which", "who", "when", "where", "why",
                  "looking", "assess", "assessment", "test", "recommend", "use",
                  "like", "want", "help", "find", "get", "add", "need", "solution",
                  "position", "role", "hire", "hiring", "job", "work", "people"}
    query_words = set(query_lower.split()) - stop_words
    filtered_result = []
    for item in result:
        name_lower = item["name"].lower()
        desc_words = set(item.get("description", "").lower().split())
        # Check if there's meaningful overlap with name or description
        name_overlap = bool(query_words & set(name_lower.split()))
        desc_overlap = len(query_words & desc_words) >= 2  # At least 2 query words in description
        # Check if it's a tech/template/core match
        is_tech = item["url"] in tech_matched
        is_template = (role_type and role_type in ROLE_TEMPLATES and
                       (item["name"] in ROLE_TEMPLATES[role_type].get("always_include", []) or
                        item["name"] in ROLE_TEMPLATES[role_type].get("optional", [])))
        is_core = any(core.lower() in name_lower for core in
                      ["OPQ32r", "Verify Interactive G+", "DSI", "Graduate Scenarios",
                       "Numerical Reasoning", "Verbal Reasoning", "Verify Interactive",
                       "Smart Interview", "SVAR"])
        # Seniority mismatch: if the item is entry-level but query is for senior/executive, skip
        seniority_mismatch = False
        if info.get("seniority") in ("senior", "executive"):
            if "entry level" in name_lower or "entry-level" in name_lower:
                seniority_mismatch = True
        # Allow through if it's a strong match, but not if there's a seniority mismatch
        if seniority_mismatch and not is_template:
            continue
        if is_tech or is_template or is_core or name_overlap or desc_overlap:
            filtered_result.append(item)

    # If filtering removed too many, fall back to scored list
    if len(filtered_result) < 3:
        filtered_result = result[:10]

    return filtered_result[:10]


def determine_missing_info(query: str) -> List[str]:
    """Determine what information is missing from the query for good recommendations."""
    info = extract_job_description(query)
    missing = []

    # Only ask for clarification if we have VERY little to go on
    # If the user has mentioned any technologies, role type, or specific assessments, 
    # we should be able to recommend something
    has_specifics = bool(info["technologies"] or info["role_type"] or info["measurement_types"])
    
    # Check if the query mentions specific assessment-related terms
    query_lower = query.lower()
    assessment_terms = [
        "test", "assessment", "battery", "screen", "measure", "evaluate",
        "java", "python", "rust", "sql", "docker", "aws", "spring", "excel",
        "opq", "verify", "svar", "dsi", "numerical", "verbal", "cognitive",
        "personality", "safety", "graduate", "simulation",
    ]
    has_assessment_context = any(t in query_lower for t in assessment_terms)
    
    if not has_specifics and not has_assessment_context:
        missing.append("role")
        # Don't ask for seniority on first vague query - too many questions
    elif not info["seniority"] and not info["technologies"] and info["role_type"] not in ["safety", "contact_center"]:
        # Only ask for seniority if we have a role type but no tech specifics
        missing.append("seniority")

    return missing


def generate_clarification_reply(query: str, missing_info: List[str]) -> str:
    """Generate a clarification response for missing information."""
    questions = []
    for info_type in missing_info[:2]:  # Ask max 2 questions at a time
        if info_type in CLARIFICATION_QUESTIONS:
            questions.append(CLARIFICATION_QUESTIONS[info_type])

    if not questions:
        return "Could you tell me more about what you're looking for?"

    if len(questions) == 1:
        return questions[0]
    else:
        return " ".join(questions)


def detect_comparison(query: str) -> Optional[tuple]:
    """Detect if the user wants to compare two assessments."""
    patterns = [
        r"difference\s+between\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+?)[\?\.,]",
        r"how\s+does\s+(.+?)\s+differ\s+from\s+(.+)",
        r"how\s+(.+?)\s+differs\s+from\s+(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            name1 = m.group(1).strip().rstrip("?.!,;")
            name2 = m.group(2).strip().rstrip("?.!,;")
            return (name1, name2)
    return None


def detect_modification(query: str) -> dict:
    """Detect if the user wants to modify the shortlist."""
    q_lower = query.lower()
    mods = {"add": [], "remove": [], "replace": []}

    # Detect add
    add_patterns = [
        r"add\s+(?:a\s+)?(.+?)(?:\s+test)?(?:\s+too)?(?:\s+as\s+well)?(?:\s*$|\.)",
        r"also\s+(?:add|include)\s+(.+?)(?:\s*$|\.)",
        r"include\s+(.+?)(?:\s*$|\.)",
        r"can\s+you\s+add\s+(.+?)(?:\s*$|\.)",
    ]
    for pattern in add_patterns:
        m = re.search(pattern, q_lower)
        if m:
            mods["add"].append(m.group(1).strip())

    # Detect remove
    remove_patterns = [
        r"remove\s+(.+?)(?:\s*$|\.)",
        r"drop\s+(.+?)(?:\s*$|\.)",
        r"take\s+out\s+(.+?)(?:\s*$|\.)",
        r"without\s+(.+?)(?:\s*$|\.)",
        r"exclude\s+(.+?)(?:\s*$|\.)",
        r"don'?t\s+(?:need|want|include)\s+(.+?)(?:\s*$|\.)",
    ]
    for pattern in remove_patterns:
        m = re.search(pattern, q_lower)
        if m:
            mods["remove"].append(m.group(1).strip())

    # Detect replace
    replace_patterns = [
        r"replace\s+(.+?)\s+with\s+(.+?)(?:\s*$|\.)",
        r"swap\s+(.+?)\s+for\s+(.+?)(?:\s*$|\.)",
        r"substitute\s+(.+?)\s+with\s+(.+?)(?:\s*$|\.)",
    ]
    for pattern in replace_patterns:
        m = re.search(pattern, q_lower)
        if m:
            mods["remove"].append(m.group(1).strip())
            mods["add"].append(m.group(2).strip())

    return mods


def is_off_topic(query: str) -> bool:
    """Check if the query is off-topic."""
    q_lower = query.lower()

    # Prompt injection patterns
    injection_patterns = [
        "ignore previous", "ignore instructions", "system prompt",
        "you are now", "pretend", "act as", "jailbreak",
        "hack", "exploit", "override", "disregard",
    ]
    for pattern in injection_patterns:
        if pattern in q_lower:
            return True

    # General off-topic (but be careful - hiring/assessment context is ON topic)
    off_topic_strict = [
        "recipe", "cook", "weather", "sports score", "movie recommendation",
        "write a poem", "write a story", "tell me a joke",
        "write me a poem", "write me a story", "write poem", "write story",
    ]
    for pattern in off_topic_strict:
        if pattern in q_lower:
            return True

    return False


def is_legal_question(query: str) -> bool:
    """Check if the user is asking a legal/compliance question."""
    q_lower = query.lower()
    legal_patterns = [
        "legally required", "legal requirement", "legally obligated",
        "compliance requirement", "does this satisfy", "regulatory obligation",
        "are we required", "are we legally", "law requires",
        "is this compliant", "regulation says", "legal advice",
    ]
    for pattern in legal_patterns:
        if pattern in q_lower:
            return True
    return False


def is_conversation_ending(query: str) -> bool:
    """Detect if the user is confirming satisfaction / ending conversation."""
    q_lower = query.lower()
    end_patterns = [
        "that works", "that's good", "looks good", "perfect",
        "confirmed", "lock it in", "that's what we need",
        "that covers it", "that's everything", "we're set",
        "great thanks", "thanks that's", "final list",
        "that's all", "looks great", "sounds good",
    ]
    for pattern in end_patterns:
        if pattern in q_lower:
            return True
    return False
