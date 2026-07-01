"""
SHL Assessment Recommender Agent.

Conversational agent that combines rule-based logic with LLM for natural language.
Core logic: recommendation engine + catalog search
LLM: used only for generating natural language replies
"""
import os
import re
import json
from typing import Optional

from .catalog_loader import get_catalog, Catalog
from .recommender import (
    get_recommendations,
    determine_missing_info,
    generate_clarification_reply,
    detect_comparison,
    detect_modification,
    is_off_topic,
    is_legal_question,
    is_conversation_ending,
    extract_job_description,
    extract_tech_keywords,
    identify_role,
    CLARIFICATION_QUESTIONS,
)

# LLM Configuration
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

MAX_TURNS = 8


def _get_llm_client():
    """Create an LLM client."""
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _has_llm() -> bool:
    """Check if LLM is configured."""
    return bool(LLM_API_KEY)


def process_chat(messages: list) -> dict:
    """
    Process a chat request and return the agent response.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}

    Returns:
        {
            "reply": str,
            "recommendations": list[dict],
            "end_of_conversation": bool
        }
    """
    catalog = get_catalog()

    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return {
            "reply": "How can I help you find the right SHL assessments today?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Turn tracking
    turn_count = sum(1 for m in messages if m["role"] in ("user", "assistant"))

    # Check for off-topic
    if is_off_topic(last_user_msg):
        return {
            "reply": "I'm here to help you select SHL assessments. I can't help with that topic. Could you tell me about the role you're hiring for, and I'll recommend suitable assessments?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Check for legal questions
    if is_legal_question(last_user_msg):
        reply = (
            "Those are legal compliance questions outside what I can advise on. "
            "I can help you select assessments, but not interpret regulatory obligations "
            "or whether a specific test satisfies a legal requirement. "
            "Your legal or compliance team is the right resource for that."
        )
        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Check for comparison request
    comparison = detect_comparison(last_user_msg)
    if comparison:
        name1, name2 = comparison
        comparison_text = catalog.compare_items(name1, name2)
        # Include previous shortlist in recommendations if any
        prev_recs = _extract_previous_recommendations(messages)
        return {
            "reply": comparison_text,
            "recommendations": prev_recs,
            "end_of_conversation": False,
        }

    # Build full conversation context
    conversation_context = " ".join(m["content"] for m in messages if m["role"] == "user")

    # Check if this is a modification to existing shortlist
    mods = detect_modification(last_user_msg)
    prev_recs = _extract_previous_recommendations(messages)

    if prev_recs and (mods["add"] or mods["remove"]):
        return _handle_modification(messages, mods, prev_recs, catalog, conversation_context)

    # Check if user is confirming/end of conversation
    if prev_recs and is_conversation_ending(last_user_msg):
        return {
            "reply": _generate_confirmation_reply(prev_recs, catalog),
            "recommendations": [catalog.format_recommendation(catalog.get_by_url(r["url"]) or r) for r in prev_recs if catalog.is_valid_url(r.get("url", ""))],
            "end_of_conversation": True,
        }

    # Check if query is too vague for recommendations
    missing = determine_missing_info(conversation_context)

    # On first turn, also check if the query is too vague even if a role was identified.
    # Key heuristic: if the user didn't specify WHAT to measure (measurement types, technologies,
    # or specific assessment names), ask clarification before recommending.
    is_vague_first_turn = False
    if turn_count <= 1 and not prev_recs:
        info = extract_job_description(conversation_context)
        has_measurement = bool(info.get("measurement_types"))
        has_tech = bool(info.get("technologies"))
        has_role = bool(info.get("role_type"))
        has_seniority = bool(info.get("seniority"))
        # The user must have SOME specificity beyond just naming a role
        has_specifics = has_measurement or has_tech
        # If they have both role AND seniority, that's some context
        has_role_context = has_role and has_seniority
        # Very vague patterns: "need a solution", "looking for assessments" without details
        q_lower = last_user_msg.lower()
        uses_vague_terms = any(p in q_lower for p in [
            "need a solution", "looking for a solution", "what should we use",
            "what assessments", "recommend assessments", "what do you recommend",
        ])
        # Clarify if: missing info, or very vague query without specifics
        if missing:
            is_vague_first_turn = True
        elif uses_vague_terms and not has_specifics:
            is_vague_first_turn = True
        elif not has_specifics and not has_role_context:
            is_vague_first_turn = True

    if is_vague_first_turn:
        # Generate contextual clarification based on what we know
        info = extract_job_description(conversation_context)
        role = info.get("role_type")
        if missing:
            clarification = _generate_smart_clarification(last_user_msg, missing, conversation_context, catalog)
        elif role == "senior_leader":
            clarification = "Happy to help narrow that down. Who is this meant for — selection of new candidates, or development feedback for existing executives?"
        elif role == "contact_center":
            clarification = "Before I shape the stack — what language are the calls in? That drives which spoken-language screen we use."
        else:
            clarification = _generate_smart_clarification(last_user_msg, ["context"], conversation_context, catalog)
        return {
            "reply": clarification,
            "recommendations": [],
            "end_of_conversation": False,
        }
    

    # Continue clarification in later turns if essential information is still missing
    info = extract_job_description(conversation_context)

    if not info.get("role_type"):
       return {
        "reply": "What role or position are you hiring for?",
        "recommendations": [],
        "end_of_conversation": False,
       }

    if not info.get("seniority"):
       return {
        "reply": "What is the seniority level — entry-level, mid-level, senior, or executive?",
        "recommendations": [],
        "end_of_conversation": False,
       }

    # If we have enough info, generate recommendations
    recommended_items = get_recommendations(
        query=last_user_msg,
        conversation_context=conversation_context,
    )

    if not recommended_items:
        # Fall back to search
        results = catalog.search(conversation_context, top_k=5)
        recommended_items = results[:5]

    if not recommended_items:
        return {
            "reply": "I couldn't find assessments matching your requirements. Could you provide more details about the role or skills you're looking to assess?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Format recommendations
    recommendations = []
    for item in recommended_items[:10]:
        if isinstance(item, dict) and "url" in item:
            rec = catalog.format_recommendation(item) if catalog.is_valid_url(item.get("url", "")) else None
            if rec:
                recommendations.append(rec)

    if not recommendations:
        return {
            "reply": "Let me search more broadly. Could you tell me more specifically about the role or skills you need to assess?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Generate reply
    reply = _generate_recommendation_reply(last_user_msg, recommended_items, catalog, conversation_context)

    # Check if conversation should end (user confirmed)
    end = False
    if is_conversation_ending(last_user_msg):
        end = True

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end,
    }


def _extract_previous_recommendations(messages: list) -> list:
    """Extract previously recommended assessments from conversation history."""
    # Look for URLs in assistant messages
    catalog = get_catalog()
    prev_recs = []
    url_pattern = re.compile(r'https://www\.shl\.com/products/product-catalog/view/[^)\s|<"\' ]+')

    for msg in messages:
        if msg["role"] == "assistant":
            urls = url_pattern.findall(msg["content"])
            for url in urls:
                item = catalog.get_by_url(url)
                if item:
                    rec = catalog.format_recommendation(item)
                    # Avoid duplicates
                    if not any(r["url"] == rec["url"] for r in prev_recs):
                        prev_recs.append(rec)

    return prev_recs


def _handle_modification(messages: list, mods: dict, prev_recs: list, catalog, conversation_context: str) -> dict:
    """Handle add/remove modifications to existing shortlist."""
    current_items = []
    for rec in prev_recs:
        item = catalog.get_by_url(rec.get("url", ""))
        if item:
            current_items.append(item)

    # Handle removals
    removed_names = []
    for name_to_remove in mods.get("remove", []):
        # Strip leading articles and whitespace
        name_clean = name_to_remove.lower().strip()
        for article in ["the ", "a ", "an "]:
            if name_clean.startswith(article):
                name_clean = name_clean[len(article):].strip()
        to_remove = []
        for item in current_items:
            item_name_lower = item["name"].lower()
            # Exact substring match
            if name_clean in item_name_lower or item_name_lower in name_clean:
                to_remove.append(item)
                removed_names.append(item["name"])
            # Acronym/abbreviation match: e.g. "opq" matches "OPQ32r"
            elif any(
                word == name_clean or (name_clean in word and len(name_clean) >= 2)
                for word in item_name_lower.replace("-", " ").split()
            ):
                to_remove.append(item)
                removed_names.append(item["name"])
            # Check if the cleaned name matches any key in the item
            elif any(name_clean in key.lower() for key in item.get("keys", [])):
                to_remove.append(item)
                removed_names.append(item["name"])
        for item in to_remove:
            current_items.remove(item)

    # Handle additions
    added_names = []
    for add_spec in mods.get("add", []):
        # Try to find assessment by name/keyword
        add_lower = add_spec.lower()

        # Check for personality test type
        if any(w in add_lower for w in ["personality", "behavioural", "behavioral"]):
            item = catalog.get_by_name("OPQ32r")
            if item and item["url"] not in {i["url"] for i in current_items}:
                current_items.append(item)
                added_names.append(item["name"])

        # Check for cognitive/ability
        elif any(w in add_lower for w in ["cognitive", "reasoning", "ability", "aptitude"]):
            item = catalog.get_by_name("SHL Verify Interactive G+")
            if item and item["url"] not in {i["url"] for i in current_items}:
                current_items.append(item)
                added_names.append(item["name"])

        # Check for SJT
        elif any(w in add_lower for w in ["situational", "judgement", "judgment", "sjt"]):
            item = catalog.get_by_name("Graduate Scenarios")
            if item and item["url"] not in {i["url"] for i in current_items}:
                current_items.append(item)
                added_names.append(item["name"])

        # Check for simulation
        elif any(w in add_lower for w in ["simulation", "practical"]):
            # Find relevant simulation
            sim_results = catalog.search(f"{add_spec} simulation", top_k=3)
            for item in sim_results:
                if "S" in item.get("test_type", "") and item["url"] not in {i["url"] for i in current_items}:
                    current_items.append(item)
                    added_names.append(item["name"])
                    break

        # Check for specific tech
        else:
            tech_items = get_recommendations(query=add_spec, conversation_context=conversation_context)
            for item in tech_items[:2]:
                if isinstance(item, dict) and "url" in item and item["url"] not in {i["url"] for i in current_items}:
                    current_items.append(item)
                    added_names.append(item["name"])

    # Generate reply
    reply_parts = []
    if removed_names:
        reply_parts.append(f"Removed: {', '.join(removed_names)}.")
    if added_names:
        reply_parts.append(f"Added: {', '.join(added_names)}.")
    reply_parts.append("Updated shortlist:")

    reply = " ".join(reply_parts)
    reply += "\n\n" + catalog.format_table(current_items)

    recommendations = [catalog.format_recommendation(item) for item in current_items[:10]]

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False,
    }


def _generate_smart_clarification(query: str, missing: list, context: str, catalog) -> str:
    """Generate a smart clarification response."""
    # Try LLM first
    if _has_llm():
        try:
            client = _get_llm_client()
            prompt = f"""The user said: "{query}"

I need to ask clarifying questions before recommending SHL assessments. Missing information: {', '.join(missing)}.

Available clarification questions:
{json.dumps(CLARIFICATION_QUESTIONS)}

Generate a brief (1-2 sentences), friendly clarification. Ask only the most important 1-2 questions. Be conversational and helpful. Do not mention assessments yet."""

            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback: rule-based clarification
    return generate_clarification_reply(query, missing)


def _generate_recommendation_reply(query: str, items: list, catalog, context: str) -> str:
    """Generate a natural language reply with recommendations."""
    # Try LLM for natural language generation
    if _has_llm():
        try:
            client = _get_llm_client()
            # Build catalog context for the items
            items_context = []
            for item in items[:10]:
                items_context.append(
                    f"- {item['name']}: Type {item['test_type']}, Keys: {', '.join(item.get('keys', []))}, "
                    f"Duration: {item.get('duration', 'N/A')}. "
                    f"Description: {item.get('description', '')[:150]}"
                )

            prompt = f"""Based on the user's query: "{query}"

Context from conversation: "{context[:500]}"

Here are the recommended assessments:
{chr(10).join(items_context)}

Generate a brief (2-4 sentences), conversational reply explaining why these assessments fit the user's needs.
- Be specific about what each assessment measures
- Reference the user's stated requirements
- Keep it under 100 words
- Do NOT include a table or list in your reply (the recommendations are sent separately)
- Just explain the reasoning naturally"""

            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            reply = completion.choices[0].message.content.strip()
            # Add the table
            reply += "\n\n" + catalog.format_table(items[:10])
            return reply
        except Exception:
            pass

    # Fallback: rule-based reply
    return _generate_rule_based_reply(query, items, catalog)


def _generate_rule_based_reply(query: str, items: list, catalog) -> str:
    """Generate a reply using rule-based logic when LLM is unavailable."""
    info = extract_job_description(query)
    role_type = info.get("role_type")

    parts = []

    # Opening based on context
    if role_type:
        role_descriptions = {
            "senior_leader": "For senior leadership roles",
            "developer": "For a technical developer role",
            "safety": "For safety-critical roles",
            "contact_center": "For contact centre screening",
            "graduate": "For graduate-level assessment",
            "finance": "For financial analyst roles",
            "admin": "For administrative positions",
            "sales": "For sales roles",
            "healthcare": "For healthcare positions",
        }
        parts.append(role_descriptions.get(role_type, "Based on your requirements"))

    # Add tech-specific context
    if info.get("technologies"):
        tech_str = ", ".join(info["technologies"])
        parts.append(f"focusing on {tech_str}")

    # Add measurement context
    if info.get("measurement_types"):
        type_names = {
            "A": "cognitive ability",
            "P": "personality",
            "K": "knowledge & skills",
            "B": "situational judgement",
            "S": "simulations",
            "C": "competencies",
            "D": "development",
        }
        meas_str = " and ".join(type_names.get(t, t) for t in info["measurement_types"])
        parts.append(f"measuring {meas_str}")

    # Combine
    if parts:
        reply = " ".join(parts) + ", here are the recommended assessments:"
    else:
        reply = "Here are assessments that match your requirements:"

    reply += "\n\n" + catalog.format_table(items[:10])

    return reply


def _generate_confirmation_reply(prev_recs: list, catalog) -> str:
    """Generate a confirmation reply when user accepts the shortlist."""
    items = []
    for rec in prev_recs:
        item = catalog.get_by_url(rec.get("url", ""))
        if item:
            items.append(item)

    reply = "Shortlist confirmed."
    if items:
        reply += "\n\n" + catalog.format_table(items)
    return reply
