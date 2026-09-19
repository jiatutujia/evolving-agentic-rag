from dataclasses import dataclass


@dataclass
class RouteDecision:
    route: str
    score: int
    reasons: list[str]


class QueryRouter:
    """
    Decide whether a query should be used directly
    or rewritten before retrieval.

    Routes:
        direct
        rewrite
    """

    def __init__(self) -> None:
        self.filler_phrases = [
            "i want to know",
            "i'd like to know",
            "can you tell me",
            "could you tell me",
            "i'm trying to understand",
            "i am trying to understand",
            "i don't really understand",
            "i do not really understand",
            "basically",
            "you know",
            "please explain",
            "can you explain",
        ]

        self.question_words = {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
        }

    def route(
        self,
        query: str,
    ) -> RouteDecision:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        normalized = query.strip().lower()

        words = normalized.replace(
            "?",
            "",
        ).split()

        score = 0
        reasons = []

        # --------------------------------------------------
        # 1. Conversational filler
        # --------------------------------------------------
        matched_fillers = [
            phrase
            for phrase in self.filler_phrases
            if phrase in normalized
        ]

        if matched_fillers:
            score += 1

            reasons.append(
                "contains conversational filler: "
                + ", ".join(matched_fillers)
            )

        # --------------------------------------------------
        # 2. Very long query
        # --------------------------------------------------
        if len(words) > 18:
            score += 1

            reasons.append(
                f"query is long ({len(words)} words)"
            )

        # --------------------------------------------------
        # 3. Question word occurs in an unusual position
        #
        # Example:
        # DSPy teleprompter how work?
        # --------------------------------------------------
        if words:
            first_word = words[0]

            contains_question_word = any(
                word in self.question_words
                for word in words
            )

            if (
                contains_question_word
                and first_word not in self.question_words
            ):
                score += 1

                reasons.append(
                    "question structure may be malformed"
                )

        # --------------------------------------------------
        # Final routing decision
        # --------------------------------------------------
        if score >= 1:
            route = "rewrite"
        else:
            route = "direct"

        if not reasons:
            reasons.append(
                "query is already concise and clear"
            )

        return RouteDecision(
            route=route,
            score=score,
            reasons=reasons,
        )