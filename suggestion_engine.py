def get_suggestions(state: str, confidence: float) -> list[str]:
    if state == "tired":
        suggestions = [
            "Enable Focus Mode",
            "Reduce notifications for 1 hour",
            "Switch UI to warm tone",
            "Suggest calm background music",
        ]
    elif state == "stressed":
        suggestions = [
            "Enable reduced interruptions",
            "Suggest a short breathing session",
            "Bundle non-urgent notifications",
            "Switch to comfort mode UI",
        ]
    elif state == "active":
        suggestions = [
            "Suggest upbeat music",
            "Switch to bright UI mode",
            "Recommend activity mode",
            "Keep notifications active",
        ]
    else:
        suggestions = [
            "Keep default settings",
            "Offer optional smart suggestions",
            "Maintain current UI mode",
        ]

    # confidence 낮으면 문구 추가
    if confidence < 0.6:
        suggestions.insert(0, "Low-confidence inference: show optional suggestions only")

    return suggestions