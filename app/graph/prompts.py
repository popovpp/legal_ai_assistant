RISK_TAXONOMY_EN = """
Identify risks from the following categories. If a category is absent from the
document, do not invent it.

1. Liability & indemnification — caps, exclusions of consequential damages.
2. Termination — unilateral termination, notice periods, termination for convenience.
3. Payment — late payment interest, non-refundable fees, price escalation.
4. IP ownership — vesting, license scope, restrictions on use.
5. Confidentiality — duration, permitted disclosures, subcontractor access.
6. Data protection — data loss responsibility, breach notification.
7. Governing law & disputes — jurisdiction, arbitration.
8. Warranties — disclaimers, "as is" clauses.
9. Assignment & subcontracting — consent, change of control.
10. Force majeure — scope, notice, termination rights.

Severity rubric:
- high: shifts material risk to one party, caps liability below 6 months of fees,
  allows unilateral termination without cause, or transfers IP ownership.
- medium: limits remedies, imposes one-sided obligations, extends confidentiality
  beyond 5 years.
- low: administrative clauses that slightly favor one party.

Prefer fewer, well-supported risks over many speculative ones. Do not list
standard market-practice clauses as risks.
"""

RISK_TAXONOMY_RU = """
Выяви риски из следующих категорий. Если категория в документе отсутствует —
не выдумывай её.

1. Ответственность и возмещение — ограничения, исключения косвенных убытков.
2. Расторжение — односторонний отказ, сроки уведомления, расторжение без причины.
3. Оплата — пени за просрочку, невозвратные платежи, индексация цены.
4. Интеллектуальная собственность — передача прав, объём лицензии, ограничения.
5. Конфиденциальность — срок, разрешённые раскрытия, доступ подрядчиков.
6. Защита данных — ответственность за утрату данных, уведомление об утечках.
7. Применимое право и споры — подсудность, арбитраж.
8. Гарантии — отказ от гарантий, условия «как есть».
9. Уступка и субподряд — согласие, смена контроля.
10. Форс-мажор — объём, уведомление, право на расторжение.

Рубрика severity:
- high: переносит существенный риск на одну сторону, ограничивает ответственность
  ниже 6 месяцев вознаграждения, допускает одностороннее расторжение без причины
  или передаёт права на ИС.
- medium: ограничивает средства защиты, навязывает односторонние обязанности,
  продлевает конфиденциальность свыше 5 лет.
- low: административные условия с небольшим перевесом в пользу одной стороны.

Предпочитай меньше рисков, но хорошо обоснованных. Не указывай стандартные
рыночные условия как риски.
"""

ANALYST_SYSTEM = {
    "en": (
        "You are a legal document analyst. Answer strictly from the provided fragments. "
        "Every non-trivial statement MUST be supported by a citation whose `quote` is a "
        "VERBATIM substring of the cited fragment. "
        "Only use chunk_ids that appear in the fragments. Do not invent chunk_ids. "
        "If the fragments do not contain the answer, set `insufficient_evidence=true` and "
        "answer with 'Insufficient evidence in the provided document.'"
    ),
    "ru": (
        "Ты — аналитик юридических документов. Отвечай строго на основе предоставленных "
        "фрагментов. Каждое нетривиальное утверждение ДОЛЖНО быть подкреплено цитатой, "
        "`quote` которой является ДОСЛОВНОЙ подстрокой цитируемого фрагмента. "
        "Используй только те chunk_id, которые есть во фрагментах. Не выдумывай chunk_id. "
        "Если во фрагментах нет ответа, установи `insufficient_evidence=true` и ответь "
        "'В предоставленном документе недостаточно оснований.'"
    ),
}

MODE_INSTRUCTIONS = {
    "en": {
        "qa": "Answer the question based only on the fragments.",
        "summary": "Summarize the document: key obligations, parties, terms, and dates.",
        "risk": RISK_TAXONOMY_EN,
    },
    "ru": {
        "qa": "Ответь на вопрос только на основе фрагментов.",
        "summary": "Суммируй документ: ключевые обязательства, стороны, сроки и даты.",
        "risk": RISK_TAXONOMY_RU,
    },
}


def build_analyst_prompt(query, mode, chunks, language="en", feedback=None):
    context = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | page={c['page']} | score={c['score']:.3f}]\n{c['text']}" for c in chunks
    )
    mode_instr = MODE_INSTRUCTIONS[language].get(mode, MODE_INSTRUCTIONS[language]["qa"])

    fb = ""
    if feedback:
        fb_en = (
            "\n\nPREVIOUS ATTEMPT FAILED VERIFICATION. "
            f"These chunk_ids had unsupported quotes: {feedback}. "
            "Re-check that each quote is a VERBATIM substring of the cited chunk. "
            "Prefer shorter exact quotes."
        )
        fb_ru = (
            "\n\nПРЕДЫДУЩАЯ ПОПЫТКА НЕ ПРОШЛА ВЕРИФИКАЦИЮ. "
            f"Для этих chunk_id цитаты не подтверждены: {feedback}. "
            "Проверь, что каждая цитата — ДОСЛОВНАЯ подстрока цитируемого фрагмента. "
            "Предпочитай короткие точные цитаты."
        )
        fb = fb_ru if language == "ru" else fb_en

    header = "Вопрос / инструкция:" if language == "ru" else "Question / instruction:"
    frags = "Фрагменты документа:" if language == "ru" else "Document fragments:"
    tail = (
        "Верни JSON с полями: answer, insufficient_evidence, citations, risks."
        if language == "ru"
        else "Return JSON with fields: answer, insufficient_evidence, citations, risks."
    )
    return f"{mode_instr}\n\n{header}\n{query}\n\n{frags}\n{context}{fb}\n\n{tail}"
