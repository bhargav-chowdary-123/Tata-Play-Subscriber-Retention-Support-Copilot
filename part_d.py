"""Part D: classification, extraction, retrieval and policy answering."""

import os
import re

import chromadb
import mock_llm
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional

from llm_client import call_llm, reset_stats, stats

POLICY_DIR = "data/policies"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
REJECTION_THRESHOLD = 0.32

LABELS = ["billing", "signal", "plan_change", "cancellation"]


class TicketRecord(BaseModel):
    """Validated structured extraction from a support ticket."""

    subscriber_id: Optional[str] = None
    category: Literal[
        "billing", "signal", "plan_change", "cancellation", "other"
    ]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool


def llm_step(prompt_value) -> str:
    """Convert a LangChain prompt value into a mock-compatible API call."""
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    messages = [
        {"role": role_map[message.type], "content": message.content}
        for message in prompt_value.to_messages()
    ]
    return call_llm(messages)


def build_classifier_chain(system_prompt: str):
    """Return a prompt, function and parser chain for ticket classification."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{ticket}"),
        ]
    )
    return prompt | llm_step | StrOutputParser()


def run_classification() -> pd.DataFrame:
    """Classify readable tickets with weak, zero-shot and few-shot prompts."""
    tickets = pd.read_csv("data/raw/tickets.csv")
    tickets = tickets[tickets["text"] != "..."].copy()
    examples = pd.read_csv("data/raw/few_shot_examples.csv")

    weak_prompt = "TICKET CLASSIFIER. Classify the ticket."

    zero_shot_prompt = """TICKET CLASSIFIER.
You are a support-ticket classifier.
Classify each customer ticket into exactly one of these labels:
billing, signal, plan_change, cancellation.
Respond with only the label.
Do not create new labels."""

    few_shot_lines = []
    for _, row in examples.iterrows():
        few_shot_lines.append(f"Ticket: {row['text']}\nLabel: {row['label']}")
    few_shot_prompt = zero_shot_prompt + "\n\n" + "\n\n".join(few_shot_lines)

    results = []

    for prompt_name, system_prompt in [
        ("weak", weak_prompt),
        ("zero_shot", zero_shot_prompt),
        ("few_shot", few_shot_prompt),
    ]:
        reset_stats()
        mock_llm.reset()
        chain = build_classifier_chain(system_prompt)

        predictions = []
        for ticket in tickets["text"]:
            prediction = chain.invoke({"ticket": ticket}).strip()
            predictions.append(prediction)

        valid = [prediction in LABELS for prediction in predictions]
        correct = [
            prediction == true
            for prediction, true in zip(predictions, tickets["true_category"])
        ]

        print(
            f"\n{prompt_name}: valid labels {sum(valid)}/{len(tickets)}, "
            f"correct {sum(correct)}/{len(tickets)}, "
            f"accuracy {sum(correct) / len(tickets):.2%}, "
            f"recovered rate limits {stats()}"
        )

        if prompt_name in ("zero_shot", "few_shot"):
            results.append(predictions)

    output = pd.DataFrame(
        {
            "ticket_id": tickets["ticket_id"].values,
            "true_category": tickets["true_category"].values,
            "zero_shot": results[0],
            "few_shot": results[1],
        }
    )
    output.to_csv("outputs/ticket_classification.csv", index=False)

    wrong = output[output["true_category"] != output["few_shot"]]
    print("\nfew-shot mistakes:")
    print(wrong.to_string(index=False))

    return output


def demonstrate_pydantic() -> None:
    """Show coercion and validation failures for TicketRecord."""
    print("\n=== PYDANTIC DEMONSTRATION ===")

    record = TicketRecord(
        subscriber_id="TP00001",
        category="billing",
        urgency="3",
        wants_refund=True,
    )
    print("string urgency coerced to:", record.urgency)

    try:
        TicketRecord(
            subscriber_id="TP00001",
            category="billing",
            urgency=9,
            wants_refund=False,
        )
    except ValidationError as error:
        print("urgency=9 error field:", error.errors()[0]["loc"])
        print("urgency=9 error type:", error.errors()[0]["type"])

    try:
        TicketRecord(
            subscriber_id="TP00001",
            category="refund",
            urgency=3,
            wants_refund=False,
        )
    except ValidationError as error:
        print("category='refund' error field:", error.errors()[0]["loc"])
        print("category='refund' error type:", error.errors()[0]["type"])


def run_extraction() -> None:
    """Extract structured fields from every ticket, retrying invalid output."""
    from langchain_core.output_parsers import PydanticOutputParser

    tickets = pd.read_csv("data/raw/tickets.csv")
    parser = PydanticOutputParser(pydantic_object=TicketRecord)

    system_prompt = (
        "TICKET EXTRACTOR.\n"
        "Extract the requested fields from the ticket.\n"
        f"{parser.get_format_instructions()}"
    )

    mock_llm.reset()
    reset_stats()

    output = []

    for row in tickets.itertuples(index=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row.text},
        ]

        attempts = 0
        record = None
        status = "manual_review"

        for attempt in range(3):
            attempts += 1
            reply = call_llm(messages)
            try:
                record = parser.parse(reply)
                status = "valid"
                break
            except (OutputParserException, ValidationError) as error:
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": f"VALIDATION ERROR: {error}",
                    }
                )

        output.append(
            {
                "ticket_id": row.ticket_id,
                "status": status,
                "attempts": attempts,
                "record": record.model_dump_json() if record else None,
            }
        )

    with open("outputs/ticket_extractions.jsonl", "w", encoding="utf-8") as f:
        for item in output:
            f.write(
                json_dumps(
                    {
                        "ticket_id": item["ticket_id"],
                        "status": item["status"],
                        "attempts": item["attempts"],
                        "record": item["record"],
                    }
                )
                + "\n"
            )

    status_counts = pd.Series([item["status"] for item in output]).value_counts()
    print("\n=== EXTRACTION OUTCOMES ===")
    print(status_counts.to_string())
    print("recovered rate limits:", stats())


def json_dumps(data) -> str:
    """Return compact JSON text for a JSONL line."""
    import json

    return json.dumps(data, ensure_ascii=False)


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list:
    """Split policy text into overlapping character chunks."""
    chunks = []
    start = 0

    while start + overlap < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap

    if not chunks:
        chunks.append(text)

    return chunks


def build_collection(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Create an in-memory Chroma collection from the policy files."""
    client = chromadb.Client()
    embedding = embedding_functions.ONNXMiniLM_L6_V2()

    collection = client.get_or_create_collection(
        name=f"policies_{chunk_size}_{chunk_overlap}",
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadata = []
    ids = []

    for file_name in sorted(os.listdir(POLICY_DIR)):
        if not file_name.endswith(".txt"):
            continue

        with open(
            os.path.join(POLICY_DIR, file_name), "r", encoding="utf-8"
        ) as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size, chunk_overlap)

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({"source": file_name, "chunk": index})
            ids.append(f"{file_name}_{index}")

    collection.add(documents=documents, metadatas=metadata, ids=ids)
    return collection


def retrieve(collection, query: str, k: int = 3) -> list:
    """Return the top-k policy chunks for a query."""
    result = collection.query(query_texts=[query], n_results=k)
    output = []

    for source, distance, text in zip(
        result["metadatas"][0],
        result["distances"][0],
        result["documents"][0],
    ):
        output.append(
            {
                "source": source["source"],
                "similarity": 1 - distance,
                "text": text,
            }
        )

    return output


def keyword_baseline(question: str, policy_files: list) -> str:
    """Return the policy file with the most shared non-stopword terms."""
    question_words = set(re.findall(r"[a-z]+", question.lower())) - mock_llm.STOP
    scores = {}

    for file_name in policy_files:
        with open(
            os.path.join(POLICY_DIR, file_name), "r", encoding="utf-8"
        ) as f:
            text = f.read().lower()

        words = set(re.findall(r"[a-z]+", text))
        scores[file_name] = len(question_words & words)

    return sorted(scores, key=lambda name: (-scores[name], name))[0]


def answer_question(collection, question: str, grounded: bool = True) -> dict:
    """Answer a policy question using retrieved context when grounded."""
    retrieved = retrieve(collection, question, k=3)
    best = retrieved[0]

    if grounded and best["similarity"] < REJECTION_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    if grounded:
        system_prompt = (
            "POLICY ASSISTANT.\n"
            "Answer using only the context below.\n"
            "If the context does not contain the answer, say that you do not "
            "have enough information."
        )
        context = "\n\n".join(
            f"[{item['source']}]\n{item['text']}" for item in retrieved
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    else:
        system_prompt = "POLICY ASSISTANT. Answer the question."
        user_prompt = question

    response = call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "answer": response,
        "sources": [item["source"] for item in retrieved],
    }


def run_rag() -> None:
    """Evaluate retrieval, grounding and the smaller-chunk configuration."""
    evaluation = pd.read_csv("data/raw/policy_eval.csv")
    policy_files = sorted(
        name for name in os.listdir(POLICY_DIR) if name.endswith(".txt")
    )

    collection = build_collection()

    print("\n=== CHUNK COUNTS ===")
    for file_name in policy_files:
        with open(
            os.path.join(POLICY_DIR, file_name), "r", encoding="utf-8"
        ) as f:
            text = f.read()
        print(file_name, len(chunk_text(text)))

    rows = []
    answerable_similarities = []
    unanswerable_similarities = []
    top_three_hits = 0
    top_one_hits = 0
    grounded_fact_hits = 0

    for row in evaluation.itertuples(index=False):
        retrieved = retrieve(collection, row.question, k=3)
        top = retrieved[0]
        sources = [item["source"] for item in retrieved]

        hit_at_three = row.expected_file in sources
        top_one = sources[0] == row.expected_file

        if pd.notna(row.expected_file):
            top_three_hits += int(hit_at_three)
            top_one_hits += int(top_one)
            answerable_similarities.append(top["similarity"])
        else:
            unanswerable_similarities.append(top["similarity"])

        grounded = answer_question(collection, row.question, grounded=True)
        ungrounded = answer_question(collection, row.question, grounded=False)

        if pd.notna(row.expected_fact) and row.expected_fact in grounded["answer"]:
            grounded_fact_hits += 1

        rows.append(
            {
                "question": row.question,
                "expected_file": row.expected_file,
                "top1_source": top["source"],
                "top1_similarity": top["similarity"],
                "hit_at_3": hit_at_three,
                "grounded_answer": grounded["answer"],
                "ungrounded_answer": ungrounded["answer"],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv("outputs/rag_eval.csv", index=False)

    print("\n=== RETRIEVAL EVALUATION ===")
    print("top-3 file hits:", top_three_hits, "of 12")
    print("top-1 file hits:", top_one_hits, "of 12")
    print("answerable top-1 similarity range:",
          min(answerable_similarities), max(answerable_similarities))
    print("unanswerable top-1 similarities:", unanswerable_similarities)

    baseline_correct = 0
    answerable = evaluation[evaluation["expected_file"].notna()]

    for row in answerable.itertuples(index=False):
        baseline_correct += int(
            keyword_baseline(row.question, policy_files) == row.expected_file
        )

    print("keyword baseline top-1 correct:", baseline_correct, "of 12")
    print("grounded answers containing expected fact:", grounded_fact_hits, "of 12")

    small_collection = build_collection(chunk_size=200, chunk_overlap=30)
    small_fact_hits = 0
    small_top_three_hits = 0

    for row in answerable.itertuples(index=False):
        retrieved = retrieve(small_collection, row.question, k=3)
        context = "\n".join(item["text"] for item in retrieved)
        small_top_three_hits += int(row.expected_fact in context)

        grounded = answer_question(small_collection, row.question, grounded=True)
        small_fact_hits += int(row.expected_fact in grounded["answer"])

    print("\n=== SMALL CHUNK EXPERIMENT ===")
    print("small chunks:", len(small_collection.get()["ids"]))
    print("expected fact in top-3 chunks:", small_top_three_hits, "of 12")
    print("grounded answers containing expected fact:", small_fact_hits, "of 12")


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    run_classification()
    demonstrate_pydantic()
    run_extraction()
    run_rag()

    print("\nPart D complete.")


if __name__ == "__main__":
    main()
