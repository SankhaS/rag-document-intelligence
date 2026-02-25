import json
import pandas as pd
from loguru import logger
from app.retrieval.rag_chain import RAGChain
from app.retrieval.vector_store import VectorStore


def load_test_set(path: str = "tests/test_questions.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_retrieval_precision(
    test_set: list[dict], vector_store: VectorStore, top_k: int = 5
) -> dict:

    correct = 0
    total = 0

    for test in test_set:
        results = vector_store.search(test["question"], top_k=top_k)
        for r in results:
            total += 1
            if r["source_file"] == test.get("expected_source_file"):
                correct += 1

    precision = correct / total if total > 0 else 0
    return {
        "retrieval_precision_at_k": round(precision, 4),
        "correct_retrievals": correct,
        "total_retrievals": total,
    }


def evaluate_answer_faithfulness(
    test_set: list[dict], rag_chain: RAGChain
) -> dict:
    faithful = 0
    hallucinated = 0
    no_answer = 0
    results_log = []

    for test in test_set:
        response = rag_chain.query(test["question"])
        answer_lower = response.answer.lower()

        # Check if expected content is in the answer
        expected = test.get("expected_answer_contains", [])
        found = sum(1 for e in expected if e.lower() in answer_lower)

        if "don't have enough information" in answer_lower:
            no_answer += 1
            status = "no_answer"
        elif found >= len(expected) / 2:  # At least half of expected terms
            faithful += 1
            status = "faithful"
        else:
            hallucinated += 1
            status = "possible_hallucination"

        results_log.append({
            "question": test["question"],
            "answer_preview": response.answer[:200],
            "status": status,
            "confidence": response.confidence,
            "expected_terms_found": f"{found}/{len(expected)}",
        })

    total = len(test_set)
    return {
        "faithfulness_rate": round(faithful / total, 4) if total else 0,
        "hallucination_rate": round(hallucinated / total, 4) if total else 0,
        "no_answer_rate": round(no_answer / total, 4) if total else 0,
        "details": results_log,
    }


def run_full_evaluation(test_set_path: str = "tests/test_questions.json"):
    logger.info("Starting RAG evaluation...")

    test_set = load_test_set(test_set_path)
    logger.info(f"Loaded {len(test_set)} test questions")

    vector_store = VectorStore()
    rag_chain = RAGChain()

    # Retrieval evaluation
    retrieval_results = evaluate_retrieval_precision(test_set, vector_store)
    logger.info(f"Retrieval Precision@5: {retrieval_results['retrieval_precision_at_k']}")

    # Faithfulness evaluation
    faith_results = evaluate_answer_faithfulness(test_set, rag_chain)
    logger.info(f"Faithfulness Rate: {faith_results['faithfulness_rate']}")
    logger.info(f"Hallucination Rate: {faith_results['hallucination_rate']}")

    # Save detailed results
    df = pd.DataFrame(faith_results["details"])
    df.to_csv("tests/evaluation_results.csv", index=False)

    print("\n" + "="*60)
    print("RAG EVALUATION RESULTS")
    print("="*60)
    print(f"Test Questions:         {len(test_set)}")
    print(f"Retrieval Precision@5:  {retrieval_results['retrieval_precision_at_k']:.2%}")
    print(f"Answer Faithfulness:    {faith_results['faithfulness_rate']:.2%}")
    print(f"Hallucination Rate:     {faith_results['hallucination_rate']:.2%}")
    print(f"No Answer Rate:         {faith_results['no_answer_rate']:.2%}")
    print("="*60)
    print("\nDetailed results saved to tests/evaluation_results.csv")


if __name__ == "__main__":
    run_full_evaluation()